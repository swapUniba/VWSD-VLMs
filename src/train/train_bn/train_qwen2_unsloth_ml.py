from unsloth import FastVisionModel # FastLanguageModel for LLMs
from datasets import load_dataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import set_seed
import PIL.Image
LANCZOS = PIL.Image.Resampling.LANCZOS


lang_complete = {
    "en": "English",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "fa": "Persian"
}


def train():

    set_seed(42)

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct",
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = False, # True or "unsloth" for long context
        cache_dir="./cache"
    )

    model = FastVisionModel.get_peft_model(
        model,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",], # Add for continual pretraining
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 8,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 42,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    FastVisionModel.for_training(model)

    ds = load_dataset("json", data_files={"train": f"./ds_train_qwen_ml.jsonl"})["train"]

    def convert_to_conversation(sample):
        user_dict = { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : sample["instruction"]}]
            }

        for i, img in enumerate(sample["img"]):
            user_dict["content"].append({"type": "text", "text": f"\n{str(i+1)}) "})
            user_dict["content"].append({"type": "image", "image": img})
        
        conversation = [
            user_dict,
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : str(sample["gold"] + 1)} ]
            },
        ]

        return { "messages" : conversation }

    ds = ds.shuffle(42)
    ds = [convert_to_conversation(x) for x in ds]

    print(len(ds))
    print(ds[0])

    coll = UnslothVisionDataCollator(model, tokenizer, resize=(336, 336), max_seq_length=16384)

    print(coll.max_seq_length)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = coll, # Must use!
        train_dataset = ds,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 64,
            num_train_epochs=1, # Set this instead of max_steps for full training runs
            learning_rate = 1e-5,
            fp16 = False,
            bf16 = True,
            logging_steps = 1,
            optim = "adamw_torch_fused",
            weight_decay = 0.1,
            lr_scheduler_type = "cosine",
            warmup_ratio=0.03,
            save_strategy= "epoch",
            save_steps=100,
            seed = 42,
            output_dir = "vwsd_qwen",
            report_to = "none",     # For Weights and Biases
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 65536,
        ),
    )

    trainer.train()
    trainer.save_model("vwsd_qwen")


if __name__ == "__main__":
    train()