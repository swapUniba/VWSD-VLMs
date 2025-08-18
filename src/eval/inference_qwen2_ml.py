from unsloth import FastVisionModel # FastLanguageModel for LLMs
import os
import json
import torch
import random
import argparse
import PIL.Image
LANCZOS = PIL.Image.Resampling.LANCZOS
from PIL import Image
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset, Dataset

lang_complete = {
    "en": "English",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "fa": "Persian"
}


def inference(model_id, ds_id):

    img_path = "./data/imgs_benchmark/images_ours/" if "original" not in ds_id else "./data/imgs_benchmark/images_original/"
    add_png = True if "original" not in ds_id else False

    random.seed(42)

    bit8 = True if "32b" in model_id else False

    print(bit8)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit = bit8, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = False, # True or "unsloth" for long context
        cache_dir="./cache"
    )

    FastVisionModel.for_inference(model)

    ds = load_dataset("json", data_files={"train": f"./data/test_qwen/{ds_id}.jsonl"})["train"]

    def convert_to_conversation(sample):
        user_dict = { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : sample["instruction"]}]
            }
        
        for i, img in enumerate(sample["options"]):
            user_dict["content"].append({"type": "text", "text": f"\n{str(i+1)}) "})
            user_dict["content"].append({"type": "image", "image": img})
        
        return tokenizer.apply_chat_template([user_dict], add_generation_prompt = True), sample["options"]

    batch_image = []
    batch_text = []

    os.makedirs(f"./reports_unsloth/{model_id}", exist_ok=True)

    with open(f"./reports_unsloth/{model_id}/{ds_id}_responses.jsonl", "w", encoding="utf8") as f: 

        with torch.no_grad():

            for x in tqdm(ds):

                txt, imgs = convert_to_conversation(x)

                print(txt)

                if add_png:
                    imgs = [Image.open(img_path + img.split("/")[-1] + ".png") for img in imgs]
                else:
                    imgs = [Image.open(img_path + img.split("/")[-1]) for img in imgs]

                batch_image.append(imgs)
                batch_text.append(txt)

                if len(batch_image) == 1:

                    inputs = tokenizer(
                        batch_image,
                        batch_text,
                        add_special_tokens = False,
                        return_tensors = "pt",
                    ).to("cuda")

                    set_seed(42)
                    generated_ids = model.generate(**inputs, do_sample=False, num_beams=1, temperature=0, max_new_tokens=8)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = tokenizer.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
            
                    x["generated_output"] = output_text

                    json.dump(x, f)
                    f.write('\n')

                    batch_image = []
                    batch_text = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-d', '--ds_name')
    args = parser.parse_args()
    
    model_name = args.model_name
    ds_name = args.ds_name

    inference(model_name, ds_name)
