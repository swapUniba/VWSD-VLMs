from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset, Dataset
import base64

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

    load_in_4bit = True if "12b" in model_id else False

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", attn_implementation="eager", load_in_4bit=load_in_4bit, torch_dtype=torch.bfloat16, cache_dir="./cache"
    ).eval()

    ds = load_dataset("json", data_files={"train": f"./data/test_qwen/{ds_id}.jsonl"})["train"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def convert_to_conversation(sample):
        user_dict = { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : sample["instruction"]}]
            }
        
        for i, img in enumerate(sample["options"]):
            user_dict["content"].append({"type": "text", "text": f"\n{str(i+1)}) "})

            if add_png:
                user_dict["content"].append({"type": "image", "url": "data:image/jpeg;base64," + encode_image(img_path + img.split("/")[-1] + ".png")})
            else:
                user_dict["content"].append({"type": "image", "url": "data:image/jpeg;base64," + encode_image(img_path + img.split("/")[-1])})
        
        return user_dict, sample["options"]

    batch_image = []
    batch_text = []

    os.makedirs(f"./reports_unsloth/{model_id}", exist_ok=True)

    with open(f"./reports_unsloth/{model_id}/{ds_id}_responses.jsonl", "w", encoding="utf8") as f: 

        with torch.no_grad():

            for x in tqdm(ds):

                txt, imgs = convert_to_conversation(x)

                batch_image.append(imgs)
                batch_text.append(txt)

                if len(batch_image) == 1:

                    inputs = processor.apply_chat_template(batch_text, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

                    set_seed(42)
                    generated_ids = model.generate(**inputs, do_sample=False, num_beams=1, temperature=0, max_new_tokens=8)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.tokenizer.batch_decode(
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
