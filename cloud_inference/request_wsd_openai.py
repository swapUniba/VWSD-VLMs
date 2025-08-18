from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

import os
import json
import time
import base64
import argparse


def main(model_id, ds_id):

    img_path = "../data/imgs_benchmark/images_ours/" if "original" not in ds_id else "../data/imgs_benchmark/images_original/"
    add_png = True if "original" not in ds_id else False

    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )

    ds = load_dataset("json", data_files={"train": f"../data/test_qwen/{ds_id}.jsonl"})["train"]

    def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    def convert_to_conversation(sample):
        user_dict = { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : sample["instruction"]}]
            }

        for i, img in enumerate(sample["options"]):
            
            if add_png:
                encoded_img = encode_image(img_path + img.split("/")[-1] + ".png")
            else:
                encoded_img = encode_image(img_path + img.split("/")[-1])
            user_dict["content"].append({"type": "text", "text": f"\n{str(i+1)}) "})
            user_dict["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_img}",
                }}
            )
        
        return user_dict
    
    os.makedirs(f"./{model_id}", exist_ok=True)

    with open(f"./{model_id}/{ds_id}_responses.jsonl", "w", encoding="utf8") as f: 

        for x in tqdm(ds):

            messages = [{"role": "system", "content": "You are a helpful assistant."}, convert_to_conversation(x)]

            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
                store=False,
                seed=42,
                temperature=0,
                n=1,
                top_p=1,
                extra_body={
                    "top_k": 1
                },
                presence_penalty=0.0,
                frequency_penalty=0.0,
                max_tokens=128
            )

            response = ""

            for chunk in stream:

                try:
                    response += chunk.choices[0].delta.content
                except Exception as e:
                    response += ""
            
            x["generated_output"] = response

            json.dump(x, f)
            f.write('\n')
            
            time.sleep(2)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-d', '--ds_name')
    args = parser.parse_args()
    
    model_name = args.model_name
    ds_name = args.ds_name

    main(model_name, ds_name)
