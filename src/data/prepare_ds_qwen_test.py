import json
import random
from datasets import load_dataset
from transformers import set_seed

lang_complete = {
    "en": "English",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "fa": "Persian"
}

def main():
    
    print("OK")

    for lang in ["en", "it", "fa", "de", "fr", "es"]:

        datasets_to_conc = []

        set_seed(42)

        ds = load_dataset("json", data_files={"train": f"./data/test/glosses_{lang}.jsonl"})["train"]
        
        for x in ds:

            for i, txt in enumerate(x["txt"]):

                instruction = f"Choose the image that represents the {lang_complete[lang]} words \"{txt}\". Generate only the number of the selected option. Options:"

                correct_img = x["img"][i]

                negative_imgs = [y for j, y in enumerate(x["img"]) if i != j]
                imgs = [correct_img] + negative_imgs
                random.shuffle(imgs)

                new_dict = {}
                assert len(set(x["img"])) == len(x["img"])
                new_dict["instruction"] = instruction
                new_dict["options"] = imgs
                new_dict["label"] = imgs.index(correct_img) + 1
                datasets_to_conc.append(new_dict)
            
        with open(f"./data/test_qwen/ds_test_qwen_{lang}_ml.jsonl", "w", encoding="utf8") as f:
            for x in datasets_to_conc:
                json.dump(x, f)
                f.write('\n')


if __name__ == "__main__":
    main()
