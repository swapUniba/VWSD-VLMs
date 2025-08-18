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

    for lang in ["en", "it", "fa"]:

        datasets_to_conc = []

        set_seed(42)

        ds = load_dataset("json", data_files={"train": f"./data/semeval-2023-task-1-V-WSD-train-v1/{lang}.json"})["train"]

        instructions = []
        options = []
        golds = []
        gold_img = []
        gold_txt = []

        for inst in ds:

            instruction = f"Choose the image that represents the {lang_complete[lang]} words \"{inst["target_text"]}\". Generate only the number of the selected option. Options:"
            
            imgs = ["./data/semeval-2023-task-1-V-WSD-train-v1/test_v1/test_images_v1/" + x for x in inst["images"]]
            gold = inst["images"].index(inst["gold"]) + 1

            instructions.append(instruction)
            options.append(imgs)
            golds.append(gold)
            gold_img.append(inst["gold"])
            gold_txt.append(inst["target_text"])
        
        for inst, opt, gold, gold_t, gold_i in zip(instructions, options, golds, gold_txt, gold_img):
            
            new_dict = {}
            new_dict["instruction"] = inst
            new_dict["options"] = opt
            new_dict["label"] = gold
            new_dict["label_txt"] = gold_t
            new_dict["label_img"] = gold_i
            datasets_to_conc.append(new_dict)
            
        with open(f"./data/test_qwen/ds_test_qwen_{lang}_original_ml.jsonl", "w", encoding="utf8") as f:
            for x in datasets_to_conc:
                json.dump(x, f)
                f.write('\n')


if __name__ == "__main__":
    main()
