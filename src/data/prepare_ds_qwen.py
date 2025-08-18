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

    max_ = {x: 0 for x in lang_complete.keys()}

    set_seed(42)

    datasets_to_conc = []
    
    print("OK")

    for lang in ["en", "it", "fa", "de", "fr", "es"]:
        ds = load_dataset("json", data_files={"train": f"./data/train/glosses_{lang}.jsonl"})["train"]

        instructions = []
        for inst in ds:
            instructions.append(f"Choose the image that represents the {lang_complete[lang]} words \"gold_txt\". Generate only the number of the selected option. Options:")

        ds = ds.add_column("instruction", instructions)
        
        for j, x in enumerate(ds):
            
            new_dict = {}
            new_dict["lang"] = lang
            new_dict["id"] = x["hyp_id"]

            if len(x["img"]) < 10:
                candidates = random.sample([l for l in range(len(ds)) if l != j], k=10-len(x["img"]))
                other_images = x["img"]
                other_texts = x["txt"]
                for a in candidates:
                    other_idx = random.sample([i for i in range(len(ds[a]["img"]))], 1)[0]
                    other_images.append(ds[a]["img"][other_idx])
                    other_texts.append(ds[a]["txt"][other_idx])
            elif len(x["img"]) > 10:
                other_images = [a for a in x["img"]][:10]
                other_texts = [a for a in x["txt"]][:10]
            else:
                other_images = [a for a in x["img"]]
                other_texts = [a for a in x["txt"]]
            
            assert len(other_images) == 10
            assert len(other_texts) == 10

            for gold_img, txt in zip(list(other_images), other_texts):

                inst = ds[j]["instruction"].replace("gold_txt", txt)

                random.shuffle(other_images)
        
                new_dict["img"] = other_images
                new_dict["instruction"] = inst
                new_dict["gold"] = other_images.index(gold_img)
                datasets_to_conc.append(new_dict)
    
    print("OK")
    
    random.shuffle(datasets_to_conc)
        
    with open("ds_train_qwen_ml.jsonl", "w", encoding="utf8") as f:
        for x in datasets_to_conc:
            
            assert len(set(x["img"])) == len(x["img"])

            if x["lang"] != "en":
                if max_[x["lang"]] < 1141:
                    max_[x["lang"]] += 1
                else:
                    continue
            
            json.dump(x, f)
            f.write('\n')
    
    print(len(datasets_to_conc))


if __name__ == "__main__":
    main()
