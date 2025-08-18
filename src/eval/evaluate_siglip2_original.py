import os
import json
import torch
import argparse
from contextlib import nullcontext

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, set_seed

from src.eval.utils import compute_metrics


def main(model_name, lang_code, report):

    set_seed(42)

    print("-" * 16)
    print(model_name)
    print(lang_code)

    model = AutoModel.from_pretrained(model_name, cache_dir="./cache").eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir="./cache")

    is_siglip2 = False

    if "siglip2" in model_name:
        is_siglip2 = True
    
    print(is_siglip2)

    ds = load_dataset("json", data_files={"train": f"./data/semeval-2023-task-1-V-WSD-train-v1/{lang_code}.json"})["train"]

    with torch.no_grad():

        score_hit = 0
        score_mrr = 0
        total = 0

        if report:
            os.makedirs(f"results_original/{model_name}/", exist_ok=True)

        with nullcontext() if not report else open(f"results_original/{model_name}/{lang_code}.jsonl", "w", encoding="utf8") as f:

            for x in tqdm(ds):

                gold_idx = x["images"].index(x["gold"])

                if is_siglip2:
                    tgt_text = x["target_text"].lower()
                else:
                    tgt_text = x["target_text"]

                inputs = processor(text=[tgt_text], images=[Image.open("./data/semeval-2023-task-1-V-WSD-train-v1/test_v1/test_images_v1/" + y).convert("RGB") for y in x["images"]], truncation=True, padding="max_length", max_length=64, return_tensors="pt").to("cuda:0")
                outputs = model(**inputs)

                img_embs = outputs.image_embeds
                txt_embs = outputs.text_embeds

                co = torch.nn.functional.cosine_similarity(txt_embs, img_embs)
                co, co_indices = torch.sort(co, descending=True)
                
                hit_1, mrr = compute_metrics(gold_idx, co_indices)
                    
                score_hit += hit_1
                score_mrr += mrr
                total += 1

                if report:
                    json.dump({"positive_txt": x["target_text"], "gold": x["gold"], "images": x["images"], "ranking": co_indices.tolist(), "hit@1": hit_1, "mrr": mrr}, f)
                    f.write('\n')
        
    print(f"HIT@1: {score_hit / total}")
    print(f"MRR: {score_mrr / total}")
    print("-" * 16)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-l', '--lang_code')
    parser.add_argument('-r', '--report', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    model_name = args.model_name
    lang_code = args.lang_code
    report = args.report

    main(model_name, lang_code, report)

