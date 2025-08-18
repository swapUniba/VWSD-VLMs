import os
import json
import torch
import random
import argparse
from contextlib import nullcontext

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, set_seed

from src.eval.utils import compute_metrics


def main(model_name, bs, lang_code, negative_type = "negative", report = False):

    set_seed(42)

    print("-" * 16)
    print(model_name)
    print(lang_code)
    print(negative_type)

    model = AutoModel.from_pretrained(model_name, cache_dir="./cache").eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir="./cache")

    is_siglip2 = False

    if "siglip2" in model_name:
        is_siglip2 = True
    
    print(is_siglip2)

    ds = load_dataset("json", data_files={"train": f"./data/test/glosses_{lang_code}.jsonl"})["train"]

    ds_map = {}

    for i, x in enumerate(ds):

        for j, img in enumerate(x["img"]):
            ds_map[img] = (i, j)

    print(ds[0])

    img_embs = []
    txt_embs = []

    img_batch = []
    txt_batch = []

    with torch.no_grad():

        for x in tqdm(ds):

            for img, txt in zip(x["img"], x["txt"]):

                img_batch.append(Image.open(img).convert("RGB"))

                if is_siglip2:
                    txt_batch.append(txt.lower())
                else:
                    txt_batch.append(txt)

            if len(img_batch) >= bs:

                inputs = processor(text=txt_batch, images=img_batch, truncation=True, padding="max_length", max_length=64, return_tensors="pt").to("cuda:0")
                outputs = model(**inputs)

                img_embs.append(outputs.image_embeds)
                txt_embs.append(outputs.text_embeds)
                img_batch = []
                txt_batch = []

        if len(img_batch) != 0:

            inputs = processor(text=txt_batch, images=img_batch, truncation=True, padding="max_length", max_length=64, return_tensors="pt").to("cuda:0")
            outputs = model(**inputs)

            img_embs.append(outputs.image_embeds)
            txt_embs.append(outputs.text_embeds)
            img_batch = []
            txt_batch = []

        img_embs = torch.vstack(img_embs)
        txt_embs = torch.vstack(txt_embs)

        score_hit = 0
        score_mrr = 0
        total = 0

        if report:
            os.makedirs(f"results/{model_name}/", exist_ok=True)

        with nullcontext() if not report else open(f"results/{model_name}/{lang_code}_{negative_type}.jsonl", "w", encoding="utf8") as f:

            for i in range(len(ds)):
                
                for j in range(10):

                    all_ids = []
                
                    if negative_type == "negative":
                        targets_imgs_idxs = list([i + a for a in range(10) if a != j])
                        all_ids = [ds[i]["img"][a] for a in range(10)]
                    else:
                        target_ids = set(ds[i]["img"])
                        all_ids_no_target = [a for a in ds_map.keys() if a not in target_ids]
                        random.shuffle(all_ids_no_target)
                        all_ids_no_target = all_ids_no_target[:9]
                        targets_imgs_idxs = []
                        all_ids.append(ds[i]["img"][j])
                        for a in all_ids_no_target:
                            idx_main, idx_sub = ds_map[a]
                            targets_imgs_idxs.append(idx_main + idx_sub)
                            all_ids.append(ds[idx_main]["img"][idx_sub])
                    
                    assert len(all_ids) == len(set(all_ids))
                
                    targets_imgs_ids = ds[i]["img"]
                    assert len(targets_imgs_ids) == len(set(targets_imgs_ids))

                    targets_imgs_idxs_with_query = [i + j] + targets_imgs_idxs
                    targets_imgs = img_embs[targets_imgs_idxs_with_query]

                    co = torch.nn.functional.cosine_similarity(txt_embs[i + j], targets_imgs)
                    co, co_indices = torch.sort(co, descending=True)
                    
                    hit_1, mrr = compute_metrics(0, co_indices)
                    
                    score_hit += hit_1
                    score_mrr += mrr
                    total += 1

                    if report:
                        json.dump({"positive_txt": ds[i]["txt"][j], "id": ds[i]["ids"][j], "ids": all_ids, "ranking": co_indices.tolist(), "hit@1": hit_1, "mrr": mrr}, f)
                        f.write('\n')
        
        print(f"HIT@1: {score_hit / total}")
        print(f"MRR: {score_mrr / total}")
        print("-" * 16)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-bs', '--mini_batch_size', type=int)
    parser.add_argument('-l', '--lang_code')
    parser.add_argument('-ne', '--negative_type')
    parser.add_argument('-r', '--report', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    model_name = args.model_name
    bs= args.mini_batch_size
    lang_code = args.lang_code
    negative_type = args.negative_type
    report = args.report

    main(model_name, bs, lang_code, negative_type, report)

