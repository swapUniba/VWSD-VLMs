import os
import json
import torch
import random
import argparse
import open_clip

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from multilingual_clip import pt_multilingual_clip

from src.eval.utils import compute_metrics


def main(bs, lang_code, negative_type = "negative"):

    set_seed(42)

    print("-" * 16)
    print(lang_code)
    print(negative_type)

    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", cache_dir="./cache").eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", cache_dir="./cache")

    image_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    image_model = image_model.to("cuda:0").eval()

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
                txt_batch.append(txt)

            if len(img_batch) >= bs:

                txt_tok = tokenizer(txt_batch, padding=True, return_tensors='pt').to(text_model.device)
                embs = text_model.transformer(**txt_tok)[0]
                att = txt_tok['attention_mask']
                embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
                output_txt_embeds = text_model.LinearTransformation(embs)

                for img in img_batch:
                    image = preprocess(img).unsqueeze(0).to(text_model.device)
                    image_features = image_model.encode_image(image)
                    img_embs.append(image_features)

                txt_embs.append(output_txt_embeds)
                img_batch = []
                txt_batch = []

        if len(img_batch) != 0:

            txt_tok = tokenizer(txt_batch, padding=True, return_tensors='pt').to(text_model.device)
            embs = text_model.transformer(**txt_tok)[0]
            att = txt_tok['attention_mask']
            embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
            output_txt_embeds = text_model.LinearTransformation(embs)

            for img in img_batch:
                image = preprocess(img).unsqueeze(0).to(text_model.device)
                image_features = image_model.encode_image(image)
                img_embs.append(image_features)

            txt_embs.append(output_txt_embeds)
            img_batch = []
            txt_batch = []

        img_embs = torch.vstack(img_embs)
        txt_embs = torch.vstack(txt_embs)

        score_hit = 0
        score_mrr = 0
        total = 0

        os.makedirs(f"results/mclip/", exist_ok=True)

        with open(f"results/mclip/{lang_code}_{negative_type}.jsonl", "w", encoding="utf8") as f:

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

                    json.dump({"positive_txt": ds[i]["txt"][j], "id": ds[i]["ids"][j], "ids": all_ids, "ranking": co_indices.tolist(), "hit@1": hit_1, "mrr": mrr}, f)
                    f.write('\n')
        
        print(f"HIT@1: {score_hit / total}")
        print(f"MRR: {score_mrr / total}")
        print("-" * 16)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--mini_batch_size', type=int)
    parser.add_argument('-l', '--lang_code')
    parser.add_argument('-ne', '--negative_type')
    args = parser.parse_args()
    
    bs= args.mini_batch_size
    lang_code = args.lang_code
    negative_type = args.negative_type

    main(bs, lang_code, negative_type)

