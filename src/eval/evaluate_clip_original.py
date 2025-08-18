import os
import json
import torch
import argparse
import open_clip

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from contextlib import nullcontext
from transformers import AutoTokenizer, set_seed
from multilingual_clip import pt_multilingual_clip

from src.eval.utils import compute_metrics


def main(lang_code, report):

    set_seed(42)

    print("-" * 16)
    print(lang_code)

    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", cache_dir="./cache").eval().to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", cache_dir="./cache")

    image_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    image_model = image_model.to("cuda:0").eval()

    ds = load_dataset("json", data_files={"train": f"./data/semeval-2023-task-1-V-WSD-train-v1/{lang_code}.json"})["train"]

    with torch.no_grad():

        score_hit = 0
        score_mrr = 0
        total = 0

        if report:
            os.makedirs(f"results_original/mclip/", exist_ok=True)

        with nullcontext() if not report else open(f"results_original/mclip/{lang_code}.jsonl", "w", encoding="utf8") as f:

            for x in tqdm(ds):

                gold_idx = x["images"].index(x["gold"])
                tgt_text = x["target_text"]

                txt_tok = tokenizer(tgt_text, padding=True, return_tensors='pt').to(text_model.device)
                embs = text_model.transformer(**txt_tok)[0]
                att = txt_tok['attention_mask']
                embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
                txt_embs = text_model.LinearTransformation(embs)

                img_embs = []
                for img in [Image.open("./data/semeval-2023-task-1-V-WSD-train-v1/test_v1/test_images_v1/" + y).convert("RGB") for y in x["images"]]:
                    image = preprocess(img).unsqueeze(0).to(text_model.device)
                    image_features = image_model.encode_image(image)
                    img_embs.append(image_features)

                img_embs = torch.vstack(img_embs)

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
    parser.add_argument('-l', '--lang_code')
    parser.add_argument('-r', '--report', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    lang_code = args.lang_code
    report = args.report

    main(lang_code, report)

