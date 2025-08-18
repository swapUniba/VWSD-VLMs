import os
import json
import random

from tqdm import tqdm

img_path_dict = {y: x for x in os.listdir("./data/img") for y in os.listdir(f"./data/img/{x}")}


def main():

    with open('./data/bn_to_lemmas.json', "r", encoding="utf8") as f:
        bn_to_lemmas = json.load(f)
    
    valid_bns = []
    with open('./data/valid_bn.jsonl', 'r', encoding='utf8') as f:

        for l in f:
            valid_bns.append(json.loads(l)["id"])
    valid_bns = set(valid_bns)

    random.seed(42)

    instances = []
    ids = {}
    co_hyp = {}

    data = []

    progress_bar = tqdm(total=20339189)
    
    with open("./data/glosses_best_wn_h_OK.jsonl", "r", encoding="utf8") as f:
        
        for l in f:

            line_data = json.loads(l)
            progress_bar.update(1)

            if len(line_data["hyps"]) == 0:
                continue

            if line_data["hyps"][0] not in bn_to_lemmas:
                continue

            if line_data["id"] not in img_path_dict:
                continue

            if line_data["id"] not in valid_bns:
                continue

            if line_data["id"] in bn_to_lemmas:
                data.append(line_data)
    
    progress_bar.close()

    for x in data:
        instances.append(x)
        ids[x["id"]] = len(ids)

        if x["hyps"][0] not in co_hyp:
            co_hyp[x["hyps"][0]] = []
        
        co_hyp[x["hyps"][0]].append(x["id"])
    
    all_idxs = list(range(len(instances)))

    print(len(all_idxs))

    test_card = 110
    test_ids = []
    test_idxs = []
    test_co_hyps = {}

    random.shuffle(all_idxs)

    for x in all_idxs:

        if x in test_idxs:
            continue

        co_hyp_single = co_hyp[instances[x]["hyps"][0]]
        co_hyp_single = [ids[y] for y in co_hyp_single if ids[y] not in test_ids]

        if len(co_hyp_single) < 10:
            continue

        test_idxs.append(x)
        test_co_hyps[instances[x]["hyps"][0]] = [x]
        considered_ids = set([instances[x]["id"]])

        for _ in range(9):

            negative_idx = x

            while negative_idx == x or instances[negative_idx]["id"] in considered_ids:
                negative_idx = random.choice(co_hyp_single)
            
            considered_ids.add(instances[negative_idx]["id"])
            test_ids.append(instances[negative_idx]["id"])
            test_idxs.append(negative_idx)
            test_co_hyps[instances[x]["hyps"][0]].append(negative_idx)

        if len(test_co_hyps) == test_card:
            break

    all_idxs = [x for x in all_idxs if x not in test_idxs]

    val_co_hyps = {}

    for x in test_co_hyps:
        val_co_hyps[x] = [y for y in test_co_hyps[x]]
        if len(val_co_hyps) == 10:
            break
    
    for x in val_co_hyps:
        del test_co_hyps[x]

    test_ids = set(test_ids)
    train_ids = []
    train_co_hyps = {}

    for x in all_idxs:

        if instances[x]["id"] in train_ids:
            continue

        if instances[x]["id"] in test_ids:
            continue

        co_hyp_single = co_hyp[instances[x]["hyps"][0]]
        co_hyp_single = [ids[y] for y in co_hyp_single if ids[y] not in test_ids]

        if len(co_hyp_single) == 1:
            continue

        negatives = []

        for negative_idx in co_hyp_single:

            current_negative_ids = [instances[neg_idx]["id"] for neg_idx in negatives]

            if negative_idx == x or instances[negative_idx]["id"] in test_ids or instances[negative_idx]["id"] in current_negative_ids:
                continue
            
            negatives.append(negative_idx)
        
        if len(negatives) == 0:
            continue
        
        train_ids.append(instances[x]["id"])
        train_co_hyps[instances[x]["hyps"][0]] = [x]
        train_ids.extend([instances[y]["id"] for y in negatives])
        train_co_hyps[instances[x]["hyps"][0]].extend(negatives)

    split_dict = {
        "train": train_co_hyps,
        "test": test_co_hyps,
        "val": val_co_hyps
    }

    print(len(train_co_hyps))
    print(len(test_co_hyps))
    print(len(val_co_hyps))

    for split in ["train", "val", "test"]:
        
        for lang in ["en", "it", "de", "fr", "es", "fa"]:

            os.makedirs(f"./data/{split}", exist_ok=True)

            with open(f"./data/{split}/glosses_{lang}.jsonl", "w", encoding="utf8") as f_out:
                
                split_hyps = split_dict[split]

                for hyp in split_hyps:

                    new_dict = {}
                    new_dict["hyp_id"] = hyp
                    new_dict["txt"] = []
                    new_dict["img"] = []
                    new_dict["ids"] = []
                    
                    hyp_lemma = random.choice(bn_to_lemmas[hyp][lang]).replace("_", " ")
                    lemmas = []

                    for idx in split_hyps[hyp]:
                        
                        lemma = random.choice(bn_to_lemmas[instances[idx]["id"]][lang]).replace("_", " ")

                        hyp_lemma_final = ""

                        for tok in hyp_lemma.split(" "):
                            if tok not in lemma:
                                hyp_lemma_final += tok + " "
                        
                        hyp_lemma_final = hyp_lemma_final[:-1]

                        if hyp_lemma not in lemma:

                            choice = random.sample([True, False], 1)[0]

                            if choice:
                                final_lemma = f"{hyp_lemma_final} {lemma}"
                            else:
                                final_lemma = f"{lemma} {hyp_lemma_final}"
                        
                        else:

                            final_lemma = lemma
                        
                        if final_lemma in lemmas and split == "train":
                            continue

                        lemmas.append(final_lemma)
                        new_dict["ids"].append(instances[idx]["id"])
                        new_dict["txt"].append(final_lemma)
                        new_dict["img"].append(os.path.join("./", "data", "img", img_path_dict[instances[idx]["id"]], instances[idx]["id"]))

                    if len(new_dict["txt"]) == 0:
                        continue
                        
                    json.dump(new_dict, f_out)
                    f_out.write('\n')


if __name__ == "__main__":
    main()