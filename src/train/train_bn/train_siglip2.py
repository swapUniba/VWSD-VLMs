import json
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.amp import autocast
from grad_cache.functional import cached, cat_input_tensor
from transformers import AutoModel, AutoProcessor, set_seed, get_linear_schedule_with_warmup

class CustomImageDataset(Dataset):
    def __init__(self, data_file, image_processor, with_negatives=False):

        data = []
        with open(data_file, "r", encoding="utf8") as f:
            for l in f:
                line_data = json.loads(l)

                if len(line_data["img"]) < 2:
                    continue

                data.append(line_data)
        
        print(len(data))
        self.data = data
        self.image_processor = image_processor
        self.with_negatives = with_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inst = self.data[idx]

        if self.with_negatives:
            imgs = inst["img"]
            txts = inst["txt"]

            target_ids = random.sample(range(len(imgs)), 2)

            positive_img = imgs[target_ids[0]]
            positive_txt = txts[target_ids[0]]
            negative_img = imgs[target_ids[1]]
            negative_txt = txts[target_ids[1]]
        else:
            target_ids = random.sample(range(len(imgs)), 1)
            positive_img = inst["img"][target_ids[0]]
            positive_txt = inst["txt"][target_ids[0]]

        new_dict = {}
        positive_img = self.image_processor(torch.from_numpy(np.array(Image.open(positive_img).convert("RGB"))), return_tensors="pt")["pixel_values"][0]

        new_dict["positive_txt"] = positive_txt.lower()
        new_dict["positive_img"] = positive_img

        if self.with_negatives:
            negative_img = self.image_processor(torch.from_numpy(np.array(Image.open(negative_img).convert("RGB"))), return_tensors="pt")["pixel_values"][0]

            new_dict["negative_txt"] = negative_txt.lower()
            new_dict["negative_img"] = negative_img
        
        return new_dict


def train_model(epochs, model_path, model_name_output, freeze_image, mini_batch_size):

    set_seed(42)

    batch_size_pos = 512

    model = AutoModel.from_pretrained(model_path, attn_implementation="flash_attention_2", cache_dir="./cache").to("cuda:0").train()

    if freeze_image:
        for param in model.vision_model.parameters():
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(model_path, cache_dir="./cache")

    total_loss = 0

    min_len = torch.inf

    datasets = {}
    ds_tags = {}

    for split in ["train"]:

        datasets[split] = []
        ds_tags[split] = []

        for lang in ["de", "en", "es", "fa", "fr", "it"]:

            train_dataset = CustomImageDataset(f"./data/{split}/glosses_{lang}.jsonl", processor.image_processor, True if split == "train" else False)

            if split == "train" and lang == "en":
                if len(train_dataset) // batch_size_pos < min_len:
                    min_len = len(train_dataset) // batch_size_pos
            
            datasets[split].append(train_dataset)
            ds_tags[split].append(f"{lang}")

    print(len(datasets["train"]))
    train_datasets = datasets["train"]

    train_tags = ds_tags["train"]

    lang_sampling = {
        x: 1 if "en" not in x else min_len for x in train_tags
    }

    sampling_steps = sum(lang_sampling.values())

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=5e-6, weight_decay=0.1)
    training_steps = sampling_steps * int(epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(training_steps / 4), training_steps)

    @cat_input_tensor
    @autocast('cuda', dtype=torch.bfloat16)
    def compute_loss(image_embeds, text_embeds):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        similarity = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
        similarity = similarity / 0.07

        caption_loss = torch.nn.functional.cross_entropy(similarity, torch.arange(len(similarity), device=similarity.device))
        image_loss = torch.nn.functional.cross_entropy(similarity.t(), torch.arange(len(similarity), device=similarity.device))

        return (caption_loss + image_loss) / 2.0
    
    @cached
    @autocast('cuda', dtype=torch.bfloat16)
    def call_model(model, inputs, is_image):
        if is_image:
            return model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            return model.get_text_features(input_ids=inputs["input_ids"])
        
    optimizer.zero_grad()
    split = batch_size_pos // mini_batch_size

    for epoch in range(int(epochs)):

        epoch_lang_sampling = {k: 0 for k in lang_sampling.keys()}

        pbar = tqdm(total=int(sampling_steps))

        train_step = 0

        cache_t = []
        cache_i = []

        t = []
        i = []

        step_mini = 0
        step_train = 0

        train_loaders = {}

        for train_ds, tag in zip(train_datasets, train_tags):
            random.shuffle(train_ds.data)
            train_loaders[tag] = iter(torch.utils.data.DataLoader(train_ds, batch_size=mini_batch_size, shuffle=True))

        print(min_len)

        while True:

            step_mini = 0

            if step_train == sampling_steps:
                break

            lang_step = random.choice(list(epoch_lang_sampling.keys()))
            epoch_lang_sampling[lang_step] += 1

            while step_mini != split:
            
                batch = next(train_loaders[lang_step])

                positive_txt = batch["positive_txt"]
                negative_txt = batch["negative_txt"]
                positive_img = batch["positive_img"]
                negative_img = batch["negative_img"]

                txt = positive_txt + negative_txt

                print(positive_txt[0])

                inputs = processor.tokenizer(text=txt, truncation=True, padding="max_length", max_length=64, return_tensors="pt").to("cuda:0")

                inputs = {
                    "input_ids": inputs.input_ids,
                    "pixel_values": torch.vstack([positive_img, negative_img]).to("cuda:0")
                }

                rt, ct = call_model(model, inputs, False)

                if freeze_image:
                    ri = model.get_image_features(pixel_values=inputs["pixel_values"])
                else:
                    ri, ci = call_model(model, inputs, True)
                    cache_i.append(ci)

                cache_t.append(ct)

                t.append(rt)
                i.append(ri)

                step_mini += 1

                if step_mini == split:

                    loss = compute_loss(t, i)

                    loss.backward()
                    loss.detach_()

                    for f, r in zip(cache_t, t):
                        f(r)

                    if not freeze_image:
                        for f, r in zip(cache_i, i):
                            f(r)

                    total_loss += loss
                    train_step += 1

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    print(total_loss.item())
                    train_step = 0
                    total_loss = 0

                    cache_t = []
                    cache_i = []

                    t = []
                    i = []

                    if epoch_lang_sampling[lang_step] == lang_sampling[lang_step]:
                        del epoch_lang_sampling[lang_step]
                    
                    pbar.update(1)
                    step_train += 1
        
        model.save_pretrained("./" + model_name_output + "_" + str(epoch))
        processor.save_pretrained("./" + model_name_output + "_" + str(epoch))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-bs', '--mini_batch_size', type=int)
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-o', '--model_name_output')
    parser.add_argument('-i', '--freeze_image', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    epochs = args.epochs
    mini_batch_size = args.mini_batch_size
    model_name = args.model_name
    model_name_output = args.model_name_output
    freeze_image = args.freeze_image

    train_model(epochs, model_name, model_name_output, freeze_image, mini_batch_size)
