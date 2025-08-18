import PIL.Image
LANCZOS = PIL.Image.Resampling.LANCZOS
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


def main():

    datasets_to_conc = []

    for lang in ["en", "it", "fa", "de", "fr", "es"]:
        ds = load_dataset("json", data_files={"train": f"../data/test/glosses_{lang}.jsonl"})["train"]
        datasets_to_conc.append(ds)

    ds = concatenate_datasets(datasets_to_conc)

    for x in tqdm(ds):

        for img_path in x["img"]:

            img = PIL.Image.open(img_path.replace("./", "../")).convert("RGB").resize((336, 336), LANCZOS)
            img.save(f"./images_ours/{img_path.split('/')[-1]}.png")
    
    for lang in ["en", "it", "fa"]:
        ds = load_dataset("json", data_files={"train": f"../data/semeval-2023-task-1-V-WSD-train-v1/{lang}.json"})["train"]

        for x in tqdm(ds):

            for img_path in x["images"]:

                img = PIL.Image.open("../data/semeval-2023-task-1-V-WSD-train-v1/test_v1/test_images_v1/" + img_path).convert("RGB").resize((336, 336), LANCZOS)
                img.save(f"./images_original/{img_path}")


if __name__ == "__main__":
    main()