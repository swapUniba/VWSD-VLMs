import os
import json

from PIL import Image

def main():

    img_path_dict = {y: x for x in os.listdir("./data/img") for y in os.listdir(f"./data/img/{x}")}

    with open('./data/valid_bn.jsonl', 'w', encoding="utf8") as f:
        for img_name, img_tag in img_path_dict.items():

            try:
                with Image.open(f"./data/img/{img_tag}/{img_name}") as img:
                    print("ok")
            except Exception as e:
                print(e)
                continue
        
            json.dump({"id": img_name}, f)
            f.write('\n')


if __name__ == "__main__":
    main()