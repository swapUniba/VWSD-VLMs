import json

for lang in ["en", "it", "fa"]:    

    data = []

    unique_texts = []
    unique_images = []

    with open(f'./test_v1/{lang}.test.data.v1.txt', 'r', encoding='utf8') as f:

        for l in f:

            l = l.replace('\n', '')
            instances = l.split('\t')

            target_text = instances[1]
            images = instances[2:]

            if target_text not in unique_texts:
                unique_texts.append(target_text)
            
            for image in images:
                if image not in unique_images:
                    unique_images.append(image)
            data.append({"target_text": target_text, "images": images})

    with open(f'./test_v1/{lang}.test.gold.v1.txt', 'r', encoding='utf8') as f:

        for i, l in enumerate(f):

            l = l.replace('\n', '')
            data[i]["gold"] = l

    with open(f'./{lang}.json', 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2)