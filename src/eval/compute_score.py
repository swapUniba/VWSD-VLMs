import re
import json
import argparse


def main(model_name, ds_name):

    score = 0
    total = 0
    with open(f"./{model_name}/{ds_name}_responses.jsonl", "r", encoding="utf8") as f:

        for l in f:
            line_data = json.loads(l)

            try:
                gen = re.findall("\\d+", line_data["generated_output"][0])[0]
                if line_data["label"] == int(gen):
                    score += 1
            except Exception as e:
                total += 1
                continue
            
            total += 1

    print(score / total)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-d', '--ds_name')
    args = parser.parse_args()
    
    model_name = args.model_name
    ds_name = args.ds_name

    main(model_name, ds_name)