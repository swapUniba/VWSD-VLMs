import os
import re
import json
import argparse

from statsmodels.stats.contingency_tables import mcnemar 


def main(results_dir_1, results_dir_2):

    for results_file in os.listdir(results_dir_1):

        acc_data1 = []
        acc_data2 = []

        with open(os.path.join(results_dir_1, results_file), "r", encoding="utf8") as f:

            for l in f:
                line_data = json.loads(l)

                if "hit@1" not in line_data:

                    try:
                        gen = re.findall("\\d+", line_data["generated_output"][0])[0]
                    except Exception as e:
                        gen = 11

                    if int(gen) == line_data["label"]:
                        acc_data1.append(1)
                    else:
                        acc_data1.append(0)
                
                else:

                    acc_data1.append(line_data["hit@1"])

        with open(os.path.join(results_dir_2, results_file), "r", encoding="utf8") as f:

            for l in f:
                line_data = json.loads(l)
                
                if not "hit@1" in line_data:

                    try:
                        gen = re.findall("\\d+", line_data["generated_output"][0])[0]
                    except Exception as e:
                        gen = 11

                    if int(gen) == line_data["label"]:
                        acc_data2.append(1)
                    else:
                        acc_data2.append(0)
                
                else:

                    acc_data2.append(line_data["hit@1"])

        acc = [0, 0, 0, 0]

        for x, y in zip(acc_data1, acc_data2):

            if x == 0 and y == 0:
                acc[0] += 1
            elif x == 0 and y == 1:
                acc[1] += 1
            elif x == 1 and y == 0:
                acc[2] += 1
            elif x == 1 and y == 1:
                acc[3] += 1

        acc = [[acc[0], acc[1]], [acc[2], acc[3]]]

        if acc[0][1] + acc[1][0] < 25:
            pval = mcnemar(acc, exact=True).pvalue
        else:
            pval = mcnemar(acc, exact=False, correction=False).pvalue
        
        print(results_file)
        print(sum(acc_data1) / len(acc_data1))
        print(sum(acc_data2) / len(acc_data2))
        print(True if pval < 0.05 else False)
        print("*" * 8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir_1", type=str)
    parser.add_argument("--results_dir_2", type=str)
    args = parser.parse_args()

    main(args.results_dir_1, args.results_dir_2)