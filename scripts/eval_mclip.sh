#!/bin/bash

declare -a arr=("negative" "random")
declare -a langs=("en" "it" "fa" "de" "fr" "es")

for lang in "${langs[@]}"
do
    for strat in "${arr[@]}"
    do
        PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=False python3 -m src.eval.evaluate_clip -l ${lang} -bs 16 -ne ${strat} >> ./results/mclip_log.txt
    done
done
