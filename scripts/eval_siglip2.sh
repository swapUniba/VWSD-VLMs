#!/bin/bash

declare -a arr=("negative" "random")
declare -a langs=("en" "it" "fa" "de" "fr" "es")

f="$(basename -- $1)"

for lang in "${langs[@]}"
do
    for strat in "${arr[@]}"
    do
        PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=False python3 -m src.eval.evaluate_siglip2 -m $1 -l ${lang} -bs 16 -ne ${strat} -r >> "./results/${f}_log.txt"
    done
done
