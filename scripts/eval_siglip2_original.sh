#!/bin/bash

declare -a langs=("en" "it" "fa")

f="$(basename -- $1)"

for lang in "${langs[@]}"
do
    PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=False python3 -m src.eval.evaluate_siglip2_original -m $1 -l ${lang} -r >> "./results_original/${f}_log.txt"
done
