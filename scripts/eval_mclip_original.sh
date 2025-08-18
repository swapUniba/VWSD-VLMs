#!/bin/bash

declare -a langs=("en" "it" "fa")

for lang in "${langs[@]}"
do
    PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=False python3 -m src.eval.evaluate_clip_original -l ${lang} -r >> ./results_original/mclip_log.txt
done
