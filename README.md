# Code repository for "Assessing and Improving the Multilingual Visual Word Sense Disambiguation ability of Vision-Language Models"

This repository contains the training, inference and evaluation scripts for VWSD using both our co-Hyphonyms benchmark and the original VWSD benchmark.

If you find any of the material provided in this repository useful, please consider citing our work.

```
TBD
```

## Requirements

Before anything else, you should obtain a copy of the required data and of the [GradCache](https://github.com/luyug/GradCache) library.

For the data do the following:

- ORIGINAL VWSD

    Download the [annotations](https://drive.google.com/file/d/10vDZsY0EhzvFFR8IF-3P_2ApOF0GIMML/view?usp=share_link) and the [images](https://drive.google.com/file/d/1rK7EskkEXzD59j5On-8orO5mIinQGUMW/view?usp=share_link) provided by the organizers of the challenge. 

    The images should be saved to the following directory "./data/semeval-2023-task-1-V-WSD-train-v1/test_v1/test_images_v1", while the annotations to the following directory "./data/semeval-2023-task-1-V-WSD-train-v1/test_v1/".

    After saving the data, use the [merge script](./data/semeval-2023-task-1-V-WSD-train-v1/merge_test.py) to create .jsonl files for the annotations.

- OUR VWSD DATASET

    Download the data from [Huggingface](https://huggingface.co/datasets/swap-uniba/VWSD-VLMs) using the following commands

    ```
    huggingface-cli download swap-uniba/VWSD-VLMs --local-dir ./data --cache-dir ./cache --repo-type dataset
    ```

For GradCache do the following:

```
git clone https://github.com/luyug/GradCache
mv GradCache/src/grad_cache .
rm -r GradCache
```

## Data

We provide the scripts used for data processing in the [data directory](/scr/data). However, these scripts are mainly provided to showcase the strategy that was used to create the data, we directly provide the processed versions of the train, val and test splits [here](https://huggingface.co/datasets/swap-uniba/VWSD-VLMs). Hence, you should download it directly.

We also provide the dataset formatted for generative VLMs (except for the dataset of the original challenge). If you want to format the data from scratch for generative VLMs, use the following commands:

```
python3 -m src.data.prepare_ds_qwen
python3 -m src.data.prepare_ds_qwen_test
python3 -m src.data.prepare_ds_qwen_test_original
```

Requirements for formatting dataset for Qwen are provided [here](requirements/unsloth_requirements.txt). 

## Train

All experiments were performed using Python 3.12.3 and CUDA 12.9, including evaluation.

Fine-tuned models have also been uploaded to [Huggingface]().

### Siglip 2

Requirements for fine-tuning of the Siglip 2 model are provided [here](requirements/train_siglip_requirements.txt). 

Afterwards, you can fine-tune the model by running this command:

```
PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=False python3 -m src.train.train_bn.train_siglip2 -m google/siglip2-large-patch16-256 -e 20 -bs 32 -o vwsd_siglip/siglip2_large_multi
```

### Qwen

Requirements for fine-tuning of the Qwen VL model are provided [here](requirements/unsloth_requirements.txt). 

Afterwards, you can fine-tune the model by running this command:

```
python3 -m src.train.train_bn.train_qwen2_unsloth_ml
```

## Evaluation

For encoder-based models, the requirements are the same as the training requirements for Siglip 2. For generative models, the requirements are the same as the training requirements for Qwen VL. 

For encoder-based models, we provide the scripts used for evaluation [here](scripts).

Regardless, inference results for both encoder-based and generative VLMs are provided [here](reports).

### Local Inference

Inference scripts are provided [here](src/eval). Below, some examples to perform inference:

Encoder VLM

```
PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=False python3 -m src.eval.evaluate_siglip2 -m google/siglip-base-patch16-256-multilingual -l en -bs 16 -ne "negative"
```

Generative VLM

```
python3 -m src.eval.inference_qwen2_ml -m unsloth/Qwen2.5-VL-7B-Instruct -d ds_test_qwen_en_original_ml
python3 -m src.eval.compute_score -m unsloth/Qwen2.5-VL-7B-Instruct -d ds_test_qwen_en_original_ml
```

### Cloud Inference

Everything related to cloud inference is provided [here](cloud_inference).

We use the [process images script](cloud_inference/process_images.py) to resize all images to 336 before sending them to the inference provider. 

We use two inference providers: Together AI and Nebius. We use Together AI for LLaMA 4, while we use Nebius for Qwen 2.5 VL 72B and Gemma 3 27B. Requirements for both inference providers are provided [here](cloud_inference/requirements/).

Use the [request wsd scripts](cloud_inference/request_wsd.py) for Together AI inference and the [request wsd openai script](cloud_inference/request_wsd_openai.py) for Nebius inference.

Examples are provided below.

Together AI

```
python3 -m request_wsd -m meta-llama/Llama-4-Scout-17B-16E-Instruct -d ds_test_qwen_en_original_ml
```

Nebius

```
python3 -m request_wsd_openai -m Qwen/Qwen2.5-VL-72B-Instruct -d ds_test_qwen_en_original_ml
```

For both Together AI and Nebius you also need to set the dedicated environment variable to specify the API key.

### Statistical Tests

We perform statistical testing using McNemar's test. The [statsmodels](https://github.com/statsmodels/statsmodels) library is required in this case. We used the 0.14.5 version.

An example to compute McNemar's test is provided below:

```
python3 -m src.eval.compute_mc_nemar --results_dir_1 ./reports/encoder/vwsd_siglip/siglip2_large_multi_9 --results_dir_2 ./reports/encoder/google/siglip2-large-patch16-256
```