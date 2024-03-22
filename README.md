# fact-checking
 **Thesis title - Multi-hop Retrieval and Reasoning with pre-trained Language Models for Fact-Checking**

This repository contains the source code of the thesis.

## Preparation

### Dependencies

- PyTorch

- See requirements.txt

### Data

This project uses HoVer dataset for claim verification ([Paper](https://arxiv.org/abs/2011.03088)). To download, run below script:

```bash
chmod +x download.sh
./download.sh
```

After running the script, the HoVer dataset will be downloaded and the directory should look as below:

```bash
fact-checking
└── datasets
    ├── hover_dev_release_v1.1.json
    ├── wiki_wo_links.db
    └── hover_train_release_v1.1.json
```

Custom datasets are derived from HoVer dataset, present in the same directory.

[Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) are used for fine-tuning and inference. These models can be downloaded only with access license from [Meta](https://llama.meta.com/llama-downloads/).

## Fine-tuning

1. To fine-tune roberta, run below script:
```
python3 scripts/roberta/finetuning.py
```
2. To fine-tune llama, download required model variant from [Meta](https://llama.meta.com/llama-downloads/). Then run below script:

```
python3 scripts/llama/finetuning.py
```

After fine-tuning, the models will be saved in ```finetuned_models/```. This repository already contains our fine-tuned models in the same directory.

## Inference

- For inference on roberta, run below script:

```
python3 scripts/roberta/inference.py
```

- For inference on llama, download required model variant from [Meta](https://llama.meta.com/llama-downloads/). Then run below script:

```
python3 scripts/llama/inference.py 
```

- For inference on fine-tuned models, run below scripts:

for roberta:

```
python3 scripts/roberta/inference_finetuned.py
```

for llama:

```
python3 scripts/llama/inference_finetuned.py
```
