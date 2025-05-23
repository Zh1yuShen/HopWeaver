# Reranker Model Training Guide

This directory contains tools and datasets for training and evaluating retrieval reranker models. Reranker models can improve the quality of document ranking in retrieval systems, which is crucial for multi-hop question answering systems.

## Directory Structure

```
train_reranker/
├── data/                        # Training and test data
│   ├── test_data.jsonl          # Test dataset
│   ├── train_data.jsonl         # Training dataset
│   └── test_data_sample.jsonl   # Test data sample
├── contrastive_data_generator.py # Contrastive learning data generator
├── ds_stage0.json               # DeepSpeed configuration file (optimizes training efficiency)
├── rerank_ablation_test.py      # Reranker model ablation study tool
└── train_reranker.py            # Reranker model training script
```

## Data Preparation

### Using Pre-generated Data

The project already includes a complete training dataset (`data/train_data.jsonl`) and test dataset (`data/test_data.jsonl`), which can be directly used for model training.

### Generating New Training Data

If you need to generate new training data, you can use the `contrastive_data_generator.py` script:

```bash
conda activate llm
python contrastive_data_generator.py
```

This script will:
1. Read documents from the corpus
2. Extract entities and generate potential queries
3. Retrieve relevant documents as positive samples
4. Generate negative samples
5. Save the results to the `./data/` directory

### Data Format for Fine-tuning

The common data format for fine-tuning reranker models within the FlagEmbedding framework is a JSONL (JSON Lines) file. Each line in this file represents a training sample, formatted as a JSON object. Each JSON object should contain the following necessary keys:

- `"query"`: Represents the search query, type string.
- `"pos"`: Contains a list of positive example documents, where each document is a string relevant to the given query.
- `"neg"`: Contains a list of negative example documents, where each document is a string irrelevant to the query. Although optional, including negative samples is highly recommended as it can significantly improve the model's ability to distinguish between relevant and irrelevant documents.

Here is a specific example of a JSON object in a JSONL file:

```json
{"query": "Explain theory of relativity.", "pos": ["The theory of relativity, proposed by Albert Einstein, describes the relationship between space and time."], "neg": ["Quantum mechanics is a fundamental theory in physics."]}
```

An example command to generate data:

```bash
python ./contrastive_data_generator.py --config ../config_lib/extract_config_wikifulldoc.yaml --num_examples 1000 --max_doc_candidates 5 --lambda1 0.85 --lambda2 0.05 --lambda3 0.1
```

## Model Training

### Full Model Training

Train using the full training dataset:

```bash
python train_reranker.py --train_data ./data/train_data.jsonl --output_dir ./output --epochs 2 --batch_size 16 --gradient_accumulation_steps 4 --learning_rate 5e-6 --use_deepspeed --ds_config ./ds_stage0.json
```

**Parameter Description**:
- `--model_path`: Base model path, defaults to `./models/bge-reranker-v2-m3`
- `--train_data`: Training data file
- `--output_dir`: Output model directory
- `--epochs`: Number of training epochs, 2-3 epochs are recommended
- `--batch_size`: Batch size, adjust according to GPU memory
- `--learning_rate`: Learning rate, 5e-6 to 7e-6 is recommended
- `--gradient_accumulation_steps`: Gradient accumulation steps, used to increase effective batch size
- `--use_deepspeed`: Enable DeepSpeed for training optimization
- `--ds_config`: DeepSpeed configuration file, use `./ds_stage0.json`

## Model Evaluation

### Ablation Study

Use `rerank_ablation_test.py` for comparative experiments of different retrieval strategies:

```bash
conda activate llm
python rerank_ablation_test.py --config_path ./config_lib/example_config.yaml --output_file ./ablation_results/results.json
```

This will test the performance of the following retrievers:
1. Base Retriever
2. Diverse Retriever
3. Zero-shot Reranker Retriever
4. Diverse + Zero-shot Reranker Retriever
5. Diverse + Fine-tuned Reranker Retriever

### Specific Test Commands

Test fine-tuned model:
```bash
conda activate llm && python test_reranker.py --model_path=./output_new --test_data=./data/test_data.jsonl --output_file=./finetune_results.json

conda activate llm && python test_reranker.py --model_path=./output_new --test_data=./data/test_data.jsonl --output_file=./new_model_results.json
```

Ablation test examples:
```bash
conda activate llm && python ./rerank_ablation_test.py --config ../config_lib/extract_config_wikifulldoc.yaml --num 50 --candidates 5

conda activate llm && CUDA_VISIBLE_DEVICES=3 python ./rerank_ablation_test.py --config ../config_lib/extract_config_wikifulldoc.yaml --num 50 --candidates 5 --single diverse

conda activate llm && CUDA_VISIBLE_DEVICES=3 python ./rerank_ablation_test.py --config ../config_lib/extract_config_wikifulldoc.yaml --num 50 --candidates 5 --doc_ids ./sampled_doc_ids.txt
```
Supported ablation types: `standard`, `diverse`, `diverse_zs`, `diverse_ft`

## Notes

1. Before running for the first time, please ensure that the base model has been downloaded in the `./models/` directory.
2. The training process may require a large amount of GPU memory. It is recommended to use a GPU with at least 16GB of video memory.
3. Using DeepSpeed can reduce video memory requirements and improve training efficiency.
4. Training results will be saved in the specified output directory and can be directly used in the retrieval system.
