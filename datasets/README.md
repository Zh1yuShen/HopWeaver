# Dataset Processing and Sampling Script (process_and_sample_datasets.py)

## Overview

This Python script is designed to process and sample questions from various multi-hop question-answering datasets: HotpotQA, 2WikiMQA, and MuSiQue. The primary goal is to extract different types of questions (e.g., bridge, comparison) from each dataset and save them as separate files. This allows for targeted evaluation of models on specific categories of human-authored questions.

## Features

-   **Processes Multiple Datasets**: Handles HotpotQA, 2WikiMQA, and MuSiQue datasets.
-   **Type-Based Extraction**: Identifies and extracts questions based on their reasoning type:
    -   **HotpotQA**:
        -   `bridge`: Questions requiring reasoning over multiple documents.
        -   `comparison`: Questions requiring comparison between entities.
    -   **2WikiMQA**:
        -   `bridge`: Similar to HotpotQA bridge.
        -   `comparison`: Similar to HotpotQA comparison.
        -   `bridge_comparison`: Questions involving both bridge and comparison reasoning.
    -   **MuSiQue**:
        -   `musique-2-steps`: Bridge questions requiring 2 reasoning steps.
        -   `musique-3-steps`: Bridge questions requiring 3 reasoning steps.
        -   `musique-4-steps`: Bridge questions requiring 4 reasoning steps.
-   **Sampling**: Allows for sampling a specified number of questions from each extracted type.
-   **Organized Output**: Saves the processed and sampled datasets into a structured output directory, with each question type in its own JSON file.

## Usage

The script is run from the command line.

### Arguments

-   `--hotpotqa` (str): Path to the HotpotQA dataset file.
    -   Default: `./data_defaults/hotpotqa/hotpot_dev_distractor_v1.json`
-   `--twowiki` (str): Path to the 2WikiMQA dataset file.
    -   Default: `./data_defaults/2wiki/dev.jsonl`
-   `--musique` (str): Path to the MuSiQue dataset file.
    -   Default: `./data_defaults/musique/dev.jsonl`
-   `--output_dir` (str): Directory where the processed datasets will be saved.
    -   Default: `./processed_datasets`
-   `--sample_size` (int): Number of samples to draw from each extracted question type. If not specified or set to `None`, all matching questions will be saved.
    -   Default: `50`
-   `--random_seed` (int): Random seed for sampling to ensure reproducibility.
    -   Default: `42`
-   `--use_separated_datasets` (bool, flag): If present, use datasets from the directory specified by `--separated_datasets_dir` instead of the paths provided by `--hotpotqa`, `--twowiki`, and `--musique` or their defaults.
    -   Default: Not present.
-   `--separated_datasets_dir` (str): Directory containing the separated dataset files (`hotpot_dev_distractor_v1.json`, `dev_2wiki.jsonl`, `dev_musique.jsonl`). Only used if `--use_separated_datasets` is active.
    -   Default: `./hopweaver_dataset_files`
-   `--log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR).
    -   Default: `INFO`
-   `--only_analyze_types` (bool, flag): If present, only analyze and log question type statistics without saving sampled data files.
    -   Default: Not present.
-   `--processed_musique_file` (str): Path to the pre-processed MuSiQue data file (`musique_data.json`) required by the MuSiQue processing logic if the default path is not suitable. This file contains pre-computed information that speeds up the analysis of MuSiQue questions.
    -   Default: `./data_defaults/dataset_mhqa/musique_data.json`

### Example Run Command

To process the datasets using custom paths (or if default paths are not set up at `./data_defaults/`), sample 100 questions of each type, and save them to a custom output directory `./processed_data`:

```bash
python process_and_sample_datasets.py \
    --hotpotqa path/to/your/hotpotqa_dev_distractor_v1.json \
    --twowiki path/to/your/dev.jsonl \
    --musique path/to/your/musique_dev.jsonl \
    --processed_musique_file path/to/your/musique_data.json \
    --output_dir ./processed_data \
    --sample_size 100 \
    --random_seed 123
```

If you have dataset files in the default locations (e.g., `./data_defaults/hotpotqa/...`), you can omit the specific dataset path arguments:

```bash
python process_and_sample_datasets.py \
    --output_dir ./processed_data \
    --sample_size 100 
    # This assumes default --hotpotqa, --twowiki, --musique, and --processed_musique_file paths are valid
```

To use separated dataset files (e.g., prepared in a specific directory like `./my_hopweaver_datasets/`) and save results to `./processed_data_separated` with a sample size of 50:

```bash
# Ensure ./my_hopweaver_datasets/ contains hotpot_dev_distractor_v1.json, dev_2wiki.jsonl, dev_musique.jsonl
python process_and_sample_datasets.py \
    --output_dir ./processed_data_separated \
    --sample_size 50 \
    --use_separated_datasets \
    --separated_datasets_dir ./my_hopweaver_datasets
```

If your separated datasets are in the default `./hopweaver_dataset_files/` directory, you can simplify the command:

```bash
# Ensure ./hopweaver_dataset_files/ contains the necessary files
python process_and_sample_datasets.py \
    --output_dir ./processed_data_separated \
    --sample_size 50 \
    --use_separated_datasets
```

**Note**: Ensure that the Python environment has necessary libraries installed (e.g., `json`, `argparse`, `logging`, `tqdm`, `collections`). The script primarily uses standard Python libraries.

## Output Structure

The script will create JSON files in the specified `output_dir`. For example:

-   `hotpotqa_bridge.json`
-   `hotpotqa_comparison.json`
-   `twowiki_bridge.json`
-   `twowiki_comparison.json`
-   `twowiki_bridge_comparison.json`
-   `musique_bridge.json`

Each file will contain a list of question objects of the corresponding type.
