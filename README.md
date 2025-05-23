<div align="center">

# 🧵 HopWeaver: Synthesizing Authentic Multi-Hop Questions Across Text Corpora

<p><strong>The first fully automated framework for synthesizing high-quality multi-hop questions from unstructured text corpora across documents without human intervention</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.15087"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Shenzy2/HopWeaver_Data"><img src="https://img.shields.io/badge/HuggingFace-Datasets-FFD21E.svg" alt="HuggingFace Datasets"></a>
  <a href="https://www.modelscope.cn/datasets/szyszy/HopWeaver_Data"><img src="https://img.shields.io/badge/ModelScope-Datasets-592EC6.svg" alt="ModelScope Datasets"></a>
  <a href="https://github.com/Zh1yuShen/HopWeaver/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made with Python"></a>
</p>

## 🌟 Key Features

- **🥇 First-of-its-kind**: Fully automated cross-document multi-hop question synthesis from unstructured corpora without manual annotation
- **💰 Cost-Effective**: Synthesizes high-quality questions at significantly lower cost than manual methods  
- **🎯 Quality Assured**: Three-dimensional evaluation system ensuring authentic multi-hop reasoning
- **🔄 Two Question Types**: Bridge questions (entity connections) and comparison questions (attribute analysis)
- **📊 Empirically Validated**: Quality comparable to or exceeding human-annotated datasets

---

**HopWeaver automatically synthesizes authentic cross-document multi-hop questions from unstructured text corpora, enabling cost-effective creation of high-quality MHQA datasets for specialized domains with scarce annotated resources.**

[English](README.md) | [中文](README_CN.md)

![Introduction](fig/intro.png)

</div>

## 📋 Table of Contents

- [🔎 Project Overview](#-project-overview)
- [🏗️ System Architecture](#-system-architecture)  
- [🔧 Core Functional Modules](#-core-functional-modules)
- [🔄 Reranker Model Training](#-reranker-model-training)
- [📚 Usage Guide](#-usage-guide)
  - [🛠️ Environment and Data Preparation](#-environment-and-data-preparation)
  - [⚙️ Configuration Files](#-configuration-files)
  - [Question Synthesis and Evaluation](#question-synthesis-and-evaluation)
  - [Self-Consistency Evaluation](#self-consistency-evaluation)
- [📝 Examples](#-examples)
  - [Bridge Question Examples](#bridge-question-examples)
  - [Comparison Question Examples](#comparison-question-examples)
- [📜 Citation](#-citation)
- [🔐 License](#-license)

## 🔎 Project Overview

The HopWeaver system is built on the FlashRAG framework and specifically designed for synthesizing and evaluating high-quality multi-hop questions. The system includes two main question synthesis paths:

1. **Bridge Question Synthesis**: Synthesizes questions requiring multi-step reasoning by extracting entities and establishing connections between them
2. **Comparison Question Synthesis**: Synthesizes questions that require comparing features of multiple entities

![Framework](fig/framework.png)


## 🏗️ System Architecture

The entire system consists of the following core components:

```
HopWeaver/
├── datasets/              # Datasets directory (containing hotpotqa, 2wiki, musique datasets)
├── fig/                   # Documentation images directory
├── flashrag/              # FlashRAG framework base code
│   ├── config/            # Base configuration module
│   ├── dataset/           # Dataset processing module
│   ├── generator/         # Generator module
│   ├── retriever/         # Retriever module
│   ├── evaluator/         # Evaluator module
│   └── utils/             # Common utility functions
│
├── hopweaver/             # HopWeaver core code
│   ├── components/        # Main components
│   │   ├── bridge/        # Bridge question components
│   │   ├── compare/       # Comparison question components
│   │   └── utils/         # Common utility functions
│   ├── config_lib/        # Configuration files directory
│   ├── evaluation_system/ # Evaluation system
│   └── train_reranker/    # Reranker model training tools
│
└── requirements.txt       # Project dependencies
```

Some functions of HopWeaver depend on the FlashRAG framework. The `flashrag` directory contains the basic framework code (with minor modifications), while the `hopweaver` directory contains specific components and functions for multi-hop question synthesis and evaluation.

## 🔧 Core Functional Modules

### 1. Bridge Question Synthesis Process

The bridge question synthesis includes the following key steps:

- **🔍 Bridge Entity Identification**: From randomly selected source documents, the system identifies bridge entities that can connect different information contexts, providing key pivots for multi-hop reasoning
  
- **🔄 Two-stage Coarse-to-Fine Retrieval**:
  - 🔎 Coarse-grained Retrieval: Using a modified maximum marginal relevance algorithm to balance query relevance, diversity from source documents, and diversity among selected documents
  
    **📊 Diverse Retrieval Scoring Function:**
    
    The diverse retrieval uses a modified Maximum Marginal Relevance (MMR) algorithm:
    
    $$\text{Score}(d_i) = \lambda_1 \cdot \text{sim}(q, d_i) - \lambda_2 \cdot \text{sim}(d_i, d_s) - \lambda_3 \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)$$
    
    Where:
    - $q$ is the query
    - $d_i$ is the candidate document  
    - $d_s$ is the source document
    - $S$ is the set of already selected documents
    - $\text{sim}(\cdot, \cdot)$ represents cosine similarity
    - $\lambda_1, \lambda_2, \lambda_3$ are weighting parameters with $\lambda_1 + \lambda_2 + \lambda_3 = 1$
    
    This formula is used by both **diverse** and **rerank** retrieval methods in their coarse retrieval stage.
  
  - 🔝 Fine-grained Reranking: Using a reranking model fine-tuned through contrastive learning to further optimize the ranking of candidate documents

- **🏗️ Multi-hop Question Construction**:
  - 📝 Sub-question Synthesis: Synthesize sub-questions from source and supplementary documents respectively, centered around the bridge entity
  - 🔄 Question Synthesis: Merge sub-questions into a single coherent multi-hop question, implying the reasoning path without directly exposing the bridge entity
  - ✅ Validation and Iteration: Ensure questions meet answerability, multi-hop nature, and no-shortcut constraints

### 2. Comparison Question Synthesis Process

Comparison question synthesis follows these steps:

- **🧩 Entity and Attribute Identification**: Identify main entities from documents and their 3-5 concise factual attribute-value pairs, filtering out attributes suitable for comparison

- **🔍 Filtering and Query Synthesis**:
  - ✓ Ensure specificity and comparability of entities and attributes
  - 🔎 Synthesize retrieval queries based on source entities, using direct recommendation or diversified search strategies

- **❓ Question Construction**:
  - 🎯 Guided Comparison: Precise comparison for specific entities and attributes
  - 🔍 Open Discovery: Find the first valid comparable pair among multiple attributes
  - 📝 Synthesize comparison questions containing information about two entities, such as "Which entity has a higher/earlier/larger attribute value?"

### 3. ✨ Question Refinement and Quality Assurance

During the bridge and comparison question synthesis process, the system implements strict quality control mechanisms:

- **🔍 Question Refinement and Validation Module**:
  - 📊 Evaluate questions for answerability, multi-hop nature, and language quality
  - 🏷️ Classify evaluation results into four categories: pass, adjust, reconstruct, or reject
  - ✅ Ensure each question involves cross-document reasoning and hides bridge entities
  - 📝 Maintain fluency without exposing intermediate reasoning steps

### 4. 🔄 Reranker Model Training and Optimization

The system synthesizes supervision signals by simulating key steps to improve retrieval quality:

- **📊 Simulated Feedback Synthesis**:
  - 📥 Extract successful and failed document examples from the bridge question synthesis process
  - 🔄 Construct contrastive training triplets (query, positive document, negative document)

- **📈 Contrastive Learning Optimization**:
  - 🧮 Use cross-entropy loss function to guide the model in distinguishing complementary documents
  - 📊 Obtain supervision signals directly from downstream task success rates

### 5. 📏 Multi-dimensional Evaluation System

The system employs a comprehensive evaluation framework to ensure question quality:

- **🤖 LLM-as-Judge Evaluation**:
  - ⭐ Use large language models as judges, employing Likert scales to evaluate each question
  - 🔄 Implement self-consistency evaluation methods to ensure stability and reproducibility of evaluation results
  - 📊 Analyze consistency of evaluation results by repeatedly evaluating the same input

- **📋 Answerability and Difficulty Evaluation**:
  - 🔍 **Q-Only Condition**: Solver receives only the question, testing baseline answerability using the solver's internal knowledge and reasoning capabilities
  - 📚 **Q+Docs Condition**: Solver receives the question and all supporting documents, simulating a golden retrieval scenario to evaluate answerability when necessary evidence is available
  - 📈 **Performance Gap Analysis**: Performance improvement from Q-Only to Q+Docs indicates whether the question is challenging and requires cross-document reasoning rather than relying solely on pre-trained knowledge

- **🔎 Evidence-Accessibility Evaluation**:
  - 📊 **Retrieval Quality Assessment**: Use multiple retrieval methods to fetch top-k documents and evaluate the accessibility of synthesized question evidence in the corpus
  - 📏 **Multi-dimensional Retrieval Metrics**: Record MAP (Mean Average Precision), RECALL@k, NDCG@k (Normalized Discounted Cumulative Gain), and Support F1 metrics
  - ✅ **Evidence Completeness Verification**: Ensure synthesized questions have complete evidence support, preventing unanswerable questions from entering the final dataset

## 🔄 Reranker Model Training
![Reranker](fig/reranker.png)
The project includes a dedicated reranker model training system to optimize the ranking of document retrieval results:

- 📊 Contrastive learning data synthesis
- ⚡ Training based on DeepSpeed
- 🧪 Reranker model ablation experiments

## 📚 Usage Guide

### 🛠️ Environment and Data Preparation

Before using HopWeaver, you need to complete the following preparation steps:

#### 1. Clone the Repository and Install Dependencies

```bash
git clone https://github.com/Zh1yuShen/HopWeaver.git
cd HopWeaver
pip install -r requirements.txt
```

#### 2. Configure LLM API

Before using the system, you need to configure the LLM API in your configuration file. Check and modify `config_lib/example_config.yaml`, focusing on these key settings:

```yaml
# API type selection (openai, azure, openrouter, anthropic, local)
api_type: "openai"

# OpenAI settings
openai_setting:
  api_keys:
    - "your-openai-api-key-1"
    - "your-openai-api-key-2"
    - "your-openai-api-key-3"

# Model selection
generator_model: "gpt-4o"
entity_extractor_model: "gpt-4o"
question_generator_model: "gpt-4o"
polisher_model: "gpt-4o"
filter_model: "gpt-4o"
```

> **Note**: Depending on the type of model you're using, you may need to modify the parameter settings and API selection logic in the `HopWeaver/flashrag/generator/openai_generator.py` file. For example, if you want to use Google's Gemini model, you need to add code similar to the following in `openai_generator.py`:

```python
# Detect model type and select corresponding configuration
if "gemini" in self.model_name.lower():
    self.openai_setting = config["google_setting"]
elif "claude" in self.model_name.lower():
    self.openai_setting = config["anthropic_setting"]
# Other model type detection...
else:
    self.openai_setting = config["openai_setting"]
```

Additionally, different models (such as GPT-4, Claude, Qwen, DeepSeek, etc.) may require different parameter configurations, such as temperature, top_p, max_tokens, etc. Please adjust accordingly based on the characteristics of your chosen model.

##### 🤖 Model Selection Recommendations

HopWeaver consists of several components that can use different models. Here are our recommendations based on our experiments:

- **polisher_model**: We recommend using DeepSeek-R1 or more advanced models for the polisher component, as it requires strong language refinement capabilities
- **Other components**: You can use the same model for other components (entity_extractor, question_generator, filter, etc.). We recommend choosing the same model for all synthesis components. In our paper, we successfully tested with various models including:
  - QwQ-32B
  - Qwen3-14B
  - GLM-9B-0414

For optimal performance, we recommend using models with at least 7B parameters. Smaller models may struggle with the complex reasoning required for multi-hop question synthesis.

##### 💻 Local Model Configuration

You can use local model support provided by [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), which supports various local model deployment methods.

##### ⚡ API Call Optimization

HopWeaver implements the following optimization mechanisms to improve the stability and efficiency of API calls:

1. **🔄 Multiple API Keys Rotation**: When multiple API Keys are configured, the system automatically rotates through them, distributing rate limits

   ```yaml
   openai_setting:
     api_keys:
       - "key1"
       - "key2"
       - "key3"  # Multiple API Key list
   ```

2. **🔄 Automatic Error Retry**: When common API errors (such as rate limits, server errors) are encountered, the system automatically retries

3. **⚡ Asynchronous Request Processing**: Supports batch asynchronous requests to maximize API call frequency

These mechanisms enable HopWeaver to utilize LLM API resources more efficiently when synthesizing a large number of multi-hop questions.

#### 3. Download Wiki Dataset

You need to download the `wiki18_fulldoc_trimmed_4096.jsonl` data file, which is our preprocessed Wiki dataset containing Wikipedia articles with document length trimmed to under 4096 characters.

Dataset download links: [huggingface](https://huggingface.co/datasets/Shenzy2/HopWeaver_Data) or [modelscope](https://www.modelscope.cn/datasets/szyszy/HopWeaver_Data)

For comparison with HotpotQA, 2wiki, and musique datasets mentioned in our paper, you can place the downloaded datasets in the `datasets` folder and use `datasets/process_and_sample_datasets.py` to process and sample arbitrary samples for subsequent comparison.

**Data Format Description**:
`wiki18_fulldoc_trimmed_4096.jsonl` is a JSONL format file, with each line containing a JSON object structured as follows:
```json
{
  "id": "591775",
  "title": "Los Ramones",
  "doc_size": 1250,
  "contents": "Los Ramones\nLos Ramones Los Ramones is the name of a municipality..."
}
```

**Field Descriptions**:
- `id`: Unique identifier for the document
- `title`: Document title
- `doc_size`: Character length of the document content
- `contents`: Complete full text content of the document

#### 4. Multi-API Provider Support

HopWeaver supports multiple API providers for enhanced flexibility and redundancy. You can configure different providers in your configuration file:

```yaml
# Multiple API Provider Configuration
api_type: "openai"  # Main API type

# OpenAI Configuration
openai_setting:
  api_keys:
    - "your-openai-api-key-1"
    - "your-openai-api-key-2"
  base_url: "https://api.openai.com/v1"

# Google Gemini Configuration
gemini_setting:
  api_keys:
    - "your-gemini-api-key-1"
    - "your-gemini-api-key-2"
  base_url: "https://generativelanguage.googleapis.com/v1"

# DeepSeek Configuration
deepseek_setting:
  api_key: "your-deepseek-api-key"
  base_url: "https://api.deepseek.com/v1"

# Claude (Anthropic) Configuration
claude_setting:
  api_key: "your-claude-api-key"
  base_url: "https://api.anthropic.com"

# OpenRouter Configuration (supports multiple models)
openrouter_setting:
  api_keys:
    - "your-openrouter-key-1"
    - "your-openrouter-key-2"
  base_url: "https://openrouter.ai/api/v1"

# GLM (SiliconFlow) Configuration
GLM_setting:
  api_keys: "your-glm-api-key"
  base_url: "https://api.siliconflow.cn/v1"
```

**Supported Models by Provider:**
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo, etc.
- **Google**: Gemini-2.0-flash, Gemini-2.5-flash-preview, etc.
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3, etc.
- **Claude**: Claude-3.5-Sonnet, etc.
- **OpenRouter**: Access to models like QwQ-32B, Gemma-3-27B, etc.
- **GLM**: GLM-4-9B, and other SiliconFlow supported models

#### 5. Global Path Mapping Configuration

HopWeaver uses global path mapping to efficiently manage model paths, indexes, and corpora:

```yaml
# Global Path Mappings
model2path:
  e5: "/path/to/e5-base-v2"
  gte: "/path/to/gte_sentence-embedding_multilingual-base"

# Pooling methods for each embedding model
model2pooling:
  e5: "mean"
  gte: "cls"

# Index paths for retrieval models
method2index:
  e5: '/path/to/e5_Flat.index'
  gte: '/path/to/gte_Flat.index'
  bm25: ~
  contriever: ~

# Corpus paths for different methods
method2corpus:
  e5: '/path/to/wiki18_fulldoc_trimmed_4096.jsonl'
  gte: '/path/to/wiki18_fulldoc_trimmed_4096.jsonl'
```

**Configuration Benefits:**
- **Centralized Management**: All model and data paths in one location
- **Easy Switching**: Change retrieval methods by modifying `retrieval_method` parameter
- **Automatic Resolution**: System automatically resolves paths based on method selection
- **Scalability**: Easy to add new models and corpora

#### 6. Advanced Retriever Parameters

For fine-tuned control over the retrieval process, configure these advanced parameters:

**📊 Retrieval Method Selection:**

HopWeaver supports three retrieval methods:
- **standard**: Standard retrieval, ranking based on query relevance only
- **diverse**: Diverse retrieval, using MMR algorithm to balance relevance and diversity
- **rerank**: Two-stage retrieval, first diverse retrieval then fine-grained reranking with trained model

```yaml
# Retriever Configuration
retriever_type: "rerank"  # Retrieval method selection, options: "standard", "diverse" or "rerank"
reranker_path: "/path/to/trained/reranker/model"  # Path to reranker model (required only for rerank method)

# Retriever Diversity Weight Parameters (applies to diverse and rerank methods' coarse retrieval stage)
lambda1: 0.87  # Query relevance weight (0-1)
lambda2: 0.03  # Original document diversity weight (0-1) 
lambda3: 0.1   # Selected document diversity weight (0-1)

# Performance Parameters
use_fp16: true              # Use FP16 for acceleration
query_max_length: 512       # Maximum query length
passage_max_length: 8196    # Maximum passage length
reranker_batch_size: 16     # Batch size for reranking (rerank method only)
reranker_normalize: false   # Whether to normalize reranker scores (rerank method only)
reranker_devices: ["cuda:0"] # Devices to use for reranking (rerank method only)

# Retrieval Cache (for performance optimization)
save_retrieval_cache: false    # Save retrieval results to cache
use_retrieval_cache: false     # Use cached retrieval results
retrieval_cache_path: ~        # Path to retrieval cache file
```

**Parameter Tuning Guide:**
- **lambda1 (0.8-0.9)**: Higher values prioritize query-document relevance
- **lambda2 (0.05-0.15)**: Controls diversity from source document
- **lambda3 (0.05-0.15)**: Controls diversity among selected documents  
- **Sum of lambda1+lambda2+lambda3 should equal 1.0**

**Performance Tips:**
- Use `use_fp16: true` for faster inference with minimal quality loss
- Adjust `reranker_batch_size` based on your GPU memory
- Enable caching for repeated experiments with same queries

#### 7. Download GTE Embedding Model

HopWeaver uses [GTE](https://huggingface.co/iic/gte_sentence-embedding_multilingual-base) multilingual model for retrieval. You can download the model directly from Hugging Face and specify the path in the configuration file:

Modify the model path in the configuration file `config_lib/example_config.yaml`:
```yaml
model2path:
  gte: "path/to/your/downloaded/GTE/model"
```

#### 8. Download or Build Index

You can choose to download our pre-built index files (
 [huggingface](https://huggingface.co/datasets/Shenzy2/HopWeaver_Data) or [modelscope](https://www.modelscope.cn/datasets/szyszy/HopWeaver_Data)), or build them yourself:

```bash
# Create index directory
mkdir -p index

# Download pre-built index (recommended)
# Index download link: [INDEX_DOWNLOAD_LINK_PLACEHOLDER]

# Or build index using FlashRAG
python -m flashrag.build_index \
  --model_name_or_path path/to/your/GTE/model \
  --corpus_path dataset/wiki18_fulldoc_trimmed_4096.jsonl \
  --index_path index/gte_Flat.index \
  --batch_size 32 \
  --model_type gte \
  --pooling_method cls \
  --use_fp16
```

Parameter descriptions:
- `--model_name_or_path`: Path to GTE model
- `--corpus_path`: Path to Wiki corpus file
- `--index_path`: Path to save the generated index
- `--batch_size`: Batch size, adjust based on your GPU memory
- `--model_type`: Model type, here it's gte
- `--pooling_method`: Pooling method, GTE uses cls
- `--use_fp16`: Use FP16 to accelerate index building

### ⚙️ Configuration Files

The project uses YAML format configuration files. Main configuration items include:

- Corpus path
- Model selection and parameters
- Data processing options
- GPU device allocation

### Question Synthesis and Evaluation

#### Bridge Question Synthesis

```bash
# Synthesize and evaluate bridge questions (basic)
python -m hopweaver.generate_and_evaluate_bridge --config ./config_lib/bridge_default_config.yaml


# Synthesize bridge questions with reranking retriever and custom weights
python -m hopweaver.generate_and_evaluate_bridge --config ./config_lib/bridge_default_config.yaml --retriever rerank --count 50 --name test_rerank --lambda1 0.87 --lambda2 0.03 --lambda3 0.1

# Synthesize bridge questions with custom configuration file
python -m hopweaver.generate_and_evaluate_bridge --config ./config_lib/your_custom_config.yaml --count 20 --name custom_test

# Evaluate existing bridge questions dataset
python -m hopweaver.generate_and_evaluate_bridge --config ./config_lib/bridge_default_config.yaml --eval-only --dataset-path ./datasets/bridge_questions.json
```

#### Comparison Question Synthesis

```bash
# Synthesize and evaluate comparison questions (basic)
python -m hopweaver.generate_and_evaluate_comparison --config ./config_lib/example_config.yaml

# Synthesize 30 comparison questions with a specific name prefix
python -m hopweaver.generate_and_evaluate_comparison --config ./config_lib/example_config.yaml --count 30 --name test_comparison

# Synthesize comparison questions with custom configuration file and output directory
python -m hopweaver.generate_and_evaluate_comparison --config ./config_lib/your_custom_config.yaml --count 50 --name test_comparison --output-dir ./output_comparison

# Evaluate existing comparison questions dataset
python -m hopweaver.generate_and_evaluate_comparison --config ./config_lib/example_config.yaml --eval-only --dataset-path ./datasets/comparison_questions.json
```

#### Generation-Only Mode (Without Evaluation)

If you only want to synthesize questions without evaluation, you can use the standalone question synthesizers:

**Bridge Question Generation Only:**

```bash
# Generate bridge questions without evaluation
python -m hopweaver.bridge_question_synthesizer --config ./config_lib/example_config.yaml --count 10

# Generate with rerank retriever
python -m hopweaver.bridge_question_synthesizer --config ./config_lib/example_config.yaml --count 15 --retriever rerank

# Generate with custom lambda parameters and rerank retriever
python -m hopweaver.bridge_question_synthesizer --config ./config_lib/example_config.yaml --count 20 --retriever rerank --lambda1 0.87 --lambda2 0.03 --lambda3 0.1

# Generate with diverse retriever (default)
python -m hopweaver.bridge_question_synthesizer --config ./config_lib/example_config.yaml --count 10 --retriever diverse
```

**Comparison Question Generation Only:**

```bash
# Generate comparison questions without evaluation
python -m hopweaver.comparison_question_synthesizer --config ./config_lib/example_config.yaml --count 10

# Generate with rerank retriever
python -m hopweaver.comparison_question_synthesizer --config ./config_lib/example_config.yaml --count 15 --retriever rerank

# Generate with custom output directory and rerank retriever
python -m hopweaver.comparison_question_synthesizer --config ./config_lib/example_config.yaml --count 20 --output-dir ./my_output --retriever rerank

# Generate with specific name prefix and diverse retriever
python -m hopweaver.comparison_question_synthesizer --config ./config_lib/example_config.yaml --count 15 --name my_comparison_test --retriever diverse
```

#### Parameters Explanation

- `--config`: Path to configuration file (default: ./config_lib/example_config.yaml)
- `--count`: Number of questions to synthesize (default: 10)
- `--name`: Dataset name prefix for distinguishing different synthesis batches
- `--retriever`: Retriever type, options: 'standard', 'diverse' or 'rerank' (default: 'diverse')
- `--eval-only`: Only evaluate existing questions without synthesizing new ones
- `--dataset-path`: Path to the dataset to evaluate (only used when eval-only is True)
- `--lambda1`: Query relevance weight (0 to 1, default: 0.8), higher values emphasize document-query relevance
- `--lambda2`: Original document diversity weight (0 to 1, default: 0.1), higher values emphasize diversity from source document
- `--lambda3`: Selected document diversity weight (0 to 1, default: 0.1), higher values emphasize diversity among selected documents

### Self-Consistency Evaluation

```bash
# Basic self-consistency evaluation
python -m hopweaver.judge_evaluation_self_con --config ./config_lib/example_config.yaml

# Self-consistency evaluation with custom parameters
python -m hopweaver.judge_evaluation_self_con \
  --config ./config_lib/example_config.yaml \
  --bridge ./datasets/bridge_questions.json \
  --comparison ./datasets/comparison_questions.json \
  --num_samples 20 \
  --repeats 5 \
  --output_dir ./eval_result/custom_stability \
  --max_workers 1

# Evaluate only specific models
python -m hopweaver.judge_evaluation_self_con \
  --models "gpt-4o-2024-11-20,claude-3-7-sonnet-20250219,gemini-2.0-flash"

# Only perform visualization and metric calculation (no new evaluations)
python -m hopweaver.judge_evaluation_self_con --results_dir ./eval_result/stability/20250521_123456
```

#### Parameter Description

- `--config`: Path to configuration file (default: ./config_lib/example_config.yaml)
- `--bridge`: Path to bridge type dataset (default: ./datasets/2wiki_bridge.json)
- `--comparison`: Path to comparison type dataset (default: ./datasets/2wiki_comparison.json)
- `--num_samples`: Number of samples to select for each type (default: 25)
- `--repeats`: Number of evaluation repetitions for each sample (default: 5)
- `--output_dir`: Output directory (default: ./eval_result/stability)
- `--max_workers`: Maximum number of parallel worker threads (default: 1)
- `--test`: Test mode, evaluate only one sample per model (flag parameter)
- `--results_dir`: Existing evaluation results directory, only perform visualization and metric calculation
- `--models`: Comma-separated list of models to evaluate

## 📝 Examples

### Bridge Question Examples

![Bridge Question Example](fig/bridge_case.png)

<details>
<summary>Click to expand for detailed explanation</summary>

#### 1. Source and Target Document Content

**Document A - Anatomy Domain Document**  
Title: Crus of diaphragm

Excerpt:
> Crus of diaphragm\nCrus of diaphragm The crus of diaphragm (pl. crura), refers to one of two tendinous structures that extends below the diaphragm to the vertebral column. There is a right crus and a left crus, which together form a tether for muscular contraction. They take their name from their leg-shaped appearance – "crus" meaning "leg" in Latin. The crura originate from the front of the bodies and intervertebral fibrocartilage of the lumbar vertebrae. They are tendinous and blend with the anterior longitudinal ligament of the vertebral column. The medial tendinous margins of the crura pass anteriorly and medialward, and meet in the middle line to form an arch across the front of the aorta known as the median arcuate ligament; this arch is often poorly defined. The area behind this arch is known as the aortic hiatus.

Key summary:
Describes the anatomical structure of the crus of diaphragm, particularly how the medial tendinous margins of the right and left crura converge in front of the aorta to form the median arcuate ligament (MAL), creating the aortic hiatus.

**Document B - Pathology Domain Document**  
Title: Median arcuate ligament syndrome

Excerpt:
> Median arcuate ligament syndrome\nMedian arcuate ligament syndrome In medicine, the median arcuate ligament syndrome (MALS, also known as celiac artery compression syndrome, celiac axis syndrome, celiac trunk compression syndrome or Dunbar syndrome) is a condition characterized by abdominal pain attributed to compression of the celiac artery and the celiac ganglia by the median arcuate ligament. The abdominal pain may be related to meals, may be accompanied by weight loss, and may be associated with an abdominal bruit heard by a clinician. The diagnosis of MALS is one of exclusion, as many healthy patients demonstrate some degree of celiac artery compression in the absence of symptoms.

Key summary:
Defines median arcuate ligament syndrome (MALS) as a condition caused by the median arcuate ligament compressing the celiac artery and ganglia, resulting in abdominal pain, weight loss, and other symptoms.

---

#### 2. Bridge Entity and Its Role

- Bridge Entity Name: **Median Arcuate Ligament**
- Type: Anatomical Structure
- Connecting Function:
    - Document A describes in detail how the "median arcuate ligament" is formed from the crus of diaphragm.
    - Document B explains how this "median arcuate ligament" can lead to a clinical syndrome (MALS) under certain conditions.
    - Thus, the "median arcuate ligament" serves as a core anatomical structure, establishing a bridge between the documents from **"what it is" (anatomical composition) to "what it does" (clinical impact)**.

---

#### 3. Sub-question Construction and Reasoning Analysis

**Reasoning Logic Connection:**  
The anatomical structure of the median arcuate ligament (from Document A) is directly related to its pathophysiological mechanism of compressing the celiac artery leading to median arcuate ligament syndrome (MALS) (from Document B).

**Example Sub-questions:**

- Sub-question 1 (from Document A):  
  Question: What arch-like structure is formed by the medial tendinous margins of the crus of diaphragm meeting in front of the aorta?  
  Answer: Median arcuate ligament  
  Source: Document A

- Sub-question 2 (from Document B):  
  Question: What syndrome is caused when the median arcuate ligament compresses the celiac artery and ganglia?  
  Answer: Median arcuate ligament syndrome (MALS)  
  Source: Document B

**Reasoning Path:**  
Document A clarifies the anatomical origin of the median arcuate ligament (formed by the crus of diaphragm, creating the aortic hiatus), providing its structural basis. Document B explains how this structure can potentially compress the celiac artery and ganglia in pathological conditions, leading to MALS.

---

#### 4. Multi-hop Question Synthesis

**Multi-hop Question:**  
Question: What syndrome is caused when the anatomical structure formed by the convergence of the crus of diaphragm at the aorta compresses the celiac artery and ganglia?  
Answer: Median arcuate ligament syndrome (MALS)

**Reasoning Path:**  
- Document A identifies the "median arcuate ligament" as the anatomical structure formed by the crus of diaphragm converging at the aorta, creating the aortic hiatus.
- Document B explains that pathological compression by this ligament on the celiac artery and ganglia leads to "median arcuate ligament syndrome."
- The question requires identifying the anatomical structure in Document A and understanding its clinical consequences in Document B, connecting them through the implied structural relationship.

</details>

### Comparison Question Examples

![Comparison Question Example](fig/comparsion_case.png)

<details>
<summary>Click to expand for detailed explanation</summary>

#### 1. Source and Target Document Content Extracts

**Source Document (Composer Biography Snippet):**  
Mihály Mosonyi (born September 4, 1815 in Boldogasszony, Austria-Hungary, died October 31, 1870 in Budapest) was a Hungarian composer. Originally named Michael Brand, he later changed his name to Mosonyi in honor of his home region of Moson. "Mihály" is the Hungarian form of Michael. He was dedicated to creating instrumental music with a Hungarian national style. His notable works include "Funeral Music" and "Purification Festival."

**Target Document (Composer Biography Snippet):**  
Adam Liszt (father of the composer) was the shepherd supervisor at the Esterházy estate and also a musician. Franz Liszt, the only son of Adam Liszt and Maria Anna, was born on October 22, 1811 in Raiding and baptized the following day. Liszt began learning music from his father at the age of six, and the family later moved to Vienna. Liszt became one of the most famous Hungarian composers of the 19th century.

---

#### 2. Connection Between Documents

- Both documents provide birth dates and early life information for their respective composers.
- Through the "birth date" attribute, a direct comparison relationship can be established.

---

#### 3. Multi-hop Reasoning Path Construction

**Reasoning Path:**  
- Information Extraction (Document A): Identify Mihály Mosonyi's birth date as September 4, 1815.
- Information Extraction (Document B): Identify Franz Liszt's birth date as October 22, 1811.
- Comparative Analysis: Compare the two dates and determine that 1811 is earlier than 1815.
- Multi-hop Question Construction: Based on the above reasoning chain, formulate the comparative question "which composer was born earlier."

---

#### 4. Final Multi-hop Comparison Question Example

**Question:**  
Which composer was born earlier: Mihály Mosonyi or Franz Liszt?

**Answer:**  
Franz Liszt

</details>

## 📜 Citation

If you use HopWeaver in your research, please cite our work:

```bibtex
@misc{shen2025hopweaversynthesizingauthenticmultihop,
      title={HopWeaver: Synthesizing Authentic Multi-Hop Questions Across Text Corpora}, 
      author={Zhiyu Shen and Jiyuan Liu and Yunhe Pang and Yanghui Rao},
      year={2025},
      eprint={2505.15087},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15087}, 
}
```

## 🔐 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

