### 🛠️ Environment and Data Preparation

Before using HopWeaver, you need to complete the following preparation steps:

#### 1. Clone the Repository and Install Dependencies

```bash
git clone https://github.com/Zh1yuShen/HopWeaver.git
cd HopWeaver
pip install -r requirements.txt
```

#### 2. Configure LLM API

To use the system, configure the LLM API in your `config_lib/example_config.yaml` (or a copy). The crucial part is that the `flashrag/generator/openai_generator.py` script selects an API configuration block from your YAML file (e.g., `openai_setting`, `google_setting`, `anthropic_setting`, `deepseek_setting`) based on **keywords found in the `generator_model` name you specify**.

- If `generator_model` includes `"gemini"` (e.g., `"gemini-1.5-pro"`), the script will attempt to use the `google_setting` block from your YAML.
- If `generator_model` includes `"claude"` (e.g., `"claude-3-sonnet-20240229"`), it will use the `anthropic_setting` block.
- If `generator_model` includes `"deepseek"` (e.g., `"deepseek-chat"`), it will use the `deepseek_setting` block.
- If none of these (or other internally defined) keywords are found, it defaults to using the `openai_setting` block (e.g., for models like `"gpt-4o"`).

You **must** ensure that the YAML configuration file contains the correctly named setting block (e.g., `google_setting`) and that this block contains the necessary API key(s) and any other required parameters for the chosen model provider.

```yaml
# Example: OpenAI settings (this block is used if generator_model is e.g., "gpt-4o", 
# or if no keywords like "gemini", "claude", "deepseek" are found in the generator_model name)
openai_setting:
  api_keys:
    - "your-openai-api-key-1"
    - "your-openai-api-key-2"
    - "your-openai-api-key-3"
  base_url: "https://api.openai.com/v1"

# Example: Google settings (this block would be used if generator_model contains "gemini")
# Make sure this block exists and is correctly filled if you set generator_model to a Gemini model.
# google_setting:
#   api_key: "YOUR_GOOGLE_API_KEY"
#   base_url: "YOUR_GOOGLE_BASE_URL" # e.g., https://generativelanguage.googleapis.com/v1

# Example: Anthropic settings (this block would be used if generator_model contains "claude")
# anthropic_setting:
#   api_key: "YOUR_ANTHROPIC_API_KEY"
#   base_url: "YOUR_ANTHROPIC_BASE_URL" # e.g., https://api.anthropic.com

# Example: DeepSeek settings (this block would be used if generator_model contains "deepseek")
# deepseek_setting:
#  api_key: "YOUR_DEEPSEEK_API_KEY"
#  base_url: "YOUR_DEEPSEEK_BASE_URL" # e.g., https://api.deepseek.com/v1

# Model selection for different components.
# The name you provide for generator_model (and other *_model fields if they also use openai_generator.py)
# dictates which <provider>_setting block above must be present and correctly configured.
generator_model: "gpt-4o"
entity_extractor_model: "gpt-4o" # Assumes this also uses a model name that maps to openai_setting or has its own logic
question_generator_model: "gpt-4o"
polisher_model: "gpt-4o"
filter_model: "gpt-4o"
```

> **Code Reference**: The specific keyword-to-setting-block mapping logic is implemented in `HopWeaver/flashrag/generator/openai_generator.py`. Review this file to see exactly how model names are parsed to select configuration sections like `config["google_setting"]`, `config["anthropic_setting"]`, etc. It defaults to `config["openai_setting"]` if no other specific keywords are matched.

Additionally, different models (such as GPT-4, Claude, Qwen, DeepSeek, etc.) may require different generation parameters (e.g., temperature, top_p, max_tokens). Ensure these are appropriately set within the chosen `*_setting` block or the `generation_params` section of your configuration, as per your model's requirements.

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

# Retriever Diversity Weight Parameters (applies to diverse and rerank methods\' coarse retrieval stage)
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
- `--model_type`: Model type, here it\'s gte
- `--pooling_method`: Pooling method, GTE uses cls
- `--use_fp16`: Use FP16 to accelerate index building 