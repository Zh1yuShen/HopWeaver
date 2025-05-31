### 🛠️ 环境与数据准备

在开始使用 HopWeaver 之前，您需要完成以下准备工作：

#### 1. 克隆代码库并安装依赖

```bash
git clone https://github.com/Zh1yuShen/HopWeaver.git
cd HopWeaver
pip install -r requirements.txt
```

#### 2. 配置 LLM API

要使用本系统，请在您的 `config_lib/example_config.yaml`（或其副本）中配置LLM API。关键在于 `flashrag/generator/openai_generator.py` 脚本会基于您在 `generator_model` 中指定的模型名称中的**关键字**，从您的YAML文件中选择一个API配置块（例如 `openai_setting`, `google_setting`, `anthropic_setting`, `deepseek_setting`）。

- 如果 `generator_model` 包含 `"gemini"` (例如, `"gemini-1.5-pro"`)，脚本将尝试使用您YAML中的 `google_setting` 配置块。
- 如果 `generator_model` 包含 `"claude"` (例如, `"claude-3-sonnet-20240229"`)，它将使用 `anthropic_setting` 配置块。
- 如果 `generator_model` 包含 `"deepseek"` (例如, `"deepseek-chat"`)，它将使用 `deepseek_setting` 配置块。
- 如果没有找到这些关键字（或其他内部定义的关键字），则默认使用 `openai_setting` 配置块 (例如, 对于像 `"gpt-4o"` 这样的模型)。

您**必须**确保配置文件中包含正确命名的设置块 (例如, `google_setting`)，并且该块包含所选模型提供商所需的API密钥和任何其他必要参数。

```yaml
# 示例：OpenAI 设置 (如果 generator_model 为例如 "gpt-4o"，
# 或在 generator_model 名称中未找到如 "gemini", "claude", "deepseek" 等关键字时，将使用此配置块)
openai_setting:
  api_keys:
    - "your-openai-api-key-1"
    - "your-openai-api-key-2"
    - "your-openai-api-key-3"
  base_url: "https://api.openai.com/v1"

# 示例：Google 设置 (如果 generator_model 包含 "gemini"，将使用此配置块)
# 如果您将 generator_model 设置为 Gemini 模型，请确保此块存在且已正确填写。
# google_setting:
#   api_key: "YOUR_GOOGLE_API_KEY"
#   base_url: "YOUR_GOOGLE_BASE_URL" # 例如 https://generativelanguage.googleapis.com/v1

# 示例：Anthropic 设置 (如果 generator_model 包含 "claude"，将使用此配置块)
# anthropic_setting:
#   api_key: "YOUR_ANTHROPIC_API_KEY"
#   base_url: "YOUR_ANTHROPIC_BASE_URL" # 例如 https://api.anthropic.com

# 示例：DeepSeek 设置 (如果 generator_model 包含 "deepseek"，将使用此配置块)
# deepseek_setting:
#   api_key: "YOUR_DEEPSEEK_API_KEY"
#   base_url: "YOUR_DEEPSEEK_BASE_URL" # 例如 https://api.deepseek.com/v1

# 不同组件的模型选择。
# 您为 generator_model (以及其他也使用 openai_generator.py 的 *_model 字段)
# 提供的名称决定了上方哪个 <provider>_setting 配置块必须存在且已正确配置。
generator_model: "gpt-4o"
entity_extractor_model: "gpt-4o" # 假设此模型名称也映射到 openai_setting 或有其自己的逻辑
question_generator_model: "gpt-4o"
polisher_model: "gpt-4o"
filter_model: "gpt-4o"
```

> **代码参考**：具体的关键字到设置块的映射逻辑在 `HopWeaver/flashrag/generator/openai_generator.py` 中实现。请查看此文件以了解模型名称是如何被解析以选择像 `config["google_setting"]`、`config["anthropic_setting"]` 等配置节的。如果未匹配到其他特定关键字，则默认为 `config["openai_setting"]`。

同时，不同模型（如 GPT-4、Claude、Qwen、DeepSeek 等）可能需要不同的生成参数（例如 temperature、top_p、max_tokens）。请根据您模型的特性，在所选的 `*_setting` 配置块或配置文件的 `generation_params` 部分进行适当设置。

##### 🤖 模型选择建议

HopWeaver 由几个可以使用不同模型的组件组成。以下是基于我们实验的建议：

- **polisher_model**：我们建议为语言润色组件使用 DeepSeek-R1 或更高级的模型，因为它需要强大的语言优化能力
- **其他组件**：您可以为其他组件（entity_extractor, question_generator, filter 等）使用相同的模型。我们建议为所有合成组件选择同一个模型。在我们的论文中，我们成功测试了各种模型，包括：
  - QwQ-32B
  - Qwen3-14B
  - GLM-9B-0414

为了获得最佳性能，我们建议使用至少 7B 参数的模型。较小的模型可能难以处理多跳问题合成所需的复杂推理。

##### 💻 本地模型配置

您可以使用[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)提供的本地模型支持，它支持多种本地模型部署方式

##### ⚡ API调用优化

HopWeaver实现了如下优化机制，提高了API调用的稳定性和效率：

1. **🔄 多个API Key轮询**: 当配置多个API Key时，系统会自动轮询使用，分散请求率限制

   ```yaml
   openai_setting:
     api_keys:
       - "key1"
       - "key2"
       - "key3"  # 多个API Key列表
   ```

2. **🔄 错误自动重试**: 当遇到常见API错误(如速率限制、服务器错误)时，系统会自动重试

3. **⚡ 异步请求处理**: 支持批量异步请求，最大化利用API调用频率

这些机制使得HopWeaver在面对大量多跳问题合成时，能更高效地利用LLM API资源。

#### 3. 多API提供商支持

HopWeaver支持多种API提供商，提供更强的灵活性和冗余能力。您可以在配置文件中配置不同的提供商：

```yaml
# 多API提供商配置
api_type: "openai"  # 主要API类型

# OpenAI 配置
openai_setting:
  api_keys:
    - "your-openai-api-key-1"
    - "your-openai-api-key-2"
  base_url: "https://api.openai.com/v1"

# Google Gemini 配置
gemini_setting:
  api_keys:
    - "your-gemini-api-key-1"
    - "your-gemini-api-key-2"
  base_url: "https://generativelanguage.googleapis.com/v1"

# DeepSeek 配置
deepseek_setting:
  api_key: "your-deepseek-api-key"
  base_url: "https://api.deepseek.com/v1"

# Claude (Anthropic) 配置
claude_setting:
  api_key: "your-claude-api-key"
  base_url: "https://api.anthropic.com"

# OpenRouter 配置（支持多种模型）
openrouter_setting:
  api_keys:
    - "your-openrouter-key-1"
    - "your-openrouter-key-2"
  base_url: "https://openrouter.ai/api/v1"

# GLM (SiliconFlow) 配置
GLM_setting:
  api_keys: "your-glm-api-key"
  base_url: "https://api.siliconflow.cn/v1"
```

**各提供商支持的模型：**
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo 等
- **Google**: Gemini-2.0-flash, Gemini-2.5-flash-preview 等
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3 等
- **Claude**: Claude-3.5-Sonnet 等
- **OpenRouter**: 可访问 QwQ-32B, Gemma-3-27B 等模型
- **GLM**: GLM-4-9B 和其他 SiliconFlow 支持的模型

#### 4. 全局路径映射配置

HopWeaver使用全局路径映射来高效管理模型路径、索引和语料库：

```yaml
# 全局路径映射
model2path:
  e5: "/path/to/e5-base-v2"
  gte: "/path/to/gte_sentence-embedding_multilingual-base"

# 各嵌入模型的池化方法
model2pooling:
  e5: "mean"
  gte: "cls"

# 检索模型的索引路径
method2index:
  e5: '/path/to/e5_Flat.index'
  gte: '/path/to/gte_Flat.index'
  bm25: ~
  contriever: ~

# 不同方法的语料库路径
method2corpus:
  e5: '/path/to/wiki18_fulldoc_trimmed_4096.jsonl'
  gte: '/path/to/wiki18_fulldoc_trimmed_4096.jsonl'
```

**配置优势：**
- **集中管理**：所有模型和数据路径集中在一个位置
- **便捷切换**：通过修改 `retrieval_method` 参数即可切换检索方法
- **自动解析**：系统根据方法选择自动解析对应路径
- **可扩展性**：易于添加新的模型和语料库

#### 5. 检索器高级参数

为了精细控制检索过程，可配置以下高级参数：

**📊 检索方法选择：**

HopWeaver支持三种检索方法：
- **standard**：标准检索，仅基于查询相关性排序
- **diverse**：多样性检索，使用MMR算法平衡相关性和多样性
- **rerank**：两阶段检索，先进行多样性检索，再使用训练好的重排模型精细排序

**🔄 重排器模型配置：**

HopWeaver支持使用重排器模型进一步优化检索结果的排序。您可以选择以下重排器模型：

**开源重排器模型：**
- **[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)**: 这是一个基于bge-m3的轻量级多语言重排器模型，具有强大的多语言能力，易于部署，推理速度快。该模型支持中文和英文，在多种基准测试中表现出色。

**自定义微调重排器：**
- **HopWeaver微调重排器**: 我们基于特定的多跳问答数据进行了微调，专门针对多跳问题的文档检索进行了优化。该模型在处理跨文档推理任务时表现更佳。（模型链接即将在HuggingFace或ModelScope上发布）

**重排器配置示例：**

```yaml
# 检索器配置
retriever_type: "rerank"  # 检索方法选择，选项："standard"、"diverse" 或 "rerank"
reranker_path: "/path/to/trained/reranker/model"  # 重排模型路径（仅在rerank方法时需要）

# 检索器多样性权重参数（适用于diverse和rerank方法的粗检索阶段）
lambda1: 0.87  # 查询相关性权重 (0-1)
lambda2: 0.03  # 原始文档多样性权重 (0-1)
lambda3: 0.1   # 已选文档多样性权重 (0-1)

# 重排器性能参数
use_reranker: true            # 启用重排器
rerank_topk: 5               # 重排后保留的文档数量
rerank_max_length: 4096      # 重排器输入的最大长度
rerank_batch_size: 256       # 重排批处理大小
rerank_use_fp16: true        # 使用FP16加速重排器推理

# 检索缓存（性能优化）
save_retrieval_cache: false    # 保存检索结果到缓存
use_retrieval_cache: false     # 使用缓存的检索结果
retrieval_cache_path: ~        # 检索缓存文件路径
```

**参数调优指南：**
- **lambda1 (0.8-0.9)**：更高的值优先考虑查询-文档相关性
- **lambda2 (0.05-0.15)**：控制与源文档的多样性
- **lambda3 (0.05-0.15)**：控制已选文档间的多样性
- **lambda1+lambda2+lambda3 的和应等于 1.0**

**性能提示：**
- 使用 `use_fp16: true` 可以获得更快的推理速度，质量损失极小
- 根据GPU内存调整 `reranker_batch_size`
- 对于重复实验相同查询，启用缓存可提高效率

#### 6. 下载Wiki数据集

您需要下载`wiki18_fulldoc_trimmed_4096.jsonl`数据文件，这是我们预处理好的Wiki数据集，包含截取了文档长度小于4096的Wiki文章。

数据集下载链接: [huggingface](https://huggingface.co/datasets/Shenzy2/HopWeaver_Data) or [modelscope](https://www.modelscope.cn/datasets/szyszy/HopWeaver_Data)

对于我们论文中比较的HotpotQA、2wiki、musique的步骤，可以将下载的数据集放入 datasets 文件夹中，并且用 datasets/process_and_sample_datasets.py 处理这些采样出任意样本，用于后续比较。

**数据格式说明**：
`wiki18_fulldoc_trimmed_4096.jsonl`是JSONL格式文件，每行包含一个JSON对象，结构如下：
```json
{
  "id": "591775",
  "title": "Los Ramones",
  "doc_size": 1250,
  "contents": "Los Ramones\nLos Ramones Los Ramones is the name of a municipality..."
}
```

**字段说明**：
- `id`: 文档的唯一标识符
- `title`: 文档标题
- `doc_size`: 文档内容的字符长度
- `contents`: 文档的完整正文内容

#### 7. 下载GTE嵌入模型

HopWeaver使用[GTE](https://huggingface.co/iic/gte_sentence-embedding_multilingual-base)多语言模型进行检索。您可以直接从Hugging Face下载该模型，并在配置文件中指定路径：

修改配置文件`config_lib/example_config.yaml`中的模型路径：
```yaml
model2path:
  gte: "您下载的GTE模型路径"
```

#### 8. 下载或构建索引

您可以选择下载我们预构建好的索引文件(
 [huggingface](https://huggingface.co/datasets/Shenzy2/HopWeaver_Data) or [modelscope](https://www.modelscope.cn/datasets/szyszy/HopWeaver_Data))，或自行构建：

```bash
# 创建索引保存目录
mkdir -p index

# 下载预构建索引（推荐）
# 索引下载链接: [INDEX_DOWNLOAD_LINK_PLACEHOLDER]

# 或者使用FlashRAG构建索引
python -m flashrag.build_index \
  --model_name_or_path 您下载的GTE模型路径 \
  --corpus_path dataset/wiki18_fulldoc_trimmed_4096.jsonl \
  --index_path index/gte_Flat.index \
  --batch_size 32 \
  --model_type gte \
  --pooling_method cls \
  --use_fp16
```

参数说明：
- `--model_name_or_path`: GTE模型路径
- `--corpus_path`: Wiki语料库文件路径
- `--index_path`: 生成的索引保存路径
- `--batch_size`: 批处理大小，可根据您的GPU内存调整
- `--model_type`: 模型类型，这里是gte
- `--pooling_method`: 池化方法，GTE使用cls
- `--use_fp16`: 使用FP16以加速索引构建

完成上述准备工作后，您就可以开始使用HopWeaver合成多跳问题了。 