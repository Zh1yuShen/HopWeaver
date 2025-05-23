### 1. Bridge Question Synthesis Process

The bridge question synthesis includes the following key steps:

- **Bridge Entity Identification**: From randomly selected source documents, the system identifies bridge entities that can connect different information contexts, providing key pivots for multi-hop reasoning
  
- **Two-stage Coarse-to-Fine Retrieval**:
  - Coarse-grained Retrieval: Using a modified maximum marginal relevance algorithm to balance query relevance, diversity from source documents, and diversity among selected documents
  
    **Diverse Retrieval Scoring Function:**
    
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
  
  - Fine-grained Reranking: Using a reranking model fine-tuned through contrastive learning to further optimize the ranking of candidate documents

- **Multi-hop Question Construction**:
  - Sub-question Synthesis: Synthesize sub-questions from source and supplementary documents respectively, centered around the bridge entity
  - Question Synthesis: Merge sub-questions into a single coherent multi-hop question, implying the reasoning path without directly exposing the bridge entity
  - Validation and Iteration: Ensure questions meet answerability, multi-hop nature, and no-shortcut constraints

### 2. Comparison Question Synthesis Process

Comparison question synthesis follows these steps:

- **Entity and Attribute Identification**: Identify main entities from documents and their 3-5 concise factual attribute-value pairs, filtering out attributes suitable for comparison

- **Filtering and Query Synthesis**:
  - Ensure specificity and comparability of entities and attributes
  - Synthesize retrieval queries based on source entities, using direct recommendation or diversified search strategies

- **Question Construction**:
  - Guided Comparison: Precise comparison for specific entities and attributes
  - Open Discovery: Find the first valid comparable pair among multiple attributes
  - Synthesize comparison questions containing information about two entities, such as "Which entity has a higher/earlier/larger attribute value?"

### 3. Question Refinement and Quality Assurance

During the bridge and comparison question synthesis process, the system implements strict quality control mechanisms:

- **Question Refinement and Validation Module**:
  - Evaluate questions for answerability, multi-hop nature, and language quality
  - Classify evaluation results into four categories: pass, adjust, reconstruct, or reject
  - Ensure each question involves cross-document reasoning and hides bridge entities
  - Maintain fluency without exposing intermediate reasoning steps

### 4. Reranker Model Training and Optimization

The system synthesizes supervision signals by simulating key steps to improve retrieval quality:

- **Simulated Feedback Synthesis**:
  - Extract successful and failed document examples from the bridge question synthesis process
  - Construct contrastive training triplets (query, positive document, negative document)

- **Contrastive Learning Optimization**:
  - Use cross-entropy loss function to guide the model in distinguishing complementary documents
  - Obtain supervision signals directly from downstream task success rates

### 5. Multi-dimensional Evaluation System

The system employs a comprehensive evaluation framework to ensure question quality:

- **LLM-as-Judge Evaluation**:
  - Use large language models as judges, employing Likert scales to evaluate each question
  - Implement self-consistency evaluation methods to ensure stability and reproducibility of evaluation results
  - Analyze consistency of evaluation results by repeatedly evaluating the same input

- **Answerability and Difficulty Evaluation**:
  - **Q-Only Condition**: Solver receives only the question, testing baseline answerability using the solver's internal knowledge and reasoning capabilities
  - **Q+Docs Condition**: Solver receives the question and all supporting documents, simulating a golden retrieval scenario to evaluate answerability when necessary evidence is available
  - **Performance Gap Analysis**: Performance improvement from Q-Only to Q+Docs indicates whether the question is challenging and requires cross-document reasoning rather than relying solely on pre-trained knowledge

- **Evidence-Accessibility Evaluation**:
  - **Retrieval Quality Assessment**: Use multiple retrieval methods to fetch top-k documents and evaluate the accessibility of synthesized question evidence in the corpus
  - **Multi-dimensional Retrieval Metrics**: Record MAP (Mean Average Precision), RECALL@k, NDCG@k (Normalized Discounted Cumulative Gain), and Support F1 metrics
  - **Evidence Completeness Verification**: Ensure synthesized questions have complete evidence support, preventing unanswerable questions from entering the final dataset 