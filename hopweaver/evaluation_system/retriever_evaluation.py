#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import time
import argparse
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Import reranker model
try:
    from FlagEmbedding import FlagReranker
    has_reranker = True
except ImportError:
    print("Warning: FlagEmbedding package not installed, hybrid retrieval functionality will be unavailable")
    has_reranker = False

# Add project root directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import FlashRAG retriever
from flashrag.retriever.retriever import DenseRetriever, BM25Retriever
from flashrag.config import Config

# Load YAML configuration file
config_file_path = os.path.join(project_root, 'config_lib/extract_config_wikifulldoc.yaml')
with open(config_file_path, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

def load_jsonl(file_path):
    """Load JSONL file
    Supports non-standard format, i.e., a JSON object can span multiple lines
    """
    # Handle non-standard JSONL files (each JSON object might span multiple lines)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to convert content to a JSON array
    try:
        # Concatenate multiple JSON objects into an array
        fixed_content = content.replace('}\n{', '},\n{')
        data = json.loads('[' + fixed_content + ']')
        print(f"Successfully loaded {len(data)} items")
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
        raise e

def hybrid_retriever(methods: List[str], top_k: int = 10, strategy="default", reranker_path=None):
    """
    Create a hybrid retriever that combines multiple retrieval methods
    
    Args:
        methods: List of retrieval methods, e.g., ['e5', 'bm25']
        top_k: Number of documents to return
        strategy: Hybrid retrieval strategy (default, interleaving, rrf, dense_first, bm25_first)
        reranker_path: (Deprecated) Kept for compatibility only
        
    Returns:
        Hybrid retriever object, supports search method
    """
    # Initialize each retriever
    retrievers = [get_retriever(method, top_k) for method in methods]
    
    # Create HybridRetriever class
    class HybridRetriever:
        """Hybrid retriever supporting multiple merging strategies"""
        def __init__(self, retrievers, top_k, strategy="default"):
            self.retrievers = retrievers
            self.top_k = top_k
            self.strategy = strategy
            
            # Categorize dense and bm25 retrievers based on retriever type
            self.dense_retrievers = [r for r in retrievers if not isinstance(r, BM25Retriever)]
            self.bm25_retrievers = [r for r in retrievers if isinstance(r, BM25Retriever)]
            
            # Build method name
            if strategy != "default":
                self.method = f"hybrid_{strategy}:{'+'.join([r.retrieval_method for r in retrievers])}"
            else:
                self.method = f"hybrid:{'+'.join([r.retrieval_method for r in retrievers])}"
            
        def extract_doc_id(self, doc):
            """Extract document IDs"""
            return str(doc.get("id", None)) or str(doc.get("doc_id", None)) or str(doc.get("docid", None))
        
        def default_fusion(self, query, num):
            """1. Default merge method: Allocate proportionally to 3, add sequentially"""
            per_retriever_num = num // len(self.retrievers) + 1  # Round up to ensure total is not less than num
            all_docs = []
            
            for retriever in self.retrievers:
                docs = retriever.search(query, num=per_retriever_num)
                all_docs.extend(docs)
            
            # Deduplicate
            unique_docs = []
            seen_ids = set()
            for doc in all_docs:
                doc_id = self.extract_doc_id(doc)
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            
            return unique_docs[:num]
        
        def interleaving_fusion(self, query, num):
            """2. Interleaving merge: Alternately select from each result set"""
            # Get a maximum of num results from each retriever
            retriever_results = []
            for retriever in self.retrievers:
                docs = retriever.search(query, num=num)
                retriever_results.append(docs)
            
            # Interleave merge
            fused_results = []
            seen_ids = set()
            max_len = max(len(results) for results in retriever_results)
            
            # Alternately add results from each retriever
            for i in range(max_len):
                for results in retriever_results:
                    if i < len(results):
                        doc = results[i]
                        doc_id = self.extract_doc_id(doc)
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            fused_results.append(doc)
                            
                        # If the required number is met, return early
                        if len(fused_results) >= num:
                            return fused_results[:num]
            
            return fused_results[:num]
        
        def rrf_fusion(self, query, num, k=60):
            """3. Reciprocal Rank Fusion (RRF)"""
            # Get a maximum of num*2 results from each retriever (to ensure sufficient diversity)
            retriever_results = []
            for retriever in self.retrievers:
                docs = retriever.search(query, num=num*2)
                retriever_results.append(docs)
            
            # Calculate RRF score for each document
            fused_scores = defaultdict(float)
            doc_objects = {}
            
            for results in retriever_results:
                for rank, doc in enumerate(results):
                    doc_id = self.extract_doc_id(doc)
                    if doc_id:
                        # RRF score formula: 1/(k + rank)
                        fused_scores[doc_id] += 1.0 / (k + rank)
                        doc_objects[doc_id] = doc
            
            # Sort by score
            sorted_doc_ids = sorted(fused_scores.keys(), key=lambda doc_id: fused_scores[doc_id], reverse=True)
            return [doc_objects[doc_id] for doc_id in sorted_doc_ids[:num]]
        
        def dense_first_fusion(self, query, num):
            """4. Dense first: Use Dense retrievers first, then supplement with BM25"""
            all_docs = []
            seen_ids = set()
            
            # Use all Dense retrievers first
            for retriever in self.dense_retrievers:
                docs = retriever.search(query, num=num)
                for doc in docs:
                    doc_id = self.extract_doc_id(doc)
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            # If not enough, supplement with BM25
            for retriever in self.bm25_retrievers:
                if len(all_docs) >= num:
                    break
                    
                docs = retriever.search(query, num=num)
                for doc in docs:
                    doc_id = self.extract_doc_id(doc)
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        
                        if len(all_docs) >= num:
                            break
            
            return all_docs[:num]
        
        def bm25_first_fusion(self, query, num):
            """5. BM25 first: Use BM25 retrievers first, then supplement with Dense"""
            all_docs = []
            seen_ids = set()
            
            # Use all BM25 retrievers first
            for retriever in self.bm25_retrievers:
                docs = retriever.search(query, num=num)
                for doc in docs:
                    doc_id = self.extract_doc_id(doc)
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            
            # If not enough, supplement with Dense
            for retriever in self.dense_retrievers:
                if len(all_docs) >= num:
                    break
                    
                docs = retriever.search(query, num=num)
                for doc in docs:
                    doc_id = self.extract_doc_id(doc)
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        
                        if len(all_docs) >= num:
                            break
            
            return all_docs[:num]
        
        def search(self, query, num=None, return_score=False):
            """Execute corresponding hybrid method based on strategy"""
            if num is None:
                num = self.top_k
                
            # Select merge method based on strategy
            if self.strategy == "interleaving":
                final_results = self.interleaving_fusion(query, num)
            elif self.strategy == "rrf":
                final_results = self.rrf_fusion(query, num)
            elif self.strategy == "dense_first":
                final_results = self.dense_first_fusion(query, num)
            elif self.strategy == "bm25_first":
                final_results = self.bm25_first_fusion(query, num)
            else:  # default
                final_results = self.default_fusion(query, num)
            
            if return_score:
                # Since there are no actual scores, use decreasing dummy scores
                fake_scores = [1.0 - 0.01 * i for i in range(len(final_results))]
                return final_results, fake_scores
            return final_results
        
        def batch_search(self, query_list, num=None, return_score=False):
            """Batch retrieval interface"""
            if isinstance(query_list, str):
                query_list = [query_list]
                
            all_results = []
            all_scores = []
            
            for query in query_list:
                results, scores = self.search(query, num=num, return_score=True)
                all_results.append(results)
                all_scores.append(scores)
                
            if return_score:
                return all_results, all_scores
            return all_results
    
    return HybridRetriever(retrievers, top_k, strategy)

def get_retriever(method: str, top_k: int = 10):
    """Initialize and return a retriever
    
    Args:
        method: Retrieval method, supports e5/gte/bm25
        top_k: Number of documents to return
        
    Returns:
        BaseRetriever: FlashRAG retriever instance
    """

    
    # Get index and corpus paths from the configuration file
    method2index = yaml_config.get("method2index", {})
    method2corpus = yaml_config.get("method2corpus", {})
    
    # Check if the method is available
    if method not in method2index:
        raise ValueError(f"Unsupported retrieval method: {method} or missing configuration in the config file")
    
    # Load retrieval settings from the configuration file
    config = {
        # Basic retrieval parameters
        "retrieval_method": method,
        "retrieval_topk": top_k,  # Override default config with the passed top_k
        "index_path": method2index[method],
        "corpus_path": method2corpus[method],
        "method2corpus": method2corpus,
        
        # Load query parameters from the configuration file
        "retrieval_query_max_length": yaml_config.get("retrieval_query_max_length", 4096),
        "retrieval_batch_size": yaml_config.get("retrieval_batch_size", 256),
        "retrieval_use_fp16": yaml_config.get("retrieval_use_fp16", True),
        
        # Cache-related parameters
        "save_retrieval_cache": yaml_config.get("save_retrieval_cache", False),
        "use_retrieval_cache": yaml_config.get("use_retrieval_cache", False),
        "retrieval_cache_path": yaml_config.get("retrieval_cache_path", "/tmp/retrieval_cache"),
        
        # Reranking-related parameters
        "use_reranker": yaml_config.get("use_reranker", False),
        
        # Hardware acceleration
        "faiss_gpu": yaml_config.get("faiss_gpu", False),
        
        # Instruction
        "instruction": yaml_config.get("instruction", "Represent the document for retrieval: "),
        
        # Add parameters specific to vector retrievers
        "use_sentence_transformer": True  # Use sentence-transformers framework
    }
    
    # Set pooling method based on model2pooling in the configuration file
    model2pooling = yaml_config.get("model2pooling", {})
    if method in model2pooling:
        config["retrieval_pooling_method"] = model2pooling.get(method, "mean")
    
    # Get model path from the configuration file
    model2path = yaml_config.get("model2path", {})
    if method in model2path and method != "bm25":
        config["retrieval_model_path"] = model2path.get(method)
    
    # For BM25 method, additional configuration is needed
    if method == "bm25":
        config["bm25_backend"] = "bm25s"  # Use faster bm25s as backend
        return BM25Retriever(config)
    else:
        # For vector retrieval methods
        return DenseRetriever(config)

def calculate_map(relevant_docs, retrieved_docs, k=10):
    """Calculate MAP (Mean Average Precision)
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        float: MAP value
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # Calculate precision at each position
    hits = 0
    sum_precisions = 0.0
    
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            hits += 1
            precision_at_i = hits / (i + 1)  # Precision at current position
            sum_precisions += precision_at_i
    
    # Calculate average precision
    return sum_precisions / len(relevant_docs) if hits > 0 else 0.0

def calculate_recall_at_k(relevant_docs, retrieved_docs, k=10):
    """Calculate Recall@k
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        float: Recall@k value
    """
    if not relevant_docs:
        return 0.0
    
    # Calculate number of relevant documents found in top k results
    hits = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)
    
    # Calculate recall
    return hits / len(relevant_docs)

def calculate_ndcg_at_k(relevant_docs, retrieved_docs, k=10):
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain at k)
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        float: NDCG@k value
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        # Simplified version: if document is relevant, rel=1, otherwise rel=0
        rel = 1 if doc_id in relevant_docs else 0
        # DCG formula: rel_i / log2(i+2)  # i starts from 0, so it's i+2
        dcg += rel / np.log2(i + 2)
    
    # Calculate IDCG (ideal situation)
    # Ideal situation is when all relevant documents are at the top
    num_rel = min(len(relevant_docs), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_rel))
    
    # If there are no relevant documents, return 0
    if idcg == 0:
        return 0.0
    
    # Calculate NDCG
    return dcg / idcg

def calculate_support_f1(relevant_docs, retrieved_docs, k=10):
    """Calculate Support F1 score
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        float: Support F1 value
    """
    if not relevant_docs:
        return 0.0
    
    # Calculate number of relevant documents found in top k results
    true_positives = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)
    
    # If there are no true positives, return 0
    if true_positives == 0:
        return 0.0
    
    # Calculate precision
    precision = true_positives / len(retrieved_docs[:k])
    
    # Calculate recall
    recall = true_positives / len(relevant_docs)
    
    # Calculate F1 score
    return 2 * precision * recall / (precision + recall)

def calculate_metrics(relevant_docs_list, retrieved_docs_list, k=20):
    """Calculate evaluation metrics for multiple documents
    
    Args:
        relevant_docs_list: List of lists of relevant document IDs
        retrieved_docs_list: List of lists of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # If there are no queries, return empty results
    if not relevant_docs_list or not retrieved_docs_list:
        return {
            "map": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "support_f1": 0.0
        }
    
    # Fixed k values
    k5 = 5
    k10 = 10
    k20 = 20
    
    # Initialize metric sums
    map_sum = 0.0
    recall_at_5_sum = 0.0
    recall_at_10_sum = 0.0
    recall_at_20_sum = 0.0
    ndcg_at_5_sum = 0.0
    ndcg_at_10_sum = 0.0
    support_f1_sum = 0.0
    
    # Iterate over each query
    valid_queries = 0
    for relevant_docs, retrieved_docs in zip(relevant_docs_list, retrieved_docs_list):
        if not relevant_docs or not retrieved_docs:
            continue
            
        # Increment valid query count
        valid_queries += 1
        
        # Calculate metrics
        map_sum += calculate_map(relevant_docs, retrieved_docs, k10)
        recall_at_5_sum += calculate_recall_at_k(relevant_docs, retrieved_docs, k5)
        recall_at_10_sum += calculate_recall_at_k(relevant_docs, retrieved_docs, k10)
        recall_at_20_sum += calculate_recall_at_k(relevant_docs, retrieved_docs, k20)
        ndcg_at_5_sum += calculate_ndcg_at_k(relevant_docs, retrieved_docs, k5)
        ndcg_at_10_sum += calculate_ndcg_at_k(relevant_docs, retrieved_docs, k10)
        support_f1_sum += calculate_support_f1(relevant_docs, retrieved_docs, k10)
    
    # If there are no valid queries, return 0
    if valid_queries == 0:
        return {
            "map": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "support_f1": 0.0
        }
    
    # Calculate averages
    return {
        "map": map_sum / valid_queries,
        "recall@5": recall_at_5_sum / valid_queries,
        "recall@10": recall_at_10_sum / valid_queries,
        "recall@20": recall_at_20_sum / valid_queries,
        "ndcg@5": ndcg_at_5_sum / valid_queries,
        "ndcg@10": ndcg_at_10_sum / valid_queries,
        "support_f1": support_f1_sum / valid_queries
    }

def find_target_document_rank(results, target_id):
    """
    Find the rank of the target document in the retrieval results
    
    Args:
        results: List of retrieval results
        target_id: ID of the target document
        
    Returns:
        int: Rank of the target document, or len(results) + 1 if not found
    """
    for i, doc in enumerate(results):
        if isinstance(doc, dict) and 'id' in doc and str(doc['id']) == str(target_id):
            return i + 1  # Rank starts from 1
        elif isinstance(doc, str) and str(doc) == str(target_id):
            return i + 1
    return len(results) + 1  # If not found, return len(results) + 1

def evaluate_retriever(data, method, top_k=20, reranker_path=None):
    """
    Evaluate the performance of a retriever
    
    Args:
        data: List of data samples
        method: Retrieval method (e.g., 'e5', 'bm25', 'hybrid:e5+bm25')
        top_k: Number of documents to retrieve
        reranker_path: (Deprecated) Path to the reranker model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Check if the method is a hybrid method
        if method.startswith("hybrid"):
            # Parse the hybrid method and strategy
            strategy = "default"
            
            if method.startswith("hybrid_"):
                # Handle hybrid methods with a specific strategy (e.g., hybrid_rrf:e5+bm25)
                strategy_parts = method.split(":")
                if len(strategy_parts) != 2:
                    raise ValueError(f"Invalid hybrid method format: {method}, should be hybrid_strategy:method1+method2+...")
                
                strategy_name = strategy_parts[0]
                strategy = strategy_name.replace("hybrid_", "")
                method_part = strategy_parts[1]
            else:
                # Handle standard hybrid methods (e.g., hybrid:e5+bm25)
                parts = method.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid hybrid method format: {method}, should be hybrid:method1+method2+...")
                
                method_part = parts[1]
            
            # Parse the retrieval methods
            retriever_methods = method_part.split("+")
            
            # Create a hybrid retriever
            print(f"Creating hybrid retriever with methods: {retriever_methods}, strategy: {strategy}")
            retriever = hybrid_retriever(retriever_methods, top_k, strategy=strategy, reranker_path=reranker_path)
        else:
            # Single retrieval method
            retriever = get_retriever(method, top_k)
        
        # Prepare evaluation results
        relevant_docs_list = []  # List of relevant documents for each query
        retrieved_docs_list = []  # List of retrieved documents for each query
        per_sample_results = []
        
        # Evaluate each sample
        for i, sample in enumerate(tqdm(data, desc=f"Evaluating {method}")):
            # Collect relevant documents for the query
            relevant_docs = []
            
            # Extract source document ID
            source_info = sample.get('source_doc', {})
            source_id = None
            if isinstance(source_info, dict) and 'id' in source_info:
                source_id = source_info.get('id')
                if source_id:  # Ensure it's valid before adding
                    relevant_docs.append(str(source_id))
            
            # Extract target document ID
            target_info = sample.get('target_doc', {})
            target_id = None
            if isinstance(target_info, dict) and 'id' in target_info:
                target_id = target_info.get('id')
                if target_id and target_id != source_id:  # Ensure it's valid and not duplicate
                    relevant_docs.append(str(target_id))
            
            # If there are no relevant documents, skip this sample
            if not relevant_docs:
                print(f"\nWarning: Sample {i} is missing relevant document IDs")
                continue
                
            # Extract the query
            if 'multi_hop_question' in sample and isinstance(sample['multi_hop_question'], dict):
                query = sample['multi_hop_question'].get('multi_hop_question', '')
            else:
                query = sample.get('query', '')  # If no multi_hop_question, try to use the query field
                
            # Ensure the query is not empty
            if not query:
                print(f"\nWarning: Sample {i} is missing the query field")
                continue
            
            # Perform retrieval
            results = retriever.search(query, num=top_k)
            
            # Extract retrieved document IDs
            retrieved_doc_ids = []
            if method == "bm25":
                # BM25 retriever results might be a list of dictionaries
                for result in results:
                    if isinstance(result, dict):
                        if "id" in result:
                            retrieved_doc_ids.append(str(result["id"]))
                        elif "docid" in result:
                            retrieved_doc_ids.append(str(result["docid"]))
                        elif "doc_id" in result:
                            retrieved_doc_ids.append(str(result["doc_id"]))
                    elif isinstance(result, str):
                        retrieved_doc_ids.append(str(result))
            else:
                # Dense retriever results are typically a list of (doc_id, score) pairs
                try:
                    retrieved_doc_ids = [str(doc_id) for doc_id, _ in results]
                except (ValueError, TypeError):
                    # If unpacking fails, try other formats
                    for result in results:
                        if isinstance(result, dict):
                            if "id" in result:
                                retrieved_doc_ids.append(str(result["id"]))
                            elif "docid" in result:
                                retrieved_doc_ids.append(str(result["docid"]))
                            elif "doc_id" in result:
                                retrieved_doc_ids.append(str(result["doc_id"]))
                        elif isinstance(result, tuple) and len(result) >= 1:
                            retrieved_doc_ids.append(str(result[0]))
                        elif isinstance(result, list) and len(result) >= 1:
                            retrieved_doc_ids.append(str(result[0]))
                        elif isinstance(result, str):
                            retrieved_doc_ids.append(str(result))
            
            # Add to evaluation results
            relevant_docs_list.append(relevant_docs)
            retrieved_docs_list.append(retrieved_doc_ids)
            
            # Calculate metrics for this sample
            # Recall calculation: number of relevant documents found / total number of relevant documents
            hits = sum(1 for doc_id in retrieved_doc_ids[:top_k] if doc_id in relevant_docs)
            recall = hits / len(relevant_docs) if relevant_docs else 0.0
            
            # Calculate F1 score for this sample
            precision = hits / top_k if top_k > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Record detailed results for this sample
            sample_result = {
                "question": query,
                "relevant_docs": relevant_docs,  # All relevant document IDs
                "retrieved_docs": retrieved_doc_ids[:top_k],  # Retrieved document IDs
                "recall@k": recall,
                "precision@k": precision,
                "f1@k": f1
            }
            per_sample_results.append(sample_result)
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(relevant_docs_list, retrieved_docs_list)
        metrics["method"] = method
        metrics["per_sample_results"] = per_sample_results
        
        return metrics
    except Exception as e:
        print(f"Error evaluating {method}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"method": method, "error": str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate retriever performance")
    parser.add_argument("--data_path", required=True, help="Path to the input JSONL file")
    parser.add_argument("--methods", nargs="+", default=["gte", "e5", "bm25"], 
                        help="Retrieval methods to evaluate (e.g., gte, e5, bm25, hybrid:e5+bm25)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory")
    parser.add_argument("--sample_num", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--reranker_path", default="./models/bge-m3", help="Path to the reranker model")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    full_data = load_jsonl(args.data_path)
    print(f"Loaded {len(full_data)} samples")
    
    # If a sample number is specified, use only that many samples
    if args.sample_num is not None and args.sample_num > 0 and args.sample_num < len(full_data):
        data = full_data[:args.sample_num]
        print(f"Using only the first {args.sample_num} samples")
    else:
        data = full_data
    
    # Print data statistics
    print(f"Evaluating {len(data)} samples")
    
    # Record evaluation results for all methods
    all_results = []
    
    # Evaluate each method
    for method in args.methods:
        print(f"\nEvaluating {method}")
        result = evaluate_retriever(data, method, args.top_k, args.reranker_path)
        all_results.append(result)
    
    # Save evaluation results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"retrieval_eval_{timestamp}.json")
    
    # Save detailed results for each sample
    per_sample_path = os.path.join(args.output_dir, f"per_sample_results_{timestamp}.json")
    
    # Print summary results
    print("\nRetriever evaluation summary:")
    metrics_table = [["Method", "MAP", "Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "Support F1"]]
    for method_result in all_results:
        if "error" in method_result:
            metrics_table.append([method_result["method"], "Error", "Error", "Error", "Error", "Error", "Error", "Error"])
        else:
            metrics_table.append([
                method_result["method"],
                f"{method_result['map']:.4f}",
                f"{method_result['recall@5']:.4f}",
                f"{method_result['recall@10']:.4f}",
                f"{method_result['recall@20']:.4f}",
                f"{method_result['ndcg@5']:.4f}",
                f"{method_result['ndcg@10']:.4f}",
                f"{method_result['support_f1']:.4f}"
            ])
    
    # Print a simple table
    col_widths = [max(len(row[i]) for row in metrics_table) for i in range(len(metrics_table[0]))]
    for row in metrics_table:
        print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    
    # Save all results
    print(f"\nSaving evaluation results to {output_path} and {per_sample_path}")
    
    # Add evaluation description
    summary_results = {}
    summary_results["evaluation_description"] = {
        "MAP": "Mean Average Precision, evaluating the ranking performance of the retriever",
        "Recall@k": "Recall at k, evaluating the proportion of relevant documents found in the top k results",
        "NDCG@k": "Normalized Discounted Cumulative Gain at k, evaluating the ranking quality of the retriever",
        "Support F1": "Support F1 score, combining precision and recall to evaluate the retriever's performance"
    }
    
    # Separate detailed results from the main results
    details_results = {}
    for method_result in all_results:
        if "per_sample_results" in method_result:
            per_sample_results = method_result.pop("per_sample_results")
            # Save detailed results to a separate file
            details_results.setdefault(method_result["method"], [])
            details_results[method_result["method"]].extend(per_sample_results)
    
    # Organize results by method and remove detailed results
    summary_results["results"] = []
    for method_result in all_results:
        if "error" in method_result:
            summary_results["results"].append({
                "method": method_result.get("method", "unknown"),
                "error": method_result.get("error", "unknown error")
            })
            continue
            
        # Extract summary metrics (without detailed results)
        summary = {
            "method": method_result.get("method", "unknown"),
            "map": method_result.get("map", 0.0),
            "recall@5": method_result.get("recall@5", 0.0),
            "recall@10": method_result.get("recall@10", 0.0),
            "recall@20": method_result.get("recall@20", 0.0),
            "ndcg@5": method_result.get("ndcg@5", 0.0),
            "ndcg@10": method_result.get("ndcg@10", 0.0),
            "support_f1": method_result.get("support_f1", 0.0),
            "sample_count": len(details_results.get(method_result["method"], []))
        }
        summary_results["results"].append(summary)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    
    with open(per_sample_path, "w", encoding="utf-8") as f:
        json.dump(details_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Detailed sample results saved to {per_sample_path}")
    
    # Print comparison table
    print("\nRetriever performance comparison:")
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Method", "MAP", "Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "Support F1"))
    print("-" * 85)
    
    for result in summary_results["results"]:
        if "error" in result:
            print("{:<10} Evaluation failed: {}".format(result["method"], result["error"]))
            continue
            
        method = result["method"]
        map_score = result.get("map", 0.0)
        recall_5 = result.get("recall@5", 0.0)
        recall_10 = result.get("recall@10", 0.0)
        recall_20 = result.get("recall@20", 0.0)
        ndcg_5 = result.get("ndcg@5", 0.0)
        ndcg_10 = result.get("ndcg@10", 0.0)
        support_f1 = result.get("support_f1", 0.0)
        
        print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            method, map_score, recall_5, recall_10, recall_20, ndcg_5, ndcg_10, support_f1))

if __name__ == "__main__":
    main()
