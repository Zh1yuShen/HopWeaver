#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from collections import defaultdict
import re
import asyncio
import warnings
warnings.filterwarnings("ignore")
import datetime

# Add project root directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import FlashRAG generator
from flashrag.generator.openai_generator import OpenaiGenerator
# Import prompt template
from ..components.utils.prompts import COMPARISON_QA_ONLY_PROMPT, COMPARISON_QA_DOCS_PROMPT

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

def normalize_answer(answer: str) -> str:
    """Normalize answer text, remove punctuation, articles, and convert to lowercase"""
    # Remove punctuation and special characters
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\s+', ' ', answer, flags=re.IGNORECASE)
    # Convert to lowercase, remove extra spaces
    answer = answer.lower().strip()
    # Replace multiple consecutive spaces with a single space
    answer = re.sub(r'\s+', ' ', answer)
    return answer

def calculate_exact_match(pred_answer: str, gold_answer: str) -> int:
    """Calculate exact match score"""
    return int(normalize_answer(pred_answer) == normalize_answer(gold_answer))

def calculate_f1_score(pred_answer: str, gold_answer: str) -> float:
    """Calculate F1 score"""
    # Normalize answers
    pred_tokens = normalize_answer(pred_answer).split()
    gold_tokens = normalize_answer(gold_answer).split()
    
    # Calculate common words
    common = set(pred_tokens) & set(gold_tokens)
    
    # If both are empty, consider it a perfect match
    if len(gold_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0
    
    # If one is empty and the other is not, F1 is 0
    if len(common) == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def create_prompts(data_item: Dict, mode: str) -> Tuple[List, str]:
    """Create prompts for comparison questions, using different prompting methods based on the mode"""
    question = data_item["multi_hop_question"]
    answer = data_item["answer"]
    
    if mode == "q_only":
        # Use only the question to test the model's internal knowledge and reasoning ability
        messages = [
            {
                "role": "system", 
                "content": COMPARISON_QA_ONLY_PROMPT
            },
            {
                "role": "user", 
                "content": question
            }
        ]
    elif mode == "q_docs":
        # Use the question and gold standard documents to test the model's reasoning upper bound under perfect retrieval conditions
        para_a = data_item["relevant_paragraph_a"]
        para_b = data_item["relevant_paragraph_b"]
        
        combined_context = f"Document 1: {para_a}\n\nDocument 2: {para_b}"
        
        messages = [
            {
                "role": "system", 
                "content": COMPARISON_QA_DOCS_PROMPT
            },
            {
                "role": "user", 
                "content": f"Based on the following documents, provide the exact answer to this question:\n\n{combined_context}\n\nQuestion: {question}\n\nRemember to provide ONLY the exact answer without any explanation or extra details."
            }
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return messages, answer

async def evaluate_llm_qa(data: List[Dict], model_config: Dict, model_name: str, mode: str, sample_size: int = None, max_retries: int = 3) -> Dict:
    """Evaluate LLM performance on comparison questions
    
    Args:
        data: Data to be evaluated
        model_config: Model configuration
        model_name: Model name
        mode: Evaluation mode (q_only or q_docs)
        sample_size: Sample size
        max_retries: Maximum number of retries for failed questions
        
    Returns:
        Dictionary containing evaluation results
    """
    if sample_size and sample_size < len(data):
        # Random sampling
        import random
        random.seed(42)  # Set random seed to ensure reproducible results
        data = random.sample(data, sample_size)
    
    # Initialize generator
    generator = OpenaiGenerator(model_config)
    
    all_em_scores = []
    all_f1_scores = []
    all_results = []
    failed_items = []  # Record failed questions
    
    print(f"Starting evaluation of model {model_name} in {mode} mode")
    
    # Prepare generation parameters, remove potentially incompatible parameters
    generation_params = model_config["generation_params"].copy()
    if "do_sample" in generation_params:
        del generation_params["do_sample"]
    
    # First round of evaluation
    for idx, item in enumerate(tqdm(data, desc=f"Evaluation progress - Round 1")):
        try:
            # Create prompt
            messages, gold_answer = create_prompts(item, mode)
            
            # Get model response
            response = await generator.get_response(messages, **generation_params)
            
            if response is None:
                print(f"Error processing item {idx+1}: Model response is empty")
                failed_items.append((idx, item))
                continue
                
            # Get text content from OpenAI's response object
            pred_answer = response.message.content.strip()
            exact_match = calculate_exact_match(pred_answer, gold_answer)
            f1_score = calculate_f1_score(pred_answer, gold_answer)
            
            # Save results
            all_em_scores.append(exact_match)
            all_f1_scores.append(f1_score)
            
            # Record more detailed information, including prompt and original answer
            result = {
                "question": item["multi_hop_question"],
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "exact_match": exact_match,
                "f1_score": f1_score,
                "mode": mode,
                "messages": messages,  # Record prompt
                "relevant_paragraph_a": item.get("relevant_paragraph_a", ""),
                "relevant_paragraph_b": item.get("relevant_paragraph_b", ""),
                "entity_a": item.get("entity_a", ""),
                "entity_b": item.get("entity_b", ""),
                "attribute_compared": item.get("attribute_compared", "")
            }
            all_results.append(result)
            
            # Display current performance every 5 questions
            if (idx + 1) % 5 == 0:
                print(f"Evaluated {idx+1}/{len(data)} questions")
                print(f"Current EM: {sum(all_em_scores) / len(all_em_scores):.4f}, F1: {sum(all_f1_scores) / len(all_f1_scores):.4f}")
                
        except Exception as e:
            print(f"Error processing item {idx+1}: {str(e)}")
            failed_items.append((idx, item))
    
    # Retry failed questions
    retry_count = 1
    while failed_items and retry_count <= max_retries:
        print(f"\nStarting retry round {retry_count+1}, {len(failed_items)} questions still not successfully processed")
        current_failed_items = []
        for idx, item in tqdm(failed_items, desc=f"Retry progress - Round {retry_count+1}"):
            try:
                # Create prompt
                messages, gold_answer = create_prompts(item, mode)
                
                # Get model response
                response = await generator.get_response(messages, **generation_params)
                
                if response is None:
                    print(f"Error retrying item {idx+1}: Model response is empty")
                    current_failed_items.append((idx, item))
                    continue
                    
                # Get text content from OpenAI's response object
                pred_answer = response.message.content.strip()
                exact_match = calculate_exact_match(pred_answer, gold_answer)
                f1_score = calculate_f1_score(pred_answer, gold_answer)
                
                # Save results
                all_em_scores.append(exact_match)
                all_f1_scores.append(f1_score)
                
                # Record more detailed information, including prompt and original answer
                result = {
                    "question": item["multi_hop_question"],
                    "gold_answer": gold_answer,
                    "predicted_answer": pred_answer,
                    "exact_match": exact_match,
                    "f1_score": f1_score,
                    "mode": mode,
                    "messages": messages,  # Record prompt
                    "relevant_paragraph_a": item.get("relevant_paragraph_a", ""),
                    "relevant_paragraph_b": item.get("relevant_paragraph_b", ""),
                    "entity_a": item.get("entity_a", ""),
                    "entity_b": item.get("entity_b", ""),
                    "attribute_compared": item.get("attribute_compared", "")
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"Error retrying item {idx+1}: {str(e)}")
                current_failed_items.append((idx, item))
        
        failed_items = current_failed_items
        retry_count += 1
    
    if failed_items:
        print(f"Warning: After {max_retries} retries, {len(failed_items)} questions still not successfully processed")
    
    # Calculate average scores
    em_avg = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0
    f1_avg = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0
    
    print(f"Final EM: {em_avg:.4f}, F1: {f1_avg:.4f}")
    print(f"Evaluated {len(all_results)}/{len(data)} questions in total")
    
    # Prepare summary results
    summary = {
        "model": model_name,
        "mode": mode,
        "sample_size": sample_size or len(data),
        "exact_match": em_avg,
        "f1_score": f1_avg,
        "num_questions": len(all_results),
        "success_rate": len(all_results) / len(data),
        "failed_count": len(data) - len(all_results)
    }
    
    results = {
        "summary": summary,
        "details": all_results
    }
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on comparison questions")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML file containing API keys and model configurations")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for results")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to evaluate, defaults to all samples")
    parser.add_argument("--modes", nargs='+', default=["q_only", "q_docs"], help="List of evaluation modes (e.g., q_only q_docs)")
    args = parser.parse_args()
    
    # Load data
    data = load_jsonl(args.data_file)
    print(f"Data loaded successfully, {len(data)} items in total")
    
    # Load main configuration file
    with open(args.config_file, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    print(f"Main configuration file '{args.config_file}' loaded successfully")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Define list of models to test (using model IDs)
    model_ids_to_test = [
        "gpt-4o-2024-11-20",
        # Temporarily comment out other models to speed up testing
        "meta-llama/llama-3.3-70b-instruct",
        "claude-3-7-sonnet-20250219",
        "qwen/qwen3-8b",
    ]

    # If sample_size is set, sample the data
    if args.sample_size and args.sample_size < len(data):
        import random
        random.seed(42)  # Ensure reproducible results
        data_subset = random.sample(data, min(args.sample_size, len(data)))
        print(f"Data has been sampled, using {len(data_subset)} items for evaluation")
    else:
        data_subset = data
        print(f"Will use all {len(data_subset)} items for evaluation")

    all_evaluation_results = []
    detailed_results_collection = defaultdict(list)
    summary_results = []
    for model_id_str in model_ids_to_test:
        print(f"\n{'='*30} Starting evaluation for model: {model_id_str} {'='*30}")
        
        # Prepare OpenaiGenerator configuration for the current model
        # OpenaiGenerator will use 'generator_model' to find corresponding API settings from main_config
        model_config = main_config.copy()
        model_config['generator_model'] = model_id_str
        
        for mode in args.modes:
            print(f"\n--- Mode: {mode} ---")
            eval_results = await evaluate_llm_qa(
                data=data_subset,
                model_config=model_config,
                model_name=model_id_str, 
                mode=mode, 
                sample_size=None,  # No need to sample again here
                max_retries=3
            )
            
            # Add results to summary list
            all_evaluation_results.append(eval_results["summary"])
            
            # Save detailed results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            details_file = os.path.join(args.output_dir, f"evaluation_details_{model_id_str.replace('/', '_')}_{mode}_{timestamp}.jsonl")
            
            with open(details_file, 'w', encoding='utf-8') as f:
                for item in eval_results["details"]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            filename_detail = f"{model_id_str.replace('/', '_')}_{mode}_detail_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath_detail = os.path.join(args.output_dir, filename_detail)
            with open(filepath_detail, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=4)
            print(f"Detailed results saved to: {filepath_detail}")
    
    # Save summary results
    filename_summary = f"summary_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath_summary = os.path.join(args.output_dir, filename_summary)
    with open(filepath_summary, 'w', encoding='utf-8') as f:
        for result in all_evaluation_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Summary results saved to: {filepath_summary}")
    
    print("\nEvaluation completed.")

if __name__ == "__main__":
    asyncio.run(main())
