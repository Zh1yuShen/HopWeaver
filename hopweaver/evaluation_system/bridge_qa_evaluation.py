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
from ..components.utils.prompts import BRIDGE_QA_ONLY_PROMPT, BRIDGE_QA_DOCS_PROMPT

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
        print(f"Successfully loaded {len(data)} records")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        raise e

def normalize_answer(answer: str) -> str:
    """Normalize answer text by removing punctuation, articles, and converting to lowercase"""
    # Remove punctuation and special characters
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', '', answer, flags=re.IGNORECASE)
    # Convert to lowercase
    answer = answer.lower()
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

def calculate_exact_match(pred_answer: str, gold_answer: str) -> int:
    """Calculate exact match score"""
    return 1 if normalize_answer(pred_answer) == normalize_answer(gold_answer) else 0

def calculate_f1_score(pred_answer: str, gold_answer: str) -> float:
    """Calculate F1 score"""
    pred_tokens = set(normalize_answer(pred_answer).split())
    gold_tokens = set(normalize_answer(gold_answer).split())
    
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If one of them is empty, F1 is 0
        return 0
    
    # Calculate number of common tokens
    common = len(pred_tokens.intersection(gold_tokens))
    
    # Calculate precision and recall
    precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = common / len(gold_tokens) if len(gold_tokens) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def create_prompts(data_item: Dict, mode: str) -> Tuple[List, str]:
    """Create prompts based on the mode with strict answer format requirements"""
    if mode == "q_only":
        # Only use the question, testing the model's internal knowledge and inference abilities
        messages = [
            {
                "role": "system", 
                "content": BRIDGE_QA_ONLY_PROMPT
            },
            {
                "role": "user", 
                "content": data_item["multi_hop_question"]["multi_hop_question"]
            }
        ]
    elif mode == "q_docs":
        # Use both the question and gold documents, testing the model's reasoning upper bound under perfect retrieval
        source_doc = data_item["source_doc"]["content"]
        target_doc = data_item["target_doc"]["content"]
        
        combined_context = f"Document 1: {source_doc}\n\nDocument 2: {target_doc}"
        
        messages = [
            {
                "role": "system", 
                "content": BRIDGE_QA_DOCS_PROMPT
            },
            {
                "role": "user", 
                "content": f"Extract the exact answer to this question based on the following documents:\n\n{combined_context}\n\nQuestion: {data_item['multi_hop_question']['multi_hop_question']}\n\nRemember to provide ONLY the exact answer without any explanation or context."
            }
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Get the ground truth answer
    gold_answer = data_item["multi_hop_question"]["answer"]
    
    return messages, gold_answer

async def evaluate_llm_qa(data: List[Dict], model_config: Dict, model_name: str, mode: str, sample_size: int = None, max_retries: int = 3) -> Dict:
    """Evaluate LLM performance on QA task
    
    Args:
        data: Data to be evaluated
        model_config: Model configuration
        model_name: Model name
        mode: Evaluation mode (q_only or q_docs)
        sample_size: Sample size
        max_retries: Maximum retry attempts for failed questions
        
    Returns:
        Dictionary containing evaluation results
    """
    if sample_size and sample_size < len(data):
        # Random sampling
        import random
        random.seed(42)  # Set random seed for reproducible results
        data = random.sample(data, sample_size)
    
    # Initialize generator
    generator = OpenaiGenerator(model_config)
    
    all_em_scores = []
    all_f1_scores = []
    all_results = []
    failed_items = []  # Record failed questions
    
    print(f"Starting evaluation for model {model_name} in {mode} mode")
    
    # Prepare generation parameters, remove potentially incompatible ones
    generation_params = model_config["generation_params"].copy()
    if "do_sample" in generation_params:
        del generation_params["do_sample"]
    
    # Create a dictionary to track results for each question
    results_by_idx = {}
    
    # First round: evaluate all questions
    for idx, item in enumerate(tqdm(data, desc=f"Evaluation Progress - Round 1")):
        try:
            # Create prompts
            messages, gold_answer = create_prompts(item, mode)
            
            # Get model response
            response = await generator.get_response(
                messages, 
                **generation_params
            )
            
            # Extract response text
            pred_answer = response.message.content.strip()
            
            # Calculate evaluation metrics
            em_score = calculate_exact_match(pred_answer, gold_answer)
            f1_score = calculate_f1_score(pred_answer, gold_answer)
            
            # Save detailed results
            question_result = {
                "question": item["multi_hop_question"]["multi_hop_question"],
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "exact_match": em_score,
                "f1_score": f1_score
            }
            
            # Record successful result
            results_by_idx[idx] = question_result
            
            # Print progress every 5 questions
            if (idx + 1) % 5 == 0:
                print(f"Evaluated {idx+1}/{len(data)} questions")
                current_em = [r["exact_match"] for r in results_by_idx.values()]
                current_f1 = [r["f1_score"] for r in results_by_idx.values()]
                print(f"Current EM: {np.mean(current_em):.4f}, F1: {np.mean(current_f1):.4f}")
                
        except Exception as e:
            print(f"Error processing question {idx+1}: {str(e)}")
            # Record failed question for later retry
            failed_items.append((idx, item))
    
    # Retry failed questions until all are processed or max retries are reached
    retry_count = 1
    while failed_items and retry_count <= max_retries:
        print(f"\nStarting retry for {len(failed_items)} failed questions, current retry round {retry_count+1}/{max_retries}")
        retry_items = failed_items.copy()
        failed_items = []  # Reset failed list
        
        for idx, item in tqdm(retry_items, desc=f"Evaluation Progress - Retry Round {retry_count+1}"):
            try:
                # Create prompts
                messages, gold_answer = create_prompts(item, mode)
                
                # Get model response
                response = await generator.get_response(
                    messages, 
                    **generation_params
                )
                
                # Extract response text
                pred_answer = response.message.content.strip()
                
                # Calculate evaluation metrics
                em_score = calculate_exact_match(pred_answer, gold_answer)
                f1_score = calculate_f1_score(pred_answer, gold_answer)
                
                # Save detailed results
                question_result = {
                    "question": item["multi_hop_question"]["multi_hop_question"],
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "exact_match": em_score,
                    "f1_score": f1_score
                }
                
                # Update results dictionary
                results_by_idx[idx] = question_result
                print(f"Successfully retried question {idx+1}")
                
            except Exception as e:
                print(f"Error during retry for question index {idx}: {str(e)}")
                # If still failed, add to failed list for next retry
                failed_items.append((idx, item))
        
        retry_count += 1
    
    # Check if there are still failed questions
    if failed_items:
        print(f"Warning: {len(failed_items)} questions still failed after maximum retries.")
        # Process questions for the current retry round
        for idx, item in failed_items:
            gold_answer = item["multi_hop_question"]["answer"]
            results_by_idx[idx] = {
                "question": item["multi_hop_question"]["multi_hop_question"],
                "gold_answer": gold_answer,
                "pred_answer": "[Processing failed]",
                "exact_match": 0,
                "f1_score": 0
            }
    
    # Print final results
    if all_em_scores:
        print(f"Final EM: {np.mean(all_em_scores):.4f}, F1: {np.mean(all_f1_scores):.4f}")
        print(f"Warning: {len(failed_items)} questions still failed after maximum retries.")
    else:
        print("Warning: No questions were successfully evaluated.")
    
    # Calculate overall EM and F1 scores
    summary = {
        "model": model_name,
        "mode": mode,
        "sample_size": len(data),
        "exact_match": float(np.mean(all_em_scores)) if all_em_scores else 0.0,
        "f1_score": float(np.mean(all_f1_scores)) if all_f1_scores else 0.0,
        "num_questions": len(all_em_scores),
        "success_rate": len(all_results) / len(data) if data else 1.0,
        "failed_count": len(data) - len(all_results) if data else 0
    }
    
    # Return results
    results = {
        "summary": summary,
        "details": all_results
    }
    
    return results

async def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM performance on HotpotQA') 
    parser.add_argument('--data_path', type=str, default='./datasets/raw_data/hotpotqa/hotpot_train_v1.1_simplified.json', help='File path for HotpotQA data')
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML file containing API keys and model configurations")
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory for evaluation results')
    parser.add_argument('--model_name', type=str, default='qwen2-7b-instruct', help='Name of the LLM model to evaluate')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for evaluation')
    parser.add_argument('--mode', type=str, default='q_only', choices=['q_only', 'q_docs'], help='Evaluation mode (q_only or q_docs)')
    args = parser.parse_args()
    
    # Load data
    data = load_jsonl(args.data_file)
    print(f"Data loaded successfully, {len(data)} records")
    
    # Model configuration (can be loaded from an external file later)
    with open(args.config_file, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    print(f"Main configuration file '{args.config_file}' loaded successfully")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Define models to test (using model IDs)
    MODELS_TO_TEST = [
        "gpt-4o-2024-11-20",
        # Commented out other models to speed up testing
        # "meta-llama/llama-3.3-70b-instruct",
        # "claude-3-7-sonnet-20250219",
        # "qwen/qwen3-8b",
    ]

    # If sample size is set, sample data
    if args.sample_size and args.sample_size < len(data):
        import random
        random.seed(42)  # Ensure reproducible results
        data_subset = random.sample(data, args.sample_size)
        print(f"Data sampled successfully, using {len(data_subset)} records for evaluation")
    else:
        data_subset = data
        print(f"Using all {len(data_subset)} records for evaluation")

    all_evaluation_results = []
    detailed_results_collection = defaultdict(list)

    for model_id_str in MODELS_TO_TEST:
        print(f"\n{'='*30} Evaluating model: {model_id_str} {'='*30}")

        # Prepare OpenaiGenerator configuration for current model
        generator_init_config = main_config.copy()
        generator_init_config['generator_model'] = model_id_str

        # Create user-friendly model name for output files and logs
        output_model_name = model_id_str.replace("/", "_").replace(":", "_")

        for mode in args.modes:
            print(f"\n--- Mode: {mode} ---")
            
            eval_results = await evaluate_llm_qa(
                data=data_subset,
                model_config=generator_init_config,  # Contains all API settings and current generator_model
                model_name=output_model_name,      # Used for logs and file names
                mode=mode,
                sample_size=None # Sampling already handled externally
            )
            
            all_evaluation_results.append(eval_results['summary'])
            detailed_results_collection[f"{output_model_name}_{mode}"] = eval_results['details']

            # Save detailed evaluation results for each model and mode combination
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_filename = os.path.join(args.output_dir, f"evaluation_details_{output_model_name}_{mode}_{timestamp}.jsonl")
            with open(detail_filename, 'w', encoding='utf-8') as f_detail:
                for res_item in eval_results['details']:
                    json.dump(res_item, f_detail, ensure_ascii=False)
                    f_detail.write('\n')
            print(f"Evaluation results saved to {detail_filename}")

    # Save evaluation summary for all models and modes
    summary_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = os.path.join(args.output_dir, f"evaluation_summary_{summary_timestamp}.json")
    with open(summary_filename, 'w', encoding='utf-8') as f_summary:
        json.dump(all_evaluation_results, f_summary, indent=4, ensure_ascii=False)
    print(f"Evaluation summary saved to {summary_filename}")
    print(f"\nOverall evaluation summary saved to: {summary_filename}")

    print("\nEvaluation completed.")

if __name__ == "__main__":
    asyncio.run(main())
