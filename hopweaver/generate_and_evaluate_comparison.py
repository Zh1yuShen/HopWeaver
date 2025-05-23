#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
One-click workflow script for comparison question generation and evaluation
'''
import os
import sys
import json
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from comparison_question_synthesizer import ComparisonQuestionSynthesizer
from evaluation_system.evaluator import QualityEvaluator
from flashrag.config import Config

# Configuration files and paths
DEFAULT_CONFIG_PATH = "./config_lib/example_config.yaml"
DATASETS_DIR = "./datasets/generated_comparison"
EVAL_RESULT_DIR = "./eval_result"

# Evaluation model list
MODELS = [
    "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-11-20",
    "gemini-2.0-flash",
    "google/gemma-3-27b-it",
    "meta-llama/llama-3.3-70b-instruct"
]

def ensure_dir_exists(dir_path):
    """Ensure directory exists, create if not exists"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def save_evaluation_stats(csv_files, dataset_name=None):
    """Save evaluation result statistics to txt file
    
    Args:
        csv_files (list): List of CSV files to save statistics for
        dataset_name (str, optional): Dataset name used for naming the statistics file
    """
    print(f"\n{'='*50}")
    print(f"Saving evaluation result statistics")
    print(f"{'='*50}")
    
    # Create result storage directory
    eval_result_dir = EVAL_RESULT_DIR
    stats_dir = os.path.join(eval_result_dir, "stats")
    ensure_dir_exists(stats_dir)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use dataset+timestamp naming logic
    if dataset_name:
        stats_file = os.path.join(stats_dir, f"{dataset_name}_stats_{timestamp}.txt")
    else:
        # Try to extract dataset name from file name
        if csv_files and len(csv_files) > 0:
            file_basename = os.path.basename(csv_files[0])
            # Remove _original_evaluation.csv or _polished_evaluation.csv suffix
            dataset_part = file_basename.replace('_original_evaluation.csv', '').replace('_polished_evaluation.csv', '').replace('_evaluation.csv', '')
            stats_file = os.path.join(stats_dir, f"{dataset_part}_stats_{timestamp}.txt")
        else:
            stats_file = os.path.join(stats_dir, f"evaluation_stats_{timestamp}.txt")
    
    # Collect statistics for all CSV files
    stats_content = ""
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                file_basename = os.path.basename(csv_file)
                
                stats_content += f"\n{file_basename}:\n{'='*60}\n"
                
                # Calculate probability of yes
                if 'multi_hop_reasoning' in df.columns:
                    stats_content += f"\n{file_basename} probability of yes:\n\n"
                    
                    # Get all models (exclude human and average score rows)
                    models = df[(df['model'] != 'human') & 
                               (df['model'] != 'Average') & 
                               (df['model'] != 'Total Average')]['model'].unique().tolist()
                    
                    yes_counts = {}
                    total_counts = {}
                    
                    for model in models:
                        yes_counts[model] = 0
                        total_counts[model] = 0
                    
                    # Calculate yes probability for each model
                    for _, row in df[(df['model'].isin(models)) & (df['multi_hop_reasoning'].notna())].iterrows():
                        model_name = row['model']
                        if pd.notna(row['multi_hop_reasoning']):
                            from evaluation_system.calculate_average_scores import compute_pass_rate
                            pass_rate = compute_pass_rate(row['multi_hop_reasoning'])
                            yes_counts[model_name] += pass_rate
                            total_counts[model_name] += 1
                    
                    # Print yes probability for each model
                    for model_name in sorted(yes_counts.keys()):
                        if total_counts[model_name] > 0:
                            yes_prob = (yes_counts[model_name] / total_counts[model_name]) * 100
                            stats_content += f"{model_name}: {yes_prob:.2f}%\n"
                    
                    # Calculate overall yes probability for all models
                    total_yes = sum(yes_counts.values())
                    total_questions = sum(total_counts.values())
                    if total_questions > 0:
                        overall_yes_prob = (total_yes / total_questions) * 100
                        stats_content += f"\nAll models: {overall_yes_prob:.2f}%\n"
                
                # Calculate average scores for each model
                if 'overall_quality' in df.columns:
                    # Calculate file total average score
                    overall_avg = 0
                    # Get total average score row
                    total_avg_row = df[df['model'] == 'Total Average']
                    if not total_avg_row.empty and not total_avg_row['overall_quality'].isna().all():
                        overall_avg = total_avg_row['overall_quality'].values[0]
                    else:
                        # If no total average row, calculate average across all models
                        model_rows = df[(df['model'] != 'human') & 
                                      (df['model'] != 'Average') & 
                                      (df['model'] != 'Total Average')]
                        if not model_rows.empty:
                            overall_avg = model_rows['overall_quality'].mean()
                    
                    stats_content += f"\n{file_basename}: {overall_avg:.4f}\n"
                    stats_content += "Average scores by model:\n"
                    
                    # Calculate average score for each model
                    for model in models:
                        model_rows = df[df['model'] == model]
                        if not model_rows.empty:
                            model_avg = model_rows['overall_quality'].mean()
                            stats_content += f"Model {model}: {model_avg:.4f}\n"
                    
                    stats_content += "Model human: nan\n"
                
                stats_content += "\n"
                
            except Exception as e:
                print(f"Error processing file {csv_file}: {str(e)}")
    
    # Write to statistics file
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats_content)
        print(f"Evaluation result statistics saved to: {stats_file}")
    except Exception as e:
        print(f"Error saving statistics: {str(e)}")
    
    return stats_file

def generate_comparison_questions(config_path, count, name=None, output_dir=None):
    """
    Generate comparison type questions
    
    Args:
        config_path (str): Configuration file path
        count (int): Number of questions to generate
        name (str, optional): Dataset name prefix
        output_dir (str, optional): Output directory
        
    Returns:
        dict: Generated question information
    """
    print(f"\n{'='*50}")
    print(f"Step 1: Using configuration {config_path} to generate {count} comparison questions")
    print(f"{'='*50}")
    
    # Initialize comparison question generator
    synthesizer = ComparisonQuestionSynthesizer(config_path, output_dir=output_dir)
    
    # Generate questions
    all_questions = synthesizer.generate_questions(count=count, save_intermediate=False)
    
    if not all_questions:
        print("Warning: No questions were successfully generated")
        return {}
    
    print(f"\nSuccessfully generated {len(all_questions)} comparison questions")
    
    # Ensure datasets directory exists
    ensure_dir_exists(DATASETS_DIR)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build dataset name
    dataset_name = f"{name}_" if name else ""
    dataset_name += f"comparison_{timestamp}"
    
    # Prepare original question data
    original_questions = []
    polished_questions = []
    
    for i, question in enumerate(all_questions):
        # Create question ID
        question_id = f"orig_{i}"
        
        # Get document content
        doc1_content = question.get("fact_entity_a", "")
        doc2_content = question.get("fact_entity_b", "")
        
        # Extract question text and answer
        q_text = question.get("multi_hop_question", "")
        answer = question.get("answer", "")
        
        # Create original question object
        original_q = {
            "id": question_id,
            "question": q_text,
            "answer": answer,
            "document1": doc1_content,
            "document2": doc2_content,
            "entity_a": question.get("entity_a", ""),
            "entity_b": question.get("entity_b", ""),
            "attribute_compared": question.get("attribute_compared", "")
        }
        original_questions.append(original_q)
        
        # Process refined questions
        if "polished_result" in question:
            polish_status = question.get("polished_result", {}).get("status", "")
            
            # Process based on polishing status
            if polish_status in ["ADJUST", "REWORKED"]:
                polish_id = f"polish_{i}"
                polished_q = {
                    "id": polish_id,
                    "question": question.get("polished_result", {}).get("refined_question", ""),
                    "answer": question.get("polished_result", {}).get("answer", ""),
                    "document1": doc1_content,
                    "document2": doc2_content,
                    "entity_a": question.get("entity_a", ""),
                    "entity_b": question.get("entity_b", ""),
                    "attribute_compared": question.get("attribute_compared", ""),
                    "status": polish_status
                }
                polished_questions.append(polished_q)
                print(f"Adding refined question with {polish_status} status: {polish_id}")
            
            # Process PASS status
            elif polish_status == "PASS":
                pass_id = f"pass_{i}"
                pass_q = {
                    "id": pass_id,
                    "question": q_text,
                    "answer": answer,
                    "document1": doc1_content,
                    "document2": doc2_content,
                    "entity_a": question.get("entity_a", ""),
                    "entity_b": question.get("entity_b", ""),
                    "attribute_compared": question.get("attribute_compared", ""),
                    "status": polish_status
                }
                polished_questions.append(pass_q)
                print(f"Adding PASS status question to refined list: {pass_id}")
    
    # Save question file
    original_questions_path = os.path.join(DATASETS_DIR, f"{dataset_name}_original.json")
    polished_questions_path = os.path.join(DATASETS_DIR, f"{dataset_name}_polished.json")
    
    with open(original_questions_path, 'w', encoding='utf-8') as f:
        json.dump(original_questions, f, ensure_ascii=False, indent=2)
    print(f"Original questions saved to: {original_questions_path}")
    
    with open(polished_questions_path, 'w', encoding='utf-8') as f:
        json.dump(polished_questions, f, ensure_ascii=False, indent=2)
    print(f"Polished questions saved to: {polished_questions_path}")
    
    return {
        "original": {
            "questions": original_questions,
            "path": original_questions_path
        },
        "polished": {
            "questions": polished_questions,
            "path": polished_questions_path
        },
        "name": dataset_name
    }

def evaluate_comparison_questions(config_path, dataset_info):
    """
    Evaluate generated comparison questions
    
    Args:
        config_path (str): Configuration file path
        dataset_info (dict): Dataset information
    """
    print(f"\n{'='*50}")
    print(f"Step 2: Evaluating generated comparison questions")
    print(f"{'='*50}")
    
    dataset_name = dataset_info["name"]
    
    # Ensure evaluation result directory exists
    ensure_dir_exists(EVAL_RESULT_DIR)
    
    # Evaluate original questions
    if dataset_info["original"]["questions"]:
        original_output_csv = os.path.join(EVAL_RESULT_DIR, f"{dataset_name}_original_evaluation.csv")
        evaluate_dataset(
            config_path, 
            dataset_info["original"]["questions"], 
            original_output_csv, 
            "original"
        )
    
    # Evaluate polished questions
    if dataset_info["polished"]["questions"]:
        polished_output_csv = os.path.join(EVAL_RESULT_DIR, f"{dataset_name}_polished_evaluation.csv")
        evaluate_dataset(
            config_path, 
            dataset_info["polished"]["questions"], 
            polished_output_csv, 
            "polished"
        )

def evaluate_dataset(config_path, questions, output_csv, question_type):
    """
    Evaluate dataset using multiple models
    
    Args:
        config_path (str): Configuration file path
        questions (list): List of questions
        output_csv (str): Output CSV file path
        question_type (str): Question type ("original" or "polished")
    """
    print(f"\nStarting evaluation for {question_type} comparison questions ({len(questions)} items)...")
    
    # Evaluation dimensions - consistent with bridge question evaluation
    dimensions = [
        'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
        'consistency', 'question_answerability', 'answer_question_consistency', 
        'information_integration_ability', 'reasoning_path_guidance',
        'logical_sophistication', 'overall_quality'
    ]
    
    # Create results DataFrame
    results = []
    
    # Evaluate each question sequentially
    for q_idx, question in enumerate(questions):
        q_id = question.get("id", f"q_{q_idx}")
        q_text = question.get("question", "")
        q_answer = question.get("answer", "")
        doc1 = question.get("document1", "")
        doc2 = question.get("document2", "")
        entity_a = question.get("entity_a", "")
        entity_b = question.get("entity_b", "")
        attribute = question.get("attribute_compared", "")
        
        print(f"\nEvaluating question {q_idx+1}/{len(questions)}: {q_id}")
        print(f"Question: {q_text[:100]}..." if len(q_text) > 100 else f"Question: {q_text}")
        print(f"Entity A: {entity_a}, Entity B: {entity_b}, Comparison Attribute: {attribute}")
        
        # Special handling for JSONL file data structure
        # If it's a comparison question JSONL file, handle different field names
        if "multi_hop_question" in question:
            q_text = question.get("multi_hop_question", "")
            # Use relevant_paragraph as the main document content
            doc1 = question.get("relevant_paragraph_a", "")
            doc2 = question.get("relevant_paragraph_b", "")
            print(f"Using JSONL special handling: Using multi_hop_question and relevant_paragraph")

        # Collect documents
        documents = [doc1, doc2]
        
        # Add additional documents (if they exist)
        for i in range(3, 11):  # Check document3 to document10
            doc_key = f"document{i}"
            if doc_key in question and question[doc_key]:
                documents.append(question[doc_key])
        
        # Build enhanced question object for evaluation - add fields specific to comparison questions
        enhanced_question = {
            "id": q_id,
            "question": q_text,
            "answer": q_answer,
            "document1": doc1,
            "document2": doc2,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "attribute_compared": attribute,
            "question_type": "comparison"  # Mark as comparison type question
        }
        
        # Print debug information to ensure multi-hop questions and document content are correctly passed
        print(f"\nQuestion content for evaluation: {enhanced_question['question']}")
        print(f"Document1 length: {len(enhanced_question['document1'])} characters")
        print(f"Document2 length: {len(enhanced_question['document2'])} characters")
        
        # Evaluate for each model
        for model_name in MODELS:
            print(f"Evaluating using model {model_name}...")
            
            try:
                # Create model configuration
                model_config = Config(config_path, {})
                model_config["evaluator_model"] = model_name
                
                # Initialize evaluator
                evaluator = QualityEvaluator(model_config)
                
                # Evaluate question
                result = evaluator.evaluate_question(enhanced_question)
                
                if result and "evaluation" in result:
                    # Create result row
                    row = {
                        "question_type": question_type,
                        "id": q_id,
                        "model": model_name,
                        "question": q_text,
                        "answer": q_answer,
                        "entity_a": entity_a,
                        "entity_b": entity_b,
                        "attribute_compared": attribute,
                        "document_count": len(documents),
                        "documents": json.dumps(documents)
                    }
                    
                    # Add evaluation dimensions
                    for dim in dimensions:
                        if dim == "multi_hop_reasoning":
                            row[dim] = "Yes" if result["evaluation"].get(dim, False) else "No"
                        else:
                            row[dim] = result["evaluation"].get(dim, "")
                    
                    # Add to results
                    results.append(row)
                    print(f"Added evaluation results for {model_name}")
                else:
                    print(f"Warning: Failed to evaluate question {q_id} using {model_name}, no valid result returned")
            except Exception as e:
                print(f"Error evaluating question {q_id} using {model_name}: {str(e)}")
        
        # Add manual evaluation row (empty row)
        human_row = {
            "question_type": question_type,
            "id": q_id,
            "model": "human",
            "question": q_text,
            "answer": q_answer,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "attribute_compared": attribute,
            "document_count": len(documents),
            "documents": json.dumps(documents)
        }
        for dim in dimensions:
            human_row[dim] = ""
        results.append(human_row)
        
        # Save results every 5 questions
        if (q_idx + 1) % 5 == 0 or q_idx == len(questions) - 1:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"Saved evaluation results to: {output_csv}")
    
    # Final save of results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Finished all evaluations and saved results to: {output_csv}")

def check_and_complete_evaluations(csv_file, max_item_retries=3):
    """Check the evaluation results file, find models with missing scores, and complete the evaluation
    
    Args:
        csv_file (str): CSV file path
        max_item_retries (int): Maximum number of retries for each item
    """
    from evaluation_system.evaluator import QualityEvaluator
    
    # Extract dataset name
    dataset_name = os.path.basename(csv_file).split('_')[0]
    print(f"Processing dataset: {dataset_name}")
    
    # Load CSV file
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"Successfully loaded {len(df)} rows of data from {csv_file}")
        else:
            print(f"File not found: {csv_file}")
            return
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return
    
    # Get all model names (excluding human and statistics rows)
    models = df['model'].unique().tolist()
    if 'human' in models:
        models.remove('human')
    if 'Average Score' in models:
        models.remove('Average Score')
    if 'Total Average Score' in models:
        models.remove('Total Average Score')
    
    # Find missing evaluations
    missing = []
    # Get all non-manual evaluation rows
    model_rows = df[df['model'] != 'human']
    
    # Check each dimension for null values
    dimensions = [
        'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
        'consistency', 'question_answerability', 'answer_question_consistency', 
        'information_integration_ability', 'reasoning_path_guidance', 'logical_sophistication',
        'overall_quality'
    ]
    
    # Check each question ID and model combination
    for question_id in df['id'].unique():
        for model in models:
            # Get the row for the current question ID and model
            row = model_rows[(model_rows['id'] == question_id) & (model_rows['model'] == model)]
            
            # If the row does not exist or any dimension is empty, consider it missing
            if row.empty:
                missing.append((question_id, model))
                print(f"Found missing evaluation: Question ID={question_id}, Model={model} (row does not exist)")
            elif row[dimensions].isna().any().any():
                missing.append((question_id, model))
                print(f"Found missing evaluation: Question ID={question_id}, Model={model} (NaN value exists)")
            elif (row[dimensions].astype(str) == '').any().any():
                missing.append((question_id, model))
                print(f"Found missing evaluation: Question ID={question_id}, Model={model} (empty string exists)")
    
    if not missing:
        print(f"No missing evaluations found, all ratings are complete!")
        return
    
    print(f"Found {len(missing)} missing evaluations, starting retries...")
    
    # Retry for each missing evaluation
    for question_id, model_name in missing:
        print(f"\n{'='*50}")
        print(f"Retrying evaluation: Question ID={question_id}, Model={model_name}")
        print(f"{'='*50}")
        
        # Get question data from CSV
        question_rows = df[df['id'] == question_id]
        
        if question_rows.empty:
            print(f"Warning: Question with ID {question_id} not found in CSV file, skipping")
            continue
        
        # Get the first row of data (question, answer, and documents should be the same for all rows)
        row = question_rows.iloc[0]
        
        # Initialize basic question data
        question = {
            'id': question_id,
            'question': row['question'],
            'answer': row['answer'],
            'question_type': 'comparison'  # Mark as comparison type question
        }
        
        # Add fields specific to comparison questions
        if 'entity_a' in row and pd.notna(row['entity_a']):
            question['entity_a'] = row['entity_a']
        if 'entity_b' in row and pd.notna(row['entity_b']):
            question['entity_b'] = row['entity_b']
        if 'attribute_compared' in row and pd.notna(row['attribute_compared']):
            question['attribute_compared'] = row['attribute_compared']
        
        # Parse document content from the 'documents' column
        try:
            if 'documents' in row and pd.notna(row['documents']):
                documents = json.loads(row['documents'])
                # Convert document list to dictionary format
                for i, doc in enumerate(documents, 1):
                    question[f'document{i}'] = doc
        except Exception as e:
            print(f"Error parsing document list: {str(e)}")

        # Use default configuration file path
        config_path = DEFAULT_CONFIG_PATH
        model_config = Config(config_path, {})
        model_config["evaluator_model"] = model_name
        model_config["generator_batch_size"] = 1
        config = model_config
        
        # Initialize evaluator
        evaluator = QualityEvaluator(config)
        
        # Add retry mechanism for this specific question and model
        success = False
        for retry in range(max_item_retries):
            try:
                print(f"Attempt {retry+1}/{max_item_retries} for Question ID={question_id}, Model={model_name}")
                # Evaluate question, each evaluation request also has an internal retry mechanism
                result = evaluator.evaluate_question(question, max_retry=3)
                
                if result and 'evaluation' in result:
                    success = True
                    break
                else:
                    # Wait a bit before retrying to prevent API rate limiting
                    import time
                    time.sleep(2)
            except Exception as e:
                print(f"Error during retry process: {str(e)}")
                if retry < max_item_retries - 1:
                    time.sleep(2)
                else:
                    print(f"All retries failed, skipping this evaluation")
        
        if not success:
            print(f"Warning: Evaluation for question {question_id} with model {model_name} failed, maximum retries reached")
            continue
        
        if result and 'evaluation' in result:
            # Update row in CSV file
            dimensions = [
                'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
                'consistency', 'question_answerability', 'answer_question_consistency', 
                'information_integration_ability', 'reasoning_path_guidance', 'logical_sophistication',
                'comparison_fairness', 'entity_relationship_clarity', 'attribute_accuracy', 'overall_quality'
            ]
            
            # Get row index
            row_idx = df[(df['id'] == question_id) & (df['model'] == model_name)].index
            
            if len(row_idx) > 0:
                # Update existing row
                evaluation = result['evaluation']
                for dim in dimensions:
                    if dim == 'multi_hop_reasoning':
                        df.loc[row_idx[0], dim] = 'Yes' if evaluation.get(dim, False) else 'No'
                    else:
                        df.loc[row_idx[0], dim] = evaluation.get(dim, '')
                
                print(f"Updated evaluation for question {question_id} with model {model_name}")
            else:
                # Create new row
                # Collect all documents
                documents = []
                for i in range(1, 11):
                    doc_key = f'document{i}'
                    if doc_key in question and question[doc_key] and question[doc_key].strip():
                        documents.append(question[doc_key])
                
                new_row = {
                    'question_type': row.get('question_type', ''),
                    'id': question_id,
                    'model': model_name,
                    'question': question.get('question', ''),
                    'answer': question.get('answer', ''),
                    'entity_a': question.get('entity_a', ''),
                    'entity_b': question.get('entity_b', ''),
                    'attribute_compared': question.get('attribute_compared', ''),
                    'document_count': len(documents),
                    'documents': json.dumps(documents)
                }
                
                evaluation = result['evaluation']
                for dim in dimensions:
                    if dim == 'multi_hop_reasoning':
                        new_row[dim] = 'Yes' if evaluation.get(dim, False) else 'No'
                    else:
                        new_row[dim] = evaluation.get(dim, '')
                
                # Add new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Added evaluation for question {question_id} with model {model_name}")
        else:
            print(f"Warning: Evaluation for question {question_id} with model {model_name} failed")
    
    # Save the updated CSV file
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\nUpdated evaluation results and saved to: {csv_file}")
    
    # Check again for any missing entries
    missing_count = 0
    for question_id in df['id'].unique():
        for model in models:
            row = df[(df['id'] == question_id) & (df['model'] == model)]
            if row.empty or row[dimensions].isna().any().any() or (row[dimensions].astype(str) == '').any().any():
                missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: Still {missing_count} missing evaluations, may need to run again")
    else:
        print(f"All evaluations are complete!")

def run_evaluation_scripts(dataset_name=None, max_item_retries=3, max_flow_retries=10):
    """Run scripts from the evaluation system for statistics and checks
    
    Args:
        dataset_name (str, optional): Dataset name, used to specify the file to process when calculating average scores
        max_item_retries (int): Maximum number of retries for each evaluation item
        max_flow_retries (int): Maximum number of repetitions for the entire check flow
    """
    print(f"\n{'='*50}")
    print(f"Step 3: Run evaluation system scripts to complete statistics and checks")
    print(f"{'='*50}")
    
    # Run script to check and complete evaluations
    try:
        # Evaluation results directory
        eval_result_dir = EVAL_RESULT_DIR
        
        # If dataset name is specified, try to process that dataset
        if dataset_name:
            # First, look for files with original and polished suffixes
            original_file = os.path.join(eval_result_dir, f"{dataset_name}_original_evaluation.csv")
            polished_file = os.path.join(eval_result_dir, f"{dataset_name}_polished_evaluation.csv")
            
            # Then, look for single-file evaluation results (possibly from eval-only mode)
            direct_file = os.path.join(eval_result_dir, f"{dataset_name}_evaluation.csv")
            
            # Select files to process
            csv_files = []
            
            # Option 1: If original and/or polished files exist, use these files
            has_standard_files = False
            if os.path.exists(original_file):
                csv_files.append(original_file)
                has_standard_files = True
            if os.path.exists(polished_file):
                csv_files.append(polished_file)
                has_standard_files = True
                
            # Option 2: If no standard files but direct_file exists, copy and use it
            if not has_standard_files and os.path.exists(direct_file):
                # If direct_file exists but no original file, copy it as the original file
                import shutil
                print(f"Copying {direct_file} to {original_file}")
                shutil.copy(direct_file, original_file)
                csv_files = [original_file]  # Use the copied file
        else:
            # Otherwise, get all evaluation result files
            csv_files = glob.glob(os.path.join(eval_result_dir, "*_evaluation.csv"))
        
        if not csv_files:
            print(f"Warning: No evaluation result files found in directory {eval_result_dir}")
            return
        
        print(f"Found {len(csv_files)} evaluation result files")
        
        # Add outer process retry loop
        for flow_retry in range(max_flow_retries):
            print(f"\nStarting check process attempt {flow_retry+1}/{max_flow_retries}")
            
            # Flag to mark if all files have completed evaluation
            all_files_completed = True
            
            # Process each CSV file
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    print(f"\n{'='*50}")
                    print(f"Processing file: {csv_file}")
                    print(f"{'='*50}")
                    
                    # Execute check and complete evaluation
                    check_and_complete_evaluations(csv_file, max_item_retries=max_item_retries)
                    
                    # Check if this file still has missing evaluations
                    try:
                        df = pd.read_csv(csv_file)
                        models = df['model'].unique().tolist()
                        if 'human' in models:
                            models.remove('human')
                        
                        dimensions = [
                            'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
                            'consistency', 'question_answerability', 'answer_question_consistency', 
                            'information_integration_ability', 'reasoning_path_guidance', 'logical_sophistication',
                            'overall_quality'
                        ]
                        
                        missing_count = 0
                        for question_id in df['id'].unique():
                            for model in models:
                                row = df[(df['id'] == question_id) & (df['model'] == model)]
                                if row.empty or row[dimensions].isna().any().any() or (row[dimensions].astype(str) == '').any().any():
                                    missing_count += 1
                        
                        if missing_count > 0:
                            print(f"File {csv_file} still has {missing_count} missing evaluations")
                            all_files_completed = False
                    except Exception as e:
                        print(f"Error checking file: {str(e)}")
                        all_files_completed = False
            
            # If all files have completed evaluation, exit the loop
            if all_files_completed:
                print(f"\nAll evaluation files have completed evaluation, ending check process")
                break
            else:
                print(f"\nStill evaluation files not completed, continuing to the next round of checks")
                # After the last retry, if evaluations are still not complete, prompt the user
                if flow_retry == max_flow_retries - 1:
                    print(f"\nWarning: Maximum process retries {max_flow_retries} reached, still incomplete evaluations")
        
        print("Evaluation check and completion process has ended")
        
        print("\nRunning calculate_average_scores.py...")
        # Pass dataset name to the calculation script to ensure correct file processing
        if dataset_name:
            # Pass dataset name as a parameter
            from evaluation_system.calculate_average_scores import main as calculate_scores
            import sys
            
            # Save original argv
            orig_argv = sys.argv
            
            # Set new argv
            sys.argv = [orig_argv[0], '--dataset', dataset_name]
            
            # Run the function to calculate average scores
            print(f"Processing dataset: {dataset_name}")
            calculate_scores()
            
            # Restore original argv
            sys.argv = orig_argv
            
            # Save evaluation result statistics
            save_evaluation_stats(csv_files, dataset_name)
        else:
            # If no dataset name, skip this step as the user only wants to process a specific dataset
            print("Skipping average score calculation as no dataset name was specified")
    except Exception as e:
        print(f"Error running evaluation scripts: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='One-click process for comparison question generation and evaluation')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Configuration file path')
    parser.add_argument('--count', type=int, default=10, help='Number of questions to generate')
    parser.add_argument('--name', type=str, default=None, help='Dataset name prefix')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory path')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate existing questions, do not generate new ones')
    parser.add_argument('--dataset-path', type=str, default=None, help='Path to the dataset to evaluate (used only when eval-only is True)')
    
    args = parser.parse_args()
    
    # Always use the same configuration file, specify retriever type via parameters
    config_path = args.config
    
    if not args.eval_only:
        # Step 1: Generate comparison questions
        dataset_info = generate_comparison_questions(config_path, args.count, args.name, args.output_dir)
        
        # Get dataset name
        dataset_name = dataset_info.get('name', None)
        print(f"Dataset name: {dataset_name}")
        
        # Step 2: Evaluate generated questions
        evaluate_comparison_questions(config_path, dataset_info)
        
        # Step 3: Run evaluation system scripts and pass the dataset name
        run_evaluation_scripts(dataset_name, max_item_retries=3, max_flow_retries=10)
    else:
        # Only evaluate existing questions
        if not args.dataset_path:
            print("Error: --dataset-path must be provided when using --eval-only")
            return
        
        try:
            # Load dataset - JSONL format, one JSON object per line
            questions = []
            with open(args.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            questions.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {len(questions)+1} of JSONL file: {str(e)}")
                
            if not questions:
                print(f"Error: Dataset {args.dataset_path} has no valid question data")
                return
                
            # Determine question type
            question_type = "original" if "original" in args.dataset_path else "polished"
            
            # Generate output CSV file path
            dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]  # Correct way to separate file extension
            print(f"Processed file name: {dataset_name}")
            output_csv = os.path.join(EVAL_RESULT_DIR, f"{dataset_name}_evaluation.csv")
            
            # Evaluate dataset
            evaluate_dataset(config_path, questions, output_csv, question_type)
            
            # Extract dataset name from file path
            dataset_basename = os.path.basename(args.dataset_path)
            dataset_name = os.path.splitext(dataset_basename)[0]
            if "_original" in dataset_name or "_polished" in dataset_name:
                # If it's an original or polished question file, extract the base name
                dataset_name = dataset_name.replace("_original", "").replace("_polished", "")
            print(f"Name extracted from dataset path: {dataset_name}")
            
            # Run evaluation system scripts and pass the dataset name
            run_evaluation_scripts(dataset_name, max_item_retries=3, max_flow_retries=10)
        except Exception as e:
            print(f"Error evaluating dataset: {str(e)}")
    
    print("\nOne-click comparison question evaluation process complete!")

if __name__ == "__main__":
    main()
