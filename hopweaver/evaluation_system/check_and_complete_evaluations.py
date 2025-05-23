#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Check all CSV files in the evaluation results folder, find models with missing scores, and use CSV data as input to retry until all evaluations are complete
'''
import os
import sys
import pandas as pd
import json
import glob
from tqdm import tqdm
# Add parent directory to path to import flashrag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add current directory to path as well
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flashrag.config import Config
from evaluator import QualityEvaluator

# Evaluation results directory
EVAL_RESULT_DIR = "./eval_result"
# Configuration file path
CONFIG_PATH = "./config_lib/example_config.yaml"
# Dataset samples directory
DATASET_SAMPLES_DIR = "./datasets/samples"

# We now read data directly from CSV files, no longer needing a mapping relationship

def load_csv_file(file_path):
    """Load CSV file
    
    Args:
        file_path: CSV file path
        
    Returns:
        DataFrame: DataFrame containing evaluation results
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {len(df)} rows of data from {file_path}")
            return df
        else:
            print(f"File does not exist: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return pd.DataFrame()

def load_questions(file_path):
    """Load question dataset
    
    Args:
        file_path: Question dataset file path
        
    Returns:
        list: List containing question data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} questions from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading question data: {str(e)}")
        return []

def filter_rejected_questions(questions):
    """Filter out rejected questions
    
    Args:
        questions: List of question data
        
    Returns:
        list: List of filtered question data
    """
    filtered = [q for q in questions if not (q.get('status') == 'REJECT' or q.get('status') == 'REJECTED')]
    print(f"Filtered to {len(filtered)} questions (removed {len(questions) - len(filtered)} rejected questions)")
    return filtered

def find_missing_evaluations(df, models):
    """Find missing evaluations
    
    Args:
        df: DataFrame of evaluation results
        models: List of model names
        
    Returns:
        list: List containing missing evaluation information, each item as (question_id, model)
    """
    missing = []
    
    # Get all non-manual evaluation rows
    model_rows = df[df['model'] != 'human']
    
    # Check each dimension for null values
    dimensions = [
        'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
        'consistency', 'question_answerability', 'answer_question_consistency', 
        'information_integration_ability', 'reasoning_path_guidance', 'logical_sophistication', 'overall_quality'
    ]
    
    # Check each question_id and model combination
    for question_id in df['id'].unique():
        for model in models:
            # Get rows for the current question_id and model
            row = model_rows[(model_rows['id'] == question_id) & (model_rows['model'] == model)]
            
            # If rows do not exist or any dimension is empty, consider it missing
            if row.empty:
                missing.append((question_id, model))
                print(f"Found missing evaluation: Question ID={question_id}, Model={model} (row does not exist)")
            elif row[dimensions].isna().any().any():
                missing.append((question_id, model))
                print(f"Found missing evaluation: Question ID={question_id}, Model={model} (NaN value exists)")
            elif (row[dimensions].astype(str) == '').any().any():
                missing.append((question_id, model))
                print(f"Found missing evaluation: Question ID={question_id}, Model={model} (empty string exists)")
    
    return missing

def get_question_data_from_csv(df, question_id):
    """Get question data from CSV data
    
    Args:
        df: DataFrame of evaluation results
        question_id: Question ID
        
    Returns:
        dict: Dictionary containing question data, returns None if not found
    """
    # Get question data
    question_rows = df[df['id'] == question_id]
    if question_rows.empty:
        return None
    
    # Get the first row of data (question, answer, and documents should be the same for all rows)
    row = question_rows.iloc[0]
    
    # Initialize basic question data
    question_data = {
        'id': question_id,
        'question': row['question'],
        'answer': row['answer']
    }
    
    # Parse document content from the 'documents' column
    try:
        if 'documents' in row and pd.notna(row['documents']):
            documents = json.loads(row['documents'])
            # Convert document list to dictionary format
            for i, doc in enumerate(documents, 1):
                question_data[f'document{i}'] = doc
    except Exception as e:
        print(f"Error parsing document list: {str(e)}")
    
    return question_data

def get_dataset_path(dataset_name):
    """Get dataset evaluation results file path
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        str: Dataset evaluation results file path
    """
    # Directly return the CSV file path
    csv_file = os.path.join(EVAL_RESULT_DIR, f"{dataset_name}_evaluation.csv")
    if os.path.exists(csv_file):
        return csv_file
    return None

def complete_evaluations_from_csv(csv_file):
    """Complete missing evaluations using CSV data
    
    Args:
        csv_file: CSV file path
    """
    # Extract dataset name
    dataset_name = os.path.basename(csv_file).split('_')[0]
    print(f"Processing dataset: {dataset_name}")
    
    # Load CSV file
    df = load_csv_file(csv_file)
    if df.empty:
        print(f"Unable to process empty CSV file: {csv_file}")
        return
    
    # Get all model names (except human)
    models = df['model'].unique().tolist()
    if 'human' in models:
        models.remove('human')
    
    # Find missing evaluations
    missing = find_missing_evaluations(df, models)
    if not missing:
        print("No missing evaluation items found. All scores are complete!")
        return
    
    print(f"Found {len(missing)} missing evaluation items. Starting to supplement...")
    
    # Retry each missing evaluation
    for question_id, model_name in tqdm(missing, desc="Completing missing evaluations"):
        print(f"\n{'='*50}")
        print(f"Evaluating Question ID: {question_id}, Model: {model_name}")
        print(f"{'='*50}")
        
        # Get question data from CSV
        question = get_question_data_from_csv(df, question_id)
        
        if not question:
            print(f"Warning: Cannot find data for Question ID {question_id} in CSV, skipping this item.")
            continue
        
        # Create configuration
        model_config = Config(CONFIG_PATH, {})
        model_config["generator_batch_size"] = 1
        model_config["evaluator_model"] = model_name
        
        # Initialize evaluator
        evaluator = QualityEvaluator(model_config)
        
        # Evaluate question
        result = evaluator.evaluate_question(question, max_retry=7)
        
        if result and 'evaluation' in result:
            # Update CSV file rows
            dimensions = [
                'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
                'consistency', 'question_answerability', 'answer_question_consistency', 
                'information_integration_ability', 'reasoning_path_guidance',
                'logical_sophistication', 'overall_quality'
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
                
                print(f"Updated evaluation for Question {question_id} with Model {model_name}")
            else:
                # Create new row
                # Collect all documents
                documents = []
                for i in range(1, 11):
                    doc_key = f'document{i}'
                    if doc_key in question and question[doc_key] and question[doc_key].strip():
                        documents.append(question[doc_key])
                
                new_row = {
                    'dataset': dataset_name,
                    'id': question_id,
                    'model': model_name,
                    'question': question.get('question', ''),
                    'answer': question.get('answer', ''),
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
                print(f"Added evaluation for Question {question_id} with Model {model_name}")
        else:
            print(f"Warning: Evaluation for Question {question_id} with Model {model_name} failed")
    
    # Save updated CSV file
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\nUpdated evaluation results and saved to: {csv_file}")
    
    # Check again if there are still missing evaluations
    missing = find_missing_evaluations(df, models)
    if missing:
        print(f"Warning: There are still {len(missing)} missing evaluations, may need to run this script again")
    else:
        print("All evaluations are complete!")

def main():
    """Main function"""
    # Get all CSV files in the evaluation results directory
    csv_files = glob.glob(os.path.join(EVAL_RESULT_DIR, "*_evaluation.csv"))
    
    if not csv_files:
        print(f"Warning: No evaluation result files found in directory {EVAL_RESULT_DIR}")
        return
    
    print(f"Found {len(csv_files)} evaluation result files")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing file: {csv_file}")
        print(f"{'='*50}")
        
        complete_evaluations_from_csv(csv_file)
    
    print("\nAll evaluation result files have been processed")

if __name__ == '__main__':
    main()
