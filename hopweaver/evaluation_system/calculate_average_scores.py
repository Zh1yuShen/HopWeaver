#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

def compute_pass_rate(multi_hop_string):
    """Calculate the pass rate in the multi_hop_reasoning column. Pass rate is calculated as: (# of outputs that are Yes or True) / (Total # of outputs that are Yes, True, No, False)."""
    valid_yes = {"Yes", "True"}
    valid_no = {"No", "False"}
    tokens = [token.strip() for token in multi_hop_string.split(",")]
    count_yes = sum(1 for token in tokens if token in valid_yes)
    count_valid = sum(1 for token in tokens if token in valid_yes or token in valid_no)
    if count_valid == 0:
        return 0.0
    return count_yes / count_valid

def process_evaluation_file(file_path):
    """
    Process evaluation files, add average score and total average score for each question
    """
    # Initialize return value to avoid undefined error
    overall_avg = 0
    model_avgs = {}
    print(f'Processing file: {file_path}')
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Backup original file
    backup_path = file_path + '.avg_backup'
    # Force backup, always overwrite
    os.system(f'cp {file_path} {backup_path}')
    print(f'Backed up original file to: {backup_path}')
    
    # Check if average score rows already exist
    has_avg_rows = any(df['model'] == '平均分') or any(df['model'] == '总平均分')
    if has_avg_rows:
        print('Average score rows already exist in the file, they will be cleared first')
        df = df[(df['model'] != '平均分') & (df['model'] != '总平均分')]
    
    # Get all unique question IDs
    unique_ids = df['id'].unique()
    print(f'File contains {len(unique_ids)} unique question IDs')
    
    # Get scoring metric columns
    score_columns = [col for col in df.columns if col in [
        'fluency', 'clarity', 'conciseness', 'relevance', 'consistency',
        'question_answerability', 'answer_question_consistency', 'information_integration_ability',
        'reasoning_path_guidance', 'logical_sophistication', 'overall_quality'
    ]]
    print(f'Scoring metric columns: {score_columns}')
    
    # Check if multi_hop_reasoning column exists, used for calculating 'yes' probability
    has_multi_hop = 'multi_hop_reasoning' in df.columns
    
    # Check if document_count column exists
    has_document_count = 'document_count' in df.columns
    
    # Recalculate overall_quality values for all models
    # Get all model names (excluding 'human')
    models = df[df['model'] != 'human']['model'].unique()
    
    # Recalculate overall_quality for each model
    for model in models:
        model_rows = df[df['model'] == model]
        
        # Check if the model has other scoring metrics
        other_scores = [col for col in score_columns if col != 'overall_quality']
        valid_scores = [col for col in other_scores if col in model_rows.columns and not model_rows[col].isna().all()]
        
        if valid_scores:
            # If other scoring metrics exist, calculate their average as overall_quality
            for idx, row in model_rows.iterrows():
                scores = []
                for col in valid_scores:
                    if pd.notna(row[col]):
                        # If it's a string type, try to convert
                        if isinstance(row[col], str):
                            score_mapping = {
                                'Very Poor': 1.0,
                                'Poor': 2.0,
                                'Fair': 3.0,
                                'Good': 4.0,
                                'Very Good': 5.0
                            }
                            if row[col] in score_mapping:
                                scores.append(score_mapping[row[col]])
                        else:
                            scores.append(row[col])
                
                if scores:
                    # Update overall_quality value
                    new_overall = sum(scores) / len(scores)
                    df.at[idx, 'overall_quality'] = new_overall

    
    # Create a new DataFrame to store results
    result_rows = []
    question_averages = {col: [] for col in score_columns}
    
    # Used for calculating 'yes' probability
    yes_counts = {}
    total_counts = {}
    
    # Used for calculating the average of document_count
    if has_document_count:
        document_count_avg = []
    
    # Process each question ID
    for question_id in unique_ids:
        # Get all rows for the current question ID
        id_rows = df[df['id'] == question_id]
        
        # Exclude rows for the 'human' model
        model_rows = id_rows[id_rows['model'] != 'human']
        
        # Calculate 'yes' probability
        if has_multi_hop and len(model_rows) > 0:
            for _, row in model_rows.iterrows():
                model_name = row['model']
                if model_name not in yes_counts:
                    yes_counts[model_name] = 0
                    total_counts[model_name] = 0
                
                if pd.notna(row['multi_hop_reasoning']):
                    pass_rate = compute_pass_rate(row['multi_hop_reasoning'])
                    yes_counts[model_name] += pass_rate
                    total_counts[model_name] += 1
        
        # Add original rows - add regardless of whether model_rows exist
        for _, row in id_rows.iterrows():
            result_rows.append(row.to_dict())
        
        # Only calculate and add average score rows if model_rows is not empty
        if len(model_rows) > 0:
            # Calculate the average score for each metric
            avg_row = id_rows.iloc[0].to_dict()
            avg_row['model'] = '平均分'
            
            # Calculate the average of document_count
            if has_document_count:
                try:
                    doc_count_avg = model_rows['document_count'].mean()
                    avg_row['document_count'] = doc_count_avg
                    document_count_avg.append(doc_count_avg)
                except Exception as e:
                    print(f"Could not calculate the average of document_count: {e}")
                    avg_row['document_count'] = np.nan
            
            # Calculate and record the average score for each metric
            for col in score_columns:
                if col in model_rows.columns:
                    # Check the data type of the column
                    if model_rows[col].dtype == 'object':
                        # Try to convert text scores to numerical values
                        score_mapping = {
                            'Very Poor': 1.0,
                            'Poor': 2.0,
                            'Fair': 3.0,
                            'Good': 4.0,
                            'Very Good': 5.0
                        }
                        
                        # Convert scores and calculate average
                        try:
                            numeric_scores = model_rows[col].map(score_mapping)
                            if not numeric_scores.isna().all():
                                avg_score = numeric_scores.mean()
                                avg_row[col] = avg_score
                                question_averages[col].append(avg_score)
                        except Exception as e:
                            print(f"Could not calculate average for column {col}: {e}")
                            avg_row[col] = np.nan
                    else:
                        # Directly calculate average for numerical columns
                        try:
                            avg_score = model_rows[col].mean()
                            avg_row[col] = avg_score
                            question_averages[col].append(avg_score)
                        except Exception as e:
                            print(f"Could not calculate average for column {col}: {e}")
                            avg_row[col] = np.nan
            
            result_rows.append(avg_row)
    
    # Create total average score row
    if any(question_averages.values()):
        # Create total average score row
        total_avg_row = df.iloc[0].to_dict()
        total_avg_row['id'] = 'All Questions'
        total_avg_row['model'] = '总平均分'
        total_avg_row['question'] = f'Average score for a total of {len(unique_ids)} questions'
        total_avg_row['answer'] = ''
        
        # Clear document fields
        if 'documents' in total_avg_row:
            total_avg_row['documents'] = ''
        
        # Calculate total average of document_count
        if has_document_count and document_count_avg:
            doc_count_total_avg = sum(document_count_avg) / len(document_count_avg)
            total_avg_row['document_count'] = doc_count_total_avg
            print(f'Total average of document_count: {doc_count_total_avg:.2f}')
        
        if 'multi_hop_reasoning' in total_avg_row:
            total_avg_row['multi_hop_reasoning'] = np.nan
        
        # Calculate total average score for each metric
        for col in score_columns:
            if question_averages[col]:
                total_avg = sum(question_averages[col]) / len(question_averages[col])
                total_avg_row[col] = total_avg
                print(f'Total average score for metric {col}: {total_avg:.4f}')
        
        result_rows.append(total_avg_row)
    
    # Create new DataFrame and save
    try:
        # If there are no result rows, skip creating and saving DataFrame
        if len(result_rows) == 0:
            print("Warning: Result rows are empty, not creating DataFrame, original file preserved")
            return overall_avg, model_avgs
            
        result_df = pd.DataFrame(result_rows)
        
        # Double-check if DataFrame is empty (just in case)
        if result_df.empty:
            print("Warning: Generated DataFrame is empty, not writing to file")
            return overall_avg, model_avgs
        
        # Write to a temporary file first
        temp_file = file_path + '.tmp'
        result_df.to_csv(temp_file, index=False)
        
        # Check if temporary file was written successfully
        if os.path.exists(temp_file):
            temp_size = os.path.getsize(temp_file)
            if temp_size > 10:
                # Temporary file written successfully, replace original file
                os.replace(temp_file, file_path)
            else:
                print(f"Warning: Temporary file is too small ({temp_size} bytes), potential write issue")
        else:
            print(f"Error: Temporary file not created successfully")
    except Exception as e:
        print(f"Error: Error creating or saving DataFrame: {str(e)}")
    
    print(f'Added question average scores and total average score, and saved to: {file_path}')
    
    # Print 'yes' probability
    if has_multi_hop and yes_counts:
        print(f"\n'Yes' probability in {os.path.basename(file_path)}:")
        for model_name in sorted(yes_counts.keys()):
            if total_counts[model_name] > 0:
                yes_prob = (yes_counts[model_name] / total_counts[model_name]) * 100
                print(f"  {model_name}: {yes_prob:.2f}%")
        
        # Calculate overall 'yes' probability for all models
        total_yes = sum(yes_counts.values())
        total_questions = sum(total_counts.values())
        if total_questions > 0:
            overall_yes_prob = (total_yes / total_questions) * 100
            print(f"  All models: {overall_yes_prob:.2f}%")
    
    # Calculate average scores for each model
    model_avgs = {}
    if 'model' in df.columns:
        model_groups = df.groupby('model')
        for model, group in model_groups:
            model_avg = group['overall_quality'].mean()
            model_avgs[model] = model_avg
        print("Average scores for each model:")
        for model, avg in model_avgs.items():
            print(f"Model {model}: {avg:.4f}")
    else:
        print("Could not find 'model' column, cannot calculate average scores for each model.")
    
    # Calculate and return total average of overall_quality and average scores for each model
    if question_averages['overall_quality']:
        overall_avg = sum(question_averages['overall_quality']) / len(question_averages['overall_quality'])
        print(f'Overall quality average score: {overall_avg:.4f}')
    else:
        print('Warning: No valid overall_quality data, total average score is 0')
    
    return overall_avg, model_avgs

def main():
    import argparse
    import glob
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate average scores for evaluation results')
    parser.add_argument('--dataset', type=str, help='Specify the dataset name to process, e.g., "mhqa_rerank_20250414_123456"')
    args = parser.parse_args()
    
    # Evaluation results directory
    eval_dir = './eval_result/'

    # Process all evaluation result files
    if args.dataset:
        # If a dataset name is specified, only process the original and polished evaluation files for that dataset
        csv_files = [
            f'{args.dataset}_original_evaluation.csv', 
            f'{args.dataset}_polished_evaluation.csv'
        ]
        print(f'Processing evaluation files for dataset {args.dataset}...')
    else:
        # Default: process all predefined files
        csv_files = [
            'test_comparison_gemini_comparison_20250417_205256_original_evaluation.csv',
            'test_comparison_gemini_comparison_20250417_205256_polished_evaluation.csv'
        ]
        print('Starting to process all evaluation result files...')
    
    all_averages = {}

    for file in csv_files:
        file_path = os.path.join(eval_dir, file)
        if os.path.exists(file_path):
            avg, model_avgs = process_evaluation_file(file_path)
            all_averages[file] = (avg, model_avgs)
            print(f'File {file} processing completed\n')
        else:
            print(f'File does not exist: {file_path}\n')

    print('All evaluation result files processed!')
    print('Total average scores for each file:')
    for file, (avg, model_avgs) in all_averages.items():
        print(f'{file}: {avg:.4f}')
        print(f'Average scores for each model:')
        for model, avg in model_avgs.items():
            print(f"  Model {model}: {avg:.4f}")

if __name__ == '__main__':
    main()
