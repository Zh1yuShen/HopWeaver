#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Model evaluation stability testing script
Performs 5 evaluations for each sample, calculates standard deviation to compare model evaluation stability
google/gemma-3-27b-it:free
meta-llama/llama-4-maverick:free
meta-llama/llama-3.3-70b-instruct:free
'''
import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import time
import random

# Custom Fleiss' Kappa calculation function
def fleiss_kappa(testData, N, k, n): # testData is the data to calculate, (N,k) is the shape of the matrix, indicating N rows and k columns, with n annotators
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0*dataMat[i, j]**2
        temp -= n
        temp /= (n-1)*n
        P0 += temp
    P0 = 1.0*P0/N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i]/sum)**2
    Pe = ysum*oneMat*1.0
    ans = (P0-Pe)/(1-Pe)
    return ans[0, 0]
# Add current directory and parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add FlashRAG main directory to path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from evaluation_system.evaluator import QualityEvaluator
from flashrag.config import Config

# Configuration files and paths
DEFAULT_CONFIG_PATH = "./config_lib/example_config.yaml"
DATASETS_DIR = "./datasets"
EVAL_RESULT_DIR = "./eval_result/stability"
STABILITY_DIR = "./eval_result/self_consistency"

# Evaluation model list
MODELS = [
    "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-11-20",
    "deepseek/deepseek-chat-v3-0324:free",
    "nvidia/llama-3.3-nemotron-super-49b-v1:free",
    "gemini-2.0-flash",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free"
]

# Evaluation dimensions
DIMENSIONS = [
    'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
    'consistency', 'question_answerability', 'answer_question_consistency', 
    'information_integration_ability', 'reasoning_path_guidance',
    'logical_sophistication', 'overall_quality'
]

# Numeric evaluation dimensions (excluding multi_hop_reasoning which is boolean)
NUMERIC_DIMENSIONS = [dim for dim in DIMENSIONS if dim != 'multi_hop_reasoning']

def ensure_dir_exists(dir_path):
    """Ensure directory exists, create if not"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def load_samples(bridge_path, comparison_path, num_samples=25):
    """
    Load dataset samples
    
    Args:
        bridge_path (str): Path to bridge type dataset
        comparison_path (str): Path to comparison type dataset
        num_samples (int): Number of samples to load for each type
        
    Returns:
        list: List of loaded samples
    """
    samples = []
    
    # Load bridge samples
    with open(bridge_path, 'r', encoding='utf-8') as f:
        bridge_data = json.load(f)
        
    if len(bridge_data) > num_samples:
        bridge_samples = random.sample(bridge_data, num_samples)
    else:
        bridge_samples = bridge_data
        
    for sample in bridge_samples:
        sample["sample_type"] = "bridge"
        samples.append(sample)
        
    # Load comparison samples
    with open(comparison_path, 'r', encoding='utf-8') as f:
        comparison_data = json.load(f)
        
    if len(comparison_data) > num_samples:
        comparison_samples = random.sample(comparison_data, num_samples)
    else:
        comparison_samples = comparison_data
        
    for sample in comparison_samples:
        sample["sample_type"] = "comparison"
        samples.append(sample)
        
    print(f"Loaded {len(samples)} samples ({len(bridge_samples)} bridge + {len(comparison_samples)} comparison)")
    return samples

def evaluate_question(config_path, model_name, question, retry_count=3):
    """
    Evaluate a question using the specified model
    
    Args:
        config_path (str): Configuration file path
        model_name (str): Model name
        question (dict): Question data
        retry_count (int): Number of retries
        
    Returns:
        dict: Evaluation result
    """
    import asyncio
    # Fix asyncio issues in multi-threaded environment
    def patch_openai_generator():
        """Patch asyncio event loop issues in openai_generator"""
        from flashrag.generator import openai_generator
        original_generate = openai_generator.OpenaiGenerator.generate
        
        def patched_generate(self, input_list, batch_size=1, return_scores=False, **generation_params):
            # Ensure a new event loop is created in child threads
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If the current thread has no event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))
            
            # The rest of the original code
            scores = []
            response_text = []
            for res in result:
                filtered_content = self._filter_thinking_chain(res.message.content)
                response_text.append(filtered_content)
                if return_scores:
                    scores.append(res.get('score', 0.0))
            
            if return_scores:
                return response_text, scores
            return response_text
        
        # Save the original method as backup
        if not hasattr(openai_generator.OpenaiGenerator, '_original_generate'):
            openai_generator.OpenaiGenerator._original_generate = original_generate
            openai_generator.OpenaiGenerator.generate = patched_generate
    
    # Apply patch
    patch_openai_generator()
    
    for attempt in range(retry_count):
        try:
            # Create model configuration
            model_config = Config(config_path, {})
            model_config["evaluator_model"] = model_name
            
            # Initialize evaluator
            evaluator = QualityEvaluator(model_config)
            
            # Evaluate question
            result = evaluator.evaluate_question(question)
            
            if result and "evaluation" in result:
                return result["evaluation"]
            else:
                print(f"Warning: Attempt {attempt+1} using {model_name} failed, no valid result returned")
        except Exception as e:
            print(f"Evaluation attempt {attempt+1}/{retry_count} failed: {str(e)}")
            if attempt < retry_count - 1:
                print(f"Retrying... ({attempt+1}/{retry_count})")
                time.sleep(2)  # Wait before retrying
    
    return None

def run_stability_evaluation(config_path, samples, output_dir, repeat_times=5, max_workers=4, test_mode=False):
    """
    Run stability evaluation
    
    Args:
        config_path (str): Configuration file path
        samples (list): List of samples
        output_dir (str): Output directory
        repeat_times (int): Number of evaluation repetitions for each sample
        max_workers (int): Maximum number of parallel worker threads
    """
    ensure_dir_exists(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory for storing results
    result_dir = os.path.join(output_dir, f"stability_eval_{timestamp}")
    ensure_dir_exists(result_dir)
    
    # Create file for raw results
    raw_results_file = os.path.join(result_dir, "raw_results.csv")
    standard_dev_file = os.path.join(result_dir, "standard_deviations.csv")
    
    all_results = []
    
    # In test mode, only use one sample
    if test_mode:
        test_samples = []
        # Ensure we have at least one bridge and one comparison sample
        bridge_sample = next((s for s in samples if s.get("sample_type") == "bridge"), None)
        comparison_sample = next((s for s in samples if s.get("sample_type") == "comparison"), None)
        
        if bridge_sample:
            test_samples.append(bridge_sample)
        if comparison_sample and (not bridge_sample or bridge_sample != comparison_sample):
            test_samples.append(comparison_sample)
            
        # If no samples yet, take the first one
        if not test_samples and samples:
            test_samples = [samples[0]]
            
        samples = test_samples
        print(f"\nTest mode: only using {len(samples)} samples")
    
    # Evaluate each sample
    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{sample_idx}")
        sample_type = sample.get("sample_type", "unknown")
        question = sample.get("question", "")
        
        print(f"\nEvaluating sample {sample_idx+1}/{len(samples)}: {sample_id} ({sample_type})")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        
        # Collect documents
        documents = []
        for i in range(1, 11):  # Check document1 through document10
            doc_key = f"document{i}"
            if doc_key in sample and sample[doc_key]:
                documents.append(sample[doc_key])
        
        # Evaluate each model multiple times
        for model_name in MODELS:
            print(f"Evaluating with model {model_name} (repeating {repeat_times} times)...")
            
            # Create thread pool for parallel processing of multiple evaluations
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all evaluation tasks
                future_to_repeat = {
                    executor.submit(evaluate_question, config_path, model_name, sample): 
                    repeat_idx for repeat_idx in range(repeat_times)
                }
                
                # Collect evaluation results
                for future in tqdm(future_to_repeat, desc=f"{model_name} evaluation progress"):
                    repeat_idx = future_to_repeat[future]
                    try:
                        evaluation = future.result()
                        
                        if evaluation:
                            # Create result row
                            row = {
                                "sample_id": sample_id,
                                "sample_type": sample_type,
                                "model": model_name,
                                "question": question,
                                "repeat_idx": repeat_idx,
                                "document_count": len(documents)
                            }
                            
                            # Add evaluation dimension results
                            for dim in DIMENSIONS:
                                if dim == "multi_hop_reasoning":
                                    is_multi_hop = evaluation.get(dim, False)
                                    row[dim] = 1 if is_multi_hop else 0
                                else:
                                    row[dim] = evaluation.get(dim, "")
                            
                            # Add to results
                            all_results.append(row)
                        else:
                            print(f"Warning: Failed to evaluate sample {sample_id} with {model_name} on attempt {repeat_idx+1}")
                    except Exception as e:
                        print(f"Error processing {model_name}'s evaluation result on attempt {repeat_idx+1}: {str(e)}")
            
        # Save results every 10 samples
        if (sample_idx + 1) % 10 == 0 or sample_idx == len(samples) - 1:
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(raw_results_file, index=False, encoding='utf-8')
                print(f"Intermediate evaluation results saved to: {raw_results_file}")
    
    # Save final raw results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(raw_results_file, index=False, encoding='utf-8')
        print(f"All raw evaluation results saved to: {raw_results_file}")
        
        # Calculate standard deviation
        calc_and_save_standard_deviations(df, standard_dev_file, result_dir)
    else:
        print("Warning: No evaluation results were collected")

def calc_and_save_standard_deviations(df, output_file, result_dir):
    """
    Calculate standard deviations and save results
    
    Args:
        df (DataFrame): Original evaluation results
        output_file (str): Output file path
        result_dir (str): Result directory
    """
    # Ensure multi_hop_reasoning column is numeric
    if 'multi_hop_reasoning' in df.columns:
        if df['multi_hop_reasoning'].dtype == object:
            df['multi_hop_reasoning'] = df['multi_hop_reasoning'].apply(
                lambda x: 1 if x == "Yes" or x is True or x == 1 else 0
            )
    
    # Map rating strings to numeric values, removing excess asterisks
    rating_map = {"Very Good": 5, "Good": 4, "Fair": 3, "Poor": 2, "Very Poor": 1}
    rating_dims = [dim for dim in DIMENSIONS if dim not in ("multi_hop_reasoning", "overall_quality")]
    for dim in rating_dims:
        if dim in df.columns:
            df[dim] = df[dim].astype(str).str.replace(r"\*+", "", regex=True)
            df[dim] = df[dim].map(rating_map)
    
    # Initialize results list
    std_results = []
    
    # Calculate standard deviation by model and dimension
    for model in MODELS:
        model_df = df[df['model'] == model]
        
        # Group by sample ID
        for sample_id in model_df['sample_id'].unique():
            sample_df = model_df[model_df['sample_id'] == sample_id]
            sample_type = sample_df['sample_type'].iloc[0] if not sample_df.empty else "unknown"
            
            # Create a result row
            result_row = {
                "model": model,
                "sample_id": sample_id,
                "sample_type": sample_type,
                "eval_count": len(sample_df)
            }
            
            # Calculate standard deviation for each dimension
            for dim in DIMENSIONS:
                # Extract numerical data
                if dim == 'multi_hop_reasoning':
                    # For multi-hop reasoning, calculate standard deviation of Yes/No results (converted to 0/1)
                    values = sample_df[dim].astype(float)
                else:
                    # For other dimensions, filter out numeric values
                    values = pd.to_numeric(sample_df[dim], errors='coerce').dropna()
                
                if len(values) > 1:  # Need at least two values to calculate standard deviation
                    std_dev = values.std()
                    result_row[f"{dim}_std"] = std_dev
                    
                    # Also calculate average
                    avg_value = values.mean()
                    result_row[f"{dim}_avg"] = avg_value
                else:
                    result_row[f"{dim}_std"] = None
                    result_row[f"{dim}_avg"] = None
            
            std_results.append(result_row)
    
    # Save standard deviation results
    std_df = pd.DataFrame(std_results)
    std_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Standard deviation results saved to: {output_file}")
    
    # Generate visualizations
    generate_visualizations(std_df, result_dir)
    
    # Summarize average standard deviation by model
    summarize_model_std_dev(std_df, result_dir)

def generate_visualizations(std_df, result_dir):
    """
    Generate standard deviation visualizations
    
    Args:
        std_df (DataFrame): Standard deviation result data
        result_dir (str): Result directory
    """
    # Create charts directory
    viz_dir = os.path.join(result_dir, "visualizations")
    ensure_dir_exists(viz_dir)
    
    # Set font to support Chinese
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative sign
    
    # Set Seaborn and Matplotlib style
    plt.rcParams.update({
        'font.size': 12,
        'figure.autolayout': True,
        'axes.titlesize': 16,
        'axes.labelsize': 14
    })
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Generate heatmap of average standard deviation for all dimensions
    generate_heatmap(std_df, viz_dir)
    
    # Generate detailed charts for overall_quality (this is the most important metric)
    generate_overall_quality_plots(std_df, viz_dir)
    
    # Generate detailed boxplots for each dimension
    generate_dimension_boxplots(std_df, viz_dir)

def generate_heatmap(std_df, viz_dir):
    """
    Generate heatmap showing standard deviations of each model across different dimensions
    """
    # Print some debug information
    print("\n------- Debugging Heat Map Data -------")
    print(f"Available columns: {std_df.columns.tolist()}")
    # Print first 5 rows to view data format
    print("\nData Sample:")
    print(std_df.head())
    
    # Check valid data count for each dimension column
    for dim in DIMENSIONS:
        std_col = f"{dim}_std"
        if std_col in std_df.columns:
            valid_count = std_df[std_col].notna().sum()
            print(f"Dimension {dim}: {valid_count} valid data points")
    
    # Load processed data and create index - Calculate average values for each model and dimension
    model_dimension_stats = []
    
    # First, group by model
    for model in MODELS:
        model_data = std_df[std_df['model'] == model]
        if model_data.empty:
            continue
            
        # Then process each dimension
        for dim in DIMENSIONS:
            std_col = f"{dim}_std"
            
            if std_col in model_data.columns:
                # Get valid data
                valid_data = model_data[std_col].dropna()
                
                # If valid data is not empty, calculate average and record
                if len(valid_data) > 0:
                    avg_std = valid_data.mean()
                    model_dimension_stats.append({
                        'model': model,
                        'dimension': dim,
                        'avg_std': avg_std,
                        'count': len(valid_data)
                    })
    
    # If no data, return
    if not model_dimension_stats:
        print("No valid model dimension data to generate heatmap")
        return
        
    # Create DataFrame for pivot table
    stats_df = pd.DataFrame(model_dimension_stats)
    
    # Print statistics
    print("\nModel-Dimension Statistics:")
    print(stats_df)
    
    # Generate heatmap
    stats_df = pd.DataFrame(model_dimension_stats)
    pivot_df = stats_df.pivot(index='dimension', columns='model', values='avg_std')
    
    # Arrange dimensions in specified order
    dimension_order = [
        'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
        'consistency', 'question_answerability', 'answer_question_consistency', 
        'information_integration_ability', 'reasoning_path_guidance',
        'logical_sophistication', 'overall_quality'
    ]
    # Filter actual existing dimensions and maintain original order
    available_dims = set(pivot_df.index.tolist())
    new_order = [d for d in dimension_order if d in available_dims]
    pivot_df = pivot_df.reindex(new_order)
    
    # Print pivot table
    print("\nPivot Table for Heatmap:")
    print(pivot_df)
    
    if not pivot_df.empty:
        plt.figure(figsize=(14, 10))
    
    # Use more suitable color mapping and handle missing values
    mask = pivot_df.isna()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # High contrast color mapping
    
    # Calculate vmax as the maximum value of the data to make the color scale consistent
    max_val = pivot_df.max().max()
    
    # Draw heatmap
    ax = sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap=cmap, mask=mask,
                    vmin=0, vmax=max_val, linewidths=0.5, square=False)
    
    # Set English titles and labels
    ax.set_title('Average Standard Deviation by Model and Dimension', fontsize=16, pad=20)
    ax.set_ylabel('Evaluation Dimension', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    
    # Adjust x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_dimension_std_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate sample count heatmap
    sample_pivot = stats_df.pivot(index='dimension', columns='model', values='count')
    
    plt.figure(figsize=(14, 10))
    mask = sample_pivot.isna()
    ax = sns.heatmap(sample_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=.5, mask=mask)
    ax.set_title("Sample Count by Model and Dimension", fontsize=16, pad=20)
    ax.set_ylabel("Evaluation Dimension", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_dimension_sample_count.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_overall_quality_plots(std_df, viz_dir):
    """
    Generate detailed plots for overall_quality dimension
    """
    # 1. Generate boxplot for overall_quality
    std_col = "overall_quality_std"
    
    # Check if column exists
    if std_col not in std_df.columns:
        print(f"Warning: Column '{std_col}' not found in data.")
        print(f"Available columns: {std_df.columns.tolist()}")
        return
        
    valid_df = std_df[std_df[std_col].notna()]
    
    if not valid_df.empty:
        # Use sorted model list, ordered by average standard deviation
        model_avg_std = valid_df.groupby('model')[std_col].mean().sort_values()
        sorted_models = model_avg_std.index.tolist()
        
        # Create new categorical column using sorted model list
        if len(sorted_models) > 0:
            valid_df['model'] = pd.Categorical(valid_df['model'], categories=sorted_models, ordered=True)
        
        # Generate boxplot
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(x="model", y=std_col, hue="model", data=valid_df.sort_values('model'), palette="Spectral", legend=False)
        
        # Add sample count text
        for i, model in enumerate(sorted_models):
            count = len(valid_df[valid_df['model'] == model])
            mean_val = valid_df[valid_df['model'] == model][std_col].mean()
            ax.text(i, mean_val, f"n={count}", horizontalalignment='center', fontsize=10, fontweight='bold')
        
        # Set English titles and labels
        ax.set_title("Standard Deviation Distribution for Overall Quality", fontsize=16, pad=20)
        ax.set_xlabel("Models (Ordered by Mean Std Dev)", fontsize=14)
        ax.set_ylabel("Standard Deviation", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        
        # Add mean value line
        avg_std = valid_df[std_col].mean()
        ax.axhline(avg_std, ls='--', color='red', label=f'Mean: {avg_std:.3f}')
        
        # Add legend
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "overall_quality_std_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Generate boxplot for overall_quality by sample type
        plt.figure(figsize=(14, 8))
        valid_df = valid_df.sort_values('model')
        ax = sns.boxplot(x="model", y=std_col, hue="sample_type", data=valid_df, palette="Set2")
        
        # Set English titles and labels
        ax.set_title("Overall Quality Standard Deviation by Sample Type", fontsize=16, pad=20)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel("Standard Deviation", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        
        # Improve legend position
        plt.legend(title="Sample Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "overall_quality_std_by_type_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Try to generate violin plot to better show distribution
        plt.figure(figsize=(14, 8))
        sns.violinplot(x="model", y=std_col, hue="model", data=valid_df, palette="Spectral", inner="quartile", legend=False)
        plt.title("Overall Quality Standard Deviation (Violin Plot)", fontsize=16, pad=20)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Standard Deviation", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "overall_quality_std_violin.png"), dpi=300, bbox_inches='tight')
        plt.close()

def generate_dimension_average_scores(score_df, viz_dir):
    """
    Generate heatmap showing average scores of each model on each evaluation dimension
    
    Args:
        score_df (DataFrame): Model average score data
        viz_dir (str): Visualization output directory
    """
    # 1. Prepare data - extract average scores for each dimension
    plot_data = {}
    
    # Get all model names
    models = score_df['model'].tolist()
    
    # Loop through each dimension, collect average scores
    for model in models:
        model_scores = []
        # Get model row
        model_row = score_df[score_df['model'] == model]
        
        if not model_row.empty:
            for dim in DIMENSIONS:
                # Get score for this dimension
                score_col = f"{dim}_score"
                if score_col in model_row.columns:
                    score = model_row[score_col].values[0]
                    model_scores.append(score)
                else:
                    model_scores.append(None)
            
            # Store this model's scores for all dimensions
            plot_data[model] = model_scores
    
    # 2. Convert to DataFrame, prepare for plotting
    # Note the transposition here! Dimensions on rows (y-axis), Models on columns (x-axis)
    heatmap_df = pd.DataFrame(plot_data, index=DIMENSIONS)
    
    # 3. Draw heatmap
    plt.figure(figsize=(16, 10))
    
    # Create mask for missing values
    mask = heatmap_df.isna()
    
    # Draw heatmap
    ax = sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='viridis', 
                  linewidths=.5, mask=mask, vmin=1, vmax=5, 
                  cbar_kws={'label': 'Average Score'})
    
    # Set title and labels
    plt.title('Average Scores of Models on Each Evaluation Dimension', fontsize=16)
    plt.ylabel('Evaluation Dimension', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    
    # Adjust label directions for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "model_dimension_average_scores.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated model dimension average scores heatmap: {os.path.join(viz_dir, 'model_dimension_average_scores.png')}")


def generate_dimension_boxplots(std_df, viz_dir):
    """
    Generate boxplots for each evaluation dimension
    """
    # For each dimension except overall_quality, generate boxplots
    remaining_dims = [dim for dim in DIMENSIONS if dim != 'overall_quality']
    
    for dim in remaining_dims:
        std_col = f"{dim}_std"
        
        # Filter out valid standard deviation values
        valid_df = std_df[std_df[std_col].notna()]
        
        if len(valid_df) > 0:
            # Create a new figure
            plt.figure(figsize=(14, 7))
            
            # Draw boxplot
            ax = sns.boxplot(x="model", y=std_col, hue="model", data=valid_df, palette="Set3", legend=False)
            
            # Set English title and labels
            ax.set_title(f"{dim} Standard Deviation Distribution", fontsize=16, pad=20)
            ax.set_xlabel("Model", fontsize=14)
            ax.set_ylabel("Standard Deviation", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            
            # Add sample count
            for i, model in enumerate(valid_df['model'].unique()):
                count = len(valid_df[valid_df['model'] == model])
                ax.text(i, valid_df[valid_df['model'] == model][std_col].max(), 
                        f"n={count}", ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{dim}_std_boxplot.png"), dpi=300, bbox_inches='tight')
            plt.close()

def summarize_model_std_dev(std_df, result_dir):
    """
    Summarize the average standard deviation and average score for each model
    
    Args:
        std_df (DataFrame): Standard deviation result data
        result_dir (str): Result directory
    """
    # 1. Prepare data storage containers
    std_summary_rows = []   # Store standard deviation summary data
    avg_score_rows = []     # Store average score summary data
    
    # 2. Process data for each model
    for model in MODELS:
        model_df = std_df[std_df['model'] == model]
        if model_df.empty:
            continue
            
        # Initialize data container for each model
        std_row = {
            "model": model, 
            "sample_count": len(model_df)
        }
        avg_row = {
            "model": model, 
            "sample_count": len(model_df)
        }
        
        # Calculate standard deviation and average score for each evaluation dimension
        for dim in DIMENSIONS:
            # Process standard deviation
            std_values = model_df[f"{dim}_std"].dropna()
            if len(std_values) > 0:
                std_row[f"{dim}_std"] = std_values.mean()
                std_row[f"{dim}_std_count"] = len(std_values)
            else:
                std_row[f"{dim}_std"] = None
                std_row[f"{dim}_std_count"] = 0
            
            # Process average scores
            avg_values = model_df[f"{dim}_avg"].dropna()
            if len(avg_values) > 0:
                avg_row[f"{dim}_score"] = avg_values.mean()
                avg_row[f"{dim}_score_count"] = len(avg_values)
            else:
                avg_row[f"{dim}_score"] = None
                avg_row[f"{dim}_score_count"] = 0
        
        # Calculate average scores separately by sample type
        for sample_type in ['bridge', 'comparison']:
            type_df = model_df[model_df['sample_type'] == sample_type]
            if not type_df.empty:
                # Calculate overall_quality average score for this sample type
                overall_values = type_df['overall_quality_avg'].dropna()
                if len(overall_values) > 0:
                    avg_row[f"overall_quality_score_{sample_type}"] = overall_values.mean()
                    avg_row[f"overall_quality_score_{sample_type}_count"] = len(overall_values)
        
        # Calculate overall_quality standard deviation and average score as summary metrics
        overall_std = model_df['overall_quality_std'].dropna()
        if len(overall_std) > 0:
            std_row["overall_std"] = overall_std.mean()
        else:
            std_row["overall_std"] = None
            
        overall_score = model_df['overall_quality_avg'].dropna()
        if len(overall_score) > 0:
            avg_row["overall_score"] = overall_score.mean()
        else:
            avg_row["overall_score"] = None
            
        # Add to summary lists
        std_summary_rows.append(std_row)
        avg_score_rows.append(avg_row)
    
    # 3. Create and save standard deviation summary dataframe
    if std_summary_rows:
        std_df = pd.DataFrame(std_summary_rows)
        
        # Sort by overall_quality standard deviation (lower is more stable)
        if 'overall_std' in std_df.columns:
            std_df = std_df.sort_values('overall_std')
        
        # Save standard deviation summary
        std_file = os.path.join(result_dir, "model_std_summary.csv")
        std_df.to_csv(std_file, index=False, encoding='utf-8')
        print(f"Standard deviation summary saved to: {std_file}")
        # Print model stability ranking
        if 'overall_std' in std_df.columns and not std_df['overall_std'].isna().all():
            print("\n===== Model Stability Ranking (Lower Standard Deviation is More Stable) =====")
            for i, (_, row) in enumerate(std_df.iterrows()):
                if pd.notna(row["overall_std"]):
                    print(f"{i+1}. {row['model']}: overall_quality standard deviation = {row['overall_std']:.4f}")
    
    # 4. Create and save average score summary dataframe
    if avg_score_rows:
        score_df = pd.DataFrame(avg_score_rows)
        
        # Sort by overall_quality score (higher is better)
        if 'overall_score' in score_df.columns:
            score_df = score_df.sort_values('overall_score', ascending=False)
        
        # Save average score summary
        score_file = os.path.join(result_dir, "model_avg_scores.csv")
        score_df.to_csv(score_file, index=False, encoding='utf-8')
        print(f"Model average score summary saved to: {score_file}")
        
        # Print model average score ranking
        if 'overall_score' in score_df.columns and not score_df['overall_score'].isna().all():
            print("\n===== Model Average Score Ranking (Higher is Better) ====")
            for i, (_, row) in enumerate(score_df.iterrows()):
                if pd.notna(row["overall_score"]):
                    print(f"{i+1}. {row['model']}: overall_quality average score = {row['overall_score']:.4f}")
    
    # 5. Generate average score visualizations
    if avg_score_rows and not all(row.get('overall_score') is None for row in avg_score_rows):
        viz_dir = os.path.join(result_dir, "visualizations")
        ensure_dir_exists(viz_dir)
        
        # Create average score bar chart
        plt.figure(figsize=(12, 8))
        
        # Prepare data - sort by score in descending order
        plot_data = score_df.sort_values('overall_score', ascending=False)
        models = plot_data['model']
        scores = plot_data['overall_score']
        
        # Draw bar chart
        bars = plt.barh(models, scores, color=plt.cm.viridis(np.linspace(0, 0.9, len(models))))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if pd.notna(width):
                plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}", 
                         ha='left', va='center', fontweight='bold')
        
        # Add title and labels
        plt.title("Average Overall Quality Score by Model", fontsize=16)
        plt.xlabel("Score", fontsize=14)
        plt.ylabel("Model", fontsize=14)
        plt.xlim(0, 5)  # Assume score range is 0-5
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "overall_quality_avg_score.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create grouped bar charts for key dimensions
        # Select some important dimensions for comparison
        key_dimensions = [
            'overall_quality', 'relevance', 'consistency', 
            'question_answerability', 'reasoning_path_guidance'
        ]
        
        # Filter out dimensions with sufficient data
        valid_dimensions = []
        for dim in key_dimensions:
            col_name = f"{dim}_score"
            if col_name in score_df.columns and not score_df[col_name].isna().all():
                valid_dimensions.append(dim)
        
        if valid_dimensions:
            # Create and sort plot data
            plot_df = score_df.sort_values('overall_score', ascending=False).copy()
            
            # Select columns to plot
            plot_columns = [f"{dim}_score" for dim in valid_dimensions]
            
            # Create new DataFrame for plotting
            plot_data = plot_df[['model'] + plot_columns].set_index('model')
            plot_data.columns = [col.replace('_score', '') for col in plot_data.columns]
            
            # Draw multi-dimension comparison chart
            plt.figure(figsize=(14, 8))
            plot_data.plot(kind='barh', figsize=(14, 8), width=0.8)
            
            plt.title("Average Score Comparison by Dimension", fontsize=16)
            plt.xlabel("Score", fontsize=14)
            plt.ylabel("Model", fontsize=14)
            plt.xlim(0, 5)  # Assume score range is 0-5
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.legend(title="Dimensions", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "dimensions_avg_score_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        # Generate average score heatmap for all dimensions
        generate_dimension_average_scores(score_df, viz_dir)

def main():
    global MODELS
    parser = argparse.ArgumentParser(description='Model evaluation stability test')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Configuration file path')
    parser.add_argument('--bridge', type=str, default=os.path.join(DATASETS_DIR, '2wiki_bridge.json'), help='Bridge type dataset path')
    parser.add_argument('--comparison', type=str, default=os.path.join(DATASETS_DIR, '2wiki_comparison.json'), help='Comparison type dataset path')
    parser.add_argument('--num_samples', type=int, default=25, help='Number of samples to select for each type')
    parser.add_argument('--repeats', type=int, default=5, help='Number of evaluation repetitions for each sample')
    parser.add_argument('--output_dir', type=str, default=STABILITY_DIR, help='Output directory')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of parallel worker threads')
    parser.add_argument('--test', action='store_true', help='Test mode, evaluate only one sample per model')
    parser.add_argument('--results_dir', type=str, default=None, help='Existing evaluation results directory, only perform visualization and metric calculation')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to evaluate')

    args = parser.parse_args()
    
    # If --models is specified, only evaluate these models
    if args.models:
        MODELS = [model.strip() for model in args.models.split(',')]
        print(f"Only evaluating models: {MODELS}")
    
    # If existing results directory is specified, only perform visualization and metric calculation
    if args.results_dir:
        result_dir = args.results_dir
        # Check raw results file
        raw_file = os.path.join(result_dir, "raw_results.csv")
        if not os.path.exists(raw_file):
            print(f"Raw results file not found: {raw_file}")
            sys.exit(1)
        # Load from raw_results and recalculate standard deviations and map ratings
        df = pd.read_csv(raw_file, encoding='utf-8')
        print(f"Loaded raw evaluation results from {raw_file}, recalculating standard deviations and generating visualizations and metrics")
        standard_dev_file = os.path.join(result_dir, "standard_deviations.csv")
        calc_and_save_standard_deviations(df, standard_dev_file, result_dir)
        std_df = pd.read_csv(standard_dev_file, encoding='utf-8')
        # Calculate Z-score normalized averages
        z_df = std_df.copy()
        for dim in DIMENSIONS:
            avg_col = f"{dim}_avg"
            if avg_col in z_df.columns:
                z_col = f"{dim}_zscore"
                # Global standardization (cross-model comparison), avoid std=1 within each model
                z_df[z_col] = (z_df[avg_col] - z_df[avg_col].mean()) / z_df[avg_col].std()
        # Save Z-score normalized results
        zscore_file = os.path.join(result_dir, "zscore_standardized.csv")
        z_df.to_csv(zscore_file, index=False, encoding='utf-8')
        print(f"Z-score normalized results saved to: {zscore_file}")
        # Generate Z-score visualizations
        z_viz_dir = os.path.join(result_dir, "zscore_visualizations")
        ensure_dir_exists(z_viz_dir)
        # Convert Z-score data to long format
        z_melt = z_df.melt(id_vars=['model'], value_vars=[f"{dim}_zscore" for dim in DIMENSIONS], var_name='variable', value_name='zscore')
        z_melt['dimension'] = z_melt['variable'].str.replace('_zscore', '', regex=False)
        # Z-score standard deviation heatmap
        pivot_z_std = z_melt.groupby(['dimension', 'model'])['zscore'].std().unstack('model')
        plt.figure(figsize=(14, 10))
        mask = pivot_z_std.isna()
        vmax_std = pivot_z_std.max().max()
        sns.heatmap(pivot_z_std, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask, vmin=0, vmax=vmax_std, linewidths=0.5)
        plt.title('Z-Score Std Dev Heatmap by Model and Dimension', fontsize=16, pad=20)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Dimension', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(z_viz_dir, 'zscore_std_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Z-score boxplots for each dimension
        for dim in DIMENSIONS:
            plt.figure(figsize=(14, 7))
            subset = z_melt[z_melt['dimension'] == dim]
            if not subset['zscore'].dropna().empty:
                sns.boxplot(x='model', y='zscore', hue='model', data=subset, palette='Set3', legend=False)
                plt.title(f"{dim} Z-Score Distribution", fontsize=16, pad=20)
                plt.xlabel('Model', fontsize=14)
                plt.ylabel('Z-Score', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(z_viz_dir, f"{dim}_zscore_boxplot.png"), dpi=300, bbox_inches='tight')
                plt.close()
        print(f"Z-score visualizations saved to: {z_viz_dir}")
            # Krippendorff's Alpha results and Fleiss' Kappa
        alpha_results = []
        fleiss_results = []
        print("\nStarting to calculate evaluator consistency metrics...\n")
        try:
            import krippendorff
        except ImportError:
            print("Missing krippendorff package, please install via pip install krippendorff")
        else:
            print(f"Saved Krippendorff's Alpha results to: {alpha_file}")
            for model in MODELS:
                model_df = df[df['model'] == model]
                for dim in DIMENSIONS:
                    matrix = model_df.pivot(index='repeat_idx', columns='sample_id', values=dim).values
                    try:
                        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')
                    except ValueError as e:
                        if "more than one value in the domain" in str(e):
                            # All ratings completely consistent, considered as highest consistency
                            print(f"{model} - {dim}: Ratings completely consistent, setting α = 1.000")
                            alpha = 1.0
                        else:
                            # Raise other errors
                            raise
                    alpha_results.append({'model': model, 'dimension': dim, 'alpha': alpha})
                    if alpha != 1.0 or "Ratings completely consistent" not in locals():
                        print(f"{model} - {dim}: α = {alpha:.3f}")
        try:
            import statsmodels.stats.inter_rater as ir
        except ImportError:
            print("Missing statsmodels package, please install it via pip install statsmodels")
        else:
            print("===== Fleiss' Kappa =====")
            # Initialize empty list to collect results
            fleiss_results = []
            
            # First convert data to numeric type to avoid string values
            for dim in DIMENSIONS:
                if dim == 'multi_hop_reasoning':
                    # Multi-hop reasoning might be boolean type, ensure it's numeric
                    df[dim] = df[dim].astype(float)
                else:
                    # Other dimensions remain as numeric values
                    df[dim] = pd.to_numeric(df[dim], errors='coerce')
            
            # Calculate for each model separately
            for model in MODELS:
                model_df = df[df['model'] == model]
                if model_df.empty:
                    print(f"Warning: No data found for model {model}")
                    continue
                    
                print(f"\nCalculating Fleiss' Kappa for {model}...")
                
                for dim in DIMENSIONS:
                    # Determine all possible category values for this dimension
                    values = model_df[dim].dropna()
                    if values.empty:
                        print(f"  Skipping {dim}: Insufficient data")
                        continue
                        
                    # Discretize ratings to reduce the number of rating categories
                    if dim == 'multi_hop_reasoning':
                        # Binary type dimension
                        categories = [0.0, 1.0]  # Ensure using float
                    else:
                        # Use integer part as category, merge similar decimal ratings
                        # For ratings in range 1-5, convert them to five categories: 1,2,3,4,5
                        categories = [1, 2, 3, 4, 5]
                        
                    # Convert ratings to discrete categories
                    def discretize_rating(rating):
                        if dim == 'multi_hop_reasoning':
                            return 1.0 if rating > 0.5 else 0.0
                        else:
                            # Round to the nearest integer
                            return round(rating)
                    
                    # Create rating frequency matrix (each row is a sample, each column is a rating category)
                    data_matrix = []
                    for sample_id in model_df['sample_id'].unique():
                        sample_data = model_df[model_df['sample_id'] == sample_id]
                        if len(sample_data) < 2:  # At least 2 raters are needed
                            continue
                            
                        # Convert ratings to discrete categories
                        ratings = [discretize_rating(r) for r in sample_data[dim].values if not pd.isna(r)]
                        
                        # Calculate frequency for each category
                        category_counts = []
                        for cat in categories:
                            count = sum(1 for r in ratings if r == cat)
                            category_counts.append(count)
                        
                        if sum(category_counts) >= 2:  # Ensure there are enough valid ratings
                            data_matrix.append(category_counts)
                    
                    # Check if there are enough sample data
                    if len(data_matrix) >= 2:
                        try:
                            # Print data matrix structure
                            n_samples = len(data_matrix)  # Number of samples
                            n_categories = len(categories)  # Number of categories
                            n_raters = int(sum(data_matrix[0]))  # Number of raters per sample
                            print(f"  {dim}: Data matrix {n_samples}x{n_categories}, {n_raters} raters per sample")
                            
                            # Check if there is enough variability
                            all_same = True
                            first_row = data_matrix[0]
                            for row in data_matrix[1:]:
                                if not np.array_equal(row, first_row):
                                    all_same = False
                                    break
                            
                            if all_same:
                                # All ratings are completely consistent, give highest Kappa value
                                kappa = 1.0
                                print(f"  {dim}: All ratings completely consistent, setting Kappa=1.0")
                            else:
                                # Use custom Fleiss' Kappa calculation function
                                try:
                                    kappa = fleiss_kappa(data_matrix, n_samples, n_categories, n_raters)
                                    print(f"  {dim}: Fleiss' Kappa calculation result (kappa = {kappa:.3f})")
                                except Exception as e:
                                    # If calculation fails, try a simpler method
                                    print(f"  Standard calculation failed, trying alternative method: {str(e)}")
                                    
                                    # Simplified method: calculate classification consistency percentage for each sample
                                    agreements = []
                                    for row in data_matrix:
                                        total = sum(row)
                                        if total > 0:
                                            # Save consistency metrics data proportion
                                            max_agreement = max(row) / total
                                            agreements.append(max_agreement)
                                    
                                    # Convert average consistency rate to Kappa value range
                                    if agreements:
                                        avg_agreement = sum(agreements) / len(agreements)
                                        # Map range [0.5, 1] to [0, 1]
                                        kappa = (avg_agreement - 0.5) * 2 if avg_agreement > 0.5 else 0
                                        print(f"  {dim}: Using alternative method, κ = {kappa:.3f} (average consistency: {avg_agreement:.3f})")
                                    else:
                                        kappa = 0
                                        print(f"  {dim}: Cannot calculate Fleiss' Kappa - ratings are the same for multiple samples or insufficient data")
                            
                            # Add to results list
                            fleiss_results.append({'model': model, 'dimension': dim, 'kappa': kappa})
                            
                        except Exception as e:
                            print(f"  - Error calculating Fleiss' Kappa: {str(e)}")
                    else:
                        print(f"  Skipping {dim}: Insufficient number of samples (only {len(data_matrix)} samples)")
                        
                    # Print empty line to increase readability
                    print()
            
            print(f"\nTotal calculated {len(fleiss_results)} Fleiss' Kappa results")
            # Save results to dataframe
            if fleiss_results:
                fleiss_df = pd.DataFrame(fleiss_results)
                print(f"Saved Fleiss' Kappa results to: {kappa_file}")
                print(f"Fleiss' Kappa results preview:\n{fleiss_df.head()}")
            else:
                print("Warning: No Fleiss' Kappa results were calculated")
                # Create empty dataframe for subsequent processing
                fleiss_df = pd.DataFrame(columns=['model', 'dimension', 'kappa'])
        # Save and visualize reliability metrics
        rel_dir = os.path.join(result_dir, 'reliability_visualizations')
        os.makedirs(rel_dir, exist_ok=True)
        # Save CSV
        pd.DataFrame(alpha_results).to_csv(os.path.join(rel_dir, 'krippendorff_alpha.csv'), index=False, encoding='utf-8')
        pd.DataFrame(fleiss_results).to_csv(os.path.join(rel_dir, 'fleiss_kappa.csv'), index=False, encoding='utf-8')
        # Alpha Heatmap
        alpha_df = pd.DataFrame(alpha_results)
        if not alpha_df.empty:
            alpha_pivot = alpha_df.pivot(index='dimension', columns='model', values='alpha')
            # Arrange dimensions in specified order
            dimension_order = [
                'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
                'consistency', 'question_answerability', 'answer_question_consistency', 
                'information_integration_ability', 'reasoning_path_guidance',
                'logical_sophistication', 'overall_quality'
            ]
            # Filter actual existing dimensions and maintain original order
            available_dims = set(alpha_pivot.index.tolist())
            new_order = [d for d in dimension_order if d in available_dims]
            alpha_pivot = alpha_pivot.reindex(new_order)
            plt.figure(figsize=(14, 10))
            mask = alpha_pivot.isna()
            cmap = sns.color_palette("YlGn", as_cmap=True)  # Use green gradient color
            # Fix value range at [0,1] to maintain consistency
            ax = sns.heatmap(alpha_pivot, annot=True, fmt='.3f', cmap=cmap, mask=mask,
                             vmin=0, vmax=1, linewidths=0.5, square=False)
            ax.set_title("Krippendorff's Alpha", fontsize=16, pad=20)
            ax.set_ylabel('Evaluation Dimension', fontsize=14)
            ax.set_xlabel('Model', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(rel_dir, 'krippendorff_alpha_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            # Fleiss' Kappa results Heatmap
        fleiss_df = pd.DataFrame(fleiss_results).dropna()
        if not fleiss_df.empty:
            # Create heatmap of model and dimension
            fleiss_pivot = fleiss_df.pivot(index='dimension', columns='model', values='kappa')
            # Arrange dimensions in specified order
            dimension_order = [
                'multi_hop_reasoning', 'fluency', 'clarity', 'conciseness', 'relevance', 
                'consistency', 'question_answerability', 'answer_question_consistency', 
                'information_integration_ability', 'reasoning_path_guidance',
                'logical_sophistication', 'overall_quality'
            ]
            # Filter actual existing dimensions and maintain original order
            available_dims = set(fleiss_pivot.index.tolist())
            new_order = [d for d in dimension_order if d in available_dims]
            fleiss_pivot = fleiss_pivot.reindex(new_order)
            plt.figure(figsize=(14, 10))
            mask = fleiss_pivot.isna()
            cmap = sns.color_palette("RdYlGn", as_cmap=True)  # Use red-yellow-green gradient color
            # Fix value range at [-1,1] to maintain consistency (Fleiss' Kappa range)
            ax = sns.heatmap(fleiss_pivot, annot=True, fmt='.3f', cmap=cmap, mask=mask,
                           vmin=-1, vmax=1, linewidths=0.5, square=False)
            ax.set_title("Fleiss' Kappa", fontsize=16, pad=20)
            ax.set_ylabel('Evaluation Dimension', fontsize=14)
            ax.set_xlabel('Model', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(rel_dir, 'fleiss_kappa_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print(f"Reliability metrics visualization saved to: {rel_dir}")
        generate_visualizations(std_df, result_dir)
        summarize_model_std_dev(std_df, result_dir)
        print("Visualization and metrics calculation completed")
        return
    
    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)
    
    # Load samples for evaluation
    samples = load_samples(args.bridge, args.comparison, args.num_samples)
    
    # Run stability evaluation
    try:
        run_stability_evaluation(
            args.config,
            samples,
            args.output_dir,
            args.repeats,
            # Limit threads to 1 in test mode to ensure reliability
            1 if args.test else args.max_workers,
            args.test  # Pass test mode parameter
        )
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
    
    print("\nStability evaluation test completed")

if __name__ == "__main__":
    main()
