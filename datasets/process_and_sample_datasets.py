#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract data from three datasets (HotpotQA, 2Wiki, MuSiQue) and preserve question type annotations.
Uses optimized algorithms to improve processing speed.
"""

import json
import os
import random
import argparse
import logging
from tqdm import tqdm
from collections import Counter

def process_hotpotqa(input_file, output_file, sample_size=None, random_seed=42, return_types=False):
    """
    Process the HotpotQA dataset, preserving question type annotations.
    The question type in HotpotQA is in the top-level 'type' field.

    Args:
    - input_file: Input file path.
    - output_file: Output file path.
    - sample_size: Number of samples to draw, None means process all data.
    - random_seed: Random seed.
    - return_types: Whether to return question type statistics.
    """
    logger.info(f"Processing HotpotQA dataset: {input_file}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"原始数据集大小: {len(data)} 条样本")
    
    # 统计问题类型分布
    type_counter = Counter()
    for item in data:
        type_counter[item.get('type', 'unknown')] += 1
    
    logger.info("问题类型分布:")
    type_stats = []
    for q_type, count in type_counter.items():
        type_info = f"{q_type}: {count} samples ({count/len(data)*100:.2f}%)"
        logger.info(f"- {type_info}")
        type_stats.append((q_type, count, f"{count/len(data)*100:.2f}%"))
    
    # 处理每个样本
    processed_data = []
    
    for item in tqdm(data):
        # 检查必需字段
        if not all(k in item for k in ['question', 'answer', 'supporting_facts', 'context']):
            continue
        
        # Get titles and sentence indices of supporting facts
        supporting_facts = item['supporting_facts']
        
        # Build a set of supporting fact titles
        supporting_titles = set([fact[0] for fact in supporting_facts])
        
        # Exactly match documents from the context
        context_dict = {ctx[0]: ctx[1] for ctx in item['context']}
        matched_docs = []
        
        # Exactly match each supporting fact title
        for title in supporting_titles:
            if title in context_dict:
                # Found exact match
                doc_content = context_dict[title]
                
                # Merge all sentences in the document
                full_text = " ".join(doc_content)
                
                # Add to the list of matched documents
                matched_docs.append((title, full_text))
        
        # 构建输出项
        output_item = {
            'id': item.get('_id', ''),
            'question': item['question'],
            'answer': item['answer'],
            'question_type': item.get('type', ''),  # Preserve question type annotation
            'level': 1 if item.get('type') == 'comparison' else 2  # Simple mapping for level
        }
        
        # 添加所有文档
        for i, (title, doc) in enumerate(matched_docs):
            output_item[f'document{i+1}'] = f"{title}: {doc}"
        
        # 只保留有至少2个文档的样本
        if len(matched_docs) >= 2:
            processed_data.append(output_item)
    
    logger.info(f"Processing complete, {len(processed_data)} data items in total")
    
    # 如果指定了采样大小，进行随机采样
    if sample_size is not None and sample_size < len(processed_data):
        processed_data = random.sample(processed_data, sample_size)
        logger.info(f"Randomly sampled {sample_size} data items")
    
    # 保存到输出文件
    if output_file is not None:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {output_file}")
    
    if return_types:
        return len(processed_data), type_counter
    else:
        return len(processed_data)

def process_2wiki(input_file, output_file, sample_size=None, random_seed=42, return_types=False):
    """
    Process the 2Wiki dataset, preserving question type annotations.
    Uses optimized algorithms to improve processing speed.

    Args:
    - input_file: Input file path.
    - output_file: Output file path.
    - sample_size: Number of samples to draw, None means process all data.
    - random_seed: Random seed.
    - return_types: Whether to return question type statistics.
    """
    logger.info(f"Processing 2Wiki dataset: {input_file}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 读取JSONL格式数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"原始数据集大小: {len(data)} 条样本")
    
    # 统计问题类型分布
    type_counter = Counter()
    for item in data:
        if 'metadata' in item and 'type' in item['metadata']:
            type_counter[item['metadata']['type']] += 1
        else:
            type_counter['unknown'] += 1
    
    logger.info("问题类型分布:")
    type_stats = []
    for q_type, count in type_counter.items():
        type_info = f"{q_type}: {count} samples ({count/len(data)*100:.2f}%)"
        logger.info(f"- {type_info}")
        type_stats.append((q_type, count, f"{count/len(data)*100:.2f}%"))
    
    # Check the processed 2Wiki dataset
    processed_file = "./data_defaults/dataset_mhqa/2wiki_data.json"  # 使用相对路径替换绝对路径
    
    if not os.path.exists(processed_file):
        logger.error(f"处理后的数据文件 {processed_file} 不存在")
        return 0
    
    # 加载处理后的数据并创建索引
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data_list = json.load(f)
    
    # 创建ID和问题到处理后数据的映射(哈希表)，用于快速查找
    proc_map = {}
    for proc_item in processed_data_list:
        item_id = proc_item.get('id', '')
        question = proc_item.get('question', '')
        key = f"{item_id}_{question}"
        proc_map[key] = proc_item
    
    logger.info(f"处理后的数据集大小: {len(processed_data_list)} 条样本")
    logger.info(f"创建了 {len(proc_map)} 个索引用于快速查找")
    
    # 处理每个样本
    processed_data = []
    
    for item in tqdm(data):
        # 检查必需字段
        if not all(k in item for k in ['question', 'golden_answers', 'metadata']):
            continue
        
        # 获取答案
        answer = None
        if 'golden_answers' in item and item['golden_answers']:
            answer = item['golden_answers'][0]
        
        if not answer:
            continue
        
        # Get question type
        question_type = ''
        level = 0
        if 'metadata' in item and 'type' in item['metadata']:
            question_type = item['metadata']['type']
            
            # Determine level based on type
            if question_type == 'comparison':
                level = 1
            elif question_type == 'bridge_comparison':
                level = 2
            elif question_type == 'compositional':
                level = 2
            elif question_type == 'inference':
                level = 3
            else:
                level = 2
        
        # 在哈希表中查找对应的处理后数据
        item_id = item.get('id', '')
        question = item.get('question', '')
        key = f"{item_id}_{question}"
        
        if key in proc_map:
            proc_item = proc_map[key]
            
            # 构建输出项
            output_item = {
                'id': item_id,
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'level': level
            }
            
            # 添加所有文档
            doc_count = 0
            for i in range(1, 10):  # 假设最多有10个文档
                doc_key = f'document{i}'
                if doc_key in proc_item:
                    output_item[doc_key] = proc_item[doc_key]
                    doc_count += 1
            
            # 只保留有至少2个文档的样本
            if doc_count >= 2:
                processed_data.append(output_item)
    
    logger.info(f"Processing complete, {len(processed_data)} data items in total")
    
    # 如果指定了采样大小，进行随机采样
    if sample_size is not None and sample_size < len(processed_data):
        processed_data = random.sample(processed_data, sample_size)
        logger.info(f"Randomly sampled {sample_size} data items")
    
    # 保存到输出文件
    if output_file is not None:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {output_file}")
    
    if return_types:
        return len(processed_data), type_counter
    else:
        return len(processed_data)

def process_musique(input_file, processed_musique_path, output_file, sample_size=None, random_seed=42, return_types=False):
    """
    Process the MuSiQue dataset, adding reasoning hop annotations.
    Uses optimized algorithms to improve processing speed.

    Args:
    - input_file: Input file path.
    - output_file: Output file path.
    - sample_size: Number of samples to draw, None means process all data.
    - random_seed: Random seed.
    - return_types: Whether to return question type statistics.
    """
    logger.info(f"Processing MuSiQue dataset: {input_file}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 读取JSONL格式数据
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"原始数据集大小: {len(data)} 条样本")
    
    # Calculate distribution of question_decomposition steps
    steps_counter = Counter()
    for item in data:
        steps = []
        if 'metadata' in item and 'question_decomposition' in item['metadata']:
            steps = item['metadata']['question_decomposition']
            steps_count = len(steps) if steps else 0
            steps_counter[steps_count] += 1
        else:
            steps_counter[0] += 1
    
    # Convert step count to question type
    type_counter = Counter()
    for steps_count, count in steps_counter.items():
        question_type = f"musique-{steps_count}-steps" if steps_count > 0 else "unknown"
        type_counter[question_type] = count
    
    logger.info("Question type distribution (based on step count):")
    type_stats = []
    for q_type, count in type_counter.items():
        type_info = f"{q_type}: {count} samples ({count/len(data)*100:.2f}%)"
        logger.info(f"- {type_info}")
        type_stats.append((q_type, count, f"{count/len(data)*100:.2f}%"))
    
    # Use the provided path for the processed MuSiQue dataset
    processed_file = processed_musique_path
    
    if not os.path.exists(processed_file):
        logger.error(f"处理后的数据文件 {processed_file} 不存在")
        return 0
    
    # 加载处理后的数据并创建索引
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data_list = json.load(f)
    
    # 创建ID和问题到处理后数据的映射(哈希表)，用于快速查找
    proc_map = {}
    for proc_item in processed_data_list:
        item_id = proc_item.get('id', '')
        question = proc_item.get('question', '')
        key = f"{item_id}_{question}"
        proc_map[key] = proc_item
    
    logger.info(f"处理后的数据集大小: {len(processed_data_list)} 条样本")
    logger.info(f"创建了 {len(proc_map)} 个索引用于快速查找")
    
    # 处理每个样本
    processed_data = []
    
    for item in tqdm(data):
        # 检查必需字段
        if not all(k in item for k in ['question', 'golden_answers', 'metadata']):
            continue
        
        # 获取答案
        answer = None
        if 'golden_answers' in item and item['golden_answers']:
            answer = item['golden_answers'][0]
        
        if not answer:
            continue
        
        # Get the number of question decomposition steps as a complexity indicator
        level = 0
        steps = []
        if 'metadata' in item and 'question_decomposition' in item['metadata']:
            steps = item['metadata']['question_decomposition']
            level = len(steps) if steps else 0
        
        question_type = f"musique-{level}-steps" if level > 0 else "unknown"
        
        # 在哈希表中查找对应的处理后数据
        item_id = item.get('id', '')
        question = item.get('question', '')
        key = f"{item_id}_{question}"
        
        if key in proc_map:
            proc_item = proc_map[key]
            
            # 构建输出项
            output_item = {
                'id': item_id,
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'level': level
            }
            
            # 添加所有文档
            doc_count = 0
            for i in range(1, 10):  # 假设最多有10个文档
                doc_key = f'document{i}'
                if doc_key in proc_item:
                    output_item[doc_key] = proc_item[doc_key]
                    doc_count += 1
            
            # 只保留有至少2个文档的样本
            if doc_count >= 2:
                processed_data.append(output_item)
    
    logger.info(f"Processing complete, {len(processed_data)} data items in total")
    
    # 如果指定了采样大小，进行随机采样
    if sample_size is not None and sample_size < len(processed_data):
        processed_data = random.sample(processed_data, sample_size)
        logger.info(f"Randomly sampled {sample_size} data items")
    
    # 保存到输出文件
    if output_file is not None:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {output_file}")
    
    if return_types:
        return len(processed_data), type_counter
    else:
        return len(processed_data)

def extract_by_type(data, types, sample_size, random_seed=42):
    """
    Extract samples from data based on type.

    Args:
    - data: List of data items.
    - types: List of types to extract.
    - sample_size: Number of samples to draw.
    - random_seed: Random seed.

    Returns:
    - Sampled data matching the specified types.
    """
    random.seed(random_seed)
    filtered_data = [item for item in data if item.get('question_type') in types]
    logger.info(f"Number of samples matching types {types}: {len(filtered_data)}")
    
    if sample_size is not None and sample_size < len(filtered_data):
        return random.sample(filtered_data, sample_size)
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description='Extract data from three datasets and classify by type')
    parser.add_argument('--hotpotqa', type=str, default='./data_defaults/hotpotqa/hotpot_dev_distractor_v1.json', help='Path to the HotpotQA dataset (default: ./data_defaults/hotpotqa/hotpot_dev_distractor_v1.json)')
    parser.add_argument('--twowiki', type=str, default='./data_defaults/2wiki/dev.jsonl', help='Path to the 2Wiki dataset (default: ./data_defaults/2wiki/dev.jsonl)')
    parser.add_argument('--musique', type=str, default='./data_defaults/musique/dev.jsonl', help='Path to the MuSiQue dataset (default: ./data_defaults/musique/dev.jsonl)')
    parser.add_argument('--output_dir', type=str, default='./processed_datasets', help='Output directory (default: ./processed_datasets)')
    parser.add_argument('--sample_size', type=int, default=50, help='Number of samples to draw from each dataset, None means no sampling')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_separated_datasets', action='store_true', help='Use datasets from the directory specified by --separated_datasets_dir')
    parser.add_argument('--separated_datasets_dir', type=str, default='./hopweaver_dataset_files', help='Directory containing separated dataset files (default: ./hopweaver_dataset_files)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--only_analyze_types', action='store_true', help='Only analyze question types, do not generate sample data')
    parser.add_argument('--processed_musique_file', type=str, default='./data_defaults/dataset_mhqa/musique_data.json', help='Path to the pre-processed MuSiQue data file (musique_data.json) needed by process_musique (default: ./data_defaults/dataset_mhqa/musique_data.json)')
    args = parser.parse_args()
    
    # Configure logging based on argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if args.use_separated_datasets:
        logger.info(f"Using separated datasets from {args.separated_datasets_dir}/")
        args.hotpotqa = os.path.join(args.separated_datasets_dir, 'hotpot_dev_distractor_v1.json')
        args.twowiki = os.path.join(args.separated_datasets_dir, 'dev_2wiki.jsonl')
        args.musique = os.path.join(args.separated_datasets_dir, 'dev_musique.jsonl')
        # Verify these files exist
        for dataset_path_attr in ['hotpotqa', 'twowiki', 'musique']:
            path_to_check = getattr(args, dataset_path_attr)
            if not os.path.exists(path_to_check):
                logger.error(f"Error: Separated dataset file not found: {path_to_check}")
                logger.error(f"Please ensure that {args.separated_datasets_dir} contains the required dataset files.")
                return # Or raise an exception
    else:
        logger.info("Using dataset paths specified in arguments or script defaults.")

    logger.setLevel(getattr(logging, args.log_level))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the three datasets
    if args.only_analyze_types:
        # Only analyze question types
        _, hotpotqa_types = process_hotpotqa(
            args.hotpotqa,
            os.path.join(args.output_dir, 'hotpotqa_typed_sample.json'),
            None,  # Do not sample
            args.random_seed,
            return_types=True
        )
        
        _, twowiki_types = process_2wiki(
            args.twowiki,
            os.path.join(args.output_dir, '2wiki_typed_sample.json'),
            None,  # Do not sample
            args.random_seed,
            return_types=True
        )
        
        _, musique_types = process_musique(
            args.musique,
            args.processed_musique_file,
            os.path.join(args.output_dir, 'musique_typed_sample.json'),
            None,  # Do not sample
            args.random_seed,
            return_types=True
        )
        
        # Detailed output of type statistics for each dataset
        logger.info("\n==== Detailed Question Type Statistics ====\n")
        
        logger.info("HotpotQA Question Types:")
        for q_type, count in sorted(hotpotqa_types.items()):
            logger.info(f"- {q_type}: {count} samples")
        
        logger.info("\n2Wiki Question Types:")
        for q_type, count in sorted(twowiki_types.items()):
            logger.info(f"- {q_type}: {count} samples")
        
        logger.info("\nMuSiQue Question Types:")
        for q_type, count in sorted(musique_types.items()):
            logger.info(f"- {q_type}: {count} samples")
        
        logger.info("\nSummary of Question Types Across All Datasets:")
        all_types = set(list(hotpotqa_types.keys()) + list(twowiki_types.keys()) + list(musique_types.keys()))
        logger.info(f"Total of {len(all_types)} different question types:")
        for type_name in sorted(all_types):
            logger.info(f"- {type_name}")
        
        return
    
    logger.info("\n==== Extract Samples by Type ====\n")
    
    # Initialize random seed
    random.seed(args.random_seed)
    
    # 1. Process HotpotQA dataset
    logger.info("\nProcessing HotpotQA dataset...")
    # First process all data, without sampling
    _, hotpotqa_all = process_hotpotqa(
        args.hotpotqa,
        None,  # Do not save temporarily
        None,  # Do not sample
        args.random_seed,
        return_types=True
    )
    
    # Read processed data from file
    with open('./data_defaults/processed_samples/hotpotqa_typed_sample.json', 'r', encoding='utf-8') as f:
        hotpotqa_data = json.load(f)
    
    # Group by type
    hotpotqa_bridge = extract_by_type(hotpotqa_data, ['bridge'], args.sample_size, args.random_seed)
    hotpotqa_comparison = extract_by_type(hotpotqa_data, ['comparison'], args.sample_size, args.random_seed)
    
    # Save data classified by type
    with open(os.path.join(args.output_dir, 'hotpotqa_bridge.json'), 'w', encoding='utf-8') as f:
        json.dump(hotpotqa_bridge, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(hotpotqa_bridge)} HotpotQA-bridge data items to {args.output_dir}/hotpotqa_bridge.json")
    
    with open(os.path.join(args.output_dir, 'hotpotqa_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(hotpotqa_comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(hotpotqa_comparison)} HotpotQA-comparison data items to {args.output_dir}/hotpotqa_comparison.json")
    
    # 2. Process 2Wiki dataset
    logger.info("\nProcessing 2Wiki dataset...")
    # First process all data, without sampling
    _, twowiki_all = process_2wiki(
        args.twowiki,
        None,  # Do not save temporarily
        None,  # Do not sample
        args.random_seed,
        return_types=True
    )
    
    # Read processed data from file
    with open('./data_defaults/processed_samples/2wiki_typed_sample.json', 'r', encoding='utf-8') as f:
        twowiki_data = json.load(f)
    
    # Group by type
    twowiki_bridge = extract_by_type(twowiki_data, ['compositional'], args.sample_size, args.random_seed)
    # Only include original 'comparison' type, not 'bridge_comparison'
    twowiki_comparison = extract_by_type(twowiki_data, ['comparison'], args.sample_size, args.random_seed)
    # Add a new 'bridge_comparison' type (Note: comment mentions 4wiki, but code uses twowiki_data)
    twowiki_bridge_comparison = extract_by_type(twowiki_data, ['bridge_comparison'], args.sample_size, args.random_seed)
    
    # Save data classified by type
    with open(os.path.join(args.output_dir, '2wiki_bridge.json'), 'w', encoding='utf-8') as f:
        json.dump(twowiki_bridge, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(twowiki_bridge)} 2Wiki-bridge data items to {args.output_dir}/2wiki_bridge.json")
    
    with open(os.path.join(args.output_dir, '2wiki_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(twowiki_comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(twowiki_comparison)} 2Wiki-comparison data items to {args.output_dir}/2wiki_comparison.json")
    
    with open(os.path.join(args.output_dir, '2wiki_bridge_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(twowiki_bridge_comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(twowiki_bridge_comparison)} 2Wiki-bridge_comparison data items to {args.output_dir}/2wiki_bridge_comparison.json")
    
    # 3. Process MuSiQue dataset
    logger.info("\nProcessing MuSiQue dataset...")
    # First process all data, without sampling
    _, musique_all = process_musique(
        args.musique,
        args.processed_musique_file,
        None,  # Do not save temporarily
        None,  # Do not sample
        args.random_seed,
        return_types=True
    )
    
    # Read processed data from file
    with open('./data_defaults/processed_samples/musique_typed_sample.json', 'r', encoding='utf-8') as f:
        musique_data = json.load(f)
    
    # All samples from MuSiQue dataset are treated as 'bridge' type (based on steps)
    musique_bridge_types = ['musique-2-steps', 'musique-3-steps', 'musique-4-steps']
    musique_bridge = extract_by_type(musique_data, musique_bridge_types, args.sample_size, args.random_seed)
    
    # Save data classified by type
    with open(os.path.join(args.output_dir, 'musique_bridge.json'), 'w', encoding='utf-8') as f:
        json.dump(musique_bridge, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(musique_bridge)} MuSiQue-bridge data items to {args.output_dir}/musique_bridge.json")
    
    # Output summary
    logger.info("\n==== Summary of Results by Type Extraction ====\n")
    logger.info(f"HotpotQA-bridge: {len(hotpotqa_bridge)} samples")
    logger.info(f"HotpotQA-comparison: {len(hotpotqa_comparison)} samples")
    logger.info(f"2Wiki-bridge: {len(twowiki_bridge)} samples")
    logger.info(f"2Wiki-comparison: {len(twowiki_comparison)} samples")
    logger.info(f"2Wiki-bridge_comparison: {len(twowiki_bridge_comparison)} samples")
    logger.info(f"MuSiQue-bridge: {len(musique_bridge)} samples")
    
    logger.info("\nAll datasets processed.")

if __name__ == "__main__":
    main()
