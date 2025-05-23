#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for training BGE Reranker model
Fine-tune BAAI/bge-reranker-v2-m3 model using FlagEmbedding framework
"""

import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Train BGE Reranker model")
    parser.add_argument("--model_path", type=str, 
                        default="./models/bge-reranker-v2-m3",
                        help="Model path")
    parser.add_argument("--train_data", type=str, 
                        default="./toy_finetune_data.jsonl",
                        help="Training data path")
    parser.add_argument("--output_dir", type=str, 
                        default="./output",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, 
                        default="./cache/model",
                        help="Model cache path")
    parser.add_argument("--cache_path", type=str, 
                        default="./cache/data",
                        help="Data cache path")
    parser.add_argument("--num_gpus", type=int, 
                        default=2,
                        help="Number of GPUs to use")
    parser.add_argument("--learning_rate", type=float, 
                        default=6e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, 
                        default=10,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, 
                        default=4,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--query_max_len", type=int, 
                        default=512,
                        help="Maximum query length")
    parser.add_argument("--passage_max_len", type=int, 
                        default=8196,
                        help="Maximum passage length")
    parser.add_argument("--train_group_size", type=int, 
                        default=4,
                        help="Training group size (1 positive sample + 3 negative samples)")
    parser.add_argument("--pad_to_multiple_of", type=int, 
                        default=8,
                        help="Pad to multiple of")
    parser.add_argument("--deepspeed_config", type=str, 
                        default="./ds_stage0.json",
                        help="DeepSpeed configuration file path")
    parser.add_argument("--warmup_ratio", type=float, 
                        default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, 
                        default=0.01,
                        help="Weight decay")
    parser.add_argument("--save_steps", type=int, 
                        default=100,
                        help="Save steps")
    parser.add_argument("--save_total_limit", type=int, 
                        default=50,
                        help="Save total limit")
    parser.add_argument("--fp16", action="store_true", 
                        default=True,
                        help="Use fp16")
    parser.add_argument("--overwrite_output_dir", action="store_true", 
                        default=True,
                        help="Overwrite output directory")
    parser.add_argument("--knowledge_distillation", type=str, 
                        default="False",
                        help="Knowledge distillation")
    parser.add_argument("--report_to", type=str, 
                        default="tensorboard",
                        help="Report training metrics in real-time, supports tensorboard")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory and cache directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    
    # Build training command
    # Control visible GPUs via CUDA_VISIBLE_DEVICES environment variable
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "-m", "FlagEmbedding.finetune.reranker.encoder_only.base",
        f"--model_name_or_path={args.model_path}",
        f"--cache_dir={args.cache_dir}",
        f"--train_data={args.train_data}",
        f"--cache_path={args.cache_path}",
        f"--train_group_size={args.train_group_size}",
        f"--query_max_len={args.query_max_len}",
        f"--passage_max_len={args.passage_max_len}",
        f"--pad_to_multiple_of={args.pad_to_multiple_of}",
        f"--knowledge_distillation={args.knowledge_distillation}",
        f"--output_dir={args.output_dir}",
        f"--report_to={args.report_to}"
    ]
    
    # Add optional parameters
    if args.overwrite_output_dir:
        cmd.append("--overwrite_output_dir")
    
    # Add other parameters
    cmd.extend([
        f"--learning_rate={args.learning_rate}",
        f"--num_train_epochs={args.num_train_epochs}",
        f"--per_device_train_batch_size={args.per_device_train_batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        "--dataloader_drop_last=True",
        "--logging_steps=1",
        f"--save_steps={args.save_steps}",
        f"--save_total_limit={args.save_total_limit}",
        "--ddp_find_unused_parameters=False",
        "--gradient_checkpointing",
        f"--weight_decay={args.weight_decay}",
        f"--deepspeed={args.deepspeed_config}",
        f"--warmup_ratio={args.warmup_ratio}"
    ])
    
    # Add fp16 or bf16 based on parameters
    if args.fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--bf16")
    
    cmd_str = " ".join(cmd)
    print(f"Executing command: {cmd_str}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Training complete!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
