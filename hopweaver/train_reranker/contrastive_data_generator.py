#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contrastive Learning Training Data Generator
This script is used to generate training data for contrastive learning, including queries, positive samples, and negative samples.
Features: Uses diverse retrieval to ensure that each query has at least one positive sample and one negative sample.
"""

import sys
import os
import json
import time
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

# Add project path
# Add parent directory to path to import flashrag
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Add current directory to path as well
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flashrag.config import Config
from components.utils.data_reader import DocumentReader
from components.bridge.bridge_entity_extractor import EntityExtractor
from components.bridge.bridge_retriever import DiverseRetrieverWrapper
from components.bridge.bridge_question_generator import QuestionGenerator


class ContrastiveDataGenerator:
    """Contrastive learning training data generator class"""
    
    def __init__(self, config_path: str, lambda1: float = 0.7, lambda2: float = 0.1, lambda3: float = 0.2):
        """
        Initialize the data generator
        
        Args:
            config_path: Path to the configuration file
            lambda1, lambda2, lambda3: Weight parameters for the diverse retriever
        """
        # Load configuration and record config file path
        self.config_path = config_path
        self.config = Config(config_path, {})
        
        # Set GPU environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["gpu_id"]) if "gpu_id" in self.config else "0"
        
        # Initialize document reader and entity extractor
        self.doc_reader = DocumentReader(self.config['corpus_path'])
        self.entity_extractor = EntityExtractor(self.config)
        
        # Initialize diverse retriever
        self.retriever = DiverseRetrieverWrapper(self.config, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        print(f"Initialized DiverseRetrieverWrapper with weight parameters: lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3}")
            
        # Initialize question generator, used to validate the usability of retrieval results
        self.question_generator = QuestionGenerator(self.config)
            
        # Output directory configuration
        self.output_dir = "./data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Result collection
        self.examples = []
    
    def generate_data(self, num_examples: int = 100, min_length: int = 300, 
                     max_doc_candidates: int = 10, doc_ids_file: str = None) -> Dict:
        """
        Generate contrastive learning training data
        
        Args:
            num_examples: Number of samples to generate
            min_length: Minimum document length
            max_doc_candidates: Number of candidate documents per retrieval
            doc_ids_file: Path to the pre-sampled document IDs file (optional)
            
        Returns:
            dict: Dictionary containing the generation results
        """
        # Use time-based random seed
        current_time = int(time.time())
        random.seed(current_time)
        np.random.seed(current_time)
        print(f"Using time seed: {current_time} to initialize random number generator")
        
        start_time = time.time()
        
        successful_examples = 0   # Number of successfully generated samples
        total_attempts = 0        # Total number of source documents attempted
        failed_attempts = 0       # Number of failed attempts
        
        # Current number of generated samples
        current_examples = 0
        
        # Output file path
        output_path = os.path.join(self.output_dir, f"contrastive_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        
        # Create output file
        with open(output_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        
        # If using pre-sampled document IDs, load them first
        source_doc_ids = []
        if doc_ids_file and os.path.exists(doc_ids_file):
            print(f"Using pre-sampled document IDs for generation: {doc_ids_file}")
            source_doc_ids = self.doc_reader.load_document_ids(doc_ids_file)
            print(f"Loaded {len(source_doc_ids)} document IDs")
        
        # Continue generating until the target number of samples is reached
        attempt_idx = 0
        pbar = tqdm(total=num_examples, desc="Generating contrastive learning training data")
        
        while current_examples < num_examples:
            try:
                attempt_idx += 1
                total_attempts += 1
                
                # Step 1: Get source document
                if source_doc_ids and attempt_idx <= len(source_doc_ids):
                    # Use pre-sampled document ID
                    doc_id = source_doc_ids[attempt_idx - 1]
                    source_doc = self.doc_reader.get_document_by_id(doc_id)
                else:
                    # Get random document
                    source_doc = self.doc_reader.get_heuristic_documents(count=1, min_length=min_length)[0]
                
                if not source_doc:
                    print(f"Attempt {attempt_idx}: Could not get a valid document, skipping")
                    continue
                
                source_doc_id = source_doc.get('id', '')
                
                # Step 2: Extract bridge entities from the document
                entities = self.entity_extractor.extract_entities(source_doc)
                
                if not entities:
                    print(f"Attempt {attempt_idx}: Failed to extract entities from the document, skipping")
                    continue
                
                # Use bridge entity (usually the first one)
                bridge_entity = entities[0] if len(entities) > 0 else None
                if bridge_entity is None:
                    print(f"Attempt {attempt_idx}: Failed to get bridge entity, skipping")
                    continue
                
                # Get bridge entity related information
                entity_name = bridge_entity['name']
                entity_type = bridge_entity.get('type', '')
                entity_query = bridge_entity.get('query', entity_name)
                entity_segments = ' '.join(bridge_entity.get('segments', []))
                
                if not entity_name or not entity_query:
                    print(f"Attempt {attempt_idx}: Entity information incomplete, skipping")
                    continue
                
                # Step 3: Retrieve a batch of related documents
                retrieved_docs = self.retriever.retrieve_with_diversity(
                    entity_query, source_doc, top_k=max_doc_candidates, doc_id=source_doc_id
                )
                
                if not retrieved_docs:
                    print(f"Attempt {attempt_idx}: Failed to retrieve any related documents, skipping")
                    failed_attempts += 1
                    continue
                
                # Step 4: Validate documents and collect positive/negative samples
                positive_samples = []
                negative_samples = []
                
                for doc_idx, candidate_doc in enumerate(retrieved_docs):
                    # If at least one positive and one negative sample have been found, stop validation
                    if positive_samples and negative_samples:
                        print(f"Found at least one positive and one negative sample, stopping validation for remaining documents")
                        break
                        
                    # Get candidate document content
                    doc_content = candidate_doc.get('contents', '')
                    doc_id = candidate_doc.get('id', 'unknown')
                    
                    print(f"\nValidating document {doc_idx+1}/{len(retrieved_docs)}: ID={doc_id}")
                    
                    # Adopt validation logic from the formal process: attempt to generate sub-questions
                    retry_count = 0
                    max_retries = 5  # Increased to 5 retries
                    retry_wait = 3  # Wait for 3 seconds
                    success = False
                    
                    while retry_count < max_retries and not success:
                        try:
                            # Call question generator to generate sub-questions
                            sub_questions = self.question_generator.generate_sub_questions(
                                bridge_entity=entity_name,
                                entity_type=entity_type,
                                doc_a_segments=entity_segments,
                                doc_b_document=doc_content
                            )
                            
                            # If sub-questions are successfully generated, the document is a positive sample
                            if sub_questions and isinstance(sub_questions, dict) and "sub_questions" in sub_questions:
                                print(f"  Document usable: Successfully generated sub-questions - adding as positive sample")
                                positive_samples.append(doc_content)
                            else:
                                print(f"  Document unusable: Failed to generate valid sub-questions - adding as negative sample")
                                negative_samples.append(doc_content)
                            success = True  # Mark as successful
                        except Exception as e:
                            retry_count += 1
                            error_msg = str(e)
                            
                            # Check if it is a quota exhaustion error
                            if "429" in error_msg or "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                                if retry_count < max_retries:
                                    print(f"  Quota exhausted, retrying after {retry_wait} seconds ({retry_count}/{max_retries})")
                                    time.sleep(retry_wait)  # Wait for 3 seconds before retrying
                                else:
                                    print(f"  Quota exhausted during document validation: Still failed after {max_retries} retries - skipping this document directly")
                                    # Skip the document directly, do not mark as negative sample
                                    break
                            else:
                                # Other errors: skip the document directly, do not mark as negative sample
                                print(f"  Error during document validation: {error_msg} - skipping this document directly")
                                break  # Do not retry, skip directly
                
                # Step 5: Check if both positive and negative samples exist
                if positive_samples and negative_samples:
                    # Generate training sample
                    example = {
                        "query": entity_query,
                        "pos": positive_samples,
                        "neg": negative_samples
                    }
                    
                    # Add to result set
                    self.examples.append(example)
                    
                    # Save to file in real-time
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                    
                    # Update counts
                    current_examples += 1
                    successful_examples += 1
                    pbar.update(1)
                    
                    print(f"Successfully generated sample {current_examples}/{num_examples}")
                else:
                    print(f"Attempt {attempt_idx}: Failed to get both positive and negative samples, skipping")
                    if positive_samples:
                        print(f"  Found {len(positive_samples)} positive samples, but no negative samples")
                    elif negative_samples:
                        print(f"  Found {len(negative_samples)} negative samples, but no positive samples")
                    else:
                        print(f"  No usable samples found")
                    failed_attempts += 1
            except Exception as e:
                print(f"Error occurred in attempt {attempt_idx}: {str(e)}")
                failed_attempts += 1
        
        pbar.close()
        end_time = time.time()
        
        # Calculate proportions for statistical results
        success_rate = successful_examples / total_attempts if total_attempts > 0 else 0
        
        # Summarize results
        summary = {
            "total_attempts": total_attempts,
            "successful_examples": successful_examples,
            "failed_attempts": failed_attempts,
            "success_rate": success_rate,
            "execution_time": end_time - start_time,
            "output_path": output_path
        }
        
        return summary
        
    def save_summary(self, summary: Dict, output_filename: Optional[str] = None) -> str:
        """
        Save generation result summary
        
        Args:
            summary: Generation result summary
            output_filename: Output filename, auto-generated if not specified
            
        Returns:
            str: Path to save the file
        """
        # Prepare result data
        result_data = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "config_path": self.config_path,
                "lambda1": getattr(self.retriever, "lambda1", None),
                "lambda2": getattr(self.retriever, "lambda2", None),
                "lambda3": getattr(self.retriever, "lambda3", None)
            }
        }
        
        # Generate output file path
        if not output_filename:
            output_filename = f"contrastive_data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save as JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"Generation result summary saved to: {output_path}")
        return output_path


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Contrastive Learning Training Data Generator')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--max_doc_candidates', type=int, default=10, help='Number of candidate documents per retrieval')
    parser.add_argument('--lambda1', type=float, default=0.7, help='Relevance weight')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Original document difference weight')
    parser.add_argument('--lambda3', type=float, default=0.2, help='Selected document difference weight')
    parser.add_argument('--output', type=str, default=None, help='Summary output filename')
    parser.add_argument('--doc_ids_file', type=str, default=None, help='Path to pre-sampled document ID file (optional)')
    args = parser.parse_args()
    
    # Print configuration
    print(f"Starting generation of contrastive learning training data:")
    print(f"- Target number of samples: {args.num_examples}")
    print(f"- Number of candidate documents: {args.max_doc_candidates}")
    print(f"- Diversity parameters: lambda1={args.lambda1}, lambda2={args.lambda2}, lambda3={args.lambda3}")
    
    try:
        # Initialize generator
        generator = ContrastiveDataGenerator(
            config_path=args.config, 
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3
        )
        
        # Execute generation
        summary = generator.generate_data(
            num_examples=args.num_examples,
            max_doc_candidates=args.max_doc_candidates,
            doc_ids_file=args.doc_ids_file
        )
        
        # Save summary
        generator.save_summary(summary, args.output)
        
        # Print results
        print("\nData generation complete:")
        print(f"- Total attempts: {summary['total_attempts']}")
        print(f"- Successful samples: {summary['successful_examples']}")
        print(f"- Failed attempts: {summary['failed_attempts']}")
        print(f"- Success rate: {summary['success_rate']:.2%}")
        print(f"- Execution time: {summary['execution_time']:.2f} seconds")
        print(f"- Output file: {summary['output_path']}")
        
    except Exception as e:
        print(f"Error occurred during data generation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
