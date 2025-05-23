#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rerank Ablation Test Script
Test the success rate and average number of attempts for different retrieval methods
"""

import os
import sys
import time
import random
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Add project path
# Add parent directory to path to import flashrag
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Add current directory to path as well
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flashrag.config import Config
from hopweaver.components.utils.data_reader import DocumentReader
from hopweaver.components.bridge.bridge_entity_extractor import EntityExtractor
from hopweaver.components.bridge.bridge_question_generator import QuestionGenerator
from hopweaver.components.bridge.bridge_retriever import RetrieverWrapper, DiverseRetrieverWrapper, RerankRetrieverWrapper


class RetrievalAblationTest:
    """Rerank Ablation Test Class"""
    
    def __init__(self, config_path: str):
        """
        Initialize the tester
        
        Args:
            config_path: Configuration file path
        """
        # Load configuration
        self.config_path = config_path
        self.config = Config(config_path, {})
        
        # Set GPU environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["gpu_id"]) if "gpu_id" in self.config else "0"
        
        # Initialize document reader and entity extractor
        self.doc_reader = DocumentReader(self.config['corpus_path'])
        self.entity_extractor = EntityExtractor(self.config)
        
        # Initialize question generator to validate the usability of retrieval results
        self.question_generator = QuestionGenerator(self.config)
        
        # Output directory configuration
        self.output_dir = os.path.join(os.path.dirname(__file__), "ablation_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Retriever dictionary for caching initialized retrievers
        self.retrievers = {}
        
    def get_retriever(self, retriever_type: str):
        """Initialize and return the specified type of retriever on demand
        
        Args:
            retriever_type: Retriever type ('standard', 'diverse', 'diverse_zs', 'diverse_ft')
            
        Returns:
            tuple: (retriever_object, retriever_name)
        """
        print(f"\n[DEBUG] Initializing retriever type: {retriever_type}")
        # If already initialized, return directly from cache
        if retriever_type in self.retrievers:
            print(f"[DEBUG] Using cached retriever: {retriever_type}")
            return self.retrievers[retriever_type]
            
        # Common parameters for diverse retrieval
        lambda1, lambda2, lambda3 = 0.87, 0.03, 0.1
        
        # Initialize retriever on demand
        if retriever_type == 'standard':
            # Standard retriever (without MMR)
            retriever = RetrieverWrapper(self.config)
            retriever_name = "Standard Retriever"
        elif retriever_type == 'diverse':
            # Diverse retriever (with MMR)
            retriever = DiverseRetrieverWrapper(
                self.config, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
            )
            retriever_name = "Diverse Retriever"
        elif retriever_type == 'diverse_zs':
            # Diverse + Zero-shot Rerank Retriever (uses default zero-shot rerank model)
            retriever = RerankRetrieverWrapper(
                self.config, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
            )
            retriever_name = "Diverse + Zero-shot Rerank Retriever (Diverse+ZS Rerank)"
        elif retriever_type == 'diverse_ft':
            # Diverse + Fine-tuned Rerank Retriever (requires specifying fine-tuned rerank model path)
            ft_reranker_path = "./output_new"
            if os.path.exists(ft_reranker_path):
                print(f"[DEBUG] Using fine-tuned rerank model path: {ft_reranker_path}")
                retriever = RerankRetrieverWrapper(
                    self.config, reranker_path=ft_reranker_path,
                    lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
                )
                print(f"[DEBUG] Fine-tuned reranker instance type: {type(retriever).__name__}")
                print(f"[DEBUG] Fine-tuned reranker has_reranker: {getattr(retriever, 'has_reranker', False)}")
                retriever_name = "Diverse + Fine-tuned Rerank Retriever (Diverse+FT Rerank)"
            else:
                print(f"Warning: Fine-tuned rerank model path {ft_reranker_path} does not exist, cannot initialize fine-tuned rerank retriever")
                return None, None
        else:
            print(f"Error: Unknown retriever type {retriever_type}")
            return None, None
            
        # Cache retriever for reuse
        self.retrievers[retriever_type] = (retriever, retriever_name)
        print(f"Initialized {retriever_name}")
        
        return retriever, retriever_name
    
    def test_retriever(self, retriever_type: str, num_examples: int = 100, 
                     min_length: int = 300, max_doc_candidates: int = 10, 
                     doc_ids_file: str = None) -> Dict:
        """
        Test the success rate and average number of attempts for a specific retriever
        
        Args:
            retriever_type: Retriever type ('standard', 'diverse', 'diverse_zs', 'diverse_ft')
            num_examples: Total number of test attempts (regardless of success)
            min_length: Minimum document length
            max_doc_candidates: Number of candidate documents per retrieval
            doc_ids_file: Path to pre-sampled document ID file (optional)
            
        Returns:
            dict: Dictionary containing test results
        """
        # Get retriever on demand
        retriever_info = self.get_retriever(retriever_type)
        if retriever_info is None:
            return None
            
        retriever, retriever_name = retriever_info
        
        print(f"\nStarting test for {retriever_name} ...")
        
        # Use time-based random seed
        current_time = int(time.time())
        random.seed(current_time)
        np.random.seed(current_time)
        print(f"Using time seed: {current_time} to initialize random number generator")
        
        start_time = time.time()
        
        successful_examples = 0   # Number of successful test samples
        total_attempts = 0        # Total number of source documents attempted
        failed_attempts = 0       # Number of failed attempts
        
        # Current number of tested samples
        current_examples = 0
        
        # Record the total number of attempts for successful samples to calculate the average number of attempts
        total_attempts_for_success = 0
        
        # If using pre-sampled document IDs, load these IDs first
        source_doc_ids = []
        if doc_ids_file and os.path.exists(doc_ids_file):
            print(f"Testing with pre-sampled document IDs: {doc_ids_file}")
            source_doc_ids = self.doc_reader.load_document_ids(doc_ids_file)
            print(f"Loaded {len(source_doc_ids)} document IDs")
        
        # Execute a fixed number of test attempts
        pbar = tqdm(total=num_examples, desc=f"Testing {retriever_name} (Overall Progress)")
        
        for attempt_idx in range(1, num_examples + 1):
            try:
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
                    print(f"Attempt {attempt_idx}: Unable to get valid document, skipping")
                    continue
                
                source_doc_id = source_doc.get('id', '')
                
                # Step 2: Extract bridge entities from the document
                entities = self.entity_extractor.extract_entities(source_doc)
                
                if not entities:
                    print(f"Attempt {attempt_idx}: Failed to extract entities from document, skipping")
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
                
                # Step 3: Retrieve a batch of related documents based on retriever type
                print(f"\n[DEBUG] Retriever type: {retriever_type}, Retriever instance type: {type(retriever).__name__}")
                print(f"[DEBUG] Bridge entity query: {entity_query[:100]}...")
                print(f"[DEBUG] Source document ID: {source_doc_id}")
                
                if retriever_type == 'standard':
                    # Standard retriever uses retrieve_related_documents
                    print("[DEBUG] Using standard retrieval method: retrieve_related_documents")
                    retrieved_docs = retriever.retrieve_related_documents(
                        entity_query, doc_id=source_doc_id, top_k=max_doc_candidates
                    )
                elif retriever_type == 'diverse':
                    # Diverse retriever uses retrieve_with_diversity
                    print("[DEBUG] Using diverse retrieval method: retrieve_with_diversity")
                    retrieved_docs = retriever.retrieve_with_diversity(
                        entity_query, source_doc, top_k=max_doc_candidates, doc_id=source_doc_id
                    )
                elif retriever_type in ['diverse_zs', 'diverse_ft']:
                    # Rerank retriever uses retrieve_with_rerank
                    print(f"[DEBUG] Using rerank retrieval method: retrieve_with_rerank (Type: {retriever_type})")
                    retrieved_docs = retriever.retrieve_with_rerank(
                        entity_query, source_doc, top_k=max_doc_candidates, doc_id=source_doc_id
                    )
                else:
                    # Use retrieve_related_documents by default
                    print(f"Warning: Unknown retriever type {retriever_type}, using standard retrieval method")
                    retrieved_docs = retriever.retrieve_related_documents(
                        entity_query, doc_id=source_doc_id, top_k=max_doc_candidates
                    )
                
                print(f"[DEBUG] Number of retrieval results: {len(retrieved_docs) if retrieved_docs else 0}")
                
                if not retrieved_docs:
                    print(f"Attempt {attempt_idx}: Failed to retrieve any related documents, skipping")
                    failed_attempts += 1
                    continue
                
                # Step 4: Validate documents sequentially until one that can successfully generate questions is found
                attempts_per_success = 0  # Record the number of documents tried before finding a successful one
                success_found = False
                
                # Validate all retrieved documents
                for doc_idx, candidate_doc in enumerate(retrieved_docs):
                    # Accumulate attempt count
                    attempts_per_success += 1
                    
                    # Get candidate document content
                    doc_content = candidate_doc.get('contents', '') or candidate_doc.get('content', '')
                    doc_id = candidate_doc.get('id', 'unknown')
                    
                    print(f"\nValidating document {doc_idx+1}/{len(retrieved_docs)}: ID={doc_id}")
                    
                    # Adopt validation logic from the formal process: attempt to generate sub-questions
                    retry_count = 0
                    max_retries = 3  # Limit retry attempts
                    retry_wait = 3  # Wait for 3 seconds
                    validation_success = False
                    
                    while retry_count < max_retries and not validation_success:
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
                                print(f"  Document usable: Successfully generated sub-questions")
                                success_found = True  # Mark that a successful document was found
                            else:
                                print(f"  Document unusable: Failed to generate valid sub-questions")
                            validation_success = True  # Mark validation process as complete
                        except Exception as e:
                            retry_count += 1
                            error_msg = str(e)
                            
                            # Check if it is a quota exhaustion error
                            if "429" in error_msg or "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                                if retry_count < max_retries:
                                    print(f"  Quota exhausted, retrying after {retry_wait} seconds ({retry_count}/{max_retries})")
                                    time.sleep(retry_wait)  # Wait for 3 seconds before retrying
                                else:
                                    print(f"  Quota exhausted during document validation: Still failed after {max_retries} retries - skipping this document")
                            else:
                                # Skip document directly for other errors
                                print(f"  Error during document validation: {error_msg} - skipping this document")
                                break  # Do not retry, skip directly
                    
                    # If a successful document is found, stop validating other documents
                    if success_found:
                        print(f"  Found a usable document after trying {attempts_per_success} documents, stopping validation of other documents")
                        break
                
                # Step 5: If a document that can successfully generate questions is found, the test is considered successful
                if success_found:
                    # Update counts
                    current_examples += 1
                    successful_examples += 1
                    # Accumulate the number of attempted documents for successful cases to calculate the average
                    total_attempts_for_success += attempts_per_success
                    print(f"Successful test sample {current_examples}/{attempt_idx} (Number of documents tried: {attempts_per_success})")
                else:
                    print(f"Attempt {attempt_idx}: Tried {attempts_per_success} documents, failed to find a usable document, skipping")
                    failed_attempts += 1
                    
                # Update progress bar after each attempt, regardless of success
                pbar.update(1)
            except Exception as e:
                print(f"Error occurred during attempt {attempt_idx}: {str(e)}")
                failed_attempts += 1
        
        pbar.close()
        end_time = time.time()
        
        # Calculate the ratio of statistical results
        success_rate = successful_examples / total_attempts if total_attempts > 0 else 0
        # Calculate the average number of documents tried (only consider successful cases)
        avg_attempts_per_doc = total_attempts_for_success / successful_examples if successful_examples > 0 else 0
        
        # Summarize results
        summary = {
            "retriever_type": retriever_type,
            "retriever_name": retriever_name,
            "total_attempts": total_attempts,
            "successful_examples": successful_examples,
            "failed_attempts": failed_attempts,
            "success_rate": success_rate,
            "avg_attempts_per_success": avg_attempts_per_doc,  # Average number of documents tried per successful case
            "execution_time": end_time - start_time
        }
        
        print(f"\n{retriever_name} Test Results:")
        print(f"  Success Rate: {success_rate:.4f} ({successful_examples}/{total_attempts})")
        print(f"  Average Documents Tried: {avg_attempts_per_doc:.2f}")
        print(f"  Execution Time: {(end_time - start_time)/60:.2f} minutes")
        
        return summary
    
    def run_all_tests(self, num_examples: int = 100, min_length: int = 300, 
                    max_doc_candidates: int = 10, doc_ids_file: str = None) -> Dict:
        """
        Run tests for all retrievers
        
        Args:
            num_examples: Number of samples to test for each retriever
            min_length: Minimum document length
            max_doc_candidates: Number of candidate documents per retrieval
            doc_ids_file: Path to pre-sampled document ID file (optional)
            
        Returns:
            dict: Dictionary containing all test results
        """
        # Set the filename for saving test results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(self.output_dir, f"rerank_ablation_test_{timestamp}.json")
        
        # Run tests for the four retriever types
        results = {
            "config_path": self.config_path,
            "num_examples": num_examples,
            "min_length": min_length,
            "max_doc_candidates": max_doc_candidates,
            "doc_ids_file": doc_ids_file,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Test standard retriever
        standard_result = self.test_retriever(
            'standard', num_examples, min_length, max_doc_candidates, doc_ids_file
        )
        if standard_result:
            results["results"]["standard"] = standard_result
        
        # Test diverse retriever
        diverse_result = self.test_retriever(
            'diverse', num_examples, min_length, max_doc_candidates, doc_ids_file
        )
        if diverse_result:
            results["results"]["diverse"] = diverse_result
        
        # Test diverse + zero-shot rerank retriever
        diverse_zs_result = self.test_retriever(
            'diverse_zs', num_examples, min_length, max_doc_candidates, doc_ids_file
        )
        if diverse_zs_result:
            results["results"]["diverse_zs"] = diverse_zs_result
        
        # Test diverse + fine-tuned rerank retriever
        diverse_ft_retriever_info = self.get_retriever('diverse_ft')
        if diverse_ft_retriever_info is not None and diverse_ft_retriever_info[0] is not None:
            diverse_ft_result = self.test_retriever(
                'diverse_ft', num_examples, min_length, max_doc_candidates, doc_ids_file
            )
            if diverse_ft_result:
                results["results"]["diverse_ft"] = diverse_ft_result
        
        # Generate summary table data
        results["summary_table"] = {
            "retrievers": [],
            "success_rates": [],
            "avg_attempts": []
        }
        
        for retriever_type, result in results["results"].items():
            results["summary_table"]["retrievers"].append(result["retriever_name"])
            results["summary_table"]["success_rates"].append(result["success_rate"])
            results["summary_table"]["avg_attempts"].append(result["avg_attempts_per_success"])
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nTest results saved to: {output_path}")
        
        # Print summary table
        print("\nRerank Ablation Test Results Summary:")
        print("------------------------------------------------------")
        print("{:<25} {:<10} {:<15}".format("Retrieval Method", "Success Rate", "Avg Attempts"))
        print("------------------------------------------------------")
        for i, retriever in enumerate(results["summary_table"]["retrievers"]):
            success_rate = results["summary_table"]["success_rates"][i]
            avg_attempts = results["summary_table"]["avg_attempts"][i]
            print("{:<25} {:<10.4f} {:<15.2f}".format(retriever, success_rate, avg_attempts))
        print("------------------------------------------------------")
        
        return results


def main():
    """Main function"""
    # Argument parsing
    parser = argparse.ArgumentParser(description='Rerank Ablation Test Script')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--num', type=int, default=50, help='Number of samples to test for each retrieval method')
    parser.add_argument('--min_length', type=int, default=300, help='Minimum document length')
    parser.add_argument('--candidates', type=int, default=10, help='Number of candidate documents per retrieval')
    parser.add_argument('--doc_ids', type=str, default=None, help='Path to pre-sampled document ID file')
    parser.add_argument('--single', type=str, default=None, 
                      choices=['standard', 'diverse', 'diverse_zs', 'diverse_ft'],
                      help='Test only the specified retrieval method')
    
    args = parser.parse_args()
    
    # Create tester
    tester = RetrievalAblationTest(args.config)
    
    # Determine whether to test a single retriever or all retrievers based on arguments
    if args.single:
        print(f"Testing only {args.single} retrieval method")
        result = tester.test_retriever(
            args.single, args.num, args.min_length, args.candidates, args.doc_ids
        )
        
        # Save single result as JSON
        if result:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(tester.output_dir, f"rerank_ablation_{args.single}_{timestamp}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Test results saved to: {output_path}")
    else:
        # Test all retrievers
        tester.run_all_tests(args.num, args.min_length, args.candidates, args.doc_ids)


if __name__ == "__main__":
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # If it needs to be imported as a module, ensure the main function does not execute automatically
    main()
