"""
Comparison Question Synthesizer - Integrates multiple components to automatically generate high-quality comparison questions

This module integrates 7 key components to implement the complete process of generating comparison questions:
1. Document Random Sampling (DocumentReader)
2. Entity and Attribute Extraction (CompareEntityExtractor)
3. Entity and Attribute Filtering (CompareEntityFilter)
4. Comparison Query Generation (CompareQueryGenerator)
5. Relevant Document Retrieval (CompareRetriever)
6. Comparison Question Construction (CompareQuestionBuilder)
7. Question Refinement (ComparisonPolisher)

The generated questions will include two entities, comparison attributes, factual basis, and answers for subsequent evaluation and application.
"""

import os
import sys
import json
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.config import Config

# Import all necessary components
from hopweaver.components.utils.data_reader import DocumentReader
from hopweaver.components.compare.compare_entity_extractor import CompareEntityExtractor
from hopweaver.components.compare.compare_entity_filter import CompareEntityFilter
from hopweaver.components.compare.compare_query_generator import CompareQueryGenerator
from hopweaver.components.compare.compare_retriever import CompareRetriever
from hopweaver.components.compare.compare_question_builder import CompareQuestionBuilder
from hopweaver.components.compare.comparison_polisher import ComparisonPolisher

class ComparisonQuestionSynthesizer:
    """
    Main class for comparison question synthesis that integrates document sampling, entity extraction,
    query generation, document retrieval, and question generation.
    """
    def __init__(self, config_path, output_dir=None, enable_logging=True):
        """
        Initialize the comparison question synthesizer
        
        Args:
            config_path (str): Path to configuration file
            output_dir (str, optional): Output directory path, if not specified, read from config file or use default
            enable_logging (bool): Whether to enable logging, default is True
        """
        # Load configuration
        self.config = Config(config_path, {})
        
        # Set GPU environment variable
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["gpu_id"])
        except KeyError:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Initialize logging
        self._setup_logging(enable_logging)
        
        # Initialize components
        self.logger.info("Initializing components...")
        self.doc_reader = DocumentReader(self.config['corpus_path'])
        self.entity_extractor = CompareEntityExtractor(self.config)
        self.entity_filter = CompareEntityFilter(self.config)
        self.query_generator = CompareQueryGenerator(self.config)
        self.retriever = CompareRetriever(self.config)
        self.question_builder = CompareQuestionBuilder(self.config)
        self.polisher = ComparisonPolisher(self.config)
        
        # Output directory config - Use parameter first, then config file, then default
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            try:
                self.output_dir = self.config["output_dir"]
            except KeyError:
                self.output_dir = "./output"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize file paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.extracted_entities_file = os.path.join(self.output_dir, f"extracted_entities_{self.timestamp}.json")
        self.filtered_entities_file = os.path.join(self.output_dir, f"filtered_entities_{self.timestamp}.json")
        self.comparison_queries_file = os.path.join(self.output_dir, f"comparison_queries_{self.timestamp}.json")
        self.retrieval_results_file = os.path.join(self.output_dir, f"retrieval_results_{self.timestamp}.json")
        self.comparison_questions_file = os.path.join(self.output_dir, f"comparison_questions_{self.timestamp}.json")
        self.polished_questions_file = os.path.join(self.output_dir, f"polished_comparison_questions_{self.timestamp}.json")
    

        
    def _setup_logging(self, enable_logging=True):
        """
        Set up logging
        
        Args:
            enable_logging (bool): Whether to enable logging, default is True
        """
        self.logger = logging.getLogger("ComparisonQuestionSynthesizer")
        
        if enable_logging:
            self.logger.setLevel(logging.INFO)
            
            # Check if handlers have already been added
            if not self.logger.handlers:
                # Create console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                
                # Create formatter
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                
                # Add handler to logger
                self.logger.addHandler(console_handler)
            
            # Prevent propagation to root logger
            self.logger.propagate = False
    
    def generate_questions(self, count=10, save_intermediate=False, verbose=True):
        """
        Generate multiple comparison questions
        
        Args:
            count (int): Number of questions to generate
            save_intermediate (bool): Whether to save intermediate results
            verbose (bool): Whether to output detailed information
            
        Returns:
            list: List of generated comparison questions
        """
        all_questions = []
        successful_count = 0
        attempt_count = 0
        max_attempts = count * 5  # Set maximum attempts to avoid infinite loops
        
        self.logger.info(f"Starting to generate {count} comparison questions...")
        
        # Create jsonl output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_output_file = os.path.join(self.output_dir, f"comparison_questions_{timestamp}.jsonl")
        
        while successful_count < count and attempt_count < max_attempts:
            attempt_count += 1
            self.logger.info(f"\n===== Generation attempt {attempt_count}/{max_attempts} (Success: {successful_count}/{count}) =====")
            
            try:
                question = self.generate_question(False, verbose)
                if question:
                    # Append result to jsonl file
                    with open(jsonl_output_file, 'a', encoding='utf-8') as f:
                        json.dump(question, f, ensure_ascii=False)
                        f.write('\n')
                    
                    all_questions.append(question)
                    successful_count += 1
                    success_rate = (successful_count/attempt_count)*100 if attempt_count > 0 else 0
                    self.logger.info(f"Success rate: {successful_count}/{attempt_count} ({success_rate:.1f}%)")
                else:
                    success_rate = (successful_count/attempt_count)*100 if attempt_count > 0 else 0
                    self.logger.warning(f"Generation failed, will retry (Success rate: {successful_count}/{attempt_count} ({success_rate:.1f}%))")
            except Exception as e:
                self.logger.error(f"Error during question generation: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"\nBatch generation completed! Generated {successful_count} comparison questions, attempt count: {attempt_count}")
        
        if successful_count > 0:
            self.logger.info(f"All results have been saved to JSONL file: {jsonl_output_file}")
            
            # Also save as JSON format (for compatibility with existing code)
            final_output_path = os.path.join(self.output_dir, "all_comparison_questions.json")
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump({"comparison_questions": all_questions}, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"All comparison questions have also been saved to JSON file: {final_output_path}")
        
        return all_questions
    
    def generate_question(self, save_intermediate=False, verbose=True):
        """
        Complete process for generating a comparison question
        
        Args:
            save_intermediate (bool): Whether to save intermediate results
            verbose (bool): Whether to output detailed information
            
        Returns:
            dict: Dictionary containing generated comparison question, answer and related information
        """
        try:
            # Step 1: Randomly get a document
            self.logger.info("Step 1: Random sampling of a document")
            source_doc = self.doc_reader.get_heuristic_documents(count=1, min_length=300)[0]
            if not source_doc:
                self.logger.error("Failed to get valid document")
                return None
            
            doc_id = source_doc.get('id', 'unknown')
            doc_title = source_doc.get('title', 'unknown')
            self.logger.info(f"Document obtained: ID={doc_id}, Title={doc_title}")
            
            if verbose:
                content_preview = source_doc.get('contents', '')[:300] + "..." if len(source_doc.get('contents', '')) > 300 else source_doc.get('contents', '')
                self.logger.info(f"Document content preview: {content_preview}")
            
            # Step 2: Extract entities and attributes from the document
            self.logger.info("Step 2: Extracting entities and attributes from the document")
            entity_data = self.entity_extractor.extract_entities(source_doc)
            
            if not entity_data or "subject_entity" not in entity_data:
                self.logger.error("Failed to extract valid entities from the document")
                return None
            
            # Add document information to entity data
            entity_data["document_id"] = doc_id
            entity_data["contents"] = source_doc.get('contents', '')
            entity_data["title"] = doc_title
            
            # Save extracted entity data
            if save_intermediate:
                with open(self.extracted_entities_file, "w", encoding="utf-8") as f:
                    json.dump(entity_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Extracted entity data saved to: {self.extracted_entities_file}")
            
            subject_entity = entity_data.get('subject_entity', {})
            subject_entity_name = subject_entity.get('name', '')
            subject_entity_type = subject_entity.get('type', '')
            
            self.logger.info(f"Extracted subject entity: {subject_entity_name} (type: {subject_entity_type})")
            
            # Step 3: Filter entities and attributes
            self.logger.info("Step 3: Evaluating comparability of entities and attributes")
            filtered_entity = self.entity_filter.filter_entity(entity_data)
            
            if not filtered_entity or "entity_score" not in filtered_entity:
                self.logger.error("Entity filtering failed")
                return None
            
            # Use newly implemented method to filter high-scoring entities and attributes
            filtered_entity_data = self.entity_filter.filter_high_score_entities_and_attributes(
                filtered_entity, entity_data, entity_threshold=5, attribute_threshold=5
            )
            
            if not filtered_entity_data or "attributes" not in filtered_entity_data or not filtered_entity_data["attributes"]:
                self.logger.error("No comparable attributes with high enough scores found")
                return None
            
            # Save filtered entity data
            if save_intermediate:
                with open(self.filtered_entities_file, "w", encoding="utf-8") as f:
                    json.dump(filtered_entity_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Filtered entity data saved to: {self.filtered_entities_file}")
            
            # Step 4: Generate comparison query
            self.logger.info("Step 4: Generating comparison query for filtered entity")
            comparison_queries = self.query_generator.generate_query(filtered_entity_data)
            
            if not comparison_queries:
                self.logger.error("Failed to generate comparison query")
                return None
            
            # Ensure entity data has contents field
            if "query_results" in comparison_queries:
                for query_result in comparison_queries["query_results"]:
                    entity_data = query_result.get("entity_data", {})
                    
                    # Ensure entity_data has contents field
                    if "contents" not in entity_data and "contents" in filtered_entity_data:
                        # Add content from filtered_entity_data if missing
                        entity_data["contents"] = filtered_entity_data.get("contents", "")
                        self.logger.info("Added contents field to entity data")
            
            # Save generated queries
            if save_intermediate:
                with open(self.comparison_queries_file, "w", encoding="utf-8") as f:
                    json.dump(comparison_queries, f, ensure_ascii=False, indent=2)
                self.logger.info("Generated comparison queries saved to: {}".format(self.comparison_queries_file))
            
            # Step 5: Execute retrieval
            self.logger.info("Step 5: Retrieving relevant documents based on generated queries")
            
            # Preprocess comparison queries - create documents array to ensure original document content is available
            if "documents" not in comparison_queries:
                comparison_queries["documents"] = []
                
                # Add source document to documents array
                if "contents" in source_doc:
                    document_entry = {
                        "id": source_doc.get("id", ""),
                        "title": source_doc.get("title", ""),
                        "contents": source_doc.get("contents", "")
                    }
                    comparison_queries["documents"].append(document_entry)
                    self.logger.info("Added source document {} to documents array".format(document_entry['id']))
            
            # Process based on query type
            retrieval_results = {"results": []}
            query_type = comparison_queries.get("query_type")

            # Choose appropriate processing method
            if "query_results" in comparison_queries:
                # Use modified retriever processing logic
                result = {"retrieval_results": []}
                temp_file = os.path.join(self.output_dir, "temp_comparison_queries.json")
                temp_result_file = os.path.join(self.output_dir, "temp_retrieval_results.json")
                
                # Save temporary file for retriever processing
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(comparison_queries, f, ensure_ascii=False, indent=2)
                
                # Use retriever to process file
                self.retriever.process_query_results(temp_file, temp_result_file)
                
                # Read and parse retrieval results from temporary file
                try:
                    with open(temp_result_file, "r", encoding="utf-8") as f:
                        retrieval_results = json.load(f)
                        self.logger.info("Successfully used retriever to process file and retrieve results")
                except Exception as e:
                    self.logger.error("Failed to process retrieval results file: {}".format(str(e)))
                    return None
                    
                # Clean up temporary files
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    if os.path.exists(temp_result_file):
                        os.remove(temp_result_file)
                except Exception as e:
                    self.logger.warning("Failed to clean up temporary files: {}".format(str(e)))
            else:
                # Use original retrieval logic
                if query_type == "recall_focused_verify":
                    # Process direct entity recommendation
                    result = self.retriever.process_recall_focused_verify(comparison_queries)
                    if result:
                        retrieval_results["results"].append(result)
                elif query_type == "search_queries":
                    # Process search queries
                    result = self.retriever.process_search_queries(comparison_queries)
                    if result:
                        retrieval_results["results"].append(result)
                else:
                    self.logger.error("Unknown query type: {}".format(query_type))
                    return None
            
            if not retrieval_results or not ("retrieval_results" in retrieval_results or "results" in retrieval_results):
                self.logger.error("Retrieval failed or no retrieval results")
                return None
                
            # Standardize result format
            if "retrieval_results" in retrieval_results and "results" not in retrieval_results:
                retrieval_results["results"] = retrieval_results["retrieval_results"]
                del retrieval_results["retrieval_results"]
            
            # Save retrieval results
            if save_intermediate:
                with open(self.retrieval_results_file, "w", encoding="utf-8") as f:
                    json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
                self.logger.info("Retrieval results saved to: {}".format(self.retrieval_results_file))
            
            # Step 6: Build comparison question
            self.logger.info("Step 6: Building comparison question based on retrieval results")
            
            # Get document B list from retrieval results and attempt to build question
            comparison_questions = []
            retrieval_list = retrieval_results.get("results", [])
            
            for result_index, result_data in enumerate(retrieval_list):
                # Get retrieved documents list
                retrieved_docs = result_data.get("retrieved_documents", [])
                
                # Iterate through each retrieved document
                for doc_index, doc_b_data in enumerate(retrieved_docs):
                    self.logger.info("Processing retrieval result {}/{} , document {}/{}".format(result_index+1, len(retrieval_list), doc_index+1, len(retrieved_docs)))
                    
                    # Get document ID and content
                    id_field = "doc_id" if "doc_id" in doc_b_data else "id" 
                    doc_b_id = doc_b_data.get(id_field, "")
                    
                    # Different retrievers may use different field names
                    content_field = None
                    for field in ["contents", "content", "text"]:
                        if field in doc_b_data and doc_b_data.get(field):
                            content_field = field
                            break
                    
                    doc_b_content = doc_b_data.get(content_field, "") if content_field else ""
                    doc_b_title = doc_b_data.get("title", "")
                
                    if not doc_b_content:
                        self.logger.warning("Document {} content is empty, skipping".format(doc_b_id))
                        continue
                    
                    # Build comparison question using entity and document data
                    question_data = self.question_builder.build_question(
                        filtered_entity_data,
                        {
                            "id": doc_b_id,
                            "title": doc_b_title,
                            "contents": doc_b_content
                        }
                    )
                    
                    if question_data and question_data.get("success", False):
                        comparison_questions.append(question_data)
                        self.logger.info("Successfully built comparison question based on document {}".format(doc_b_id))
                        # Only need one successful question
                        break
            
            if not comparison_questions:
                self.logger.error("Failed to build any valid comparison questions")
                return None
            
            # Save built comparison questions
            if save_intermediate:
                with open(self.comparison_questions_file, "w", encoding="utf-8") as f:
                    json.dump({"comparison_questions": comparison_questions}, f, ensure_ascii=False, indent=2)
                self.logger.info("Built comparison questions saved to: {}".format(self.comparison_questions_file))
            
            # Step 7: Polish comparison question
            self.logger.info("Step 7: Polishing comparison question")
            
            # Directly pass comparison question data to polisher
            questions_data = {"comparison_questions": comparison_questions}
            
            # Decide whether to pass output file path based on save_intermediate parameter
            if save_intermediate:
                polished_questions = self.polisher.polish_questions_from_data(
                    questions_data, 
                    self.polished_questions_file, 
                    save_output=True
                )
            else:
                # When not saving intermediate results, do not pass output file path
                polished_questions = self.polisher.polish_questions_from_data(
                    questions_data, 
                    output_file=None, 
                    save_output=False
                )
            
            if not polished_questions or not polished_questions.get("polished_questions", []):
                self.logger.error("Failed to polish comparison question")
                return None
            
            # Get polished question
            polished_question = polished_questions.get("polished_questions", [])[0]
            
            # Combine original question and polished results to form final output
            final_question = comparison_questions[0].copy()  # Use basic information from the original question
            
            # Add polishing results
            if "polished_result" in polished_question:
                final_question["polished_result"] = polished_question["polished_result"]
            
            self.logger.info("Comparison question generation and polishing successful!")
            return final_question
            
        except Exception as e:
            self.logger.error(f"Error occurred during question generation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Comparison question generation tool")
    parser.add_argument("-c", "--config", default="./config_lib/example_config.yaml", help="Configuration file path")
    parser.add_argument("-n", "--num", type=int, default=30, help="Number of questions to generate")
    parser.add_argument("-o", "--output", default="./output_compare", help="Output directory")
    
    # Compatible with the original command line parameter format
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Old format: directly specify the number as the first parameter
        try:
            num_questions = int(sys.argv[1])
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output_compare"
            config_path = "./config_lib/example_config.yaml"
        except ValueError:
            args = parser.parse_args()
            num_questions = args.num
            output_dir = args.output
            config_path = args.config
    else:
        # New format: use argument parser
        args = parser.parse_args()
        num_questions = args.num
        output_dir = args.output
        config_path = args.config
    
    print(f"Using configuration file: {config_path}")
    print(f"Generating {num_questions} comparison questions")
    print(f"Output directory: {output_dir}")
    
    # Create comparison question synthesizer - directly pass output directory
    synthesizer = ComparisonQuestionSynthesizer(config_path, output_dir=output_dir)
    timestamp = synthesizer.timestamp
    
    # Generate comparison questions - set not to save intermediate results
    questions = synthesizer.generate_questions(count=num_questions, save_intermediate=False)
    
    if questions:
        print(f"Successfully generated {len(questions)} comparison questions")
        
        # Save as jsonl format, one question per line, in the same format as multi-hop questions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"comparison_questions_{timestamp}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for question in questions:
                # Convert to multi-hop question format
                formatted_question = {
                    "source_doc": {
                        "id": question.get("document_a", {}).get("document_id", ""),
                        "title": question.get("document_a", {}).get("title", ""),
                        "content": question.get("fact_entity_a", "")
                    },
                    "target_doc": {
                        "id": question.get("document_b", {}).get("id", ""),
                        "title": question.get("document_b", {}).get("title", ""),
                        "content": question.get("fact_entity_b", "")
                    },
                    "comparison_question": {
                        "question": question.get("multi_hop_question", ""),
                        "answer": question.get("answer", ""),
                        "entity_a": question.get("entity_a", ""),
                        "entity_b": question.get("entity_b", ""),
                        "attribute_compared": question.get("attribute_compared", "")
                    },
                    "facts": {
                        "fact_entity_a": question.get("fact_entity_a", ""),
                        "fact_entity_b": question.get("fact_entity_b", "")
                    },
                    "relevant_paragraphs": {
                        "relevant_paragraph_a": question.get("relevant_paragraph_a", ""),
                        "relevant_paragraph_b": question.get("relevant_paragraph_b", "")
                    }
                }
                
                # If there are polishing results, add them
                if "polished_result" in question:
                    polish_result = question["polished_result"]
                    formatted_question["polish_result"] = polish_result
                    
                    # If the polishing process refined the fact, add it
                    if "refined_fact_a" in polish_result:
                        formatted_question["polish_facts"] = {
                            "refined_fact_a": polish_result.get("refined_fact_a", ""),
                            "refined_fact_b": polish_result.get("refined_fact_b", "")
                        }
                
                # Write in jsonl format, but use indentation to enhance readability
                f.write(json.dumps(formatted_question, ensure_ascii=False, indent=2) + "\n")
                
        print(f"Comparison questions saved to: {output_file}")
    else:
        print("Failed to generate any questions")
    print("Processing completed!")
