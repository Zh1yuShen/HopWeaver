"""
Comparison Question Optimizer - Comparison Question Polisher Module

This module is responsible for optimizing comparison questions, ensuring logical correctness, and adding necessary background information to improve question quality.
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
from hopweaver.components.utils.prompts import COMPARISON_POLISHER_PROMPT


class ComparisonPolisher:
    """
    Comparison question optimizer, used to improve the quality, logic, and background information of generated comparison questions.
    """
    
    def __init__(self, config_or_path):
        """
        Initialize the comparison question optimizer.
        
        Args:
            config_or_path: Configuration object or configuration file path
        """
        if isinstance(config_or_path, str):
            self.config = Config(config_or_path)
        else:
            self.config = config_or_path
        
        # Set model configuration
        if "polisher_model" in self.config:
            self.config["generator_model"] = self.config["polisher_model"]
        elif "generator_model" not in self.config:
            self.config["generator_model"] = "gpt-4o"
            
        # Set maximum token count
        if "generation_params" not in self.config:
            self.config["generation_params"] = {}
        self.config["generation_params"]["max_tokens"] = 4096
        
        # Initialize generator
        self.generator = self._initialize_model()
        
        # Set output directory
        self.output_dir = self.config["output_dir"] if "output_dir" in self.config else "./output"
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def _initialize_model(self):
        """Initialize OpenAI generator"""
        # Use FlashRAG's OpenaiGenerator
        return OpenaiGenerator(self.config)
    
    def polish_questions_from_data(self, questions_data, output_file=None, save_output=True):
        """
        Directly process comparison question data in memory.
        
        Args:
            questions_data: List or dictionary containing comparison questions
            output_file: Output file path (optional, auto-generated if not provided)
            save_output: Whether to save the output file, defaults to True
            
        Returns:
            dict: Dictionary containing optimized questions
        """
        # Generate file path only when output needs to be saved
        if save_output and output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"polished_comparison_questions_{timestamp}.json"
            )
        
        # Handle different input formats
        if isinstance(questions_data, dict):
            # If it's a dictionary, try to get the questions list
            questions = questions_data.get("comparison_questions", [])
        elif isinstance(questions_data, list):
            # If a list is passed directly
            questions = questions_data
        else:
            print(f"Error: Unsupported question data format: {type(questions_data)}")
            return None
        
        if not questions:
            print("Warning: No valid comparison questions found")
            return None
        
        print(f"Found {len(questions)} comparison questions to optimize")
        
        # Process each question
        polished_questions = []
        for i, question in enumerate(questions, 1):
            print(f"Optimizing question {i}/{len(questions)}")
            
            # Extract question information
            entity_a = question.get("entity_a", "")
            entity_b = question.get("entity_b", "")
            attribute = question.get("attribute_compared", "")
            original_question = question.get("multi_hop_question", "")
            original_answer = question.get("answer", "")
            fact_a = question.get("fact_entity_a", "")
            fact_b = question.get("fact_entity_b", "")
            
            # Get document context - prioritize relevant paragraphs
            # If relevant_paragraph field exists, use it directly, otherwise extract from full text
            if "relevant_paragraph_a" in question and question["relevant_paragraph_a"]:
                doc_a_context = question["relevant_paragraph_a"]
            else:
                doc_a_context = self._extract_document_context(question.get("document_a_text", ""))
                
            if "relevant_paragraph_b" in question and question["relevant_paragraph_b"]:
                doc_b_context = question["relevant_paragraph_b"]
            else:
                doc_b_context = self._extract_document_context(question.get("document_b", {}).get("contents", ""))
            
            # Optimize question
            polished_result = self._polish_question(
                entity_a, entity_b, attribute,
                original_question, original_answer,
                fact_a, fact_b,
                doc_a_context, doc_b_context
            )
            
            if polished_result:
                # Create a complete result containing the original and polished question
                complete_result = {
                    "success": True,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "attribute_compared": attribute,
                    "multi_hop_question": original_question,
                    "answer": original_answer,
                    "fact_entity_a": fact_a,
                    "fact_entity_b": fact_b,
                    # Copy other fields of the original question
                    "document_a": question.get("document_a", {}),
                    "document_b": question.get("document_b", {}),
                    # Add polishing results
                    "polished_result": polished_result
                }
                polished_questions.append(complete_result)
                print(f"Question {i} optimized successfully")
            else:
                # Even if optimization fails, retain original question information
                complete_result = {
                    "success": False,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "attribute_compared": attribute,
                    "multi_hop_question": original_question,
                    "answer": original_answer,
                    "fact_entity_a": fact_a,
                    "fact_entity_b": fact_b,
                    "document_a": question.get("document_a", {}),
                    "document_b": question.get("document_b", {}),
                    "polished_result": {"status": "FAILED"}
                }
                polished_questions.append(complete_result)
                print(f"Warning: Question {i} optimization failed")
        
        # Save results
        result = {"polished_questions": polished_questions}
        
        # Decide whether to save to file based on save_output parameter
        if save_output and output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Optimization results saved to: {output_file}")
        elif save_output and not output_file:
            print("Warning: Output file path not specified, cannot save optimization results")
        
        print(f"Successfully optimized {len(polished_questions)}/{len(questions)} questions")
        
        return result
    
    def polish_questions(self, input_file, output_file=None):
        """
        Process and optimize all questions in the comparison question file.
        
        Args:
            input_file: Input file path (containing comparison questions)
            output_file: Output file path (optional, auto-generated if not provided)
            
        Returns:
            str: Output file path
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"polished_comparison_questions_{timestamp}.json"
            )
        
        print(f"Starting to process comparison question file: {input_file}")
        try:
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get question list
            questions = data.get("question_results", [])
            
            if not questions:
                print("Warning: No valid comparison questions found in the input file")
                return None
            
            print(f"Found {len(questions)} comparison questions")
            
            # Process each question
            polished_questions = []
            for i, question in enumerate(questions, 1):
                print(f"Optimizing question {i}/{len(questions)}")
                
                # Extract question information
                entity_a = question.get("entity_a", "")
                entity_b = question.get("entity_b", "")
                attribute = question.get("attribute_compared", "")
                original_question = question.get("multi_hop_question", "")
                original_answer = question.get("answer", "")
                fact_a = question.get("fact_entity_a", "")
                fact_b = question.get("fact_entity_b", "")
                
                # Get document context - prioritize relevant paragraphs
                # If relevant_paragraph field exists, use it directly, otherwise extract from full text
                if "relevant_paragraph_a" in question and question["relevant_paragraph_a"]:
                    doc_a_context = question["relevant_paragraph_a"]
                else:
                    doc_a_context = self._extract_document_context(question.get("document_a_text", ""))
                    
                if "relevant_paragraph_b" in question and question["relevant_paragraph_b"]:
                    doc_b_context = question["relevant_paragraph_b"]
                else:
                    doc_b_context = self._extract_document_context(question.get("document_b", {}).get("contents", ""))
                
                # Optimize question
                polished_result = self._polish_question(
                    entity_a, entity_b, attribute,
                    original_question, original_answer,
                    fact_a, fact_b,
                    doc_a_context, doc_b_context
                )
                
                if polished_result:
                    # Retain original question information and add optimized content
                    polished_question_data = question.copy() # Use a different variable name to avoid confusion
                    polished_question_data["polished_result"] = polished_result
                    polished_questions.append(polished_question_data)
                    
                    # Print original question and optimization results for real-time review
                    print("\n" + "=" * 80)
                    print(f"Question {i}/{len(questions)}:")
                    print(f"Original question: {original_question}")
                    print(f"Original answer: {original_answer}")
                    print("\nOptimization result:")
                    print(json.dumps(polished_result, indent=2, ensure_ascii=False))
                    print("=" * 80)
                    
                    if polished_result.get("status") == "PASS":
                        print("Question passed validation, no modification needed")
                    elif polished_result.get("status") == "ADJUST":
                        print("Question has been fine-tuned")
                    elif polished_result.get("status") == "REWORKED":
                        print("Question has been rewritten")
                    else:
                        print("Question was rejected")
                else:
                    print(f"Warning: Question {i} optimization failed")
            
            # Save results
            output_data = {
                "polished_questions": polished_questions,
                "timestamp": datetime.now().isoformat(),
                "original_file": input_file
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"Successfully optimized {len(polished_questions)}/{len(questions)} questions")
            print(f"Results saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"Error: An error occurred during processing: {str(e)}")
            return None
    
    def _extract_document_context(self, doc_text, max_length=500):
        """
        Extract context from document text.
        
        Args:
            doc_text: Document text
            max_length: Maximum context length
            
        Returns:
            str: Extracted context
        """
        if not doc_text:
            return ""
        
        # Simply truncate the first max_length characters as context
        context = doc_text[:max_length]
        
        # Ensure truncation at the end of a complete sentence
        last_period = context.rfind(".")
        if last_period > 0 and last_period < len(context) - 10:  # Keep at least 10 characters
            context = context[:last_period + 1]
            
        return context
    
    def _polish_question(self, entity_a, entity_b, attribute,
                        original_question, original_answer,
                        fact_a, fact_b,
                        doc_a_context, doc_b_context):
        """
        Optimize a single comparison question.
        
        Args:
            entity_a: Name of entity A
            entity_b: Name of entity B
            attribute: Attribute being compared
            original_question: Original question
            original_answer: Original answer
            fact_a: Relevant facts for entity A
            fact_b: Relevant facts for entity B
            doc_a_context: Context for document A
            doc_b_context: Context for document B
            
        Returns:
            dict: Optimization result
        """
        try:
            # Prepare input data for the prompt
            prompt = COMPARISON_POLISHER_PROMPT.format(
                entity_a_name=entity_a,
                entity_b_name=entity_b,
                attribute_compared=attribute,
                original_question=original_question,
                original_answer=original_answer,
                fact_entity_a=fact_a,
                fact_entity_b=fact_b,
                document_a_context=doc_a_context,
                document_b_context=doc_b_context
            )
            
            # Prepare message format
            messages = [{"role": "user", "content": prompt}]
            
            # Call OpenAI to generate optimization results
            response = self.generator.generate([messages])
            
            if not response or response[0] is None:
                print("Warning: Generator returned an empty response")
                return None
                
            # Parse response
            return self._parse_response(response[0])
            
        except Exception as e:
            print(f"Error: An error occurred while optimizing the question: {str(e)}")
            return None
    
    def _parse_response(self, response):
        """
        Parse OpenAI's response and extract optimization results.
        
        Args:
            response: OpenAI's response text
            
        Returns:
            dict: Parsed result
        """
        if not response:
            return None
        
        response = response.strip()
        
        # Check optimization result type
        if "[PASS]" in response:
            return {"status": "PASS"}
        
        elif "[ADJUST]" in response:
            result = {"status": "ADJUST"}
            
            # Directly extract the optimized question, no longer extract background information separately
            q_start = response.find("REFINED_QUESTION:") 
            if q_start != -1:
                q_end = response.find("REFINED_ANSWER:", q_start)
                if q_end != -1:
                    result["refined_question"] = response[q_start + 17:q_end].strip()
            
            # Extract the optimized answer
            a_start = response.find("REFINED_ANSWER:") 
            if a_start != -1:
                result["refined_answer"] = response[a_start + 15:].strip()
            
            return result
        
        elif "[REWORKED]" in response:
            result = {"status": "REWORKED"}
            
            # Directly extract the optimized question, no longer extract background information separately
            q_start = response.find("REFINED_QUESTION:") 
            if q_start != -1:
                q_end = response.find("REFINED_ANSWER:", q_start)
                if q_end != -1:
                    result["refined_question"] = response[q_start + 17:q_end].strip()
            
            # Extract the optimized answer
            a_start = response.find("REFINED_ANSWER:") 
            if a_start != -1:
                a_end = response.find("REFINED_FACT_A:", a_start)
                if a_end != -1:
                    result["refined_answer"] = response[a_start + 15:a_end].strip()
                else:
                    result["refined_answer"] = response[a_start + 15:].strip()
            
            # Extract optimized fact A
            fa_start = response.find("REFINED_FACT_A:") 
            if fa_start != -1:
                fa_end = response.find("REFINED_FACT_B:", fa_start)
                if fa_end != -1:
                    result["refined_fact_a"] = response[fa_start + 15:fa_end].strip()
                else:
                    result["refined_fact_a"] = response[fa_start + 15:].strip()
            
            # Extract optimized fact B
            fb_start = response.find("REFINED_FACT_B:") 
            if fb_start != -1:
                result["refined_fact_b"] = response[fb_start + 15:].strip()
            
            return result
        
        elif "[REJECTED]" in response:
            result = {"status": "REJECTED"}
            
            # Extract rejection reason
            r_start = response.find("REASON:") 
            if r_start != -1:
                result["reason"] = response[r_start + 7:].strip()
            
            return result
        
        # Unrecognized response format
        return {"status": "UNKNOWN", "raw_response": response}


if __name__ == "__main__":
    # Configure path directly, do not use command line arguments
    config_path = "./config_lib/extract_config_wikifulldoc.yaml" 
    input_file = "./comparison_questions.json"
    output_file = "./polished_comparison_questions.json"
    
    print(f"Using configuration file: {config_path}")
    print(f"Processing input file: {input_file}")
    
    polisher = ComparisonPolisher(config_path)
    output_path = polisher.polish_questions(input_file, output_file)
    
    if output_path:
        print(f"Processing complete, results saved to: {output_path}")
    else:
        print("Processing failed")
