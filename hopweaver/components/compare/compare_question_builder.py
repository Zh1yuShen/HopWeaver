"""
Comparison Question Builder - Processes retrieval results to generate inter-entity comparison questions

This module receives the output of the comparison retriever, analyzes the retrieved documents, and attempts to:
1. Identify the main entity in Document B (Entity B)
2. Find comparable attribute pairs between Entity A and Entity B
3. Generate natural language comparison questions, answers, and supporting facts

As a component in a multi-hop question answering system, this module helps the system generate high-quality comparison questions.
"""

import sys
import os
import json
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
from hopweaver.components.utils.prompts import COMPARE_QUESTION_BUILDER_PROMPT

# Prompt template for comparison question generation



class CompareQuestionBuilder:
    """Comparison Question Builder: Generates entity comparison questions based on retrieval results"""
    
    def __init__(self, config):
        """
        Initialize the comparison question builder
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.config["generator_model"] = self.config["entity_extractor_model"]
        self.config["generation_params"]["max_tokens"] = 4096
        # Initialize generator
        self.generator = OpenaiGenerator(config)
    
    def _format_attributes_list(self, attributes):
        """
        Format the attribute list into a string
        
        Args:
            attributes: List of attributes
            
        Returns:
            str: Formatted attribute list string
        """
        attributes_list = ""
        for attr in attributes:
            attributes_list += f"- Attribute Name: {attr.get('name', '')}, Attribute Value: {attr.get('value', '')}\n"
        return attributes_list
    
    def build_question(self, entity_data, candidate_doc, document_a_text=None, max_retries=3):
        """
        Build comparison question
        
        Args:
            entity_data: Entity A data, including entity information and attributes
            candidate_doc: Candidate Document B, containing possible Entity B
            document_a_text: Text content of Document A (if None, will be obtained from entity_data)
            max_retries: Maximum number of retries
            
        Returns:
            dict: Dictionary containing the comparison question, or None if failed
        """
        # Print log
        subject_entity = entity_data.get("subject_entity", {})
        subject_entity_name = subject_entity.get("name", "Unknown Entity")
        subject_entity_type = subject_entity.get("type", "Unknown Type")
        
        print(f"\nAttempting to build entity comparison question:")
        print(f"  Entity A: {subject_entity_name} (Type: {subject_entity_type})")
        print(f"  Candidate Document B Title: {candidate_doc.get('title', 'Unknown')}")
        
        # Get Attribute List for Entity A
        attributes = entity_data.get("attributes", [])
        attributes_list_str = self._format_attributes_list(attributes)
        
        # Get content of Document A
        if document_a_text is None:
            # Try various possible fields to get document content
            content_field = None
            for field in ["contents", "content", "text", "document_content"]:  # Prioritize using 'contents'
                if field in entity_data and entity_data.get(field):
                    content_field = field
                    break
                    
            if content_field:
                document_a_text = entity_data.get(content_field)
                print(f"  Getting Document A content from entity_data.{content_field}, length: {len(document_a_text)}")
            else:
                # If entity data does not contain document content, try using the content of the candidate document
                if candidate_doc and "contents" in candidate_doc:
                    print(f"  Warning: Document content not found in entity data, trying to use candidate document content")
                    document_a_text = candidate_doc.get("contents", "")
                    if document_a_text:
                        print(f"  Successfully used candidate document content as Document A content, length: {len(document_a_text)}")
                    else:
                        raise ValueError(f"Could not find content for Document A, cannot process comparison question for entity {subject_entity_name}")
                else:
                    raise ValueError(f"Could not find content for Document A, cannot process comparison question for entity {subject_entity_name}")
        else:
            print(f"  Getting Document A content from passed parameter, length: {len(document_a_text)}")
        
        # Get content of Document B
        document_b_text = candidate_doc.get("contents", "")
        print(f"  Document B content length: {len(document_b_text)}")
        print(f"  Attribute list length: {len(attributes_list_str)}")
        print(f"  Number of attributes: {len(attributes)}")
        
        # Format prompt
        try:
            # Determine if the query type is recall_focused_verify
            query_result = entity_data.get("query_result", {})
            query_type = query_result.get("query_type", "")
            
            # Prepare prompt parameters
            prompt_params = {
                "subject_entity_name": subject_entity_name,
                "subject_entity_type": subject_entity_type,
                "document_a_text": document_a_text,
                "attributes_list_str_a": attributes_list_str,
                "document_b_text": document_b_text
            }
            
            # Print parameter filling status
            print(f"\nPrompt parameter filling status:")
            print(f"  subject_entity_name: {subject_entity_name}")
            print(f"  subject_entity_type: {subject_entity_type}")
            print(f"  document_a_text length: {len(document_a_text)}")
            print(f"  attributes_list_str_a length: {len(attributes_list_str)}")
            print(f"  document_b_text length: {len(document_b_text)}")
            
            # If query_type is recall_focused_verify, add Entity B name and target attribute
            if query_type == "recall_focused_verify":
                entity_b_name = query_result.get("entity_b_name", "")
                attribute_x = query_result.get("attribute_x", "")
                if entity_b_name and attribute_x:
                    prompt_params["suggested_entity_b_name"] = entity_b_name
                    prompt_params["target_attribute_x_name"] = attribute_x
                    print(f"  Using known comparison information: Target Entity B: {entity_b_name}, Target Attribute: {attribute_x}")
            
            # Process conditional input
            target_info = ""
            if "suggested_entity_b_name" in prompt_params and "target_attribute_x_name" in prompt_params:
                target_info = f"*   **Focused Comparison Target:** The suggested entity for comparison is **{prompt_params['suggested_entity_b_name']}**, and the primary attribute suggested for comparison is **{prompt_params['target_attribute_x_name']}**. Prioritize identifying this entity and checking this attribute pair in step 2."
            
            prompt_params["target_info"] = target_info
            
            # Format prompt
            prompt = COMPARE_QUESTION_BUILDER_PROMPT.format(**prompt_params)
            
            # Print the beginning of the prompt (first 100 characters only)
            print(f"  Prompt beginning: {prompt[:100]}...")
            
            # Print overview of Document A and Document B content
            if document_a_text:
                print(f"  Document A content preview: {document_a_text[:50]}...")
            else:
                print(f"  Document A content is empty")
            
            if document_b_text:
                print(f"  Document B content preview: {document_b_text[:50]}...")
            else:
                print(f"  Document B content is empty")
            
            # Debug code removed
        except Exception as e:
            print(f"Error formatting prompt: {str(e)}")
            return None
        
        # Create message format
        messages = [{"role": "user", "content": prompt}]
        
        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Question building retry attempt {attempt + 1}...")
                
                # Generate response
                response = self.generator.generate([messages])
                
                # Check if the response is valid
                if not response or response[0] is None:
                    print(f"Warning: Generator returned an empty response (attempt {attempt+1}/{max_retries})")
                    continue
                
                # Determine if it's a failure response, skip if so
                if "FAIL" in response[0] and len(response[0].strip()) < 10:
                    print(f"Generated response indicates inability to build comparison question")
                    return None
                
                # Parse response
                question_result = self._parse_question_response(response[0], entity_data, candidate_doc)
                
                # If question building is successful, break the retry loop
                if question_result:
                    return question_result
                    
                # If JSON parsing fails, try to extract information from text且不是最后一次尝试，则继续重试
                if attempt < max_retries - 1:
                    print(f"Response parsing failed, will retry...")
                    
            except Exception as e:
                print(f"Error during question building: {str(e)}")
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def _parse_question_response(self, response, entity_data, candidate_doc):
        """
        Parse the generated question response
        
        Args:
            response: Raw response from the generator
            entity_data: Entity A data
            candidate_doc: Candidate Document B
            
        Returns:
            dict: Parsed question result, or None if failed
        """
        # Clean response (remove backticks)
        response = response.strip()
        
        # Remove possible Markdown code block markers
        if response.startswith('```') and '```' in response[3:]:
            # Find the end of the Markdown code block
            end_pos = response.rfind('```')
            if end_pos > 3:
                # Extract the code block content, skipping the opening backticks and possible language identifier line
                content_start = response.find('\n') + 1
                content = response[content_start:end_pos].strip()
                response = content
        
        # Check if it's a failure response
        if response.startswith("FAIL"):
            print("No comparable attribute pairs found")
            return None
        
        # Try to parse JSON formatted response
        # Initialize result dictionary
        result = {
            "success": True,
            "entity_a": entity_data.get("subject_entity", {}).get("name", ""),
            "entity_b": "",
            "attribute_compared": "",
            "multi_hop_question": "",
            "answer": "",
            "fact_entity_a": "",
            "fact_entity_b": "",
            "relevant_paragraph_a": "",
            "relevant_paragraph_b": "",
            "document_a": entity_data,
            "document_b": candidate_doc
        }
        
        # Remove possible PASS prefix
        if response.strip().startswith("PASS"):
            pass_end = response.find("\n", response.find("PASS"))
            if pass_end != -1:
                response = response[pass_end:].strip()
            else:
                response = response.replace("PASS", "").strip()
        
        # Find the start position of each field
        field_markers = [
            "entity_a:", "entity_b:", "attribute_compared:", "attribute:", 
            "multi_hop_question:", "question:", "answer:", 
            "fact_entity_a:", "fact_a:", "fact_entity_b:", "fact_b:",
            "relevant_paragraph_a:", "relevant_paragraph_b:"
        ]
        
        # Build field position mapping
        field_positions = {}
        for marker in field_markers:
            pos = response.find(marker)
            if pos != -1:
                field_positions[marker] = pos
        
        # Sort field markers by position
        sorted_markers = sorted(field_positions.items(), key=lambda x: x[1])
        
        # Extract the content of each field (considering multiple lines)
        for i, (marker, start_pos) in enumerate(sorted_markers):
            # Find the start position of the field content (after the colon)
            content_start = start_pos + len(marker)
            
            # Find the start position of the next field (if any)
            content_end = len(response)
            if i < len(sorted_markers) - 1:
                content_end = sorted_markers[i+1][1]
            
            # Extract the content (including possible multiple lines)
            content = response[content_start:content_end].strip()
            
            # Save the content to the result dictionary based on the field name
            key = marker.replace(":", "").lower()
            if key == "entity_a":
                result["entity_a"] = content
            elif key == "entity_b":
                result["entity_b"] = content
            elif key in ["attribute_compared", "attribute"]:
                result["attribute_compared"] = content
            elif key in ["multi_hop_question", "question"]:
                result["multi_hop_question"] = content
            elif key == "answer":
                result["answer"] = content
            elif key in ["fact_entity_a", "fact_a"]:
                result["fact_entity_a"] = content
            elif key in ["fact_entity_b", "fact_b"]:
                result["fact_entity_b"] = content
            elif key == "relevant_paragraph_a":
                result["relevant_paragraph_a"] = content
            elif key == "relevant_paragraph_b":
                result["relevant_paragraph_b"] = content
        
        # Check if all required fields are filled
        required_fields = ["entity_a", "entity_b", "attribute_compared", "multi_hop_question", "answer", "fact_entity_a", "fact_entity_b", "relevant_paragraph_a", "relevant_paragraph_b"]
        
        # Check required fields
        missing_fields = []
        for field in required_fields:
            if not result.get(field):
                missing_fields.append(field)
        
        # If required fields are missing, return None to indicate retry
        if missing_fields:
            print(f"Warning: Missing required fields {', '.join(missing_fields)}")
            return None
        
        # Print parsed result
        # Check if the raw response starts with PASS
        is_standard_format = response.strip().startswith("PASS")
        if is_standard_format:
            print(f"Successfully built comparison question:\n  Question: {result['multi_hop_question']}")
        else:
            print(f"Successfully parsed non-standard response, built comparison question:\n  Question: {result['multi_hop_question']}")
        
        return result
    
    def process_retrieval_results(self, input_file, output_file):
        """
        Process retrieval results file, build comparison questions
        
        Args:
            input_file: Input file path, containing entity query results and retrieved documents
            output_file: Output file path, to save generated comparison questions
            
        Returns:
            bool: Processing success
        """
        try:
            print(f"\nReading retrieval results file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get retrieval results list
            retrieval_results = data.get("retrieval_results", [])
            
            if not retrieval_results:
                print("No retrieval results found in input file")
                return False
            
            print(f"Successfully read {len(retrieval_results)} retrieval results")
            
            # Get original document content
            documents = {}
            if "documents" in data:
                for doc in data.get("documents", []):
                    doc_id = doc.get("id")
                    if doc_id:
                        documents[doc_id] = doc
            
            # Process all entity query results and generate comparison questions
            question_results = []
            
            for i, retrieval_result in enumerate(retrieval_results, 1):
                print(f"\nProcessing retrieval result {i}/{len(retrieval_results)}")
                
                # Load input data
                entity_data = retrieval_result.get("entity_data")
                
                if not entity_data:
                    print("  Skipping: Missing entity data")
                    continue
                
                # Get entity document
                
                # Assume we already have the output from CompareQueryGenerator and a DocumentReader
                if "contents" not in entity_data:
                    raise ValueError(f"contents field not found, cannot process entity {entity_data.get('subject_entity', {}).get('name', 'Unknown')} query result")
                
                document_a_text = entity_data.get("contents")
                if not document_a_text:
                    raise ValueError(f"contents field is empty, cannot process entity {entity_data.get('subject_entity', {}).get('name', 'Unknown')} query result")
                print(f"  Using loaded document content for Entity A, length: {len(document_a_text)}")
                
                # Try to get from loaded document data
                retrieved_docs = retrieval_result.get("retrieved_documents", [])
                
                if not retrieved_docs:
                    print("  Skipping: No documents retrieved")
                    continue
                
                # Process each retrieved document to build questions until successful
                success = False
                for j, candidate_doc in enumerate(retrieved_docs, 1):
                    print(f"  Processing candidate document {j+1}/{len(candidate_docs)}: {candidate_doc.get('title', candidate_doc.get('id', 'Unknown'))}")
                    
                    # Build question
                    question_result = self.build_question(entity_data, candidate_doc, document_a_text)
                    
                    if question_result:
                        # Add query type and entity information
                        question_result["query_type"] = retrieval_result.get("query_type")
                        question_result["entity_a_data"] = entity_data
                        question_result["document_b_id"] = candidate_doc.get("id")
                        question_result["document_a_text"] = document_a_text # <--- Add this line

                        # Add to result list
                        question_results.append(question_result)
                        success = True
                        break
                
                if not success:
                    print("  Failed to build comparison question")
            
            # Save results to JSON file
            if question_results:
                output_data = {
                    "question_results": question_results,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                print(f"Comparison question building completed! Successfully generated {len(question_results)}/{len(retrieval_results)} comparison questions")
                print(f"Results saved to: {output_file}")
                return True
            else:
                print("\nProcessing completed, but failed to build any comparison questions")
                return False
                
        except Exception as e:
            import traceback
            print(f"Error during processing: {str(e)}")
            print(traceback.format_exc())
            return False


if __name__ == "__main__":
    # Module functional test
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Create CompareQuestionBuilder instance
    question_builder = CompareQuestionBuilder(config)
    
    # Input file path (from CompareQueryGenerator output)
    input_file = "./comparison_retrieval_results.json"
    
    # Output file path
    output_file = "./comparison_questions.json"
    
    # Process retrieval results
    question_builder.process_retrieval_results(input_file, output_file)
