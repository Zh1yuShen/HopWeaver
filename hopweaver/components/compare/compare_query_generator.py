"""
Comparison Query Generator - Generates queries for Entity A to find possible comparison Entity B

This module receives filtered high-quality entity data and generates for each entity:
1. Directly recommend a specific Entity B and a verification query (if possible)
2. Or generate 3 diversified search queries

As a component in a multi-hop question answering system, this module helps the system find other entities that can be compared with the current entity.
"""

import sys
import os
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
from hopweaver.components.utils.prompts import COMPARE_QUERY_GENERATOR_PROMPT

# Prompt template for comparison query generation



class CompareQueryGenerator:
    """Comparison Query Generator: Generates queries for finding comparison Entity B based on high-quality Entity A"""
    
    def __init__(self, config):
        """
        Initialize the comparison query generator
        
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
        
    def generate_query(self, entity_data, max_retries=3):
        """
        Generate queries for Entity A to find Entity B
        
        Args:
            entity_data: Entity data, including entity information, attributes, and document content
            max_retries: Maximum number of retries
            
        Returns:
            dict: Dictionary containing query information, in the format:
                  {
                      "entity_a": {Entity A information},
                      "query_type": "recall_focused_verify" or "search_queries",
                      "entity_b_name": Recommended Entity B name (if recall_focused_verify),
                      "attribute_x": Comparison attribute (if recall_focused_verify),
                      "verification_query": Verification query (if recall_focused_verify),
                      "search_queries": [Query 1, Query 2, Query 3] (if search_queries)
                  }
        """
        # Print log
        print(f"\nProcessing entity data: {entity_data.get('subject_entity', {}).get('name', '?')}")
        
        # Extract entity information
        subject_entity = entity_data.get("subject_entity", {})
        subject_entity_name = subject_entity.get("name", "Unknown Entity")
        subject_entity_type = subject_entity.get("type", "Unknown Type")
        document_id = entity_data.get("document_id", "unknown")
        
        # Format attribute list
        attributes = entity_data.get("attributes", [])
        attributes_list_str = self._format_attributes_list(attributes)
        
        # Try to get document content
        contents = entity_data.get("contents", "")
        document_a_text = contents
        
        # If 'contents' is not present, try to get document content from other possible fields
        if not document_a_text and "contents" in entity_data:
            document_a_text = entity_data.get("contents", "")
        
        document_title = entity_data.get("title", "")
        
        # Add document title as context information (if available)
        if document_title:
            document_a_text = f"Title: {document_title}\n\n{document_a_text}"
            
        # Use prompt template and format
        try:
            prompt = COMPARE_QUERY_GENERATOR_PROMPT.format(
                subject_entity_name=subject_entity_name,
                subject_entity_type=subject_entity_type,
                document_a_text=document_a_text,
                attributes_list_str_a=attributes_list_str
            )
        except Exception as e:
            print(f"Error formatting prompt: {str(e)}")
            print(f"  subject_entity_name: {subject_entity_name}")
            print(f"  subject_entity_type: {subject_entity_type}")
            print(f"  attributes_list_str_a: {attributes_list_str}")
            print(f"  document_a_text: [Document content too long to display]")
            return None
        
        # Create message format
        messages = [{"role": "user", "content": prompt}]
        
        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Query generation retry attempt {attempt + 1}...")
                
                # Print log
                print(f"\nGenerating comparison queries for entity {subject_entity_name} (type: {subject_entity_type})...")
                
                # Generate response
                response = self.generator.generate([messages])
                
                # Check if the response is valid
                if not response or response[0] is None:
                    print(f"Warning: Generator returned an empty response (attempt {attempt+1}/{max_retries})")
                    continue
                
                # Parse response
                query_result = self._parse_query_response(response[0], entity_data)
                
                # If query generation is successful, break the retry loop
                if query_result:
                    return query_result
                    
            except Exception as e:
                print(f"Error during query generation: {str(e)}")
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def _parse_query_response(self, response, entity_data):
        """
        Parse the generated query response
        
        Args:
            response: Raw response from the generator
            entity_data: Entity data
            
        Returns:
            dict: Parsed query result
        """
        # Initialize result
        result = {
            "entity_a": {
                "name": entity_data.get("subject_entity", {}).get("name", ""),
                "type": entity_data.get("subject_entity", {}).get("type", ""),
                "document_id": entity_data.get("document_id", "")
            }
        }
        
        # Remove possible completion markers
        response = response.replace("<|COMPLETE|>", "").strip()
        
        # Print raw response
        print(f"Raw response: {response}")
        
        try:
            # Check if it's a direct recommendation path (recall_focused_verify)
            if "recall_focused_verify" in response:
                # Extract content within parentheses
                content = response.strip('()').strip()
                parts = [p.strip() for p in content.split("<|>")]
                
                # Check if the number of parts is correct
                if len(parts) >= 4:
                    result["query_type"] = "recall_focused_verify"
                    result["entity_b_name"] = parts[1]
                    result["attribute_x"] = parts[2]
                    result["verification_query"] = parts[3]
                    print(f"Generated direct recommendation:\n  Entity B: {result['entity_b_name']}\n  Attribute X: {result['attribute_x']}\n  Verification Query: {result['verification_query']}")
                    return result
            
            # Check if it's a search query path (search_queries)
            elif "search_queries" in response:
                # Extract content within parentheses
                content = response.strip('()').strip()
                parts = [p.strip() for p in content.split("<|>")]
                
                # Check if the number of parts is correct
                if len(parts) >= 4:
                    result["query_type"] = "search_queries"
                    result["search_queries"] = [parts[1], parts[2], parts[3]]
                    print(f"Generated search queries:\n  Query 1: {parts[1]}\n  Query 2: {parts[2]}\n  Query 3: {parts[3]}")
                    return result
            
            # If no query type is recognized, try to extract information from the response
            else:
                print("Unrecognized response format, trying to extract queries from text...")
                
                # Try to extract search queries
                if "query" in response.lower() or "search" in response.lower():
                    # Split into lines and find at least 3 query statements
                    lines = [line.strip() for line in response.split("\n") if line.strip()]
                    queries = []
                    
                    for line in lines:
                        # Find possible queries, usually with quotes or colons
                        if (":" in line or "\"" in line or "'" in line) and not line.startswith("*") and not line.startswith("#"):
                            # Clean up line numbers, quotes, etc.
                            cleaned_line = line
                            for prefix in ["1.", "2.", "3.", "query 1:", "query 2:", "query 3:", "query:", "- "]:
                                if cleaned_line.lower().startswith(prefix):
                                    cleaned_line = cleaned_line[len(prefix):].strip()
                            
                            # Remove quotes
                            cleaned_line = cleaned_line.strip('"\'').strip()
                            
                            if cleaned_line and len(cleaned_line) > 10:  # Ensure the query has sufficient length
                                queries.append(cleaned_line)
                    
                    # If at least 3 queries are found, return the result
                    if len(queries) >= 3:
                        result["query_type"] = "search_queries"
                        result["search_queries"] = queries[:3]  # Take the first three
                        print(f"Extracted search queries from non-standard format:\n  Query 1: {queries[0]}\n  Query 2: {queries[1]}\n  Query 3: {queries[2]}")
                        return result
                
                # If no queries are recognized, generate default search queries
                entity_name = entity_data.get("subject_entity", {}).get("name", "")
                entity_type = entity_data.get("subject_entity", {}).get("type", "")
                
                if entity_name and entity_type:
                    result["query_type"] = "search_queries"
                    result["search_queries"] = [
                        f"Other {entity_type.lower()} similar to {entity_name}",
                        f"Famous {entity_type.lower()} with characteristics like {entity_name}",
                        f"Notable {entity_type.lower()} in the same field as {entity_name}"
                    ]
                    print(f"Generated default search queries:\n  Query 1: {result['search_queries'][0]}\n  Query 2: {result['search_queries'][1]}\n  Query 3: {result['search_queries'][2]}")
                    return result
        
        except Exception as e:
            print(f"Error parsing query response: {str(e)}")
        
        return None
    
    def process_entities_batch(self, input_file, output_file):
        """
        Process a batch of entities and generate comparison queries
        
        Args:
            input_file: Input file path (containing high-quality entity data)
            output_file: Output file path
            
        Returns:
            bool: Whether the processing was successful
        """
        try:
            print(f"\nReading high-quality entity file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Try different data structure formats
            filtered_entities = data.get("filtered_entities", [])
            if filtered_entities:
                print(f"Found filtered_entities format data")
                entity_data_list = [entity_item.get("entity_data", {}) for entity_item in filtered_entities if "entity_data" in entity_item]
            else:
                entity_data_list = data.get("entity_data", [])
            
            documents = data.get("documents", [])
            
            if not entity_data_list:
                print("No entity data found in the input file")
                return False
            
            print(f"Successfully read {len(entity_data_list)} entity data")
            
            # Process each entity and generate queries
            query_results = []
            
            for i, entity_data in enumerate(entity_data_list, 1):
                print(f"\nProcessing entity {i}/{len(entity_data_list)}")
                
                if "subject_entity" not in entity_data:
                    print("  Skipping: Missing subject entity information")
                    continue
                
                # Generate query
                query_result = self.generate_query(entity_data)
                
                if query_result:
                    query_results.append({
                        "entity_data": entity_data,
                        "query_result": query_result
                    })
                else:
                    print("  Query generation failed, skipping this entity")
            
            # Save results to JSON file
            if query_results:
                output_data = {
                    "query_results": query_results,
                    "documents": documents
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                print(f"\nQuery generation completed! Successfully processed {len(query_results)}/{len(entity_data_list)} entities")
                print(f"Results saved to: {output_file}")
                return True
            else:
                print("\nProcessing completed, but no queries were generated")
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
    
    # Create CompareQueryGenerator instance
    query_generator = CompareQueryGenerator(config)
    
    # Example input file path (containing high-quality entity data)
    input_file = "./filtered_compare_entities_high_quality.json"
    
    # Output file path
    output_file = "./comparison_queries.json"
    
    # Process batch of entities
    query_generator.process_entities_batch(input_file, output_file)
