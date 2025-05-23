import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
import json
from hopweaver.components.utils.data_reader import DocumentReader
from hopweaver.components.utils.prompts import ENTITY_EXTRACTION_PROMPT

example_document = {
"id": "climate_doc_2023",
"title": "Climate Change Impacts",
"contents": """
Climate Change Impacts on Global Agricultural Systems: A Meta-Analysis
Climate change poses significant threats to global agricultural systems. This meta-analysis synthesizes findings from 157 studies published between 2000 and 2022, examining the impacts of changing climate patterns on crop yields, farming practices, and food security across different geographical regions. Results indicate that while some high-latitude regions may experience positive effects, tropical and subtropical agricultural zones face severe threats from increased temperature, altered precipitation patterns, and extreme weather events. The Intergovernmental Panel on Climate Change (IPCC) projections suggest that without substantial adaptation strategies, global crop yields could decline by 2-6% per decade as temperatures rise. This paper also evaluates the effectiveness of various adaptation strategies, including drought-resistant crops, precision farming techniques, and policy interventions by organizations such as the Food and Agriculture Organization (FAO).
"""
}



class EntityExtractor:
    """Extract entities and related information from documents"""
    def __init__(self, config):
        self.config = config
        self.config["generator_model"] = self.config["entity_extractor_model"]
        self.config["generation_params"]["max_tokens"] = 4096
        # Initialize generator
        self.generator = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize OpenAI generator"""
        # Use FlashRAG's OpenaiGenerator
        return OpenaiGenerator(self.config)
    
    def extract_entities(self, document, max_retries=3):
        """Extract entities and related information from the document
        
        Args:
            document: Document dictionary containing 'contents' field
            max_retries: Maximum number of retries
            
        Returns:
            list: List of dictionaries containing entity information, each entity includes document ID
        """
        if not document or 'contents' not in document:
            print("Document is empty or does not contain content")
            return []
            
        input_text = document['contents']
        document_id = document.get('id', 'unknown_id')  # Get document ID, use default value if not present
        document_title = document.get('title', '')  # Get document title, use empty string if not present
        
        # Use and format the prompt template imported from prompts.py
        prompt = ENTITY_EXTRACTION_PROMPT.format(document_title=document_title, input_text=input_text)
        
        # Create message format
        messages = [{"role": "user", "content": prompt}]
        
        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Entity extraction retry attempt {attempt + 1}...")
                
                response = self.generator.generate([messages])
                print(response)
                
                # Check if the response is valid
                if not response or response[0] is None:
                    print(f"Warning: Generator returned an empty response (attempt {attempt+1}/{max_retries})")
                    continue
                    
                # Parse response and add document ID
                entities = self._parse_entity_response(response[0])
                
                # If entities are successfully extracted, check if paragraph length meets requirements
                if entities:
                    # Check if each entity's paragraph contains at least 20 words
                    segments_too_short = False
                    for entity in entities:
                        if "segments" in entity and entity["segments"]:
                            for i, segment in enumerate(entity["segments"]):
                                segment_words = len(segment.split())
                                if segment_words < 20:
                                    print(f"Warning: Paragraph {i+1} of entity '{entity['name']}' has only {segment_words} words, less than the required 20 words")
                                    segments_too_short = True
                                    break
                        if segments_too_short:
                            break
                    
                    # If there are paragraphs with insufficient length, retry generation
                    if segments_too_short:
                        print(f"Paragraph length insufficient, attempting to regenerate... (attempt {attempt+1}/{max_retries})")
                        continue
                    
                    # All checks passed, add document ID to each entity and return
                    for entity in entities:
                        entity["document_id"] = document_id
                    return entities
                else:
                    print(f"Failed to extract entities from response (attempt {attempt+1}/{max_retries})")
                    
            except Exception as e:
                print(f"Error during entity extraction (attempt {attempt+1}/{max_retries}): {str(e)}")
        
        # All retries failed
        print(f"Entity extraction failed after {max_retries} attempts")
        return []
    
    def _parse_entity_response(self, response):
        """Parse LLM response and extract entity information
        
        Args:
            response: LLM response text
            
        Returns:
            list: List of dictionaries containing entity information
        """
        # Remove possible completion markers
        response = response.replace("<|COMPLETE|>", "").strip()
        
        # Handle possible code block formats (e.g., ```json ... ```)
        # Check if it contains code block markers
        if '```' in response:
            # Attempt to extract code block content
            try:
                # Find the start and end of the first code block
                start_idx = response.find('```')
                if start_idx != -1:
                    # Skip start marker and possible language identifier
                    content_start = response.find('\n', start_idx) + 1
                    # Find the end marker
                    end_idx = response.find('```', content_start)
                    if end_idx != -1:
                        # Extract code block content
                        code_content = response[content_start:end_idx].strip()
                        # Check if it is JSON format
                        if code_content.startswith('[') or code_content.startswith('{'):
                            try:
                                # Attempt to parse JSON
                                json_data = json.loads(code_content)
                                # If it is a list, take the first element as the response content
                                if isinstance(json_data, list) and len(json_data) > 0:
                                    response = json_data[0]
                                    print("Successfully extracted JSON content from code block")
                            except json.JSONDecodeError:
                                print("Code block content is not valid JSON format")
            except Exception as e:
                print(f"Error processing code block: {e}")
        
        # Directly attempt to extract main entity information from the entire response
        # If content in the format (entity"<|>"name"<|>"type") is found, create entity directly
        import re
        
        # First, attempt to extract basic information for the bridge entity
        entity_pattern = r'\(?(?:["\']?bridge_entity["\']?<\|>|\(["\']?bridge_entity["\']?<\|>)["\']?([^<\|>"]+)["\']?<\|>["\']?([^<\|>"]+)["\']?\)?'
        entity_matches = re.findall(entity_pattern, response)
        
        # Attempt to extract associated segment information
        segments_pattern = r'(?:["\']?relevant_segments["\']?<\|>|\(["\']?relevant_segments["\']?<\|>)["\']?([^<\|>"]+)["\']?<\|>["\']?([^<\|>"\)]+)["\']?'
        segments_matches = re.findall(segments_pattern, response)
        
        # Attempt to extract query information
        query_pattern = r'(?:["\']?query["\']?<\|>|\(["\']?query["\']?<\|>)["\']?([^<\|>"]+)["\']?<\|>["\']?([^<\|>"\)]+)["\']?'
        query_matches = re.findall(query_pattern, response)
        
        # If entity information is found through regular expressions, build the entity directly
        entity_dict = {}
        
        # Process basic entity information
        for name, entity_type in entity_matches:
            entity_dict[name.strip()] = {
                "name": name.strip(),
                "type": entity_type.strip(),
                "segments": [],
                "query": ""
            }
        
        # Process paragraph information
        for name, segment_text in segments_matches:
            name = name.strip()
            if name in entity_dict:
                # Ensure paragraph is not empty
                if segment_text.strip():
                    entity_dict[name]["segments"] = [segment_text.strip()]
        
        # Process query information
        for name, query in query_matches:
            name = name.strip()
            if name in entity_dict:
                entity_dict[name]["query"] = query.strip()
        
        # If no valid entity is found through the above regular expressions, try the traditional segmented parsing method
        if not entity_dict:
            print("Attempting to parse response using traditional segmented parsing method")
            # Split different parts by delimiter
            sections = response.split("##")
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Check if it is a bridge entity definition, related paragraph, or query
                if 'bridge_entity<|' in section or '"bridge_entity"<|' in section:
                    try:
                        # Handle square bracket or parenthesis format
                        content = section.strip('[]()').strip()
                        parts = [p.strip().strip('"').strip('>').strip('"') for p in content.split('<|')]
                        entity_type = parts[0]  # Should be "bridge_entity"
                        entity_name = parts[1]
                        entity_category = parts[2] if len(parts) > 2 else "Unknown type"
                        
                        # Create new entity entry
                        entity_dict[entity_name] = {
                            "name": entity_name,
                            "type": entity_category,
                            "segments": [],
                            "query": ""
                        }
                    except Exception as e:
                        print(f"Error parsing bridge entity definition: {e}")
                        
                elif 'relevant_segments<|' in section or '"relevant_segments"<|' in section:
                    try:
                        # Handle square bracket or parenthesis format
                        content = section.strip('[]()').strip()
                        parts = [p.strip().strip('"').strip('>').strip('"') for p in content.split('<|')]
                        segment_type = parts[0]  # Should be "relevant_segments"
                        entity_name = parts[1]
                        
                        # Extract all paragraphs, remove redundant quotes
                        segments = []
                        for segment in parts[2:]:
                            cleaned_segment = segment.strip().strip('"').strip('>').strip('"')
                            if cleaned_segment:
                                segments.append(cleaned_segment)
                        
                        # Add paragraphs to the corresponding entity
                        if entity_name in entity_dict:
                            entity_dict[entity_name]["segments"] = segments
                    except Exception as e:
                        print(f"Error parsing segments: {e}")
                    
                elif 'query<|' in section or '"query"<|' in section:
                    try:
                        # Handle square bracket or parenthesis format
                        content = section.strip('[]()').strip()
                        parts = [p.strip().strip('"').strip('>').strip('"') for p in content.split('<|')]
                        query_type = parts[0]  # Should be "query"
                        entity_name = parts[1]
                        query = parts[2] if len(parts) > 2 else ""
                        
                        # Add query to the corresponding entity
                        if entity_name in entity_dict:
                            entity_dict[entity_name]["query"] = query
                    except Exception as e:
                        print(f"Error parsing query: {e}")
        
        # If entity name, type, or paragraphs are missing from the entity dictionary, try to extract from the original response
        if not entity_dict:
            print("Attempting last resort parsing: directly extract entity information from response")
            # Directly search for entity name in the original text
            try:
                # First, try to extract from triplet format
                triple_pattern = r'\(?"?([^"<\|>]+)"?<\|>"?([^"<\|>]+)"?\)?'
                triples = re.findall(triple_pattern, response)
                
                # If triplets are found, try to identify entity name and type
                if triples and len(triples) >= 2:
                    # Assume the first triplet is the entity name and type
                    entity_name = triples[0][1].strip()
                    entity_type = triples[0][0].strip() if triples[0][0] != "bridge_entity" else triples[1][0].strip()
                    
                    # Create a basic entity
                    entity_dict[entity_name] = {
                        "name": entity_name,
                        "type": entity_type,
                        "segments": [],
                        "query": ""
                    }
                    
                    # Extract possible paragraphs from the response
                    content_parts = response.split('"##"')
                    for part in content_parts:
                        if entity_name in part and len(part) > 50:  # Assume at least 50 characters for a valid paragraph
                            entity_dict[entity_name]["segments"] = [part.strip()]
                            break
            except Exception as e:
                print(f"Failed to directly extract entity information: {e}")
        
        # Convert to list form and perform final validation
        entities = list(entity_dict.values())
        
        # Final validation: ensure each entity has at least a name and type
        validated_entities = []
        for entity in entities:
            if entity["name"] and (entity["type"] or entity.get("segments")):
                # If type is missing but segments exist, set default type
                if not entity["type"] and entity.get("segments"):
                    entity["type"] = "Concept"  # Default type
                validated_entities.append(entity)
        
        return validated_entities
    
    def get_entity_names(self, document):
        """Extract only the list of entity names from the document (backward compatible with original functionality)
        
        Args:
            document: Document dictionary containing 'contents' field
            
        Returns:
            list: List of extracted entity names and document IDs
        """
        entities_info = self.extract_entities(document)
        # Return a list of dictionaries containing entity names and document IDs
        return [{"name": entity["name"], "document_id": entity["document_id"]} for entity in entities_info]


if __name__ == "__main__":
    # Module functionality test
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Initialize DocumentReader, use corpus_path from config
    doc_reader = DocumentReader(config['corpus_path'])
    
    # Initialize entity extractor
    extractor = EntityExtractor(config)
    
    # Set test mode
    # Set batch_mode to True to enable batch testing, False for single document testing
    batch_mode = True
    # Number of documents for batch testing
    batch_count = 5
    # Output JSON file path
    output_file = "./extracted_entities.json"
    
    # Batch testing mode
    if batch_mode:
        print(f"\nStarting batch test, will process {batch_count} documents...")
        all_entities = []
        all_documents = []  # Used to store original documents
        processed_docs = 0
        max_attempts = batch_count * 3  # Set maximum attempts to avoid infinite loop
        attempts = 0
        
        while processed_docs < batch_count and attempts < max_attempts:
            attempts += 1
            # Randomly get a document from the knowledge base
            sample_document = doc_reader.get_random_document()
            
            if not sample_document:
                print(f"Cannot get document from knowledge base, skipping...")
                continue
                
            print(f"\nProcessing document {processed_docs+1}/{batch_count}:")
            print(f"- Document ID: {sample_document.get('id', 'unknown')}")
            print(f"- Title: {sample_document.get('title', 'unknown')}") 
            print(f"- Size: {len(sample_document.get('contents', ''))} characters")
            
            # Extract entities
            print("Starting to extract entity information...")
            entities = extractor.extract_entities(sample_document)
            
            if entities:
                print(f"Successfully extracted {len(entities)} entities")
                all_entities.extend(entities)
                
                # Save original document (keep only necessary fields to reduce file size)
                doc_to_save = {
                    "id": sample_document.get("id", "unknown"),
                    "title": sample_document.get("title", "unknown"),
                    "contents": sample_document.get("contents", "")
                }
                all_documents.append(doc_to_save)
                
                processed_docs += 1
                
                # Display brief information for the first entity
                if entities:
                    entity = entities[0]
                    print(f"Example entity: {entity['name']} (Type: {entity['type']}, Document ID: {entity['document_id']})")
            else:
                print("No entities extracted, skipping this document")
        
        # Save results to JSON file
        if all_entities:
            # Create a results dictionary containing entities and original documents
            result = {
                "entities": all_entities,
                "documents": all_documents
            }
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, indent=2, ensure_ascii=False, fp=f)
            
            print(f"\nBatch test complete! Processed {processed_docs} documents, extracted {len(all_entities)} entities")
            print(f"Results saved to: {output_file}")
        else:
            print(f"\nBatch test complete, but no entities were extracted")
    
    # Single document test mode (original functionality)
    else:
        # Maximum number of retries for the retry mechanism
        max_retry = 3
        current_retry = 0
        entities = []
        
        while current_retry < max_retry:
            # Randomly get a document from the knowledge base
            sample_document = doc_reader.get_random_document()
            
            # Print information of the read document
            if sample_document:
                print(f"\nRetrieved document from knowledge base:")
                print(f"- Document ID: {sample_document.get('id', 'unknown')}")
                print(f"- Title: {sample_document.get('title', 'unknown')}") 
                print(f"- Size: {len(sample_document.get('contents', ''))} characters")
                print(f"- Attempt: {current_retry + 1}/{max_retry}")
            else:
                print(f"\nCould not retrieve document from knowledge base, using fallback example... (Attempt: {current_retry + 1}/{max_retry})")
                sample_document = example_document  # Use predefined sample document
            
            # Extract entities
            print("\nStarting to extract entity information...")
            entities = extractor.extract_entities(sample_document)
            
            # Determine if entities were extracted
            if entities:
                print(f"Successfully extracted {len(entities)} entities")
                print(sample_document['contents'])
                break
            else:
                print(f"No entities extracted, retrying attempt {current_retry + 1}")
                current_retry += 1
        
        # Output extraction results
        if entities:
            # Format output results
            print("\nExtracted entity information:")
            for entity in entities:
                print(f"\nEntity: {entity['name']} (Type: {entity['type']}, Document ID: {entity['document_id']})")
                if 'segments' in entity and entity['segments']:
                    print("Related text segments:")
                    for i, segment in enumerate(entity['segments'], 1):
                        print(f"  {i}. {segment[:200]}..." if len(segment) > 200 else f"  {i}. {segment}")
                if 'query' in entity and entity['query']:
                    print(f"Expanded query: {entity['query']}")
            
            # Full information can also be displayed in JSON format
            print("\nFull entity information in JSON format:")
            print(json.dumps(entities, indent=2, ensure_ascii=False))
        else:
            print(f"\nFailed to extract valid entities after {max_retry} attempts, please check document content or API settings")
