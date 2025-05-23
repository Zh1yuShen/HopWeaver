import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import flashrag module
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
import json
from hopweaver.components.utils.data_reader import DocumentReader
from hopweaver.components.utils.prompts import COMPARE_ENTITY_EXTRACTION_PROMPT



example_document = {
    "id": "tokyo_profile_2023",
    "title": "Tokyo Metropolitan City",
    "contents": """
    Tokyo, officially the Tokyo Metropolis, is the capital and most populous city of Japan. Located at the head of Tokyo Bay, the Tokyo Metropolitan Area is the most populous metropolitan area in the world, with a population of 37.468 million as of 2018. Tokyo is the political and economic center of Japan, and houses the seat of the Emperor of Japan and the national government. The greater Tokyo area has a GDP of $2 trillion as of 2020, making it the largest metropolitan economy in the world.

    Tokyo was originally a small fishing village called Edo. It became a prominent political and cultural center when Tokugawa Ieyasu established the Tokugawa shogunate there in 1603. The city grew to become one of the largest cities in the world and the most populous urban area in Japan during the Edo period (1603-1867). Following the Meiji Restoration in 1868, the imperial capital in Kyoto was relocated to Edo and renamed Tokyo ("Eastern Capital"). Tokyo served as the venue for the 1964 Summer Olympics, the 2020 Summer Olympics (postponed to 2021 due to the COVID-19 pandemic), and the 2020 Summer Paralympics.
    """
}

class CompareEntityExtractor:
    """Extract subject entities and their comparable attributes from documents"""
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
        """Extract subject entities and their comparable attributes from the document
        
        Args:
            document: Document dictionary containing 'contents' field
            max_retries: Maximum number of retries
            
        Returns:
            dict: Dictionary containing subject entity and attribute information
        """
        if not document or 'contents' not in document:
            print("Document is empty or does not contain content")
            return {}
            
        input_text = document['contents']
        document_id = document.get('id', 'unknown_id')  # Get document ID, use default value if not present
        document_title = document.get('title', '')  # Get document title, use empty string if not present
        
        # Use prompt template and format
        prompt = COMPARE_ENTITY_EXTRACTION_PROMPT.format(document_title=document_title, input_text=input_text)
        
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
                entity_data = self._parse_entity_response(response[0])
                
                # If entities are successfully extracted, break the retry loop
                if entity_data and "subject_entity" in entity_data:
                    # Add document ID
                    entity_data["document_id"] = document_id
                    return entity_data
                else:
                    print(f"Failed to extract entity data from response (attempt {attempt+1}/{max_retries})")
                    
            except Exception as e:
                print(f"Error during entity extraction (attempt {attempt+1}/{max_retries}): {str(e)}")
        
        # All retries failed
        print(f"Entity extraction failed after {max_retries} attempts")
        return {}
    
    def _parse_entity_response(self, response):
        """Parse LLM response to extract subject entity and attribute information
        
        Args:
            response: LLM response text
            
        Returns:
            dict: Dictionary containing subject entity and attribute information
        """
        # Remove possible completion markers
        response = response.replace("<|COMPLETE|>", "").strip()
        
        # Handle possible code block format (e.g., ```json ... ```)
        if '```' in response:
            try:
                # Find the start and end of the first code block
                start_idx = response.find('```')
                if start_idx != -1:
                    # Skip the start marker and possible language identifier
                    content_start = response.find('\n', start_idx) + 1
                    # Find the end marker
                    end_idx = response.find('```', content_start)
                    if end_idx != -1:
                        # Extract code block content
                        code_content = response[content_start:end_idx].strip()
                        # Check if it is JSON format
                        if code_content.startswith('{') or code_content.startswith('['):
                            try:
                                json_data = json.loads(code_content)
                                if isinstance(json_data, dict) or (isinstance(json_data, list) and len(json_data) > 0):
                                    response = json_data if isinstance(json_data, dict) else json_data[0]
                                    print("Successfully extracted JSON content from code block")
                            except json.JSONDecodeError:
                                print("Code block content is not valid JSON format")
            except Exception as e:
                print(f"Error processing code block: {e}")
        
        # Split different parts by delimiter
        sections = response.split(" ## ")
        
        result = {}
        attributes = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Check if it is a subject entity definition
            if 'subject_entity<|' in section or '"subject_entity"<|' in section:
                try:
                    # Handle square bracket or parenthesis format
                    content = section.strip('[]()').strip()
                    parts = [p.strip().strip('"').strip('>').strip('"') for p in content.split('<|')]
                    entity_type = parts[0]  # Should be "subject_entity"
                    entity_name = parts[1]
                    entity_category = parts[2]
                    
                    # Create subject entity entry
                    result["subject_entity"] = {
                        "name": entity_name,
                        "type": entity_category
                    }
                except Exception as e:
                    print(f"Error parsing subject entity definition: {e}")
                    
            # Check if it is an attribute definition
            elif 'attribute<|' in section or '"attribute"<|' in section:
                try:
                    # Handle square bracket or parenthesis format
                    content = section.strip('[]()').strip()
                    parts = [p.strip().strip('"').strip('>').strip('"') for p in content.split('<|')]
                    attr_type = parts[0]  # Should be "attribute"
                    attr_name = parts[1]
                    attr_value = parts[2]
                    
                    # Create attribute entry (query field removed)
                    attribute = {
                        "name": attr_name,
                        "value": attr_value
                    }
                    attributes.append(attribute)
                except Exception as e:
                    print(f"Error parsing attribute definition: {e}")
        
        # Add attributes to the result
        if attributes:
            result["attributes"] = attributes
        
        return result
    
    def get_entity_and_attributes(self, document):
        """Get subject entity and comparable attributes from the document (simplified interface)
        
        Args:
            document: Document dictionary containing 'contents' field
            
        Returns:
            tuple: (subject_entity, list_of_attributes)
        """
        entity_data = self.extract_entities(document)
        
        if not entity_data or "subject_entity" not in entity_data:
            return None, []
            
        subject_entity = entity_data["subject_entity"]
        attributes = entity_data.get("attributes", [])
        
        return subject_entity, attributes


if __name__ == "__main__":
    # Module functionality test
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Initialize DocumentReader, using corpus_path from config
    doc_reader = DocumentReader(config['corpus_path'])
    
    # Initialize compare entity extractor
    extractor = CompareEntityExtractor(config)
    
    # Set test mode
    # Set batch_mode to True to enable batch testing, False for single document testing
    batch_mode = True
    # Number of documents for batch testing
    batch_count = 5
    # Output JSON file path
    output_file = "./extracted_compare_entities.json"
    
    # Batch test mode
    if batch_mode:
        print(f"\nStarting batch test, will process {batch_count} documents...")
        all_entity_data = []
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
            print("Starting to extract subject entity and attribute information...")
            entity_data = extractor.extract_entities(sample_document)
            
            if entity_data and "subject_entity" in entity_data:
                print(f"Successfully extracted subject entity and {len(entity_data.get('attributes', []))} attributes")
                all_entity_data.append(entity_data)
                
                # Save original document (only keep necessary fields to reduce file size)
                doc_to_save = {
                    "id": sample_document.get("id", "unknown"),
                    "title": sample_document.get("title", "unknown"),
                    "contents": sample_document.get("contents", "")
                }
                all_documents.append(doc_to_save)
                
                processed_docs += 1
                
                # Display brief information of the subject entity and the first attribute
                if "subject_entity" in entity_data:
                    subject = entity_data["subject_entity"]
                    print(f"Subject Entity: {subject['name']} (Type: {subject['type']}, Document ID: {entity_data['document_id']})")
                    
                    attributes = entity_data.get("attributes", [])
                    if attributes:
                        attr = attributes[0]
                        print(f"Sample Attribute: {attr['name']} = {attr['value']}")
            else:
                print("Subject entity and attributes not extracted, skipping this document")
        
        # Save results to JSON file
        if all_entity_data:
            # Create a result dictionary containing entity data and original document
            result = {
                "entity_data": all_entity_data,
                "documents": all_documents
            }
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, indent=2, ensure_ascii=False, fp=f)
            
            print(f"\nBatch test completed! Processed {processed_docs} documents in total")
            print(f"Results saved to: {output_file}")
        else:
            print(f"\nBatch test completed, but no entity data was extracted")
    
    # Single document test mode
    else:
        # Maximum number of retries for the retry mechanism
        max_retry = 3
        current_retry = 0
        entity_data = {}
        
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
                print(f"\nCannot get document from knowledge base, using fallback example... (Attempt: {current_retry + 1}/{max_retry})")
                sample_document = example_document  # Use predefined sample document
            
            # Extract entities
            print("\nStarting to extract subject entity and attribute information...")
            entity_data = extractor.extract_entities(sample_document)
            
            # Check if entities were extracted
            if entity_data and "subject_entity" in entity_data:
                print(f"Successfully extracted subject entity and {len(entity_data.get('attributes', []))} attributes")
                break
            else:
                print(f"Subject entity and attributes not extracted, retrying attempt {current_retry + 1}")
                current_retry += 1
        
        # Output extraction results
        if entity_data and "subject_entity" in entity_data:
            # Format output results
            print("\nExtracted subject entity and attribute information:")
            subject = entity_data["subject_entity"]
            print(f"\nSubject Entity: {subject['name']} (Type: {subject['type']}, Document ID: {entity_data['document_id']})")
            
            attributes = entity_data.get("attributes", [])
            if attributes:
                print("\nAttributes List:")
                for i, attr in enumerate(attributes, 1):
                    print(f"\n{i}. Attribute: {attr['name']}")
                    print(f"   Value: {attr['value']}")
            else:
                print("\nNo comparable attributes found")
            
            # Display full original document content for reference
            print("\nOriginal document content:")
            print(sample_document['contents'])
        else:
            print("\nFailed to extract any subject entity and attributes")
