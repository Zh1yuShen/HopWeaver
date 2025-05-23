import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
import json
from hopweaver.components.utils.prompts import COMPARE_ENTITY_FILTER_PROMPT

class CompareEntityFilter:
    """Comparison Entity Filter: Evaluates entity specificity and attribute value comparability"""
    def __init__(self, config, filter_model=None):
        self.config = config
        # If filter_model is specified, use it as the fixed filtering model
        # Otherwise use the entity_extractor_model from the configuration

        self.config["generator_model"] = self.config["filter_model"]
        print(f"Using model from configuration file: {self.config['generator_model']}")
        
        self.config["generation_params"]["max_tokens"] = 4096
        # Initialize generator
        self.generator = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize OpenAI generator"""
        # Use OpenaiGenerator from FlashRAG
        return OpenaiGenerator(self.config)
    
    def filter_entity(self, entity_data, max_retries=3):
        """Evaluate entity specificity and attribute value comparability
        
        Args:
            entity_data: Dictionary containing subject entity and attribute information
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: Dictionary containing entity and attribute scores
        """
        if not entity_data or "subject_entity" not in entity_data:
            print("Entity data is empty or does not contain subject entity")
            return {}
            
        document_id = entity_data.get('document_id', 'unknown_id')
        subject_entity = entity_data.get('subject_entity', {})
        subject_entity_name = subject_entity.get('name', '')
        subject_entity_type = subject_entity.get('type', '')
        attributes = entity_data.get('attributes', [])
        
        # Format attribute list for prompt use
        attributes_list = ""
        for i, attr in enumerate(attributes, 1):
            attributes_list += f"- Attribute Name: {attr.get('name', '')}, Attribute Value: {attr.get('value', '')}\n"
        
        # Get original document content
        document_content = entity_data.get("contents", "")
        document_title = entity_data.get("title", "")
        
        # Add document title as context information (if available)
        if document_title:
            document_content = f"Title: {document_title}\n\n{document_content}"
            
        # Use prompt template and format
        # Check formatting parameters
        print(f"Formatting parameters:")
        print(f"  subject_entity_name: {subject_entity_name}")
        print(f"  subject_entity_type: {subject_entity_type}")
        print(f"  attributes_list: {attributes_list}")
        # Don't print full document content to avoid large logs
        print(f"  document_content: [Document content included]")
        
        # Format prompt with correct parameters
        prompt = COMPARE_ENTITY_FILTER_PROMPT.format(
            subject_entity_name=subject_entity_name,
            subject_entity_type=subject_entity_type,
            attributes_list=attributes_list,
            document_content=document_content
        )
        
        # Create message format
        messages = [{"role": "user", "content": prompt}]
        
        # Add retry mechanism
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Entity evaluation retry {attempt + 1}...")
                
                response = self.generator.generate([messages])
                
                # Check if response is valid
                if not response or response[0] is None:
                    print(f"Warning: Generator returned empty response (attempt {attempt+1}/{max_retries})")
                    continue
                    
                # Get document ID
                document_id = entity_data.get("document_id", "unknown")
                
                # Parse response, pass document_id parameter
                filter_result = self._parse_filter_response(response[0], document_id)
                
                # If scores were successfully extracted, break retry loop
                if filter_result and "entity_score" in filter_result:
                    # Add entity information
                    filter_result["entity_score"]["entity_name"] = subject_entity_name
                    filter_result["entity_score"]["entity_type"] = subject_entity_type
                    # 添加原始文档内容
                    if "contents" in entity_data and entity_data["contents"]:
                        filter_result["document_content"] = entity_data["contents"]
                    # 添加文档标题
                    if "title" in entity_data and entity_data["title"]:
                        filter_result["document_title"] = entity_data["title"]
                    return filter_result
                else:
                    print(f"Failed to extract score data from response (attempt {attempt+1}/{max_retries})")
                    
            except Exception as e:
                print(f"Entity evaluation error (attempt {attempt+1}/{max_retries}): {str(e)}")
        
        # All retries failed
        print(f"Entity evaluation failed after {max_retries} attempts")
        result = {
            "entity_score": {
                "document_id": document_id,
                "entity_name": subject_entity_name,
                "entity_type": subject_entity_type,
                "concreteness_score": 0
            },
            "attribute_scores": []
        }
        
        # Add original document content (even if evaluation failed)
        if "contents" in entity_data and entity_data["contents"]:
            result["document_content"] = entity_data["contents"]
        # 添加文档标题
        if "title" in entity_data and entity_data["title"]:
            result["document_title"] = entity_data["title"]
            
        return result
    
    def _parse_filter_response(self, response, document_id="unknown"):
        """Parse LLM response, extract scoring information
        
        Args:
            response: LLM response text
            document_id: Current document ID being processed
            
        Returns:
            dict: Dictionary containing entity and attribute scores
        """
        # Note: Entity name and type are handled by the calling method filter_entity
        # Here we only parse score values, not entity detail information
        # Print raw response for debugging
        print(f"Raw response:\n{response}\n")
        
        # Initialize result data structure
        result = {
            "entity_score": {
                "document_id": document_id,
                "entity_name": "unknown",
                "entity_type": "unknown",
                "concreteness_score": 0
            },
            "attribute_scores": []
        }
        
        # Clean response text, remove code block markers and completion tokens
        response = response.replace("<|COMPLETE|>", "").strip()
        
        # If response is wrapped in quotes, remove outer quotes
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1].strip()
        
        # Try to parse format starting with number followed by ##, e.g.: 3 ## ("Date of Birth"<|>"4 October 1832"<|>5)
        if '##' in response:
            try:
                # 分割每个部分
                parts = response.split('##')
                first_part = parts[0].strip()
                
                # If the first part is a number, use it as entity score
                if first_part.isdigit():
                    concreteness_score = int(first_part)
                    result["entity_score"]["concreteness_score"] = concreteness_score
                    print(f"Successfully parsed entity score (number format): document ID={document_id}, specificity score={concreteness_score}")
                    
                    # Process attribute scores in the following parts
                    for attr_part in parts[1:]: 
                        attr_part = attr_part.strip()
                        if '<|>' in attr_part:
                            # Parse attribute information ("attribute name"<|>"attribute value"<|>score)
                            attr_segments = attr_part.split('<|>')
                            if len(attr_segments) >= 3:
                                attr_name = attr_segments[0].strip().strip('(').strip('"')
                                attr_value = attr_segments[1].strip().strip('"')
                                score_str = attr_segments[2].strip().rstrip(')').strip('"')
                                if score_str.isdigit():
                                    comparability_score = int(score_str)
                                    attribute_score = {
                                        "name": attr_name,
                                        "value": attr_value,
                                        "comparability_score": comparability_score
                                    }
                                    result["attribute_scores"].append(attribute_score)
                                    print(f"Successfully parsed attribute score (number format): name={attr_name}, value={attr_value}, comparability score={comparability_score}")
                
                # If entity score and attribute scores were successfully parsed, return result
                if result["entity_score"]["concreteness_score"] > 0:
                    return result
            except Exception as e:
                print(f"Error parsing number format: {e}")
        
        # Try to parse new format ("entity_score"<|>5) ## ("attribute_score"<|>"Name"<|>"Value"<|>5)
        if '<|>' in response and '##' in response:
            try:
                # 分割每个部分
                parts = response.split('##')
                
                # Process entity score part
                entity_part = parts[0].strip()
                if '"entity_score"<|>' in entity_part:
                    # Extract entity score
                    score_str = entity_part.split('<|>')[1].strip().rstrip(')')
                    if score_str.isdigit():
                        concreteness_score = int(score_str)
                        result["entity_score"]["concreteness_score"] = concreteness_score
                        print(f"Successfully parsed entity score (new format): document ID={document_id}, specificity score={concreteness_score}")
                
                # Process attribute score parts
                for attr_part in parts[1:]:  # 跳过第一个实体部分
                    attr_part = attr_part.strip()
                    if '"attribute_score"<|>' in attr_part:
                        # Parse attribute information ("attribute_score"<|>"Name"<|>"Value"<|>5)
                        attr_segments = attr_part.split('<|>')
                        if len(attr_segments) >= 4:
                            attr_name = attr_segments[1].strip('"')
                            attr_value = attr_segments[2].strip('"')
                            score_str = attr_segments[3].strip().rstrip(')')
                            if score_str.isdigit():
                                comparability_score = int(score_str)
                                attribute_score = {
                                    "name": attr_name,
                                    "value": attr_value,
                                    "comparability_score": comparability_score
                                }
                                result["attribute_scores"].append(attribute_score)
                                print(f"Successfully parsed attribute score (new format): name={attr_name}, value={attr_value}, comparability score={comparability_score}")
            except Exception as e:
                print(f"Error parsing new format score: {e}")
        
        # Try to parse simple separator format like "5|5|5|5"
        elif '|' in response and not '{' in response:
            simple_format = response.strip().split('\n')[0].strip()
            parts = simple_format.split('|')
            if len(parts) > 0 and parts[0].isdigit():
                try:
                    concreteness_score = int(parts[0])
                    result["entity_score"]["concreteness_score"] = concreteness_score
                    print(f"Successfully parsed entity score (simple format): document ID={document_id}, specificity score={concreteness_score}")
                    
                    # If there are attribute scores
                    if len(parts) > 1:
                        for i, score in enumerate(parts[1:], 0):
                            if score.isdigit():
                                comparability_score = int(score)
                                attribute_name = f"Attribute {i+1}" if i < len(attributes) else f"Unknown Attribute {i+1}"
                                attribute_value = ""
                                if i < len(attributes):
                                    attribute_name = attributes[i].get('name', f"Attribute {i+1}")
                                    attribute_value = attributes[i].get('value', "")
                                
                                attribute_score = {
                                    "name": attribute_name,
                                    "value": attribute_value,
                                    "comparability_score": comparability_score
                                }
                                result["attribute_scores"].append(attribute_score)
                                print(f"Successfully parsed attribute score (simple format): name={attribute_name}, comparability score={comparability_score}")
                except Exception as e:
                    print(f"Error parsing simple format score: {e}")
                    
        
        # Process code block format
        if response.startswith("```") and "```" in response:
            # Remove language markers at the beginning, such as ```json or ```text
            first_line_end = response.find("\n")
            if first_line_end > 3:  # If the first line contains language marker
                response = response[first_line_end:].strip()
            
            # Remove remaining code block markers
            response = response.replace("```", "").strip()
        
        # Remove other possible markers
        response = response.replace("Output:", "").strip()
        
        # Check if it's JSON format
        if response.strip().startswith('{') and response.strip().endswith('}'): 
            try:
                import json
                json_data = json.loads(response)
                
                # Extract entity score
                if "concreteness_score" in json_data:
                    concreteness_score = int(json_data["concreteness_score"])
                    result["entity_score"]["concreteness_score"] = concreteness_score
                    print(f"Successfully parsed entity score (JSON): document ID={document_id}, specificity score={concreteness_score}")
                
                # Extract attribute scores
                if "attribute_scores" in json_data and isinstance(json_data["attribute_scores"], list):
                    for attr in json_data["attribute_scores"]:
                        if "attribute_name" in attr and "comparability_score" in attr:
                            attribute_score = {
                                "name": attr["attribute_name"],
                                "value": attr.get("attribute_value", ""),
                                "comparability_score": int(attr["comparability_score"])
                            }
                            result["attribute_scores"].append(attribute_score)
                            print(f"Successfully parsed attribute score (JSON): name={attribute_score['name']}, value={attribute_score['value']}, comparability score={attribute_score['comparability_score']}")
                
                # If JSON was successfully parsed, return result directly
                if result["entity_score"]["concreteness_score"] > 0 or result["attribute_scores"]:
                    return result
            except Exception as e:
                print(f"JSON parsing failed: {e}, trying other formats")
        
        # Try parsing line by line (new format)
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Other parsing logic remains unchanged...
            
        return result
        
    def filter_high_score_entities_and_attributes(self, filtered_entity, entity_data, 
                                               entity_threshold=4, attribute_threshold=4):
        """
        Filter out high-scoring entities and attributes
        
        Args:
            filtered_entity: Dictionary containing entity and attribute scores (result from filter_entity method)
            entity_data: Original entity data (containing all attribute information)
            entity_threshold: Entity score threshold, default is 4
            attribute_threshold: Attribute score threshold, default is 4
            
        Returns:
            dict: Dictionary containing entity scores, high-scoring attributes, and original document information; returns None if nothing passes filtering
        """
        if not filtered_entity or "entity_score" not in filtered_entity:
            print("Entity score data invalid")
            return None
            
        # 检查实体评分
        entity_score = filtered_entity.get("entity_score", {})
        entity_concreteness = entity_score.get("concreteness_score", 0)
        
        if entity_concreteness < entity_threshold:
            print(f"Entity specificity score ({entity_concreteness}) below threshold ({entity_threshold})")
            return None
            
        # Filter high-scoring attributes
        attribute_scores = filtered_entity.get("attribute_scores", [])
        high_score_attributes = []
        
        for attr_score in attribute_scores:
            name = attr_score.get("name")
            score = attr_score.get("comparability_score", 0)
            
            if score >= attribute_threshold:  # 属性评分阈值
                for attr in entity_data.get("attributes", []):
                    if attr.get("name") == name:
                        high_score_attributes.append(attr)
                        break
        
        if not high_score_attributes:
            print(f"No comparable attributes found with scores meeting threshold ({attribute_threshold})")
            return None
            
        # Build result
        doc_id = entity_data.get("document_id", "unknown")
        result = {
            "document_id": doc_id,
            "contents": entity_data.get("contents", ""),
            "title": entity_data.get("title", ""),
            "subject_entity": entity_data.get("subject_entity", {}),
            "attributes": high_score_attributes,
            "entity_score": entity_score,
            "attribute_scores": attribute_scores
        }
        
        return result


def filter_high_quality_entities(input_file, output_file, entity_score_threshold=5, attribute_score_threshold=5):
    """
    Filter high-quality entities and attributes, only keeping content where both entity scores and attribute scores meet thresholds
    
    Args:
        input_file (str): Input JSON file path (result processed by compare_entity_filter)
        output_file (str): Output JSON file path
        entity_score_threshold (int): Entity specificity score threshold, default is 5
        attribute_score_threshold (int): Attribute comparability score threshold, default is 5
    """
    print(f"\nReading filtered entity file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        filtered_entities = data.get("filtered_entities", [])
        documents = data.get("documents", [])
        
        if not filtered_entities:
            print("No filtered entity data found in input file")
            return False
            
        print(f"Read {len(filtered_entities)} filtered entities")
        
        # Only keep high-quality entities and their high-quality attributes
        high_quality_entities = []
        retained_entity_count = 0
        
        for entity_entry in filtered_entities:
            entity_data = entity_entry.get("entity_data", {})
            filter_result = entity_entry.get("filter_result", {})
            
            # 检查实体评分
            entity_score = filter_result.get("entity_score", {})
            concreteness_score = entity_score.get("concreteness_score", 0)
            
            if concreteness_score < entity_score_threshold:
                print(f"Skipping entity: {entity_score.get('entity_name', 'unknown')} - score {concreteness_score} < {entity_score_threshold}")
                continue
                
            # For high-quality entities, only keep high-quality attributes
            attribute_scores = filter_result.get("attribute_scores", [])
            high_quality_attributes = []
            
            for attr in attribute_scores:
                comparability_score = attr.get("comparability_score", 0)
                if comparability_score >= attribute_score_threshold:
                    # 找到原始属性数据
                    attr_name = attr.get("name", "").replace("{", "").replace("}", "")
                    original_attrs = []
                    
                    for orig_attr in entity_data.get("attributes", []):
                        if orig_attr.get("name", "") == attr_name or orig_attr.get("name", "") == attr.get("name", ""):
                            original_attrs.append(orig_attr)
                    
                    # 添加高质量属性
                    high_quality_attributes.extend(original_attrs)
                else:
                    print(f"  Skipping attribute: {attr.get('name', 'unknown')} - score {comparability_score} < {attribute_score_threshold}")
            
            # If there are high-quality attributes, keep this entity
            if high_quality_attributes:
                # Create new entity data structure
                new_entity_data = {
                    "subject_entity": entity_data.get("subject_entity", {}),
                    "attributes": high_quality_attributes,
                    "document_id": entity_data.get("document_id", "")
                }
                
                # If document content needs to be kept, it can be retrieved from filter_result
                if "document_content" in filter_result:
                    new_entity_data["contents"] = filter_result.get("document_content", "")
                if "document_title" in filter_result:
                    new_entity_data["title"] = filter_result.get("document_title", "")
                
                high_quality_entities.append(new_entity_data)
                retained_entity_count += 1
                print(f"Keeping entity: {entity_score.get('entity_name', 'unknown')} - specificity score: {concreteness_score}/5, kept {len(high_quality_attributes)} high-quality attributes")
        
        # Create output data structure
        output_data = {
            "entity_data": high_quality_entities,
            "documents": documents  # 保留原始文档列表
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"\nHigh-quality filtering complete! Kept {retained_entity_count} high-quality entities out of {len(filtered_entities)}")
        print(f"结果已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return False

if __name__ == "__main__":
    # Module functionality test
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Initialize comparison entity filter
    filter_tool = CompareEntityFilter(config)
    
    # Input JSON file path (from results generated by comparison entity extractor)
    input_file = "./extracted_compare_entities.json"
    
    # Output JSON file path
    output_file = "./filtered_compare_entities.json"
    
    try:
        # Read extracted entity data
        print(f"\nReading file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entity_data_list = data.get("entity_data", [])
        documents = data.get("documents", [])
        
        if not entity_data_list:
            print("No entity data found in input file")
            sys.exit(1)
            
        print(f"Successfully read {len(entity_data_list)} entity data entries")
        
        # Process each entity data
        filtered_results = []
        
        for i, entity_data in enumerate(entity_data_list, 1):
            print(f"\nProcessing entity {i}/{len(entity_data_list)}")
            
            if "subject_entity" not in entity_data:
                print("  Skipping: Missing subject entity information")
                continue
                
            subject_entity = entity_data["subject_entity"]
            document_id = entity_data.get("document_id", "unknown")
            
            print(f"  Subject entity: {subject_entity.get('name', '')} (type: {subject_entity.get('type', '')})")
            print(f"  Document ID: {document_id}")
            print(f"  Number of attributes: {len(entity_data.get('attributes', []))}")
            
            # 评估实体
            print("  Evaluating entity...")
            filter_result = filter_tool.filter_entity(entity_data)
            
            if filter_result:
                # Merge original entity data and score results
                # If filter_result already contains document content, remove it from entity_data to avoid redundancy
                entity_data_copy = entity_data.copy()
                if "document_content" in filter_result:
                    # Remove content from entity_data to avoid duplication
                    if "contents" in entity_data_copy:
                        del entity_data_copy["contents"]
                
                filtered_entity = {
                    "entity_data": entity_data_copy,
                    "filter_result": filter_result
                }
                filtered_results.append(filtered_entity)
                
                # Print score result summary
                entity_score = filter_result.get("entity_score", {})
                attribute_scores = filter_result.get("attribute_scores", [])
                
                concreteness_score = entity_score.get('concreteness_score', 'unknown')
                print(f"  Entity specificity score: {concreteness_score}/5")
                
                if attribute_scores:
                    avg_attr_score = sum(attr.get('comparability_score', 0) for attr in attribute_scores) / len(attribute_scores)
                    print(f"  Attribute comparability average score: {avg_attr_score:.1f}/5")
                else:
                    print("  No attribute scores")
            else:
                print("  Evaluation failed, skipping this entity")
        
        # Save results to JSON file
        if filtered_results:
            # Create output data containing original documents and filtered results
            output_data = {
                "filtered_entities": filtered_results,
                "documents": documents
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, indent=2, ensure_ascii=False, fp=f)
                
            print(f"\nFiltering complete! Successfully processed {len(filtered_results)}/{len(entity_data_list)} entities")
            print(f"结果已保存到: {output_file}")
            
            # Further perform high-quality entity filtering
            high_quality_output_file = output_file.replace('.json', '_high_quality.json')
            print("\nStarting high-quality entity filtering...")
            print("Only keeping entities with specificity score of 5, and their attributes with comparability score of 5")
            filter_high_quality_entities(output_file, high_quality_output_file)
        else:
            print("\nFiltering complete, but no entities were successfully evaluated")
            
    except Exception as e:
        import traceback
        print(f"Error during processing: {str(e)}")
        print(traceback.format_exc())
