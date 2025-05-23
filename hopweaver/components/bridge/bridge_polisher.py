import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
from hopweaver.components.utils.prompts import POLISHER_PROMPT

class Polisher:
    """Multi-hop question polisher to improve the quality of multi-hop questions, supporting PASS, ADJUST, REWORKED, and REJECTED statuses"""
    def __init__(self, config):
        self.config = config
        
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
        
    def _initialize_model(self):
        """Initialize OpenAI generator"""
        # Use FlashRAG's OpenaiGenerator
        return OpenaiGenerator(self.config)
        
    def polish_question(self, multi_hop_question, answer, reasoning_path, doc_a_seg, doc_b_seg, sub_question_1="", sub_question_2="", max_retry=3):
        """Refine multi-hop questions to improve quality
        
        Args:
            multi_hop_question (str): Original multi-hop question
            answer (str): Answer to the question
            reasoning_path (str): Reasoning path
            doc_a_seg (str): Document A segment
            doc_b_seg (str): Document B segment
            sub_question_1 (str, optional): Sub-question for Document A
            sub_question_2 (str, optional): Sub-question for Document B
            max_retry (int): Maximum retry attempts
            
        Returns:
            dict: Refinement result, including status and possible modified content
        """
        # Parameter validation
        if not multi_hop_question or not answer or not doc_a_seg or not doc_b_seg:
            print("Warning: Missing necessary input parameters")
            return {"status": "ERROR", "message": "Missing necessary input parameters"}
            
        # Prepare prompt template
        prompt = POLISHER_PROMPT.format(
            multi_hop_question=multi_hop_question,
            answer=answer,
            reasoning_path=reasoning_path if reasoning_path else "",
            sub_question_1=sub_question_1 if sub_question_1 else "No sub-question information",
            sub_question_2=sub_question_2 if sub_question_2 else "No sub-question information",
            doc_a_seg=doc_a_seg,
            doc_b_seg=doc_b_seg
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Implement retry mechanism
        for attempt in range(max_retry):
            try:
                if attempt > 0:
                    print(f"Retrying attempt {attempt + 1}...")
                
                # Call generator
                response = self.generator.generate([messages])
                
                if not response or response[0] is None:
                    print(f"Warning: Generator returned empty response (attempt {attempt+1}/{max_retry})")
                    continue
                
                # Print full response for debugging
                print("Full raw response:")
                print("-" * 80)
                print(response[0])
                print("-" * 80)
                
                # Parse response
                result = self._parse_polish_response(response[0])
                
                # Check if the result is valid
                if result and result.get('status') != 'ERROR':
                    return result
                elif result and result.get('status') == 'ERROR':
                    print(f"Parsing error: {result.get('message', '')}, will retry")
                    # If it's the last retry, return an error result
                    if attempt == max_retry - 1:
                        return result
                    
            except Exception as e:
                print(f"Error refining question (attempt {attempt+1}/{max_retry}): {str(e)}")
                
        # All retries failed
        return {"status": "ERROR", "message": f"Refinement failed after {max_retry} attempts"}
    
    def _parse_polish_response(self, response):
        """Parse refinement response, extract status and content
        
        Args:
            response (str): Generator's response
            
        Returns:
            dict: Contains status and possible modified content, or error message
        """
        try:
            # Remove possible completion markers and normalize line breaks
            response = response.replace("<|COMPLETE|>", "").replace("\r\n", "\n").strip()
            # Print processed response for debugging
            print("Preprocessed response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
            # Check standard format tags
            if "[PASS]" in response:
                return {"status": "PASS"}
                
            elif "[ADJUST]" in response:
                # Use regular expressions to extract fields
                import re
                print("Starting to parse ADJUST response...")
                
                # Preprocess response, remove extra spaces and tabs
                cleaned_response = re.sub(r'[ \t]+\n', '\n', response)  # Remove trailing spaces from lines
                cleaned_response = re.sub(r'\n[ \t]+', '\n', cleaned_response)  # Remove leading spaces from lines
                
                # Extract modified reasoning path using a regex that handles multi-line content
                # First, try using the original response
                reasoning_match = re.search(r'REFINED_REASONING_PATH:[ \t]*(.*(?:\n[ \t]*(?!REFINED_QUESTION:|REFINED_ANSWER:).*)*)', response, re.IGNORECASE | re.DOTALL)
                
                if not reasoning_match:
                    # Try using the cleaned response
                    reasoning_match = re.search(r'REFINED_REASONING_PATH:[ \t]*(.*(?:\n(?!REFINED_QUESTION:|REFINED_ANSWER:).*)*)', cleaned_response, re.IGNORECASE | re.DOTALL)
                
                if not reasoning_match:
                    # Try using a more lenient matching pattern to handle cases without a colon
                    reasoning_match = re.search(r'REFINED_REASONING_PATH[ \t]*:[ \t]*(.*(?:\n[ \t]*(?!REFINED_QUESTION|REFINED_ANSWER).*)*)', response, re.IGNORECASE | re.DOTALL)
                    
                if not reasoning_match:
                    print("Could not find REFINED_REASONING_PATH")
                    # Print partial response content for debugging
                    print(f"Response content snippet: {response[:200]}...")
                    return {"status": "ERROR", "message": "Could not find REFINED_REASONING_PATH"}
                refined_reasoning_path = reasoning_match.group(1).strip()
                
                # Extract modified question using a regex that handles multi-line content
                question_match = re.search(r'REFINED_QUESTION:[ \t]*(.*(?:\n[ \t]*(?!REFINED_ANSWER:).*)*?)(?=\n[ \t]*REFINED_ANSWER:|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                if not question_match:
                    # Try using the cleaned response
                    question_match = re.search(r'REFINED_QUESTION:[ \t]*(.*(?:\n(?!REFINED_ANSWER:).*)*?)(?=\n[ \t]*REFINED_ANSWER:|\n\n|$)', cleaned_response, re.IGNORECASE | re.DOTALL)
                if not question_match:
                    # Try using a more lenient matching pattern
                    question_match = re.search(r'REFINED_QUESTION[ \t]*:[ \t]*(.*(?:\n[ \t]*(?!REFINED_ANSWER).*)*?)(?=\n[ \t]*REFINED_ANSWER|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                
                if not question_match:
                    print("Could not find REFINED_QUESTION")
                    # Print partial response content for debugging
                    print(f"Response content snippet: {response[:200]}...")
                    return {"status": "ERROR", "message": "Could not find REFINED_QUESTION"}
                refined_question = question_match.group(1).strip()
                
                # Ensure the question does not contain REFINED_ANSWER
                if "REFINED_ANSWER:" in refined_question:
                    refined_question = refined_question.split("REFINED_ANSWER:")[0].strip()
                
                # Extract answer using a regex that handles multi-line content
                answer_match = re.search(r'REFINED_ANSWER:[ \t]*(.*(?:\n[ \t]*(?!\n|\[).*)*?)(?=\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                if not answer_match:
                    # Try using the cleaned response
                    answer_match = re.search(r'REFINED_ANSWER:[ \t]*(.*(?:\n(?!\n|\[).*)*?)(?=\n\n|$)', cleaned_response, re.IGNORECASE | re.DOTALL)
                if not answer_match:
                    # Try using a more lenient matching pattern
                    answer_match = re.search(r'REFINED_ANSWER[ \t]*:[ \t]*(.*(?:\n[ \t]*(?!\n|\[).*)*?)(?=\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                
                if not answer_match:
                    print("Could not find REFINED_ANSWER")
                    # Print partial response content for debugging
                    print(f"Response content snippet: {response[:200]}...")
                    return {"status": "ERROR", "message": "Could not find REFINED_ANSWER"}
                answer = answer_match.group(1).strip()
                
                print(f"Successfully extracted ADJUST fields:")
                print(f"- Reasoning path: {refined_reasoning_path[:50]}...")
                print(f"- Question: {refined_question}")
                print(f"- Answer: {answer}")
                
                return {
                    "status": "ADJUST",
                    "refined_reasoning_path": refined_reasoning_path,
                    "refined_question": refined_question,
                    "answer": answer
                }
                    
            elif "[REWORKED]" in response:
                # Use regular expressions to extract fields
                import re
                print("Starting to parse REWORKED response...")
                
                # Preprocess response, remove extra spaces and tabs
                cleaned_response = re.sub(r'[ \t]+\n', '\n', response)  # Remove trailing spaces from lines
                cleaned_response = re.sub(r'\n[ \t]+', '\n', cleaned_response)  # Remove leading spaces from lines
                
                # Extract modified reasoning path using a regex that handles multi-line content
                # First, try using the original response
                reasoning_match = re.search(r'REFINED_REASONING_PATH:[ \t]*(.*(?:\n[ \t]*(?!REFINED_QUESTION:|REFINED_ANSWER:).*)*)', response, re.IGNORECASE | re.DOTALL)
                
                if not reasoning_match:
                    # Try using the cleaned response
                    reasoning_match = re.search(r'REFINED_REASONING_PATH:[ \t]*(.*(?:\n(?!REFINED_QUESTION:|REFINED_ANSWER:).*)*)', cleaned_response, re.IGNORECASE | re.DOTALL)
                
                if not reasoning_match:
                    # Try using a more lenient matching pattern to handle cases without a colon
                    reasoning_match = re.search(r'REFINED_REASONING_PATH[ \t]*:[ \t]*(.*(?:\n[ \t]*(?!REFINED_QUESTION|REFINED_ANSWER).*)*)', response, re.IGNORECASE | re.DOTALL)
                    
                if not reasoning_match:
                    print("Could not find REFINED_REASONING_PATH")
                    # Print partial response content for debugging
                    print(f"Response content snippet: {response[:200]}...")
                    return {"status": "ERROR", "message": "Could not find REFINED_REASONING_PATH"}
                refined_reasoning_path = reasoning_match.group(1).strip()
                
                # Extract modified question using a regex that handles multi-line content
                question_match = re.search(r'REFINED_QUESTION:[ \t]*(.*(?:\n[ \t]*(?!REFINED_ANSWER:).*)*?)(?=\n[ \t]*REFINED_ANSWER:|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                if not question_match:
                    # Try using the cleaned response
                    question_match = re.search(r'REFINED_QUESTION:[ \t]*(.*(?:\n(?!REFINED_ANSWER:).*)*?)(?=\n[ \t]*REFINED_ANSWER:|\n\n|$)', cleaned_response, re.IGNORECASE | re.DOTALL)
                if not question_match:
                    # Try using a more lenient matching pattern
                    question_match = re.search(r'REFINED_QUESTION[ \t]*:[ \t]*(.*(?:\n[ \t]*(?!REFINED_ANSWER).*)*?)(?=\n[ \t]*REFINED_ANSWER|\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                
                if not question_match:
                    print("Could not find REFINED_QUESTION")
                    # Print partial response content for debugging
                    print(f"Response content snippet: {response[:200]}...")
                    return {"status": "ERROR", "message": "Could not find REFINED_QUESTION"}
                refined_question = question_match.group(1).strip()
                
                # Ensure the question does not contain REFINED_ANSWER
                if "REFINED_ANSWER:" in refined_question:
                    refined_question = refined_question.split("REFINED_ANSWER:")[0].strip()
                
                # Extract answer using a regex that handles multi-line content
                answer_match = re.search(r'REFINED_ANSWER:[ \t]*(.*(?:\n[ \t]*(?!\n|\[).*)*?)(?=\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                if not answer_match:
                    # Try using the cleaned response
                    answer_match = re.search(r'REFINED_ANSWER:[ \t]*(.*(?:\n(?!\n|\[).*)*?)(?=\n\n|$)', cleaned_response, re.IGNORECASE | re.DOTALL)
                if not answer_match:
                    # Try using a more lenient matching pattern
                    answer_match = re.search(r'REFINED_ANSWER[ \t]*:[ \t]*(.*(?:\n[ \t]*(?!\n|\[).*)*?)(?=\n\n|$)', response, re.IGNORECASE | re.DOTALL)
                
                if not answer_match:
                    print("Could not find REFINED_ANSWER")
                    # Print partial response content for debugging
                    print(f"Response content snippet: {response[:200]}...")
                    return {"status": "ERROR", "message": "Could not find REFINED_ANSWER"}
                answer = answer_match.group(1).strip()
                
                print(f"Successfully extracted REWORKED fields:")
                print(f"- Reasoning path: {refined_reasoning_path[:50]}...")
                print(f"- Question: {refined_question}")
                print(f"- Answer: {answer}")
                
                return {
                    "status": "REWORKED",
                    "refined_reasoning_path": refined_reasoning_path,
                    "refined_question": refined_question,
                    "answer": answer
                }
                
            elif "[REJECTED]" in response:
                return {"status": "REJECTED"}
            
            else:
                # If there are no standard format tags, return an error
                return {"status": "ERROR", "message": "Response format does not meet expectations, missing standard tags"}
        
        except Exception as e:
            print(f"Error parsing refinement response: {str(e)}")
            return {"status": "ERROR", "message": f"Parsing error: {str(e)}"}


if __name__ == "__main__":
    # Test code
    
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Create polisher
    polisher = Polisher(config)
    
    # Test input
    test_question = "Thorborg Rappe was a pioneer in Swedish intellectual disability education. In what Swedish city did another pioneer in this area establish their first institution?"
    test_answer = "Gothenburg"
    test_reasoning = "1. Document A mentions Emanuella Carlbeck as a pioneer in the education of students with intellectual disabilities in Sweden, alongside Thorborg Rappe.\n2. Document B reveals that Emanuella Carlbeck founded the first institution for people with intellectual disabilities in Gothenburg in 1866.\n3. By connecting the information from both documents, we understand that the \"another pioneer\" mentioned in the multi-hop question refers to Emanuella Carlbeck, and Document B provides the information that she founded her first institution in Gothenburg."
    test_doc_a = """Alongside Emanuella Carlbeck, Thorborg Rappe is counted as a pioneer in the education of students with Intellectual disability in Sweden."""
    test_doc_b = """Emanuella Carlbeck Emanuella Ottiliana Carlbeck (24 August 1829 – 10 September 1901), was a Swedish pedagogue. She is counted as a pioneer in the education of students with Intellectual disability. She founded the first institution for people with Intellectual disability in Gothenburg in 1866."""
    test_sub_question_1 = "Who is mentioned alongside Thorborg Rappe as a pioneer in Swedish intellectual disability education?\nAnswer: Emanuella Carlbeck"
    test_sub_question_2 = "Where did Emanuella Carlbeck found her first institution?\nAnswer: Gothenburg, Sweden"
    
    # Test polishing
    result = polisher.polish_question(test_question, test_answer, test_reasoning, test_doc_a, test_doc_b, test_sub_question_1, test_sub_question_2)
    
    # Print results
    print("Polishing results:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
