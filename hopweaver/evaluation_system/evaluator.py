#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Multi-hop Question Quality Evaluator
'''
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
from hopweaver.components.utils.prompts import MHQA_QUALITY_ASSESSMENT_PROMPT, COMPARE_QA_QUALITY_ASSESSMENT_PROMPT


class QualityEvaluator:
    """Multi-hop Question Quality Evaluator"""
    def __init__(self, config):
        self.config = config
        
        # Set model configuration
        if "evaluator_model" in self.config:
            self.config["generator_model"] = self.config["evaluator_model"]
        elif "generator_model" not in self.config:
            self.config["generator_model"] = "gpt-4o"
            
        # Set maximum tokens
        if "generation_params" not in self.config:
            self.config["generation_params"] = {}
        self.config["generation_params"]["max_tokens"] = 1024
        
        # Add necessary batch processing parameters
        if "generator_batch_size" not in self.config:
            self.config["generator_batch_size"] = 1
        
        # Initialize generator
        self.generator = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize OpenAI generator"""
        # Use FlashRAG's OpenaiGenerator
        return OpenaiGenerator(self.config)
        
    def evaluate_question(self, question_data, max_retry=3):
        """Evaluate the quality of multi-hop questions
        
        Args:
            question_data: Dictionary containing the question, answer, and related documents
            max_retry: Maximum number of retries
            
        Returns:
            dict: Dictionary containing the evaluation results
        """
        # Parameter validation - must have at least question, answer, and two documents
        required_fields = ['question', 'answer', 'document1', 'document2']
        if not all(field in question_data for field in required_fields):
            print(f"Warning: Input data is missing necessary fields: {required_fields}")
            return None
            
        # Dynamically select prompt based on question type
        if question_data.get('question_type') == 'comparison':
            prompt_template = COMPARE_QA_QUALITY_ASSESSMENT_PROMPT
            print("  Using comparison question evaluation template")
        else:
            prompt_template = MHQA_QUALITY_ASSESSMENT_PROMPT
            print("  Using default multi-hop question evaluation template")
            
        # Custom prompt template to resolve all formatting issues
        # Handle {yes/no} formatting issues
        prompt_template = prompt_template.replace("{yes/no}", "yes or no")
        # Handle {rating} formatting issues
        prompt_template = prompt_template.replace("{rating}", "rating")
        
        # Output formatting parameters in the template
        import re
        format_params = re.findall(r'\{([^}]+)\}', prompt_template)
        print(f"  Formatting parameters: {format_params}")
        
        # Prepare formatting parameter dictionary
        format_args = {
            'question': question_data['question'],
            'answer': question_data['answer']
        }
        
        # Dynamically add all documents
        doc_count = 1
        while f'document{doc_count}' in question_data:
            format_args[f'document{doc_count}'] = question_data[f'document{doc_count}']
            doc_count += 1
            
        # Add empty strings for unprovided documents to avoid formatting errors
        for i in range(doc_count, 11):  # Supports up to 10 documents
            format_args[f'document{i}'] = ""  # Only add empty strings, no extra text
            
        # Format using the modified template
        prompt = prompt_template.format(**format_args)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Implement retry mechanism
        for attempt in range(max_retry):
            try:
                # Generate evaluation results
                response = self.generator.generate([messages])
                
                # Debug output, display original response
                print(f"\n=== Original response from model ===\nType: {type(response)}\nContent: {response}\n==========================\n")
                print(response)
                # Handle list return value
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = response
                    
                # Ensure response is a string
                if response_text:
                    response_text = str(response_text).strip()
                    result = self._parse_evaluation_result(response_text, question_data)
                    
                    # Check for missing rating dimensions
                    if result:
                        # Rating dimensions to check
                        required_dimensions = ['fluency', 'clarity', 'conciseness', 'relevance', 
                                             'consistency', 'question_answerability', 
                                             'answer_question_consistency', 'information_integration_ability',
                                             'reasoning_path_guidance', 'logical_sophistication']
                        
                        # Check for missing dimensions
                        missing_dimensions = [dim for dim in required_dimensions 
                                            if dim not in result['evaluation'] or not result['evaluation'][dim]]
                        
                        if missing_dimensions and attempt < max_retry - 1:
                            print(f"Warning: Missing rating dimensions: {missing_dimensions}, will retry... (Attempt {attempt+1}/{max_retry})")
                            # If there are missing dimensions and retries are left, continue retrying
                            continue
                        elif missing_dimensions:
                            print(f"Warning: Missing rating dimensions: {missing_dimensions}, but max retries reached, will use incomplete results")
                    
                    return result
                    
            except Exception as e:
                print(f"Evaluation attempt {attempt+1}/{max_retry} failed: {str(e)}")  
                import traceback
                traceback.print_exc()
                
            if attempt < max_retry - 1:
                print(f"Retrying... ({attempt+1}/{max_retry})")
                
        print(f"Warning: Failed to get valid evaluation results after {max_retry} attempts")
        return None
        
    def _parse_evaluation_result(self, response, question_data):
        """Parse evaluation results
        
        Args:
            response: Evaluation result text returned by the model
            question_data: Original question data
            
        Returns:
            dict: Dictionary containing parsed evaluation results
        """
        try:
            print(f"\n=== Starting to parse evaluation results ===\nOriginal response:\n{response}\n")
            
            # Initialize results dictionary, including original question data
            result = {
                'id': question_data.get('id', ''),
                'question': question_data['question'],
                'answer': question_data['answer'],
                'evaluation': {},
                'raw_response': response  # Store original response for debugging
            }
            
            # Dynamically add all documents
            doc_count = 1
            while f'document{doc_count}' in question_data:
                result[f'document{doc_count}'] = question_data[f'document{doc_count}']
                doc_count += 1
            
            # Remove <|COMPLETE|> marker
            response = response.replace('<|COMPLETE|>', '').strip()
            
            # Parse evaluation results
            lines = response.strip().split('\n')
            print(f"\nResponse after splitting lines ({len(lines)} lines):\n")
            for i, line in enumerate(lines):
                print(f"  Line {i+1}: '{line}'")
                
            # Process results after splitting lines    
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Handle dashes before key names
                    key = key.strip().lstrip('-').strip()
                    value = value.strip()
                    print(f"  Parsing result - processed - Key: '{key}', Value: '{value}'")
                    
                    # Store evaluation dimensions in the results dictionary
                    if key == 'Multi-Hop Reasoning Requirement':
                        result['evaluation']['multi_hop_reasoning'] = value.lower() == 'yes'
                    elif key in ['Fluency', 'Clarity', 'Conciseness', 'Relevance', 
                               'Consistency', 'Question Answerability', 
                               'Answer-Question Consistency', 'Information Integration Ability',
                               'Reasoning Path Guidance', 'Logical Sophistication']:
                        # Convert rating to lowercase and replace spaces with underscores
                        norm_key = key.lower().replace(' ', '_').replace('-', '_')
                        # Keep as is even if the value is empty
                        result['evaluation'][norm_key] = value
            
            # Calculate overall quality score (simple average of all dimensions except multi_hop_reasoning)
            rating_map = {
                'very poor': 1,
                'poor': 2,
                'fair': 3,
                'good': 4,
                'very good': 5
            }
            
            print(f"\nEvaluation dimension results:")
            for key, value in result['evaluation'].items():
                print(f"  - {key}: {value}")
            
            quality_scores = []
            for key, value in result['evaluation'].items():
                if key != 'multi_hop_reasoning' and isinstance(value, str):
                    value_lower = value.lower().strip()
                    if value_lower and value_lower in rating_map:
                        quality_scores.append(rating_map[value_lower])
                        print(f"  Scoring: {key} = {value_lower} gets {rating_map[value_lower]} points")
                    elif not value_lower:
                        print(f"  Skipping: {key} field is empty, not included in total score")
            
            if quality_scores:
                result['evaluation']['overall_quality'] = sum(quality_scores) / len(quality_scores)
                print(f"  Overall quality score: {result['evaluation']['overall_quality']:.2f}/5.0")
            else:
                result['evaluation']['overall_quality'] = 0
                print("  Warning: No ratings available")
                
            return result
            
        except Exception as e:
            print(f"Error parsing evaluation results: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Original response: {response}")
            return None
            
    def evaluate_dataset(self, dataset, output_file=None, sample_size=None):
        """Evaluate all questions in the dataset
        
        Args:
            dataset: List or file path containing question data
            output_file: Optional, output file path for results
            sample_size: Optional, number of samples to evaluate
            
        Returns:
            list: List containing evaluation results
        """
        # Read dataset
        if isinstance(dataset, str):
            if not os.path.exists(dataset):
                print(f"Error: File not found - {dataset}")
                return []
                
            try:
                with open(dataset, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading dataset file: {str(e)}")
                return []
        else:
            data = dataset
            
        # Limit sample size
        if sample_size and sample_size < len(data):
            print(f"Evaluating {sample_size} samples from {len(data)} data points")
            import random
            random.seed(42)  # Set random seed for reproducibility
            data = random.sample(data, sample_size)
        
        # Evaluate all questions
        results = []
        for i, question_data in enumerate(data):
            print(f"Evaluating question {i+1}/{len(data)}: {question_data.get('id', i)}")
            
            result = self.evaluate_question(question_data)
            if result:
                results.append(result)
                
                # Display evaluation progress and results summary
                multi_hop = result['evaluation'].get('multi_hop_reasoning', False)
                quality = result['evaluation'].get('overall_quality', 0)
                print(f"  - Multi-hop requirement: {'Yes' if multi_hop else 'No'}, Overall quality: {quality:.2f}/5.0")
            
        # Save results
        if output_file and results:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Evaluation results saved to {output_file}")
            except Exception as e:
                print(f"Error saving evaluation results: {str(e)}")
                
        # Print evaluation summary
        self._print_evaluation_summary(results)
                
        return results
        
    def _print_evaluation_summary(self, results):
        """Print evaluation results summary
        
        Args:
            results: List of evaluation results
        """
        if not results:
            print("No evaluation results to summarize")
            return
            
        # Calculate multi-hop question ratio
        multi_hop_count = sum(1 for r in results if r['evaluation'].get('multi_hop_reasoning', False))
        multi_hop_ratio = multi_hop_count / len(results) if results else 0
        
        # Calculate average score for each dimension
        dimension_scores = {}
        rating_map = {
            'very poor': 1,
            'poor': 2,
            'fair': 3,
            'good': 4,
            'very good': 5
        }
        
        for result in results:
            for key, value in result['evaluation'].items():
                if key != 'multi_hop_reasoning' and key != 'overall_quality':
                    if key not in dimension_scores:
                        dimension_scores[key] = []
                    
                    if value.lower() in rating_map:
                        dimension_scores[key].append(rating_map[value.lower()])
        
        # Calculate average score
        dimension_averages = {k: sum(v)/len(v) if v else 0 for k, v in dimension_scores.items()}
        
        # Calculate overall average score
        overall_avg = sum(r['evaluation'].get('overall_quality', 0) for r in results) / len(results) if results else 0
        
        # Print summary
        print("\n" + "="*50)
        print(f"Evaluation summary (Total {len(results)} data points)")
        print("="*50)
        print(f"Multi-hop question ratio: {multi_hop_ratio:.2%} ({multi_hop_count}/{len(results)})")
        print(f"Overall quality average score: {overall_avg:.2f}/5.0")
        print("\nAverage score for each dimension:")
        for dim, score in sorted(dimension_averages.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {dim}: {score:.2f}/5.0")
        print("="*50)


def main():
    """Main function to demonstrate the use of the evaluator"""
    # Use Config to load configuration file
    from flashrag.config import Config
    config = Config("./config_lib/example_config.yaml", {})
    
    config["generator_batch_size"] = 1
    
    # Directly load the first piece of data from 2wiki_sample.json as an example
    sample_path = '../dataset/samples/2wiki_sample.json'
    with open(sample_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
        example_question = sample_data[0]
    print(f"Successfully loaded the first piece of data from {sample_path}")
    print(f"Example question: {example_question['question']}")
    
    # Initialize evaluator
    print("Initializing multi-hop question evaluator...")
    evaluator = QualityEvaluator(config)
    
    result = evaluator.evaluate_question(example_question)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    

if __name__ == '__main__':
    main()
