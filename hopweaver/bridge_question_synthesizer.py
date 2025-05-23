import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.config import Config
from hopweaver.components.utils.data_reader import DocumentReader
from hopweaver.components.bridge.bridge_entity_extractor import EntityExtractor
from hopweaver.components.bridge.bridge_retriever import RetrieverWrapper, DiverseRetrieverWrapper, RerankRetrieverWrapper
from hopweaver.components.bridge.bridge_question_generator import QuestionGenerator
from hopweaver.components.bridge.bridge_polisher import Polisher

class QuestionSynthesizer:
    """
    Multi-hop question synthesizer main class that integrates document reading,
    entity extraction, document retrieval, and question generation.
    """
    def __init__(self, config_path, lambda1=None, lambda2=None, lambda3=None):
        """
        Initialize the question synthesizer
        
        Args:
            config_path (str): Path to configuration file
            lambda1 (float, optional): Query relevance weight
            lambda2 (float, optional): Source document diversity weight
            lambda3 (float, optional): Selected document set diversity weight
        """
        # Load configuration
        self.config = Config(config_path, {})
        
        # Set GPU environment variable
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["gpu_id"])
        
        # Initialize components
        self.doc_reader = DocumentReader(self.config['corpus_path'])
        self.entity_extractor = EntityExtractor(self.config)
        
        # Select retriever based on configuration
        self.retriever_type = self.config["retriever_type"] if "retriever_type" in self.config else "diverse"  # Default: diverse retriever
        
        # Use provided lambda parameters first, then config values, or defaults
        l1 = lambda1 if lambda1 is not None else (self.config["lambda1"] if "lambda1" in self.config else 0.7)
        l2 = lambda2 if lambda2 is not None else (self.config["lambda2"] if "lambda2" in self.config else 0.1)
        l3 = lambda3 if lambda3 is not None else (self.config["lambda3"] if "lambda3" in self.config else 0.2)
        
        if self.retriever_type == "rerank":
            # Use rerank retriever
            reranker_path = self.config["reranker_path"] if "reranker_path" in self.config else "./models/bge-reranker-v2-m3"
            self.retriever = RerankRetrieverWrapper(
                self.config, 
                reranker_path=reranker_path,
                lambda1=l1, 
                lambda2=l2, 
                lambda3=l3,
                use_fp16=self.config["use_fp16"] if "use_fp16" in self.config else True
            )
            print(f"Using rerank retriever: RerankRetrieverWrapper, reranker model path: {reranker_path}")
        elif self.retriever_type == "standard":
            # Use standard retriever
            self.retriever = RetrieverWrapper(self.config)
            print("Using standard retriever: RetrieverWrapper")
        else:
            # Use diversity retriever
            self.retriever = DiverseRetrieverWrapper(
                self.config, 
                lambda1=l1, 
                lambda2=l2, 
                lambda3=l3
            )
            print("Using diversity retriever: DiverseRetrieverWrapper")
        
        self.question_generator = QuestionGenerator(self.config)
        self.polisher = Polisher(self.config)
        
        # Configure output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_question(self, save_output=True, verbose=True):
        """
        Complete process for generating multi-hop questions
        
        Args:
            save_output (bool): Whether to save output to file
            verbose (bool): Whether to print detailed information
            
        Returns:
            dict: Dictionary containing generated question, answer and related information
        """
        try:
            # Step 1: Randomly get a document
            print("Step 1: Random sampling of a document")
            source_doc = self.doc_reader.get_heuristic_documents(count=1, min_length=300)[0]
            if not source_doc:
                print("Failed to get valid document")
                return None
            
            print(f"Document obtained: ID={source_doc.get('id', 'unknown')}, Title={source_doc.get('title', 'unknown')}")
            if verbose:
                print(f"Document preview: {source_doc.get('contents', '')[:300]}..." if len(source_doc.get('contents', '')) > 300 else f"Document content: {source_doc.get('contents', '')}")
            
            # Step 2: Extract bridge entities from the document
            print("\nStep 2: Extracting bridge entities from the document")
            entities = self.entity_extractor.extract_entities(source_doc)
            
            if not entities:
                print("Failed to extract entities from the document")
                return None
            
            # Use bridge entity (should be only one)
            bridge_entity = entities[0] if len(entities) > 0 else None
            if bridge_entity is None:
                print("Failed to get bridge entity")
                return None
            print(f"Using bridge entity: {bridge_entity['name']} (Type: {bridge_entity.get('type', 'unknown')})")
            if verbose and bridge_entity.get('segments'):
                print("Entity related segments:")
                for i, segment in enumerate(bridge_entity.get('segments', [])):
                    print(f"  Segment {i+1}: {segment[:200]}..." if len(segment) > 200 else f"  Segment {i+1}: {segment}")
            
            # Get bridge entity information
            entity_name = bridge_entity['name']
            entity_type = bridge_entity.get('type', '')
            entity_query = bridge_entity.get('query', entity_name)
            entity_segments = ' '.join(bridge_entity.get('segments', []))
            source_doc_id = source_doc.get('id', '')
            
            # Check if any variable is empty, if empty then re-extract entity information
            if not entity_name or not entity_type or not entity_query or not entity_segments or not source_doc_id:
                print("Bridge entity information incomplete, re-extracting entity information")
                # Re-extract entity
                entities = self.entity_extractor.extract_entities(source_doc)
                if not entities:
                    print("Failed to re-extract entities")
                    return None
                
                # Update bridge entity
                bridge_entity = entities[0] if len(entities) > 0 else None
                if bridge_entity is None:
                    print("Still failed to get bridge entity after re-extraction")
                    return None
                
                print(f"Re-extracted bridge entity: {bridge_entity['name']} (Type: {bridge_entity.get('type', 'unknown')})")
                
                # Re-get entity information
                entity_name = bridge_entity['name']
                entity_type = bridge_entity.get('type', '')
                entity_query = bridge_entity.get('query', entity_name)
                entity_segments = ' '.join(bridge_entity.get('segments', []))
                
                # If still empty values, give up processing
                if not entity_name or not entity_type or not entity_query or not entity_segments:
                    print("Entity information still incomplete after re-extraction, giving up")
                    return None
            
            # Step 3: Retrieve related documents
            print("\nStep 3: Retrieving related documents based on bridge entity")
            # Choose retrieval method based on retriever type
            if isinstance(self.retriever, RerankRetrieverWrapper) and hasattr(self.retriever, 'retrieve_with_rerank'):
                print("Using rerank retrieval method")
                retrieved_docs = self.retriever.retrieve_with_rerank(
                    entity_query, 
                    entity_segments,  # Use entity-related paragraphs instead of entire original document
                    top_k=5, 
                    doc_id=source_doc_id
                )
            else:
                print("Using diversity retrieval method")
                retrieved_docs = self.retriever.retrieve_with_diversity(
                    entity_query, 
                    entity_segments,  # Entity-related paragraphs
                    top_k=5, 
                    doc_id=source_doc_id
                )
            
            if not retrieved_docs:
                print("Failed to retrieve related documents")
                return None
            
            # Try each retrieved document until finding one that can generate valid questions
            success = False
            
            for doc_index, target_doc in enumerate(retrieved_docs):
                target_doc_id = target_doc.get('id', '')
                target_doc_content = target_doc.get('contents', '')
                
                print(f"Trying document {doc_index+1}/{len(retrieved_docs)}: ID={target_doc_id}")
                if verbose:
                    print(f"Document preview: {target_doc_content[:300]}..." if len(target_doc_content) > 300 else f"Document content: {target_doc_content}")
                
                # Step 4: Generate sub-questions for current document
                print(f"\nStep 4: Generating sub-questions for document {doc_index+1}")
                sub_questions = self.question_generator.generate_sub_questions(
                    entity_name,
                    entity_type,
                    entity_segments,
                    target_doc_content
                )
                
                if not sub_questions:
                    print(f"Failed to generate sub-questions for document {doc_index+1}, possibly due to invalid bridge entity connection or other issues")
                    # If there are more documents, continue trying
                    if doc_index < len(retrieved_docs) - 1:
                        print(f"Trying next retrieved document...")
                        continue
                    else:
                        print("Tried all retrieved documents, unable to generate valid sub-questions")
                        return None
                
                # Found a document that can generate sub-questions
                success = True
                print(f"Successfully generated sub-questions for document {doc_index+1}")
                break
            
            # If all documents failed, return None
            if not success:
                return None
                
            print("Successfully generated sub-questions")
            if verbose and sub_questions:
                # Display analysis section
                if 'analysis' in sub_questions:
                    print("\nSub-question analysis:")
                    for key, value in sub_questions['analysis'].items():
                        print(f"  {key.replace('_', ' ').title()}: {value}")
                
                # Display sub-questions section
                if 'sub_questions' in sub_questions:
                    print("\nGenerated sub-questions:")
                    for i, q in enumerate(sub_questions['sub_questions']):
                        print(f"  Sub-question {i+1}: {q.get('question', '')}")
                        print(f"  Answer {i+1}: {q.get('answer', '')}")
                        print(f"  Source: {q.get('source', '')}\n")
            
            # Step 5: Synthesize multi-hop question
            print("\nStep 5: Synthesizing multi-hop question")
            multi_hop_result = self.question_generator.synthesize_multi_hop_question(sub_questions)
            
            if not multi_hop_result:
                print("Failed to synthesize multi-hop question - sub-questions may not combine into a valid multi-hop question")
                # If there are other documents to try, should go back to step 3 to try the next document
                # But since we've tried them all, return None
                return None
                
            print(f"Successfully synthesized multi-hop question: {multi_hop_result.get('multi_hop_question', '')}")
            if verbose:
                print(f"  Answer: {multi_hop_result.get('answer', '')}")
                print(f"  Reasoning path: {multi_hop_result.get('reasoning_path', '')}")
                print(f"  Sources: {multi_hop_result.get('sources', '')}")
            
            # Step 6: Polish the multi-hop question
            print("\nStep 6: Polishing multi-hop question")
            # Get required input parameters
            multi_hop_question = multi_hop_result.get('multi_hop_question', '')
            answer = multi_hop_result.get('answer', '')
            reasoning_path = multi_hop_result.get('reasoning_path', '')
            doc_a_seg = sub_questions['analysis'].get('doc_a_seg', '')
            doc_b_seg = sub_questions['analysis'].get('doc_b_seg', '')
            
            # Get sub-questions and sub-answers
            sub_question_1 = ""
            sub_question_2 = ""
            
            if len(sub_questions['sub_questions']) > 0:
                q1 = sub_questions['sub_questions'][0]['question']
                a1 = sub_questions['sub_questions'][0]['answer']
                sub_question_1 = f"{q1}\nAnswer: {a1}"
                
            if len(sub_questions['sub_questions']) > 1:
                q2 = sub_questions['sub_questions'][1]['question']
                a2 = sub_questions['sub_questions'][1]['answer']
                sub_question_2 = f"{q2}\nAnswer: {a2}"
            
            # Call polisher for refinement
            polish_result = self.polisher.polish_question(
                multi_hop_question, 
                answer, 
                reasoning_path, 
                doc_a_seg, 
                doc_b_seg, 
                sub_question_1, 
                sub_question_2
            )
            
            if not polish_result:
                print("Failed to polish multi-hop question - using original multi-hop question")
            else:
                print(f"Polish status: {polish_result.get('status', 'UNKNOWN')}")
                if polish_result.get('status') in ['ADJUST', 'REWORKED']:
                    if verbose:
                        print(f"  Original question: {multi_hop_question}")
                        print(f"  Polished question: {polish_result.get('refined_question', '')}")
                        print(f"  Polished reasoning path: {polish_result.get('refined_reasoning_path', '')}")
                        print(f"  Polished answer: {polish_result.get('answer', '')}")
                    
            
            # Construct complete result
            result = {
                "source_doc": {
                    "id": source_doc_id,
                    "title": source_doc.get('title', ''),
                    "content": source_doc.get('contents', '')[:1000] + "..." if len(source_doc.get('contents', '')) > 1000 else source_doc.get('contents', '')
                },
                "target_doc": {
                    "id": target_doc_id,
                    "title": target_doc.get('title', ''),
                    "content": target_doc_content[:1000] + "..." if len(target_doc_content) > 1000 else target_doc_content
                },
                "bridge_entity": {
                    "name": entity_name,
                    "type": entity_type,
                    "segments": bridge_entity.get('segments', []),
                    "query": entity_query
                },
                "sub_questions": sub_questions,
                "multi_hop_question": multi_hop_result,
                "polish_result": polish_result
            }
            
            # Save result
            if save_output:
                self._save_result(result, mode="single")
                
            return result
            
        except Exception as e:
            print(f"Error occurred during question generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_result(self, result, mode="single", jsonl_file=None):
        """Save generated results to file
        
        Args:
            result (dict): Generated result dictionary
            mode (str): Save mode, options are "single" (single question) or "batch" (batch generation)
            jsonl_file (str, optional): jsonl file path, only used in batch mode
        """
        if mode == "single":
            # Single question mode
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            question_str = result['multi_hop_question'].get('multi_hop_question', '')
            question_id = ''.join(e for e in question_str[:10] if e.isalnum())
            output_file = os.path.join(self.output_dir, f"multi_hop_question_{timestamp}_{question_id}.json")
            
            # Save complete result in JSON format
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Create readable text summary file
            text_output_file = os.path.join(self.output_dir, f"multi_hop_question_{timestamp}_{question_id}_summary.txt")
            
            # Save readable text summary
            with open(text_output_file, 'w', encoding='utf-8') as f:
                f.write("=== Multi-hop Question Generation Summary ===\n\n")
                
                # Document information
                f.write("=== Document Information ===\n")
                f.write(f"Document A ID: {result['source_doc']['id']}\n")
                f.write(f"Document A Title: {result['source_doc']['title']}\n")
                f.write(f"Document B ID: {result['target_doc']['id']}\n")
                f.write(f"Document B Title: {result['target_doc'].get('title', 'No Title')}\n\n")
                
                # Bridge entity information
                f.write("=== Bridge Entity ===\n")
                f.write(f"Entity name: {result['bridge_entity']['name']}\n")
                f.write(f"Entity type: {result['bridge_entity']['type']}\n")
                
                # Document segment information
                f.write("=== Document Segment Information ===\n")
                
                # Document A segment
                f.write("--- Document A Segment ---\n")
                if result['sub_questions'] and 'analysis' in result['sub_questions'] and 'doc_a_seg' in result['sub_questions']['analysis']:
                    f.write(f"{result['sub_questions']['analysis']['doc_a_seg']}\n")
                else:
                    # If no new format segment information, use bridge entity segments
                    for i, segment in enumerate(result['bridge_entity'].get('segments', [])):
                        f.write(f"Segment {i+1}: {segment}\n")
                
                # Document B segment
                f.write("\n--- Document B Segment ---\n")
                if result['sub_questions'] and 'analysis' in result['sub_questions'] and 'doc_b_seg' in result['sub_questions']['analysis']:
                    f.write(f"{result['sub_questions']['analysis']['doc_b_seg']}\n")
                
                f.write("\n")
                
                # Sub-question information
                f.write("=== Sub-questions ===\n")
                if result.get('sub_questions') and result['sub_questions'].get('sub_questions'):
                    for i, q in enumerate(result['sub_questions']['sub_questions']):
                        f.write(f"\nSub-question {i+1}: {q.get('question', '')}\n")
                        f.write(f"Answer: {q.get('answer', '')}\n")
                        f.write(f"Source: {q.get('source', '')}\n")
                
                # Analysis information
                f.write("=== Analysis Information ===\n")
                if result.get('sub_questions') and result['sub_questions'].get('analysis'):
                    analysis = result['sub_questions']['analysis']
                    f.write(f"Bridge connection: {analysis.get('bridge_connection', '')}\n")
                    f.write(f"Document A information: {analysis.get('doc_a_seg', '')}\n")
                    f.write(f"Document B information: {analysis.get('doc_b_seg', '')}\n")
                    f.write(f"Reasoning path: {analysis.get('reasoning_path', '')}\n\n")
                
                # Multi-hop question
                f.write("=== Multi-hop Question ===\n")
                f.write(f"Question: {result['multi_hop_question'].get('multi_hop_question', '')}\n")
                f.write(f"Answer: {result['multi_hop_question'].get('answer', '')}\n")
                f.write(f"Reasoning path: {result['multi_hop_question'].get('reasoning_path', '')}\n")
                f.write(f"Sources: {result['multi_hop_question'].get('sources', '')}")
                
                # Polish result
                f.write("\n=== Polish Result ===\n")
                if result.get('polish_result'):
                    f.write(f"Status: {result['polish_result'].get('status', 'UNKNOWN')}\n")
                    if result['polish_result'].get('status') in ['ADJUST', 'REWORKED']:
                        f.write(f"Polished question: {result['polish_result'].get('refined_question', '')}\n")
                        f.write(f"Polished reasoning path: {result['polish_result'].get('refined_reasoning_path', '')}\n")
                        f.write(f"Polished answer: {result['polish_result'].get('answer', '')}")
                else:
                    f.write("No polish or polish failed\n")
                
            print(f"\nResult saved to:\n- JSON: {output_file}\n- Readable summary: {text_output_file}")
            
        elif mode == "batch":
            # Batch mode, append result to jsonl file, do not generate summary
            with open(jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    
    def batch_generate(self, count=5, save_all=True):
        """Batch generate multi-hop questions
        
        Args:
            count (int): Number of questions to generate
            save_all (bool): Whether to save all generated results
            
        Returns:
            list: List of successfully generated questions
        """
        results = []
        success_count = 0
        attempt_count = 0
        max_attempts = count * 3  # Set maximum attempts to avoid infinite loop
        
        print(f"Starting batch generation of {count} multi-hop questions...")
        
        # Create jsonl output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_output_file = os.path.join(self.output_dir, f"multihop_batch_{timestamp}.jsonl")
        
        while success_count < count and attempt_count < max_attempts:
            attempt_count += 1
            print(f"\n===== Generation attempt {attempt_count}/{max_attempts} (Success: {success_count}/{count}) =====")
            
            # Generate question, but do not save (we will handle saving here)
            result = self.generate_question(save_output=False)
            
            if result:
                # If save_all, append result to jsonl file
                if save_all:
                    self._save_result(result, mode="batch", jsonl_file=jsonl_output_file)
                
                results.append(result)
                success_count += 1
                print(f"Successfully generated {success_count} questions, stopping generation")
            else:
                print("Generation failed, trying next...")
        
        print(f"\nBatch generation complete: Total attempts {attempt_count}, successful {success_count}")
        
        if save_all and success_count > 0:
            print(f"All results saved to: {jsonl_output_file}")
        
        return results


if __name__ == "__main__":
    # Configuration file path
    config_path = "./config_lib/example_config.yaml"
    
    # Create question synthesizer
    synthesizer = QuestionSynthesizer(config_path)
    
    # Select running mode
    # Single question generation mode (default)
    single_mode = False
    
    if single_mode:
        print("=== Single Question Generation Mode ===")
        # Generate a question and print result
        result = synthesizer.generate_question(save_output=True)
        
        if result:
            print("\nGeneration successful!\n")
            print("=== Document Information ===")
            print(f"Document A ID: {result['source_doc']['id']}")
            print(f"Document A Title: {result['source_doc']['title']}")
            print(f"Document B ID: {result['target_doc']['id']}")
            print(f"Document B Title: {result['target_doc'].get('title', 'No Title')}")
            
            print("\n=== Bridge Entity Information ===")
            print(f"Entity name: {result['bridge_entity']['name']}")
            print(f"Entity type: {result['bridge_entity']['type']}")
            print("\n=== Entity Related Segments ===")
            for i, segment in enumerate(result['bridge_entity'].get('segments', [])):
                print(f"Segment {i+1}: {segment[:200]}..." if len(segment) > 200 else f"Segment {i+1}: {segment}")
            
            print("\n=== Sub-question Information ===")
            if result.get('sub_questions') and result['sub_questions'].get('sub_questions'):
                for i, sub_q in enumerate(result['sub_questions']['sub_questions']):
                    print(f"\nSub-question {i+1}: {sub_q.get('question', '')}")
                    print(f"Answer: {sub_q.get('answer', '')}")
                    print(f"Source: {sub_q.get('source', '')}\n")
            
            print("=== Analysis Information ===")
            if result.get('sub_questions') and result['sub_questions'].get('analysis'):
                analysis = result['sub_questions']['analysis']
                print(f"Bridge connection: {analysis.get('bridge_connection', '')}")
                print(f"Document A information: {analysis.get('doc_a_seg', '')}")
                print(f"Document B information: {analysis.get('doc_b_seg', '')}")
                print(f"Reasoning path: {analysis.get('reasoning_path', '')}\n")
            
            print("=== Multi-hop Question ===")
            print(f"Question: {result['multi_hop_question'].get('multi_hop_question', '')}")
            print(f"Answer: {result['multi_hop_question'].get('answer', '')}")
            print(f"Reasoning path: {result['multi_hop_question'].get('reasoning_path', '')}")
            print(f"Sources: {result['multi_hop_question'].get('sources', '')}")
            
            print("\n=== Polish Result ===")
            if result.get('polish_result'):
                print(f"Status: {result['polish_result'].get('status', 'UNKNOWN')}")
                if result['polish_result'].get('status') in ['ADJUST', 'REWORKED']:
                    print(f"Polished question: {result['polish_result'].get('refined_question', '')}")
                    print(f"Polished reasoning path: {result['polish_result'].get('refined_reasoning_path', '')}")
                    print(f"Polished answer: {result['polish_result'].get('answer', '')}")
            else:
                print("No polish or polish failed")
        else:
            print("\nGeneration failed")
    else:
        # Batch generation mode
        print("=== Batch Question Generation Mode ===")
        # Batch size
        batch_count = 15
        results = synthesizer.batch_generate(count=batch_count)
        
        # Print statistics
        print(f"\nBatch generation complete: Total attempts {len(results)}/{batch_count}")
