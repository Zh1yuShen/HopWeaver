import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.retriever import DenseRetriever
from flashrag.config import Config
import json
import numpy as np

# Import reranker model

from FlagEmbedding import FlagReranker


class RetrieverWrapper:
    """Wrapper class for retrieving relevant documents"""
    def __init__(self, config):
        self.config = config
        self.config["corpus_path"] = self.config["method2corpus"][self.config["retrieval_method"]]
        # Initialize retriever during actual use
        # Import retriever from FlashRAG
        self.retriever = DenseRetriever(config)
        
    def retrieve_related_documents(self, query, doc_id=None, top_k=5):
        """Retrieve documents related to the query
        
        Args:
            query: Query text
            doc_id: Document ID to exclude (usually the source document)
            top_k: Maximum number of documents to return
            
        Returns:
            list: List of relevant documents
        """
        # Call the search method of the actual retriever
        results = self.retriever.search(query, num=top_k*2)
        
        # Filter out the input document itself
        if doc_id:
            # Select filter field based on retrieval method
            id_field = "doc_id" if self.config["retrieval_method"] == "e5_instruct" else "id"
            results = [doc for doc in results if doc.get(id_field) != doc_id]
            
        return results[:top_k]
        
    def retrieve_with_query(self, query_text, top_k=5, return_score=False, doc_id=None):
        """Retrieve relevant documents using query text
        
        Args:
            query_text: Query text obtained from entity_extractor
            top_k: Maximum number of documents to return
            return_score: Whether to return similarity scores
            doc_id: Document ID to exclude (usually the source document)
            
        Returns:
            list or tuple: List of retrieved documents, also returns scores if return_score is True
        """
        try:
            if return_score:
                results, scores = self.retriever.search(query_text, num=top_k + (1 if doc_id else 0), return_score=True)
                
                # Filter out the input document itself
                if doc_id:
                    # Select filter field based on retrieval method
                    id_field = "doc_id" if self.config["retrieval_method"] == "e5_instruct" else "id"
                    filtered_results = []
                    filtered_scores = []
                    for doc, score in zip(results, scores):
                        if doc.get(id_field) != doc_id:
                            filtered_results.append(doc)
                            filtered_scores.append(score)
                    results = filtered_results[:top_k]
                    scores = filtered_scores[:top_k]
                
                return results, scores
            else:
                results = self.retriever.search(query_text, num=top_k + (1 if doc_id else 0))
                
                # Filter out the input document itself
                if doc_id:
                    # Choose the filtering field based on the retrieval method
                    id_field = "doc_id" if self.config["retrieval_method"] == "e5_instruct" else "id"
                    results = [doc for doc in results if doc.get(id_field) != doc_id]
                    results = results[:top_k]
                
                return results
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            # Return empty result
            if return_score:
                return [], []
            else:
                return []


class DiverseRetrieverWrapper(RetrieverWrapper):
    """Diversity retrieval wrapper class based on MMR (Maximum Marginal Relevance) algorithm, inherits from RetrieverWrapper"""
    def __init__(self, config, lambda1=0.8, lambda2=0.1, lambda3=0.1):
        """
        Initialization method
        
        Args:
            config: Configuration parameters
            lambda1: Query relevance weight, range 0 to 1, higher value emphasizes relevance of document to query
            lambda2: Original document diversity weight, range 0 to 1, higher value emphasizes diversity of document from original document
            lambda3: Selected document set diversity weight, range 0 to 1, higher value emphasizes diversity of document from selected documents
            Note: lambda1 + lambda2 + lambda3 should equal 1
        """
        super().__init__(config)
        # Ensure the sum of the three weight parameters is 1
        total = lambda1 + lambda2 + lambda3
        self.lambda1 = lambda1 / total
        self.lambda2 = lambda2 / total
        self.lambda3 = lambda3 / total

    def _get_doc_embedding(self, doc):
        """Get the embedding vector of the document"""
        # If input is a string, encode directly
        if isinstance(doc, str):
            # Use retriever.encoder instead of retriever to encode text
            embedding = self.retriever.encoder.encode(doc)
            return np.array(embedding)
        
        # If it is a dictionary, check if it has an embedding vector
        if isinstance(doc, dict):
            # Assume the document contains an embedding vector field (usually generated by DenseRetriever)
            # Depending on your retrieval_method, you may need to adjust the field name
            if "embedding" in doc:
                return np.array(doc["embedding"])
            else:
                # If the document does not have an embedding vector, it may need to be re-encoded
                # Use retriever.encoder instead of retriever to encode text
                doc_text = doc.get("content", "") or doc.get("contents", "")
                embedding = self.retriever.encoder.encode(doc_text)
                return np.array(embedding)
        
        # If it is neither a string nor a dictionary, throw an exception
        raise ValueError("orig_doc must be a string or dictionary type")

    def _compute_similarity(self, doc1_vec, doc2_vec):
        """Calculate the cosine similarity between two document vectors"""
        # Ensure the vector is one-dimensional
        doc1_vec = doc1_vec.flatten()
        doc2_vec = doc2_vec.flatten()
        if np.linalg.norm(doc1_vec) == 0 or np.linalg.norm(doc2_vec) == 0:
            return 0.0
        return np.dot(doc1_vec, doc2_vec) / (np.linalg.norm(doc1_vec) * np.linalg.norm(doc2_vec))

    def retrieve_with_diversity(self, query_text, orig_doc, top_k=5, return_score=False, doc_id=None):
        """Retrieve relevant and diverse documents using the MMR (Maximum Marginal Relevance) algorithm
        
        MMR algorithm formula: Score(di) = λ1 × rel(q, di) − λ2 × sim(di, d0) − λ3 × max sim(di, dj)
        Where:
        - rel(q, di) represents the relevance of query q to document di
        - sim(di, d0) represents the similarity of document di to the original document d0
        - max sim(di, dj) represents the maximum similarity of document di to any document dj in the selected document set S
        - λ1、λ2、λ3 are the weights for balancing the three components
        
        Args:
            query_text: Query text from entity_extractor
            orig_doc: Original document, used to calculate repetition penalty
            top_k: Maximum number of documents to return
            return_score: Whether to return similarity scores
            doc_id: ID of the document to exclude (usually the source document)
            
        Returns:
            list or tuple: List of retrieved documents, and scores if return_score is True
        """
        try:
            # Step 1: Get the embedding vector of the original document
            orig_doc_vec = self._get_doc_embedding(orig_doc)

            # Step 2: Use the original retrieval method to get candidate documents
            # To increase the candidate document pool, the initial retrieval quantity needs to be larger
            candidate_num = top_k * 3  # Increase the number of candidate documents to ensure sufficient diversity
            candidates, candidate_scores = self.retriever.search(query_text, num=candidate_num, return_score=True)

            # Filter out the input document itself (if there is a doc_id)
            if doc_id:
                id_field = "doc_id" if self.config["retrieval_method"] == "e5_instruct" else "id"
                filtered_candidates = []
                filtered_scores = []
                for doc, score in zip(candidates, candidate_scores):
                    if doc.get(id_field) != doc_id:
                        filtered_candidates.append(doc)
                        filtered_scores.append(score)
                candidates = filtered_candidates
                candidate_scores = filtered_scores

            if not candidates:
                if return_score:
                    return [], []
                return []

            # Step 3: Pre-calculate the embedding vector of each candidate document to avoid repeated calculations
            candidate_vecs = []
            for candidate in candidates:
                candidate_vec = self._get_doc_embedding(candidate)
                candidate_vecs.append(candidate_vec)
            
            # Step 4: Implement the MMR algorithm for document sorting and selection
            selected_indices = []
            selected_scores = []
            
            # Initialize selected document list and score list
            S = []
            
            # Iteratively select top_k documents
            for _ in range(min(top_k, len(candidates))):
                best_score = float('-inf')
                best_idx = -1
                
                # Calculate the MMR score for each candidate document
                for i in range(len(candidates)):
                    # Skip already selected documents
                    if i in selected_indices:
                        continue
                    
                    # Calculate the three components of the MMR score
                    # 1. Query relevance score
                    rel_score = candidate_scores[i]
                    
                    # 2. Similarity to the original document (for diversity)
                    orig_sim = self._compute_similarity(candidate_vecs[i], orig_doc_vec)
                    
                    # 3. Maximum similarity to the selected document set (for diversity)
                    max_sim_to_selected = 0.0
                    if S:  # If the selected document set is not empty
                        for j in selected_indices:
                            sim_to_selected = self._compute_similarity(candidate_vecs[i], candidate_vecs[j])
                            max_sim_to_selected = max(max_sim_to_selected, sim_to_selected)
                    
                    # Apply the MMR formula: Score(di) = λ1 × rel(q, di) − λ2 × sim(di, d0) − λ3 × max sim(di, dj)
                    mmr_score = (self.lambda1 * rel_score) - (self.lambda2 * orig_sim) - (self.lambda3 * max_sim_to_selected)
                    
                    # Update the best document
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                
                # Add the selected document to the result list and remove it from the candidate list
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    selected_scores.append(best_score)
                    S.append(candidates[best_idx])
            
            # Get the final results
            final_results = [candidates[i] for i in selected_indices]
            final_scores = selected_scores

            if return_score:
                return final_results, final_scores
            return final_results

        except Exception as e:
            print(f"Error during diversity retrieval: {str(e)}")
            if return_score:
                return [], []
            return []


            
class RerankRetrieverWrapper(DiverseRetrieverWrapper):
    """Retrieval wrapper class combining diversity retrieval and reranker model, inherits from DiverseRetrieverWrapper"""
    
    def __init__(self, config, reranker_path='./models/bge-reranker-v2-m3', 
             lambda1=0.8, lambda2=0.1, lambda3=0.1, use_fp16=True):
        """
        Initialization method
        
        Args:
            config: Configuration parameters
            reranker_path: Reranker model path
            lambda1: Query relevance weight, range 0 to 1
            lambda2: Original document diversity weight, range 0 to 1
            lambda3: Selected document set diversity weight, range 0 to 1
            use_fp16: Whether to use FP16 acceleration
        """
        super().__init__(config, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        
        # Initialize reranker model
        if FlagReranker is not None and os.path.exists(reranker_path):
            try:
                # Get advanced parameters from configuration, use default values if not present
                query_max_length = self.config["query_max_length"] if "query_max_length" in self.config else 512  # Default query length 512
                passage_max_length = self.config["passage_max_length"] if "passage_max_length" in self.config else 8196  # Default document length 8196
                batch_size = self.config["reranker_batch_size"] if "reranker_batch_size" in self.config else 16  # Default batch size 16
                normalize = self.config["reranker_normalize"] if "reranker_normalize" in self.config else False  # Default no normalization
                devices = self.config["reranker_devices"] if "reranker_devices" in self.config else None  # Default use all available devices
                
                # Initialize reranker model with more parameters
                self.reranker = FlagReranker(
                    reranker_path,
                    use_fp16=use_fp16,
                    query_max_length=query_max_length,
                    max_length=passage_max_length,
                    batch_size=batch_size,
                    normalize=normalize,
                    devices=devices
                )
                self.has_reranker = True
                print(f"Successfully loaded reranker model: {reranker_path}")
                print(f"Reranker parameters: query_max_length={query_max_length}, passage_max_length={passage_max_length}, batch_size={batch_size}")
            except Exception as e:
                print(f"Failed to load reranker model: {str(e)}")
                self.has_reranker = False
        else:
            print(f"Reranker model path does not exist or FlagEmbedding module is unavailable: {reranker_path}")
            self.has_reranker = False
            
    def _prepare_rerank_pairs(self, query, candidates):
        """
        Prepare query-document pairs for reranking
        
        Args:
            query: Query text
            candidates: List of candidate documents
            
        Returns:
            list: List of query-document pairs
        """
        pairs = []
        for doc in candidates:
            # Get the document content
            doc_content = doc.get("content", "") or doc.get("contents", "")
            # Add the query-document pair
            pairs.append([query, doc_content])
        return pairs
    
    def retrieve_with_rerank(self, query_text, orig_doc, top_k=5, return_score=False, doc_id=None, normalize=True):
        """
        First use diversity retrieval to get top_k*3 documents, then use reranker model to reorder and select top_k
        
        Args:
            query_text: Query text
            orig_doc: Original document, used to calculate diversity
            top_k: Maximum number of documents to return
            return_score: Whether to return similarity scores
            doc_id: ID of the document to exclude (usually the source document)
            normalize: Whether to normalize rerank scores to the 0-1 range
            
        Returns:
            list or tuple: List of retrieved documents, and scores if return_score is True
        """
        try:
            # If there are no candidate documents or the reranker is unavailable, return the diversity retrieval results directly
            if not self.has_reranker:
                print("No candidate documents or reranker unavailable, returning diversity retrieval results")
                return self.retrieve_with_diversity(query_text, orig_doc, top_k, return_score, doc_id)
            
            # Step 1: Use diversity retrieval to get candidate documents, 3 times the final required number, to provide enough candidates for reranking
            expanded_top_k = top_k   # Expand the retrieval range to 3 times the final required number
            print(f"Reranker using expanded retrieval range: {expanded_top_k} candidate documents")
            diverse_results, diverse_scores = self.retrieve_with_diversity(query_text, orig_doc, top_k=expanded_top_k, return_score=True, doc_id=doc_id)
            
            if not diverse_results:
                if return_score:
                    return [], []
                return []
            
            # Step 2: Prepare query-document pairs for reranking
            rerank_pairs = self._prepare_rerank_pairs(query_text, diverse_results)
            
            # Step 3: Use reranker model to calculate scores
            print("Using reranker model to calculate scores...")
            rerank_scores = self.reranker.compute_score(rerank_pairs, normalize=normalize)
            
            # Step 4: Reorder documents based on rerank scores
            reranked_results = []
            for doc, score in zip(diverse_results, rerank_scores):
                reranked_results.append((doc, score))
            
            # Sort the documents by rerank score in descending order
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # Get the final results
            final_results = [doc for doc, _ in reranked_results[:top_k]]
            final_scores = [score for _, score in reranked_results[:top_k]]

            if return_score:
                return final_results, final_scores
            return final_results
            
        except Exception as e:
            print(f"Error during rerank retrieval: {str(e)}")
            if return_score:
                return [], []
            return []


if __name__ == "__main__":
    import json
    from datetime import datetime
    
    # Module functionality test
    # Configuration and data for testing
    config_path = "./config_lib/example_config.yaml"  # Your config file path
    config = Config(config_path, {})
    
    # Create retriever
    print("Testing basic retriever...")
    retriever = RetrieverWrapper(config) # Create retriever instance
    print("Initialization complete")
    # Test query
    test_query = "Charlie Wilson's personal life instead of his role in funding the Afghan Mujahideen."
    
    # ID of the document to exclude
    exclude_doc_id = "2393795"
    
    print(f"\nUsing query for retrieval (excluding document ID: {exclude_doc_id}): '{test_query}'")
    
    # Original document - string type
    orig_doc = "CIA's relationship with the United States Congress\nCIA's relationship with the United States Congress There have been various arrangements to handle the Central Intelligence Agency's relationship with the United States Congress.  The formal liaison began some time before the 1960s, with a single position named the 'legislative liaison'. This later became the 'legislative counsel'. In the 1960s, an actual office was created for this purpose - the Office of Legislative Counsel.  In the 1970s, the Central Intelligence Agency (CIA) ramped up its congressional-liaison staff to deal with the large number of investigations coming from the Congress. It was the era of the Rockefeller Commission, the Church Committee, and the Pike Committee, all of which requested large amounts of information from the agency. In the 1980s, there were several reorganizations and renaming of the office. Near the end of the 1980s, the office was renamed the Office of Congressional Affairs and has kept that name, as of 2009.  In the early 2000s (decade), the relationship became more intense, with debates about the Global war on terror and controversies surrounding it. For example, the CIA planned a secret program in 2001 but did not inform congress until much later. This time line is based on information found in Snider, \"The Agency and the Hill\", Chapter 4 (available online, see below under 'sources'). It lists the liaison, or the head of the liaison office, along with brief mentions of some significant events, reorganizations, and name changes.   During much of the 1980s a unique and unusual relationship evolved between Congress and the CIA in the person of Texas congressman Charlie Wilson from Texas's 2nd congressional district. Using his position on various House appropriations committees, and in partnership with CIA agent Gust Avrakotos, Wilson was able to increase CIA's funding the Afghan Mujahideen to several hundred million dollars a year during the Soviet Afghan war. Author George Crile would describe Wilson as eventually becoming the \"Agency's station chief on the Hill\". He eventually got a position on the Intelligence Committee and was supposed to be overseeing the CIA."
    
    # Perform diversity retrieval and return scores (excluding specified ID document by default)
    results, scores = retriever.retrieve_with_diversity(test_query, orig_doc, top_k=3, return_score=True, doc_id=exclude_doc_id)
    
    # Prepare JSON data
    json_results = {
        "query": test_query,
        "exclude_doc_id": exclude_doc_id,
        "timestamp": datetime.now().isoformat(),
        "results": []
    }
    
    # Output retrieval results
    print("\nRetrieval results:")
    if results:
        for i, (doc, score) in enumerate(zip(results, scores)):
            print(f"\nResult #{i+1} (similarity-diversity score: {score:.4f}):")
            print(f"  ID: {doc.get('id', 'N/A')}, Title: {doc.get('title', 'N/A')}, Score: {doc.get('score', 'N/A')}")
            print(f"  Content: {doc.get('contents', '')[:3000]}...")
            
            # Add to JSON results
            json_results["results"].append({
                "rank": i + 1,
                "doc_id": doc.get('id', 'unknown'),
                "score": float(score),
                "contents": doc.get('contents', '')
            })
    else:
        print("No relevant documents found")
    
    # Save as JSON file
    output_file = f"./retrieval_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"Retrieval results saved to: {output_file}")
