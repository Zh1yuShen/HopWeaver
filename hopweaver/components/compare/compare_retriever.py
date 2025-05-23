"""
Comparison Query Retriever - Processes queries from comparison_queries.json file

This module receives the output of the comparison query generator and performs retrieval based on the query type:
1. For direct entity recommendations (recall_focused_verify), retrieve using verification queries
2. For search queries (search_queries), retrieve using 3 queries separately and merge the results

As a component in a multi-hop question answering system, this module helps the system find documents relevant to the query.
"""

import sys
import os
import json
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.retriever import DenseRetriever
from flashrag.config import Config


class CompareRetriever:
    """Comparison Query Retriever: Processes comparison queries and returns relevant documents"""
    
    def __init__(self, config):
        """
        Initialize the comparison query retriever
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.config["corpus_path"] = self.config["method2corpus"][self.config["retrieval_method"]]
        # Initialize retriever
        self.retriever = DenseRetriever(config)
        
        # Cache document content
        self.documents_cache = {}
        
    def retrieve_documents(self, query, doc_id=None, top_k=3):
        """
        Retrieve documents related to the query
        
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
            
            # Filter out the input document itself
            results = [doc for doc in results if doc.get(id_field) != doc_id]
            
            # Add retrieved documents to cache
            for doc in results:
                if doc.get(id_field):
                    self.documents_cache[doc.get(id_field)] = doc
        
        return results[:top_k]
        

        

    
    def process_recall_focused_verify(self, query_result, top_k=3):
        """
        Process direct entity recommendation type queries
        
        Args:
            query_result: Query result containing verification query
            top_k: Maximum number of documents to return
            
        Returns:
            dict: Dictionary containing retrieval results
        """
        # Get verification query
        verification_query = query_result.get("verification_query", "")
        if not verification_query:
            print("Warning: Verification query not found")
            return None
        
        # Get source document ID
        doc_id = query_result.get("entity_a", {}).get("document_id", "")
        
        # Print log
        print(f"\nProcessing direct entity recommendation query:")
        print(f"  Entity A: {query_result.get('entity_a', {}).get('name', '')}")
        print(f"  Entity B: {query_result.get('entity_b_name', '')}")
        print(f"  Attribute X: {query_result.get('attribute_x', '')}")
        print(f"  Verification Query: {verification_query}")
        
        # Retrieve documents using verification query
        results = self.retrieve_documents(verification_query, doc_id=doc_id, top_k=top_k)
        
        # Return results
        return {
            "query_type": "recall_focused_verify",
            "entity_a": query_result.get("entity_a", {}),
            "entity_b_name": query_result.get("entity_b_name", ""),
            "attribute_x": query_result.get("attribute_x", ""),
            "verification_query": verification_query,
            "retrieved_documents": results
        }
    
    def process_search_queries(self, query_result, top_k=3):
        """
        Process search query type queries
        
        Args:
            query_result: Query result containing search queries
            top_k: Maximum number of documents to return for each query
            
        Returns:
            dict: Dictionary containing retrieval results
        """
        # Get search query list
        search_queries = query_result.get("search_queries", [])
        if not search_queries:
            print("Warning: Search queries not found")
            return None
        
        # Get source document ID
        doc_id = query_result.get("entity_a", {}).get("document_id", "")
        
        # Print log
        print(f"\nProcessing search queries:")
        print(f"  Entity A: {query_result.get('entity_a', {}).get('name', '')}")
        for i, query in enumerate(search_queries, 1):
            print(f"  Query {i}: {query}")
        
        # Retrieve documents for each query
        all_results = []
        
        for i, query in enumerate(search_queries, 1):
            print(f"\nExecuting query {i}/{len(search_queries)}: {query}")
            results = self.retrieve_documents(query, doc_id=doc_id, top_k=top_k)
            print(f"  Found {len(results)} documents")
            
            # Add results to the list
            for doc in results:
                # Check if the document is already in the results list
                if not any(existing_doc.get("id") == doc.get("id") for existing_doc in all_results):
                    all_results.append(doc)
        
        # Limit total number of results
        all_results = all_results[:top_k]
        
        # Return results
        return {
            "query_type": "search_queries",
            "entity_a": query_result.get("entity_a", {}),
            "search_queries": search_queries,
            "retrieved_documents": all_results
        }
    
    def process_query_results(self, input_file, output_file, top_k=5):
        """
        Process comparison query result file and retrieve relevant documents
        
        Args:
            input_file: Input file path (comparison query results)
            output_file: Output file path
            top_k: Maximum number of documents to return for each query
            
        Returns:
            bool: Whether processing was successful
        """
        try:
            print(f"\nReading comparison query file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process query results
            query_results = data.get("query_results", [])
            
            if not query_results:
                print("Query results not found in input file")
                return False
            
            print(f"Successfully read {len(query_results)} query results")
            
            # Save document content to local cache
            doc_map = {}
            if "documents" in data:
                print(f"Saving {len(data['documents'])} documents to cache")
                for doc in data["documents"]:
                    if "id" in doc and "contents" in doc:
                        doc_id = doc["id"]
                        doc_map[doc_id] = doc
                        self.documents_cache[doc_id] = doc
                        print(f"  Caching document {doc_id}: {doc['title'] if 'title' in doc else ''}")
            
            # Process each query result and retrieve relevant documents
            retrieval_results = []
            
            for i, result_item in enumerate(query_results, 1):
                print(f"\nProcessing query result {i}/{len(query_results)}")
                
                # Get query result
                query_result = result_item.get("query_result")
                
                if not query_result:
                    print("  Skipping: No query result found")
                    continue
                
                # Process based on query type
                query_type = query_result.get("query_type")
                
                if query_type == "recall_focused_verify":
                    # Process direct entity recommendation
                    retrieval_result = self.process_recall_focused_verify(query_result, top_k=top_k)
                elif query_type == "search_queries":
                    # Process search queries
                    retrieval_result = self.process_search_queries(query_result, top_k=top_k)
                else:
                    print(f"  Skipping: Unknown or invalid query type {query_type}")
                    continue
                
                if retrieval_result:
                    # Add entity data
                    entity_data = result_item.get("entity_data", {})
                    retrieval_result["entity_data"] = entity_data
                    
                    # Keep document A content
                    doc_id = entity_data.get("document_id", "")
                    if doc_id:
                        print(f"  Getting content of document A (document ID: {doc_id})")
                        
                        # Check if document exists in cache (from input file's documents array)
                        if doc_id in doc_map:
                            document_a = doc_map[doc_id]
                            if "contents" in document_a:
                                # Add document A content to entity data
                                entity_data["contents"] = document_a.get("contents", "")
                                print(f"  Successfully retrieved content of document A (length: {len(document_a.get('contents', ''))})")
                        else:
                            # If document is not in documents array, log warning
                            print(f"  Warning: Document {doc_id} not found in documents array, cannot retrieve content")
                    
                    retrieval_results.append(retrieval_result)
                else:
                    print("  Retrieval failed, skipping this query result")
            
            # Save processing results to output file
            if retrieval_results:
                # Collect cached documents
                documents = []
                for doc_id, doc in self.documents_cache.items():
                    if doc not in documents:
                        documents.append(doc)
                
                output_data = {
                    "retrieval_results": retrieval_results,
                    "timestamp": datetime.now().isoformat(),
                    "top_k": top_k,
                    "documents": documents
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                print(f"\nComparison query retrieval completed. Successfully processed {len(retrieval_results)}/{len(query_results)} query results")
                print(f"Results saved to: {output_file}")
                return True
            else:
                print("\nProcessing completed, but no documents were successfully retrieved")
                return False
                
        except Exception as e:
            import traceback
            print(f"Error occurred during processing: {str(e)}")
            print(traceback.format_exc())
            return False


if __name__ == "__main__":
    # Module functional test
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Create CompareRetriever instance
    compare_retriever = CompareRetriever(config)
    
    # Example input and output file paths
    input_file = "./comparison_queries.json"
    
    # Output file path
    output_file = "./comparison_retrieval_results.json"
    
    # Execute processing
    compare_retriever.process_query_results(input_file, output_file, top_k=3)
