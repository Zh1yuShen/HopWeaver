import json
import random
import os
from typing import List, Dict, Optional, Union, Any

class DocumentReader:
    """Document reader class, supports random document retrieval"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.index_filepath = filepath + '.index'
        self.line_positions = []
        self._build_or_load_index()
    
    def _build_or_load_index(self):
        """Build or load line index"""
        if os.path.exists(self.index_filepath):
            # Load existing index
            with open(self.index_filepath, 'r') as f:
                self.line_positions = json.load(f)
        else:
            # Build new index
            with open(self.filepath, 'rb') as f:
                position = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    self.line_positions.append(position)
                    position += len(line)
            
            # Save index
            with open(self.index_filepath, 'w') as f:
                json.dump(self.line_positions, f)
    
    def get_random_documents(self, count=1):
        """Randomly retrieve multiple documents"""
        docs = []
        for _ in range(count):
            doc = self.get_random_document()
            if doc:
                docs.append(doc)
        return docs
    
    def get_random_document(self, exclude_id=None):
        """
        Randomly retrieve a document, optionally excluding a specific ID
        
        Args:
            exclude_id: Document ID to exclude
            
        Returns:
        """
        # If no documents are loaded, return None
        if not self.line_positions:
            return None
        
        # Maximum number of attempts to avoid infinite loop
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            random_idx = random.randint(0, len(self.line_positions) - 1)
            position = self.line_positions[random_idx]
            
            with open(self.filepath, 'rb') as f:
                f.seek(position)
                line = f.readline()
                try:
                    doc = json.loads(line.decode('utf-8'))
                    # Validate that the document has necessary fields
                    if 'id' not in doc or 'contents' not in doc:
                        print(f"Warning: Document is missing necessary fields (id or contents), skipping this document")
                        continue  # Continue loop to try the next document
                    if exclude_id and doc.get('id') == exclude_id:
                        continue  # If this ID needs to be excluded, continue loop
                    return doc
                except json.JSONDecodeError:
                    print(f"Warning: Cannot parse document line, trying to get another document")
                    continue  # Continue loop to try the next document
        
        # If maximum attempts are reached and no valid document is found, return None
        print(f"Warning: Still no valid document found after {max_attempts} attempts")
        return None
            
    def get_document_by_id(self, doc_id):
        """Get document by ID"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if doc.get("id") == doc_id:
                        return doc
                except json.JSONDecodeError:
                    continue
        return None
    
    def sample_and_save_document_ids(self, count=1000, min_length=300, output_file="sampled_doc_ids.txt"):
        """Sample a specified number of document IDs and save them to a file
        
        Args:
            count: Number of documents to sample
            min_length: Minimum length of document content (characters)
            output_file: Output file path
            
        Returns:
            bool: Whether saving was successful
        """
        # Sample document IDs
        doc_ids = []
        attempts = 0
        max_attempts = count * 10  # Set maximum number of attempts
        
        print(f"Sampling {count} document IDs...")
        while len(doc_ids) < count and attempts < max_attempts:
            attempts += 1
            doc = self.get_random_document()
            
            # Check if the document is valid
            if doc and 'contents' in doc and 'id' in doc and len(doc['contents']) >= min_length:
                doc_id = doc['id']
                # Ensure no duplicate IDs
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                    if len(doc_ids) % 100 == 0:
                        print(f"  Sampled: {len(doc_ids)}/{count}")
        
        # Save IDs to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(doc_ids))
            print(f"Successfully sampled and saved {len(doc_ids)} document IDs to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving document IDs: {str(e)}")
            return False
    
    def load_document_ids(self, input_file="sampled_doc_ids.txt") -> List[str]:
        """Load a list of document IDs from a file
        
        Args:
            input_file: Input file path
            
        Returns:
            List[str]: List of document IDs
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                doc_ids = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(doc_ids)} document IDs from {input_file}")
            return doc_ids
        except Exception as e:
            print(f"Error loading document IDs: {str(e)}")
            return []
    
    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict]:
        """Get multiple documents by a list of IDs
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List[Dict]: List of documents
        """
        documents = []
        for doc_id in doc_ids:
            doc = self.get_document_by_id(doc_id)
            if doc:
                documents.append(doc)
        return documents
    
    def get_heuristic_documents(self, count=1, min_length=500):
        """Heuristically select documents to improve the generation of effective question chains
        
        Selecting documents of moderate length and correct format is more likely to contain useful entity relationships
        
        Args:
            count: Number of documents to select
            min_length: Minimum length of document content (characters)
            
        Returns:
            list: List of selected documents
        """
        selected_docs = []
        attempts = 0
        max_attempts = count * 5  # Set maximum number of attempts to avoid infinite loop
        
        while len(selected_docs) < count and attempts < max_attempts:
            doc = self.get_random_document()
            attempts += 1
            
            if not doc:
                continue
                
            # Check if content length meets requirements
            content = doc.get('contents', '')
            if isinstance(content, str) and len(content) >= min_length:
                selected_docs.append(doc)
                
        return selected_docs


if __name__ == "__main__":
    # Test code
    import sys
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Document Reader Tool')
    parser.add_argument('--action', type=str, required=True, choices=['sample', 'load', 'test'], 
                        help='Action to perform: sample (sample document IDs), load (load document IDs), test (test functionality)')
    parser.add_argument('--corpus', type=str, required=True, help='Corpus path or configuration file path')
    parser.add_argument('--corpus_file', type=str, help='Directly specify corpus file path, overrides --corpus')
    parser.add_argument('--output', type=str, default='sampled_doc_ids.txt', help='Output file path')
    parser.add_argument('--count', type=int, default=1000, help='Number of samples')
    parser.add_argument('--min_length', type=int, default=300, help='Minimum document length')
    parser.add_argument('--test_count', type=int, default=5, help='Number of documents to test loading')
    
    args = parser.parse_args()
    
    # Determine corpus file path
    corpus_file = args.corpus_file if args.corpus_file else args.corpus
    
    # Check if it is a YAML configuration file
    if corpus_file.endswith('.yaml') or corpus_file.endswith('.yml'):
        try:
            # Try to load corpus path from configuration file
            print(f"Attempting to load corpus path from configuration file {corpus_file}...")
            try:
                from flashrag.config import Config
                config = Config.from_file(corpus_file)
                corpus_file = config['corpus_filepath']
                print(f"Corpus path loaded from configuration file: {corpus_file}")
            except (ImportError, KeyError) as e:
                print(f"Error using configuration file: {str(e)}")
                # Try to use the original path
                if os.path.exists(corpus_file) and os.path.isfile(corpus_file):
                    print(f"Using original path as corpus file")
                else:
                    print(f"Error: Cannot determine corpus file path")
                    sys.exit(1)
        except Exception as e:
            print(f"Error processing configuration file: {str(e)}")
            sys.exit(1)
    
    # Check if corpus file exists
    if not os.path.exists(corpus_file) or not os.path.isfile(corpus_file):
        print(f"Error: Corpus file does not exist - {corpus_file}")
        sys.exit(1)
    
    # Initialize document reader
    print(f"Using corpus file: {corpus_file}")
    reader = DocumentReader(corpus_file)
    
    if args.action == "sample":
        # Sample document IDs and save
        print(f"Sampling {args.count} document IDs...")
        success = reader.sample_and_save_document_ids(
            count=args.count, 
            min_length=args.min_length, 
            output_file=args.output
        )
        if success:
            print(f"Sampling complete, document IDs saved to {args.output}")
        else:
            print("Sampling failed")
    
    elif args.action == "load":
        # Test loading document IDs and retrieving documents
        doc_ids = reader.load_document_ids(args.output)
        if not doc_ids:
            print(f"Failed to load any document IDs from {args.output}")
            sys.exit(1)
            
        # Select the first N for testing
        test_count = min(args.test_count, len(doc_ids))
        test_ids = doc_ids[:test_count]
        
        print(f"\nTesting reading {test_count} documents:")
        docs = reader.get_documents_by_ids(test_ids)
        
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}/{len(docs)}:")
            print(f"ID: {doc.get('id', 'unknown')}")
            print(f"Title: {doc.get('title', 'unknown')}")
            content = doc.get('contents', '')
            print(f"Content length: {len(content)} characters")
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Content preview: {preview}")
    
    elif args.action == "test":
        # Test basic functionality
        print(f"\nTesting random document reading functionality...")
        doc = reader.get_random_document()
        if doc:
            print(f"\nRandom document:")
            print(f"ID: {doc.get('id', 'unknown')}")
            print(f"Title: {doc.get('title', 'unknown')}")
            content = doc.get('contents', '')
            print(f"Content length: {len(content)} characters")
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"Content preview: {preview}")
        else:
            print("\nFailed to retrieve random document")