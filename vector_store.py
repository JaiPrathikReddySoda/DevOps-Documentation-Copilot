"""
Vector Store Module

This module handles vector storage and retrieval using FAISS (Facebook AI Similarity Search).
It provides efficient similarity search capabilities for document chunks and their embeddings.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path


class VectorStore:
    """
    FAISS-based vector store for document embeddings.
    
    This class handles:
    - Creating embeddings from text chunks
    - Storing embeddings in FAISS index
    - Performing similarity search
    - Persisting and loading the index
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "vector_index",
                 dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
            index_path: Path to store the FAISS index and metadata
            dimension: Dimension of the embeddings
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.dimension = dimension
        
        # Initialize the sentence transformer model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Store metadata for each vector
        self.metadata_store = []
        
        # Create directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata' keys
        """
        if not chunks:
            return
        
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata_store.extend(chunks)
        
        print(f"Added {len(chunks)} chunks to vector store. Total vectors: {self.index.ntotal}")
    
    def search(self, 
               query: str, 
               k: int = 5, 
               threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing similar chunks with scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        # Filter results by threshold and format output
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.metadata_store):
                result = {
                    'content': self.metadata_store[idx]['content'],
                    'metadata': self.metadata_store[idx]['metadata'],
                    'score': float(score)
                }
                results.append(result)
        
        return results
    
    def search_with_filters(self, 
                           query: str, 
                           k: int = 5,
                           threshold: float = 0.5,
                           file_types: Optional[List[str]] = None,
                           sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search with additional filters.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity score threshold
            file_types: Filter by file types
            sources: Filter by source files
            
        Returns:
            List of dictionaries containing similar chunks with scores
        """
        # Get initial search results
        results = self.search(query, k * 2, threshold)  # Get more results for filtering
        
        # Apply filters
        filtered_results = []
        for result in results:
            metadata = result['metadata']
            
            # Check file type filter
            if file_types and metadata.get('type') not in file_types:
                continue
            
            # Check source filter
            if sources and metadata.get('source') not in sources:
                continue
            
            filtered_results.append(result)
            
            # Stop if we have enough results
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'model_name': self.model_name,
            'file_types': {},
            'sources': set(),
            'total_tokens': 0
        }
        
        for chunk in self.metadata_store:
            file_type = chunk['metadata']['type']
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            stats['sources'].add(chunk['metadata']['source'])
            stats['total_tokens'] += chunk['metadata'].get('token_count', 0)
        
        stats['sources'] = list(stats['sources'])
        stats['unique_sources'] = len(stats['sources'])
        
        return stats
    
    def save(self) -> None:
        """Save the vector store to disk."""
        # Save FAISS index
        index_file = self.index_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata as JSON
        metadata_file = self.index_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2)
        
        # Save configuration
        config_file = self.index_path / "config.json"
        config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Vector store saved to {self.index_path}")
    
    def load(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.json"
        config_file = self.index_path / "config.json"
        
        if not all(f.exists() for f in [index_file, metadata_file, config_file]):
            # Clean up potentially corrupted old pickle files
            old_pickle_file = self.index_path / "metadata.pkl"
            if old_pickle_file.exists():
                old_pickle_file.unlink()
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata from JSON
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata_store = json.load(f)
            
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.model_name = config['model_name']
                self.dimension = config['dimension']
            
            print(f"Vector store loaded from {self.index_path}")
            print(f"Total vectors: {self.index.ntotal}")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        print("Vector store cleared")
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all vectors from a specific source.
        
        Args:
            source: Source to delete
            
        Returns:
            Number of vectors deleted
        """
        if self.index.ntotal == 0:
            return 0
        
        # Find indices to keep
        keep_indices = []
        for i, chunk in enumerate(self.metadata_store):
            if chunk['metadata']['source'] != source:
                keep_indices.append(i)
        
        if len(keep_indices) == len(self.metadata_store):
            return 0  # No vectors to delete
        
        # Rebuild index with kept vectors
        new_index = faiss.IndexFlatIP(self.dimension)
        new_metadata = []
        
        # Get embeddings for kept vectors
        kept_texts = [self.metadata_store[i]['content'] for i in keep_indices]
        if kept_texts:
            embeddings = self.embedding_model.encode(kept_texts)
            faiss.normalize_L2(embeddings)
            new_index.add(embeddings.astype('float32'))
            new_metadata = [self.metadata_store[i] for i in keep_indices]
        
        # Replace old index and metadata
        self.index = new_index
        self.metadata_store = new_metadata
        
        deleted_count = len(self.metadata_store) - len(keep_indices)
        print(f"Deleted {deleted_count} vectors from source: {source}")
        
        return deleted_count
    
    def get_sources(self) -> List[str]:
        """
        Get list of all sources in the vector store.
        
        Returns:
            List of source paths/URLs
        """
        return list(set(chunk['metadata']['source'] for chunk in self.metadata_store))
    
    def get_file_types(self) -> Dict[str, int]:
        """
        Get count of chunks by file type.
        
        Returns:
            Dictionary mapping file types to counts
        """
        file_types = {}
        for chunk in self.metadata_store:
            file_type = chunk['metadata']['type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        return file_types 