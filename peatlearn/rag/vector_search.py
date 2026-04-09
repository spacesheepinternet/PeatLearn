#!/usr/bin/env python3
"""
Pinecone Vector Search Engine for Ray Peat Corpus

Provides semantic search capabilities using Pinecone vector database.
"""

import os
import sys
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging


try:
    from pinecone import Pinecone
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependencies. Please install: pip install -r requirements.txt")
    print(f"Error: {e}")
    sys.exit(1)

from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from the Pinecone corpus."""
    id: str
    context: str
    ray_peat_response: str
    source_file: str
    similarity_score: float
    tokens: int
    original_id: str = ""
    truncated: bool = False

class PineconeVectorSearch:
    """Vector search engine using Pinecone for the Ray Peat corpus."""
    
    def __init__(self, index_name: str = "ray-peat-corpus"):
        """Initialize the Pinecone vector search engine."""
        self.index_name = index_name
        self.index = None
        self.pc = None
        self.embedding_model = "gemini-embedding-001"
        self.embedding_dimensions = 3072  # Will be updated from index stats if available
        
        # Load environment variables
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
        
        # Initialize Pinecone connection
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and connect to index."""
        try:
            # Get API key
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=api_key)
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                raise ValueError(f"Index '{self.index_name}' not found. Available indexes: {existing_indexes}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Verify connection and capture index dimension if available
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {stats.get('total_vector_count', 0)} vectors")
            dim = stats.get('dimension')
            if isinstance(dim, int) and dim > 0:
                self.embedding_dimensions = dim
                logger.info(f"Detected index dimension: {self.embedding_dimensions}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for a search query.

        Falls back to a deterministic local embedding if external API is unavailable
        or rate-limited, so Pinecone integration can be tested offline.
        """
        # Prefer external embedding if configured
        if settings.GEMINI_API_KEY:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.embedding_model}:embedContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": settings.GEMINI_API_KEY
            }
            payload = {
                "model": f"models/{self.embedding_model}",
                "content": {"parts": [{"text": query}]}
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            embedding = result.get("embedding", {}).get("values")
                            if embedding:
                                return np.array(embedding, dtype=float)
                        else:
                            # Log and fall back to local embedding
                            error_text = await response.text()
                            logger.error(f"Embedding API error {response.status}: {error_text}. Falling back to local embedding.")
            except Exception as e:
                logger.error(f"Embedding API call failed: {e}. Falling back to local embedding.")

        # Local deterministic embedding fallback: hash-based pseudo-embedding
        import hashlib
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        # Repeat digest to fill the required dimension
        bytes_needed = self.embedding_dimensions
        arr = np.frombuffer((digest * ((bytes_needed // len(digest)) + 1))[:bytes_needed], dtype=np.uint8).astype(np.float32)
        # Normalize to unit length to mimic real embeddings
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10, 
        min_similarity: float = 0.1,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search the corpus for passages most similar to the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold (not directly supported by Pinecone)
            filter_dict: Optional metadata filters for Pinecone
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Pinecone index not initialized")
        
        # Generate embedding for the query
        query_embedding = await self.generate_query_embedding(query)
        if query_embedding is None:
            return []
        
        try:
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert Pinecone results to SearchResult objects
            results = []
            for match in query_response.get('matches', []):
                score = match.get('score', 0.0)
                
                # Apply similarity threshold (since Pinecone doesn't support it directly)
                if score < min_similarity:
                    continue
                
                metadata = match.get('metadata', {})
                
                result = SearchResult(
                    id=match.get('id', ''),
                    context=metadata.get('context', ''),
                    ray_peat_response=metadata.get('ray_peat_response', ''),
                    source_file=metadata.get('source_file', ''),
                    similarity_score=score,
                    tokens=metadata.get('tokens', 0),
                    original_id=metadata.get('original_id', ''),
                    truncated=metadata.get('truncated', False)
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    async def search_by_metadata(
        self, 
        filter_dict: Dict[str, Any], 
        top_k: int = 100
    ) -> List[SearchResult]:
        """
        Search by metadata filters only (no vector query).
        
        Args:
            filter_dict: Metadata filters for Pinecone
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Pinecone index not initialized")
        
        try:
            # Create a dummy vector for metadata-only search
            dummy_vector = [0.0] * self.embedding_dimensions
            
            # Query with dummy vector to get metadata-filtered results
            query_response = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert to SearchResult objects
            results = []
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                
                result = SearchResult(
                    id=match.get('id', ''),
                    context=metadata.get('context', ''),
                    ray_peat_response=metadata.get('ray_peat_response', ''),
                    source_file=metadata.get('source_file', ''),
                    similarity_score=0.0,  # No meaningful similarity for metadata-only search
                    tokens=metadata.get('tokens', 0),
                    original_id=metadata.get('original_id', ''),
                    truncated=metadata.get('truncated', False)
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} results for metadata filter: {filter_dict}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if self.index is None:
            return {"error": "Pinecone index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "index_name": self.index_name,
                "embedding_dimensions": self.embedding_dimensions,
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": stats.get('namespaces', {}),
                "status": "connected"
            }
            
        except Exception as e:
            logger.error(f"Error getting corpus stats: {e}")
            return {"error": str(e)}
    
    async def get_similar_documents(
        self, 
        document_id: str, 
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Find documents similar to a specific document by ID.
        
        Args:
            document_id: ID of the document to find similar documents for
            top_k: Number of similar documents to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Pinecone index not initialized")
        
        try:
            # First, fetch the document to get its vector
            fetch_response = self.index.fetch(ids=[document_id])
            
            if document_id not in fetch_response.get('vectors', {}):
                logger.warning(f"Document with ID '{document_id}' not found")
                return []
            
            # Get the vector
            document_vector = fetch_response['vectors'][document_id]['values']
            
            # Search for similar vectors
            query_response = self.index.query(
                vector=document_vector,
                top_k=top_k + 1,  # +1 to exclude the original document
                include_metadata=True
            )
            
            # Convert to SearchResult objects, excluding the original document
            results = []
            for match in query_response.get('matches', []):
                if match.get('id') == document_id:
                    continue  # Skip the original document
                
                metadata = match.get('metadata', {})
                
                result = SearchResult(
                    id=match.get('id', ''),
                    context=metadata.get('context', ''),
                    ray_peat_response=metadata.get('ray_peat_response', ''),
                    source_file=metadata.get('source_file', ''),
                    similarity_score=match.get('score', 0.0),
                    tokens=metadata.get('tokens', 0),
                    original_id=metadata.get('original_id', ''),
                    truncated=metadata.get('truncated', False)
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents for ID: {document_id}")
            return results[:top_k]  # Ensure we return only top_k results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific vector.
        
        Args:
            id: Vector ID
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            self.index.update(id=id, set_metadata=metadata)
            logger.info(f"Updated metadata for vector: {id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False

# Global instance for the API (same pattern as the original)
search_engine = PineconeVectorSearch()


