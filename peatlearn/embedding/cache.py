"""
Embedding Cache System for PeatLearn
Avoids re-embedding documents that haven't changed
Uses content hashing to detect changes
"""
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime

class EmbeddingCache:
    def __init__(self, cache_dir='embedding/cache'):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_file = self.cache_dir / 'embeddings.pkl'
        self.metadata_file = self.cache_dir / 'metadata.json'
        
        # Load existing cache
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()
        
        print(f"📦 Embedding cache initialized")
        print(f"   Cached embeddings: {len(self.embeddings)}")
        print(f"   Cache location: {self.cache_dir}")
    
    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _load_embeddings(self) -> Dict:
        """Load embeddings from cache"""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️  Error loading embeddings cache: {e}")
                return {}
        return {}
    
    def _load_metadata(self) -> Dict:
        """Load metadata from cache"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading metadata cache: {e}")
                return {}
        return {}
    
    def _save_embeddings(self):
        """Save embeddings to cache"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def _save_metadata(self):
        """Save metadata to cache"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get(self, content: str, doc_id: str = None) -> Optional[np.ndarray]:
        """
        Get cached embedding for content
        
        Args:
            content: Document content
            doc_id: Optional document identifier
            
        Returns:
            Cached embedding or None if not found
        """
        content_hash = self._compute_hash(content)
        
        if content_hash in self.embeddings:
            # Update access time
            if content_hash in self.metadata:
                self.metadata[content_hash]['last_accessed'] = datetime.now().isoformat()
                self.metadata[content_hash]['access_count'] = self.metadata[content_hash].get('access_count', 0) + 1
            
            return self.embeddings[content_hash]
        
        return None
    
    def set(
        self,
        content: str,
        embedding: np.ndarray,
        doc_id: str = None,
        metadata: Dict = None
    ):
        """
        Cache an embedding
        
        Args:
            content: Document content
            embedding: Embedding vector
            doc_id: Optional document identifier
            metadata: Optional additional metadata
        """
        content_hash = self._compute_hash(content)
        
        # Store embedding
        self.embeddings[content_hash] = embedding
        
        # Store metadata
        self.metadata[content_hash] = {
            'doc_id': doc_id,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1,
            'content_length': len(content),
            'embedding_dim': len(embedding),
            **(metadata or {})
        }
        
        # Persist to disk
        self._save_embeddings()
        self._save_metadata()
    
    def needs_embedding(self, content: str) -> bool:
        """
        Check if content needs to be embedded
        
        Args:
            content: Document content
            
        Returns:
            True if embedding needed, False if cached
        """
        content_hash = self._compute_hash(content)
        return content_hash not in self.embeddings
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size_mb = 0
        if self.embeddings:
            # Estimate size
            sample_embedding = next(iter(self.embeddings.values()))
            bytes_per_embedding = sample_embedding.nbytes
            total_size_mb = (len(self.embeddings) * bytes_per_embedding) / (1024 * 1024)
        
        return {
            'total_cached': len(self.embeddings),
            'cache_size_mb': round(total_size_mb, 2),
            'cache_location': str(self.cache_dir),
            'oldest_entry': min(
                (m['created'] for m in self.metadata.values()),
                default=None
            ),
            'most_accessed': max(
                self.metadata.items(),
                key=lambda x: x[1].get('access_count', 0),
                default=(None, {'access_count': 0})
            )[1].get('doc_id')
        }
    
    def clear(self):
        """Clear the entire cache"""
        self.embeddings = {}
        self.metadata = {}
        self._save_embeddings()
        self._save_metadata()
        print("🗑️  Cache cleared")


def demo():
    """Demo of embedding cache"""
    cache = EmbeddingCache()
    
    # Simulate embedding
    content = "Ray Peat discusses thyroid function..."
    embedding = np.random.rand(3072)  # 3072-dim vector
    
    # Check if needs embedding
    if cache.needs_embedding(content):
        print("🆕 Content not cached, need to embed")
        # Simulate API call
        cache.set(content, embedding, doc_id="doc_123")
    else:
        print("✅ Content found in cache!")
        cached_embedding = cache.get(content)
        print(f"   Retrieved embedding: {cached_embedding.shape}")
    
    # Stats
    stats = cache.get_stats()
    print(f"\n📊 Cache Stats:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demo()