#!/usr/bin/env python3
"""
Check the current state of generated embeddings and vectors.
"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from peatlearn.embedding.embed_corpus import QAPair

def check_vectors():
    """Check what vectors have been generated so far."""
    
    checkpoint_file = Path("data/embeddings/vectors/checkpoint.json")
    completed_pairs_file = Path("data/embeddings/vectors/completed_pairs.pkl")
    
    print("🔍 Ray Peat Embedding Vector Status")
    print("=" * 50)
    
    # Check checkpoint
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        completed = checkpoint.get("completed_count", 0)
        total = checkpoint.get("total_count", 0)
        cost = checkpoint.get("total_cost", 0.0)
        last_updated = checkpoint.get("last_updated", "Unknown")
        model = checkpoint.get("model", "Unknown")
        
        progress_pct = (completed / total * 100) if total > 0 else 0
        
        print(f"📊 Checkpoint Status:")
        print(f"   • Completed: {completed:,}/{total:,} pairs ({progress_pct:.1f}%)")
        print(f"   • Cost so far: ${cost:.4f}")
        print(f"   • Model used: {model}")
        print(f"   • Last updated: {last_updated}")
        print()
        
    else:
        print("❌ No checkpoint found")
        return
    
    # Check actual vectors
    if completed_pairs_file.exists():
        print(f"📦 Loading completed pairs...")
        
        with open(completed_pairs_file, 'rb') as f:
            completed_pairs = pickle.load(f)
        
        print(f"✅ Found {len(completed_pairs)} completed Q&A pairs with embeddings")
        
        # Analyze first few pairs
        if completed_pairs:
            first_pair = completed_pairs[0]
            print(f"\n🔍 Sample Analysis:")
            print(f"   • Embedding dimensions: {len(first_pair.embedding) if first_pair.embedding is not None else 'None'}")
            
            if first_pair.embedding is not None:
                embedding_array = np.array(first_pair.embedding)
                print(f"   • Embedding dtype: {embedding_array.dtype}")
                print(f"   • Min/Max values: {embedding_array.min():.4f} / {embedding_array.max():.4f}")
                print(f"   • Mean value: {embedding_array.mean():.4f}")
                print(f"   • Standard deviation: {embedding_array.std():.4f}")
            
            print(f"\n📝 Sample Q&A Pair:")
            print(f"   • Source: {first_pair.source_file}")
            print(f"   • Context: {first_pair.context[:100]}...")
            print(f"   • Response: {first_pair.ray_peat_response[:100]}...")
            print(f"   • Tokens: {first_pair.tokens}")
        
        # Count successful embeddings
        successful = sum(1 for pair in completed_pairs if pair.embedding is not None)
        failed = len(completed_pairs) - successful
        
        print(f"\n📈 Embedding Success Rate:")
        print(f"   • Successful: {successful:,}")
        print(f"   • Failed: {failed:,}")
        print(f"   • Success rate: {(successful/len(completed_pairs)*100):.1f}%")
        
        # Estimate storage size
        if successful > 0:
            embedding_size = len(completed_pairs[0].embedding) if completed_pairs[0].embedding is not None else 0
            total_vectors = successful * embedding_size * 4  # 4 bytes per float32
            total_mb = total_vectors / (1024 * 1024)
            
            print(f"\n💾 Storage Analysis:")
            print(f"   • Vector size: {embedding_size} dimensions")
            print(f"   • Current storage: {total_mb:.2f} MB")
            
            # Project full corpus
            if total > 0:
                projected_mb = (total_mb / successful) * total
                print(f"   • Projected full corpus: {projected_mb:.2f} MB")
        
    else:
        print("❌ No completed pairs file found")

if __name__ == "__main__":
    check_vectors() 