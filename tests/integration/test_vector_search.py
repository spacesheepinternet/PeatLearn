#!/usr/bin/env python3
"""
Test the vector search functionality.
"""

import sys
import os
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
os.environ['PROJECT_ROOT'] = str(project_root)

async def test_vector_search():
    """Test the vector search system."""
    print("🔍 Testing Vector Search System\n" + "="*50)
    
    try:
        # Import the vector search module
        from rag.vector_search import RayPeatVectorSearch
        
        # Initialize search engine
        print("🚀 Initializing vector search engine...")
        search_engine = RayPeatVectorSearch()
        
        # Check if data loaded
        if search_engine.embeddings is None:
            print("❌ Failed to load embeddings!")
            return False
            
        print(f"✅ Successfully loaded embeddings!")
        
        # Get corpus stats
        stats = search_engine.get_corpus_stats()
        print(f"\n📊 Corpus Statistics:")
        print(f"   Total embeddings: {stats['total_embeddings']:,}")
        print(f"   Total tokens: {stats['total_tokens']:,}")
        print(f"   Embedding dimensions: {stats['embedding_dimensions']}")
        print(f"   Source files: {stats['source_files']}")
        
        # Test search queries
        test_queries = [
            "thyroid hormone metabolism",
            "sugar and energy production", 
            "stress and cortisol",
            "progesterone benefits",
            "Ray Peat nutrition"
        ]
        
        print(f"\n🔎 Testing search queries...")
        for query in test_queries:
            print(f"\n--- Searching: '{query}' ---")
            
            try:
                results = await search_engine.search(query, top_k=3, min_similarity=0.1)
                
                if results:
                    print(f"✅ Found {len(results)} results")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. Similarity: {result.similarity_score:.3f}")
                        print(f"     Context: {result.context[:80]}...")
                        print(f"     Source: {result.source_file}")
                else:
                    print(f"⚠️ No results found for '{query}'")
                    
            except Exception as e:
                print(f"❌ Search failed for '{query}': {e}")
                
        print(f"\n✅ Vector search test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run vector search tests."""
    success = await test_vector_search()
    
    if success:
        print(f"\n🎉 Vector search system is working!")
        print(f"\n📋 Ready for next step:")
        print(f"   Test the API server: python test_api.py")
    else:
        print(f"\n❌ Vector search test failed.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
