#!/usr/bin/env python3
"""
Simple test script for the RAG system.
"""

import sys
import os
import asyncio
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
os.environ['PROJECT_ROOT'] = str(project_root)

async def test_basic_functionality():
    """Test basic functionality without full imports."""
    print("🧪 Testing Ray Peat RAG System\n" + "="*50)
    
    try:
        # Test 1: Check if embedding files exist
        vectors_dir = project_root / "embedding" / "vectors"
        print(f"📁 Checking vectors directory: {vectors_dir}")
        
        if not vectors_dir.exists():
            print("❌ Vectors directory not found!")
            return False
            
        # Check for embedding files
        embedding_files = list(vectors_dir.glob("embeddings_*.npy"))
        metadata_files = list(vectors_dir.glob("metadata_*.json"))
        
        print(f"📊 Found {len(embedding_files)} embedding files")
        print(f"📊 Found {len(metadata_files)} metadata files")
        
        if not embedding_files or not metadata_files:
            print("❌ No embedding or metadata files found!")
            return False
            
        # Test 2: Load and check metadata
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        print(f"📄 Loading metadata from: {latest_metadata.name}")
        
        with open(latest_metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        print(f"✅ Loaded {len(metadata)} metadata entries")
        
        # Show a sample entry
        if metadata:
            sample = metadata[0]
            print(f"\n📝 Sample entry:")
            print(f"   ID: {sample['id']}")
            print(f"   Context: {sample['context'][:100]}...")
            print(f"   Source: {sample['source_file']}")
            print(f"   Tokens: {sample['tokens']}")
            
        print(f"\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_imports():
    """Test if we can import our modules."""
    print(f"\n🔧 Testing imports...")
    
    try:
        # Test numpy and sklearn
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ NumPy and scikit-learn imported successfully")
        
        # Test basic vector operations
        vec1 = np.random.random((1, 768))
        vec2 = np.random.random((1, 768))
        similarity = cosine_similarity(vec1, vec2)[0][0]
        print(f"✅ Cosine similarity test: {similarity:.3f}")
        
        # Test aiohttp
        import aiohttp
        print("✅ aiohttp imported successfully")
        
        # Test fastapi
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

async def main():
    """Run all tests."""
    success = True
    
    success &= await test_basic_functionality()
    success &= await test_imports()
    
    if success:
        print(f"\n🎉 All tests passed! The RAG system is ready.")
        print(f"\n📋 Next steps:")
        print(f"   1. Test vector search: python test_vector_search.py")
        print(f"   2. Start the API server: python inference/backend/app.py")
        print(f"   3. Visit http://localhost:8000/docs for API documentation")
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
