#!/usr/bin/env python3
"""
Test the FastAPI server and RAG endpoints.
"""

import sys
import os
import asyncio
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
os.environ['PROJECT_ROOT'] = str(project_root)

async def test_api_imports():
    """Test if we can import and create the API."""
    print("🔧 Testing API imports and initialization...")
    
    try:
        # Import the app
        from app.api import app
        print("✅ FastAPI app imported successfully")
        
        # Test if the app is configured
        print(f"   App title: {app.title}")
        print(f"   Routes: {len(app.routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ API import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_system():
    """Test the RAG system directly."""
    print(f"\n🤖 Testing RAG System...")
    
    try:
        from rag.rag_system import RayPeatRAG
        
        # Initialize RAG system
        rag = RayPeatRAG()
        print("✅ RAG system initialized")
        
        # Test question answering
        question = "What does Ray Peat say about thyroid function?"
        print(f"\n❓ Testing question: '{question}'")
        
        print("   Generating answer... (this may take 10-15 seconds)")
        start_time = time.time()
        
        response = await rag.answer_question(question, max_sources=3)
        
        elapsed = time.time() - start_time
        print(f"✅ Answer generated in {elapsed:.1f} seconds")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Sources used: {len(response.sources)}")
        
        print(f"\n📝 Answer:")
        print(f"   {response.answer[:200]}...")
        
        print(f"\n📚 Sources:")
        for i, source in enumerate(response.sources, 1):
            print(f"   {i}. {source.source_file} (similarity: {source.similarity_score:.3f})")
            
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run API tests."""
    print("🧪 Testing RAG API System\n" + "="*50)
    
    success = True
    success &= await test_api_imports()
    success &= await test_rag_system()
    
    if success:
        print(f"\n🎉 All API tests passed!")
        print(f"\n📋 Ready to start the server:")
        print(f"   cd inference/backend")
        print(f"   source ../../venv/bin/activate")
        print(f"   python app.py")
        print(f"\n   Then visit: http://localhost:8000/docs")
    else:
        print(f"\n❌ Some API tests failed.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
