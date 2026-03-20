#!/usr/bin/env python3
"""
Quick test to verify Gemini embedding API is working correctly.
"""

import sys
import asyncio
import aiohttp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings

async def test_single_embedding():
    """Test generating a single embedding to verify API connectivity."""
    
    if not settings.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in environment")
        return False
    
    test_text = "Ray Peat discusses the importance of thyroid function for metabolism."
    model = settings.EMBEDDING_MODEL
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.GEMINI_API_KEY
    }
    
    payload = {
        "model": f"models/{model}",
        "content": {
            "parts": [{"text": test_text}]
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"🧪 Testing embedding with model: {model}")
            print(f"📝 Test text: {test_text}")
            print("🌐 Making API call...")
            
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    embedding = data.get('embedding', {}).get('values', [])
                    
                    print(f"✅ Success! Received embedding")
                    print(f"📊 Embedding dimensions: {len(embedding)}")
                    print(f"🔢 First 5 values: {embedding[:5]}")
                    print(f"📈 Min/Max values: {min(embedding):.4f} / {max(embedding):.4f}")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ API Error {response.status}: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

async def main():
    """Run the embedding test."""
    print("🚀 Gemini Embedding API Test")
    print("=" * 40)
    
    success = await test_single_embedding()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Embedding API is working correctly!")
        print("🔄 Your main embedding process should be working fine.")
    else:
        print("❌ Embedding API test failed!")
        print("🔧 Check your API key and network connection.")

if __name__ == "__main__":
    asyncio.run(main()) 