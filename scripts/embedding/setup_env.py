#!/usr/bin/env python3
"""
Ray Peat Legacy - Environment Setup for Embedding Generation

This script helps set up the required environment variables and dependencies
for running the embedding pipeline.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = {
        'GEMINI_API_KEY': 'Google Gemini API key for embedding generation'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  • {var}: {description}")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        print("\n".join(missing_vars))
        print("\n💡 To set up environment variables:")
        print("   1. Create a .env file in the project root")
        print("   2. Add: GEMINI_API_KEY=your_api_key_here")
        print("   3. Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    else:
        print("✅ All required environment variables are set!")
        return True

def install_dependencies():
    """Install required dependencies for embedding generation."""
    print("📦 Installing embedding dependencies...")
    import subprocess
    
    try:
        # Install from embedding requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(Path(__file__).parent / "requirements.txt")
        ], check=True, capture_output=True, text=True)
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🔧 Ray Peat Legacy - Embedding Environment Setup")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    try:
        import aiohttp
        import numpy
        print("✅ Core dependencies already installed")
    except ImportError:
        print("⚠️  Some dependencies missing, installing...")
        if not install_dependencies():
            return
    
    # Check environment variables
    print("\n2. Checking environment variables...")
    if not check_environment():
        return
    
    # Test API connection (optional)
    print("\n3. Testing API connection...")
    try:
        from config.settings import settings
        if settings.GEMINI_API_KEY:
            print("✅ Gemini API key found in configuration")
        else:
            print("❌ Gemini API key not found in settings")
            return
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    print("\n🎉 Environment setup complete!")
    print("You can now run: python embedding/embed_corpus.py")

if __name__ == "__main__":
    main() 