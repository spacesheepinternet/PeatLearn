#!/usr/bin/env python3
"""
Pinecone Migration - Complete Setup Script

This script guides you through the entire process of migrating your 
Ray Peat embeddings to Pinecone vector database.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if embeddings exist
    vectors_dir = Path(__file__).parent.parent / "vectors"
    embedding_files = list(vectors_dir.glob("embeddings_*.npy"))
    metadata_files = list(vectors_dir.glob("metadata_*.json"))
    
    if not embedding_files or not metadata_files:
        print("❌ No existing embeddings found!")
        print("   Please run 'python embedding/embed_corpus.py' first to generate embeddings.")
        return False
    
    print(f"✅ Found {len(embedding_files)} embedding files")
    
    # Check for .env file
    env_file = Path(__file__).parent.parent.parent / ".env"
    if not env_file.exists():
        print("⚠️ .env file not found. Creating one...")
        with open(env_file, 'w') as f:
            f.write("# Environment variables for PeatLearn project\n")
            f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
            f.write("PINECONE_API_KEY=your_pinecone_api_key_here\n")
        print(f"✅ Created .env file at: {env_file}")
    
    # Load environment and check for API keys
    load_dotenv(env_file)
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    
    if not gemini_key or gemini_key == 'your_gemini_api_key_here':
        print("⚠️ GEMINI_API_KEY not set in .env file")
    else:
        print("✅ GEMINI_API_KEY found")
    
    if not pinecone_key or pinecone_key == 'your_pinecone_api_key_here':
        print("❌ PINECONE_API_KEY not set in .env file")
        print("   Please add your Pinecone API key to the .env file")
        return False
    else:
        print("✅ PINECONE_API_KEY found")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Pinecone dependencies...")
    
    import subprocess
    
    try:
        # Install Pinecone requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(Path(__file__).parent / "requirements.txt")
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

async def run_upload():
    """Run the upload process."""
    print("\n⬆️ Starting upload to Pinecone...")
    
    try:
        # Import upload script and run it
        from .upload import RayPeatPineconeUploader, PineconeConfig
        
        # Get API key
        api_key = os.getenv('PINECONE_API_KEY')
        config = PineconeConfig(api_key=api_key)
        
        print(f"📋 Upload configuration:")
        print(f"   • Index: {config.index_name}")
        print(f"   • Dimensions: {config.vector_dimension}")
        print(f"   • Batch size: {config.batch_size}")
        
        # Initialize uploader
        uploader = RayPeatPineconeUploader(config)
        
        # Load and upload
        vectors, metadata = uploader.load_latest_embeddings()
        prepared_vectors = uploader.prepare_vectors_for_upload(vectors, metadata)
        upload_stats = uploader.upload_vectors(prepared_vectors)
        
        # Verify
        verification_passed = uploader.verify_upload(len(vectors))
        uploader.save_upload_report(upload_stats, len(vectors), len(metadata))
        
        if verification_passed:
            print("✅ Upload completed successfully!")
            return True
        else:
            print("⚠️ Upload completed but verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

async def run_tests():
    """Run comprehensive tests."""
    print("\n🧪 Running tests...")
    
    try:
        from .test_pinecone import PineconeTestSuite
        
        test_suite = PineconeTestSuite()
        report = await test_suite.run_all_tests()
        
        success_rate = report['summary']['success_rate']
        
        if success_rate == 1.0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"⚠️ Tests partially passed ({success_rate:.1%})")
            return False
            
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def integrate_with_existing_code():
    """Integrate Pinecone with existing RAG systems."""
    print("\n🔗 Integrating with existing code...")
    
    try:
        from .integrate_pinecone import PineconeIntegrator
        
        integrator = PineconeIntegrator()
        success = integrator.run_integration()
        
        if success:
            print("✅ Integration completed!")
            return True
        else:
            print("⚠️ Integration partially completed")
            return False
            
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False

async def main():
    """Main migration workflow."""
    print("🚀 Ray Peat Corpus - Complete Pinecone Migration")
    print("=" * 60)
    
    print("""
This script will help you migrate your Ray Peat embeddings to Pinecone.

Steps:
1. Check prerequisites
2. Install dependencies
3. Upload embeddings to Pinecone
4. Run comprehensive tests
5. Integrate with existing code

Let's get started!
""")
    
    # Step 1: Prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return
    
    # Ask for confirmation
    response = input("\n🚀 Ready to proceed with migration? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Migration cancelled.")
        return
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please install manually:")
        print("   pip install -r embedding/pinecone/requirements.txt")
        return
    
    # Step 3: Upload embeddings
    upload_success = await run_upload()
    if not upload_success:
        print("\n❌ Upload failed. Check the logs above for details.")
        return
    
    # Step 4: Run tests
    test_success = await run_tests()
    if not test_success:
        print("\n⚠️ Some tests failed, but migration may still be functional.")
        response = input("Continue with integration? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Migration stopped. You can run tests manually later with:")
            print("   python embedding/pinecone/test_pinecone.py")
            return
    
    # Step 5: Integration
    integration_success = integrate_with_existing_code()
    
    # Final summary
    print(f"\n🎉 Migration Summary:")
    print(f"   ✅ Prerequisites: OK")
    print(f"   ✅ Dependencies: OK")
    print(f"   {'✅' if upload_success else '❌'} Upload: {'OK' if upload_success else 'FAILED'}")
    print(f"   {'✅' if test_success else '⚠️'} Tests: {'OK' if test_success else 'PARTIAL'}")
    print(f"   {'✅' if integration_success else '⚠️'} Integration: {'OK' if integration_success else 'PARTIAL'}")
    
    if upload_success:
        print(f"\n📚 Next Steps:")
        print(f"   1. Your embeddings are now in Pinecone!")
        print(f"   2. Update your applications to use the new backend")
        print(f"   3. Check PINECONE_MIGRATION.md for details")
        print(f"   4. Use embedding/pinecone/utils.py for management")
        
        print(f"\n🛠️ Management Commands:")
        print(f"   • python embedding/pinecone/utils.py          # Interactive management")
        print(f"   • python embedding/pinecone/test_pinecone.py  # Run tests")
        print(f"   • python embedding/pinecone/upload.py         # Re-upload if needed")
        
        print(f"\n🔍 Quick Test:")
        print(f"   Try this in Python:")
        print(f"   from embedding.pinecone.rag_system import rag_system")
        print(f"   response = await rag_system.answer_question('What is metabolism?')")
        print(f"   print(response.answer)")
    else:
        print(f"\n❌ Migration incomplete. Check logs and try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏸️ Migration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        sys.exit(1)


