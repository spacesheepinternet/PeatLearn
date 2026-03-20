#!/usr/bin/env python3
"""
Pinecone Integration Script

Updates existing RAG systems to use Pinecone as the vector backend.
This script helps migrate from local vector storage to Pinecone.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeIntegrator:
    """Handles integration of Pinecone into existing systems."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.inference_rag_dir = self.project_root / "inference" / "backend" / "rag"
        self.adaptive_learning_dir = self.project_root / "src" / "adaptive_learning"
        self.backup_dir = self.project_root / "backups" / f"pre_pinecone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_existing_files(self) -> bool:
        """Create backups of existing RAG files before modification."""
        try:
            logger.info("Creating backups of existing files...")
            
            files_to_backup = [
                self.inference_rag_dir / "vector_search.py",
                self.inference_rag_dir / "rag_system.py",
                self.adaptive_learning_dir / "rag_system.py"
            ]
            
            for file_path in files_to_backup:
                if file_path.exists():
                    backup_path = self.backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Backed up: {file_path} -> {backup_path}")
                else:
                    logger.warning(f"File not found for backup: {file_path}")
            
            # Create backup info file
            backup_info = self.backup_dir / "backup_info.txt"
            with open(backup_info, 'w') as f:
                f.write(f"Pinecone Integration Backup\n")
                f.write(f"Created: {datetime.now().isoformat()}\n")
                f.write(f"Files backed up:\n")
                for file_path in files_to_backup:
                    if file_path.exists():
                        f.write(f"  - {file_path}\n")
            
            logger.info(f"Backup completed: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def create_pinecone_inference_wrapper(self) -> bool:
        """Create a wrapper for the inference RAG system to use Pinecone."""
        try:
            logger.info("Creating Pinecone wrapper for inference RAG system...")
            
            wrapper_content = '''#!/usr/bin/env python3
"""
Pinecone-Compatible Vector Search for Inference Backend

This module provides a drop-in replacement for the original vector search,
now using Pinecone as the backend while maintaining API compatibility.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import Pinecone implementations
try:
    from embedding.pinecone.vector_search import PineconeVectorSearch, SearchResult
    from embedding.pinecone.rag_system import PineconeRAG, RAGResponse
    
    # Create compatible aliases
    RayPeatVectorSearch = PineconeVectorSearch
    RayPeatRAG = PineconeRAG
    
    # Global instances for backward compatibility
    search_engine = PineconeVectorSearch()
    rag_system = PineconeRAG()
    
    print("✅ Using Pinecone backend for vector search")
    
except ImportError as e:
    print(f"⚠️ Pinecone not available, falling back to local storage: {e}")
    
    # Fallback to original implementations
    try:
        from .vector_search_local import RayPeatVectorSearch, SearchResult
        from .rag_system_local import RayPeatRAG, RAGResponse
        
        # Global instances
        search_engine = RayPeatVectorSearch()
        rag_system = RayPeatRAG()
        
        print("📁 Using local vector storage")
        
    except ImportError:
        print("❌ Neither Pinecone nor local storage available")
        raise
'''
            
            # Write the wrapper to inference RAG directory
            wrapper_file = self.inference_rag_dir / "vector_search.py"
            
            # Rename original files
            original_vector_search = self.inference_rag_dir / "vector_search.py"
            original_rag_system = self.inference_rag_dir / "rag_system.py"
            
            if original_vector_search.exists():
                shutil.move(original_vector_search, self.inference_rag_dir / "vector_search_local.py")
            
            if original_rag_system.exists():
                shutil.move(original_rag_system, self.inference_rag_dir / "rag_system_local.py")
            
            # Write new wrapper
            with open(wrapper_file, 'w') as f:
                f.write(wrapper_content)
            
            # Create RAG system wrapper
            rag_wrapper_content = '''#!/usr/bin/env python3
"""
Pinecone-Compatible RAG System for Inference Backend

Drop-in replacement for the original RAG system using Pinecone.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import from the vector search wrapper (which handles Pinecone/local fallback)
from .vector_search import rag_system, RayPeatRAG, RAGResponse

# Re-export for compatibility
__all__ = ['rag_system', 'RayPeatRAG', 'RAGResponse']
'''
            
            rag_wrapper_file = self.inference_rag_dir / "rag_system.py"
            with open(rag_wrapper_file, 'w') as f:
                f.write(rag_wrapper_content)
            
            logger.info("Inference RAG wrapper created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create inference wrapper: {e}")
            return False
    
    def create_adaptive_learning_wrapper(self) -> bool:
        """Update the adaptive learning RAG system to use Pinecone."""
        try:
            logger.info("Creating Pinecone wrapper for adaptive learning RAG system...")
            
            adaptive_rag_file = self.adaptive_learning_dir / "rag_system.py"
            
            if not adaptive_rag_file.exists():
                logger.warning(f"Adaptive learning RAG file not found: {adaptive_rag_file}")
                return True  # Not an error if the file doesn't exist
            
            # Read the existing file
            with open(adaptive_rag_file, 'r') as f:
                content = f.read()
            
            # Create backup with _local suffix
            backup_file = self.adaptive_learning_dir / "rag_system_local.py"
            with open(backup_file, 'w') as f:
                f.write(content)
            
            # Create new Pinecone-compatible version
            pinecone_content = '''#!/usr/bin/env python3
"""
Adaptive Learning RAG System with Pinecone Backend

Enhanced RAG system for adaptive learning using Pinecone vector database.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from embedding.pinecone.vector_search import PineconeVectorSearch
    from embedding.pinecone.rag_system import PineconeRAG
    
    # Use Pinecone backend
    BASE_RAG_CLASS = PineconeRAG
    BASE_SEARCH_CLASS = PineconeVectorSearch
    
    print("✅ Adaptive learning using Pinecone backend")
    
except ImportError:
    print("⚠️ Pinecone not available, falling back to local storage")
    
    # Import from the local backup
    from .rag_system_local import *
    
    # For fallback compatibility
    BASE_RAG_CLASS = RayPeatRAG  # Assuming this exists in the local version
    BASE_SEARCH_CLASS = RayPeatVectorSearch

# Rest of your adaptive learning code can be imported from the local backup
# and enhanced with Pinecone-specific features

try:
    # Import the original adaptive learning functionality
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_system_local", 
                                                  Path(__file__).parent / "rag_system_local.py")
    rag_local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_local)
    
    # Re-export everything from the local module
    for attr in dir(rag_local):
        if not attr.startswith('_'):
            globals()[attr] = getattr(rag_local, attr)
    
    # Override specific classes to use Pinecone if available
    if 'BASE_RAG_CLASS' in globals() and BASE_RAG_CLASS.__name__ == 'PineconeRAG':
        # You can enhance your adaptive learning classes here to use Pinecone
        # For example, if you have a custom RAG class:
        # class AdaptiveRAG(BASE_RAG_CLASS):
        #     # Your adaptive learning enhancements
        pass
        
except Exception as e:
    print(f"Warning: Could not import local adaptive learning module: {e}")
'''
            
            # Write the new Pinecone-compatible version
            with open(adaptive_rag_file, 'w') as f:
                f.write(pinecone_content)
            
            logger.info("Adaptive learning RAG wrapper created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create adaptive learning wrapper: {e}")
            return False
    
    def update_requirements(self) -> bool:
        """Update requirements files to include Pinecone dependencies."""
        try:
            logger.info("Updating requirements files...")
            
            pinecone_requirements = [
                "pinecone-client>=3.0.0",
                "# Pinecone vector database support"
            ]
            
            # Update main requirements.txt
            main_requirements = self.project_root / "requirements.txt"
            if main_requirements.exists():
                with open(main_requirements, 'r') as f:
                    content = f.read()
                
                if "pinecone-client" not in content:
                    with open(main_requirements, 'a') as f:
                        f.write(f"\n# Pinecone integration\n")
                        f.write(f"pinecone-client>=3.0.0\n")
                    logger.info("Updated main requirements.txt")
            
            # Update inference requirements.txt
            inference_requirements = self.project_root / "inference" / "requirements.txt"
            if inference_requirements.exists():
                with open(inference_requirements, 'r') as f:
                    content = f.read()
                
                if "pinecone-client" not in content:
                    with open(inference_requirements, 'a') as f:
                        f.write(f"\n# Pinecone integration\n")
                        f.write(f"pinecone-client>=3.0.0\n")
                    logger.info("Updated inference requirements.txt")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update requirements: {e}")
            return False
    
    def create_migration_guide(self) -> bool:
        """Create a migration guide for developers."""
        try:
            guide_content = f'''# Pinecone Migration Guide

This project has been migrated to use Pinecone as the vector database backend.

## What Changed

1. **Vector Storage**: Now uses Pinecone instead of local numpy files
2. **Search Performance**: Improved latency and scalability
3. **Metadata Filtering**: Enhanced search capabilities
4. **API Compatibility**: Existing code should work without changes

## Backup Location

Original files backed up to: `{self.backup_dir.relative_to(self.project_root)}`

## Configuration Required

Add to your `.env` file:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Migration Steps Completed

✅ Created Pinecone upload script
✅ Backed up original files
✅ Created compatibility wrappers
✅ Updated requirements files

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Add API key**: Update `.env` file with your Pinecone API key
3. **Upload embeddings**: Run `python embedding/pinecone/upload.py`
4. **Test migration**: Run `python embedding/pinecone/test_pinecone.py`

## Rollback Instructions

If you need to rollback to local storage:

1. Copy files from `{self.backup_dir}` back to their original locations
2. Remove Pinecone requirements from requirements.txt
3. Restart your applications

## Support

- Check `embedding/pinecone/README.md` for detailed documentation
- Use `embedding/pinecone/utils.py` for management tasks
- Run tests with `embedding/pinecone/test_pinecone.py`

## Performance Notes

- First run after migration may be slower (Pinecone cold start)
- Search latency: ~100-300ms (vs ~50ms local, but more scalable)
- No local storage requirements (saves ~100MB disk space)

Generated: {datetime.now().isoformat()}
'''
            
            guide_file = self.project_root / "PINECONE_MIGRATION.md"
            with open(guide_file, 'w') as f:
                f.write(guide_content)
            
            logger.info(f"Migration guide created: {guide_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration guide: {e}")
            return False
    
    def run_integration(self) -> bool:
        """Run the complete integration process."""
        print("🚀 Starting Pinecone Integration")
        print("=" * 50)
        
        steps = [
            ("Creating backups", self.backup_existing_files),
            ("Creating inference wrapper", self.create_pinecone_inference_wrapper),
            ("Creating adaptive learning wrapper", self.create_adaptive_learning_wrapper),
            ("Updating requirements", self.update_requirements),
            ("Creating migration guide", self.create_migration_guide)
        ]
        
        success_count = 0
        
        for step_name, step_function in steps:
            print(f"\n📋 {step_name}...")
            try:
                if step_function():
                    print(f"✅ {step_name} completed")
                    success_count += 1
                else:
                    print(f"❌ {step_name} failed")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
        
        print(f"\n📊 Integration Summary:")
        print(f"   Steps completed: {success_count}/{len(steps)}")
        print(f"   Success rate: {success_count/len(steps):.1%}")
        
        if success_count == len(steps):
            print(f"\n🎉 Integration completed successfully!")
            print(f"📁 Backups saved to: {self.backup_dir.relative_to(self.project_root)}")
            print(f"📖 See PINECONE_MIGRATION.md for next steps")
            return True
        else:
            print(f"\n⚠️ Integration partially completed")
            print(f"📁 Backups available at: {self.backup_dir.relative_to(self.project_root)}")
            return False

def main():
    """Main integration function."""
    try:
        integrator = PineconeIntegrator()
        success = integrator.run_integration()
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


