#!/usr/bin/env python3
"""
Pinecone Utility Functions for Ray Peat Corpus

Provides management and verification utilities for the Pinecone vector database.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging


try:
    from pinecone import Pinecone
    from dotenv import load_dotenv
    from tqdm.auto import tqdm
except ImportError as e:
    print(f"Missing dependencies. Please install: pip install -r requirements.txt")
    print(f"Error: {e}")
    sys.exit(1)

from .vector_search import PineconeVectorSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeManager:
    """Utility class for managing Pinecone operations."""
    
    def __init__(self, index_name: str = "ray-peat-corpus"):
        self.index_name = index_name
        self.pc = None
        self.index = None
        self.dimension: int = 3072
        
        # Load environment variables
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
        
        # Initialize Pinecone
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
            
            # Connect to index if it exists
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name in existing_indexes:
                self.index = self.pc.Index(self.index_name)
                # Detect index dimension
                stats = self.index.describe_index_stats()
                dim = stats.get('dimension')
                if isinstance(dim, int) and dim > 0:
                    self.dimension = dim
                logger.info(f"Connected to index: {self.index_name} (dimension={self.dimension})")
            else:
                logger.warning(f"Index '{self.index_name}' not found. Available: {existing_indexes}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get detailed information about the index."""
        if not self.index:
            return {"error": "Index not connected"}
        
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "index_name": self.index_name,
                "total_vector_count": stats.get('total_vector_count', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "dimension": stats.get('dimension', self.dimension),
                "namespaces": stats.get('namespaces', {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {"error": str(e)}
    
    def sample_vectors(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Sample random vectors from the index for verification."""
        if not self.index:
            return []
        
        try:
            # Get some vector IDs first
            stats = self.index.describe_index_stats()
            if stats.get('total_vector_count', 0) == 0:
                logger.warning("No vectors found in index")
                return []
            
            # For sampling, we'll query with a random vector to get some results
            import numpy as np
            random_vector = np.random.rand(self.dimension).tolist()
            
            query_response = self.index.query(
                vector=random_vector,
                top_k=num_samples,
                include_metadata=True
            )
            
            samples = []
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                samples.append({
                    "id": match.get('id'),
                    "score": match.get('score'),
                    "context": metadata.get('context', '')[:100] + "...",
                    "source_file": metadata.get('source_file', ''),
                    "tokens": metadata.get('tokens', 0),
                    "truncated": metadata.get('truncated', False)
                })
            
            return samples
            
        except Exception as e:
            logger.error(f"Error sampling vectors: {e}")
            return []
    
    def verify_vector_integrity(self, sample_size: int = 100) -> Dict[str, Any]:
        """Verify the integrity of vectors in the index."""
        if not self.index:
            return {"error": "Index not connected"}
        
        try:
            # Sample some vectors
            samples = self.sample_vectors(sample_size)
            
            if not samples:
                return {"error": "No samples available"}
            
            # Analyze the samples
            verification_results = {
                "total_sampled": len(samples),
                "metadata_completeness": {
                    "has_context": sum(1 for s in samples if s["context"]),
                    "has_source_file": sum(1 for s in samples if s["source_file"]),
                    "has_tokens": sum(1 for s in samples if s["tokens"] > 0),
                    "truncated_count": sum(1 for s in samples if s.get("truncated", False))
                },
                "score_distribution": {
                    "min_score": min(s["score"] for s in samples),
                    "max_score": max(s["score"] for s in samples),
                    "avg_score": sum(s["score"] for s in samples) / len(samples)
                },
                "sample_data": samples[:5],  # First 5 samples for inspection
                "verification_time": datetime.now().isoformat()
            }
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error verifying vector integrity: {e}")
            return {"error": str(e)}
    
    def search_by_source_file(self, source_file: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all vectors from a specific source file."""
        if not self.index:
            return []
        
        try:
            # Use metadata filtering to find vectors from specific source
            import numpy as np
            dummy_vector = np.zeros(self.dimension).tolist()
            
            query_response = self.index.query(
                vector=dummy_vector,
                top_k=limit,
                include_metadata=True,
                filter={"source_file": source_file}
            )
            
            results = []
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                results.append({
                    "id": match.get('id'),
                    "context": metadata.get('context', ''),
                    "ray_peat_response": metadata.get('ray_peat_response', ''),
                    "tokens": metadata.get('tokens', 0),
                    "truncated": metadata.get('truncated', False)
                })
            
            logger.info(f"Found {len(results)} vectors from source: {source_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by source file: {e}")
            return []
    
    def get_unique_source_files(self) -> List[str]:
        """Get a list of all unique source files in the index."""
        if not self.index:
            return []
        
        try:
            # Sample a large number of vectors to get source file diversity
            import numpy as np
            random_vector = np.random.rand(self.dimension).tolist()
            
            query_response = self.index.query(
                vector=random_vector,
                top_k=1000,  # Large sample
                include_metadata=True
            )
            
            source_files = set()
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                source_file = metadata.get('source_file', '')
                if source_file:
                    source_files.add(source_file)
            
            return sorted(list(source_files))
            
        except Exception as e:
            logger.error(f"Error getting unique source files: {e}")
            return []
    
    def delete_by_source_file(self, source_file: str) -> bool:
        """Delete all vectors from a specific source file."""
        if not self.index:
            logger.error("Index not connected")
            return False
        
        try:
            # First, get all vector IDs from this source file
            vectors_from_source = self.search_by_source_file(source_file, limit=10000)
            
            if not vectors_from_source:
                logger.warning(f"No vectors found for source file: {source_file}")
                return True
            
            # Extract IDs
            ids_to_delete = [v["id"] for v in vectors_from_source]
            
            # Delete in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(ids_to_delete), batch_size):
                batch_ids = ids_to_delete[i:i + batch_size]
                self.index.delete(ids=batch_ids)
                deleted_count += len(batch_ids)
                logger.info(f"Deleted batch {i//batch_size + 1}, total deleted: {deleted_count}")
            
            logger.info(f"Successfully deleted {deleted_count} vectors from source: {source_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting by source file: {e}")
            return False
    
    def backup_index_metadata(self, output_file: Optional[Path] = None) -> bool:
        """Create a backup of all metadata in the index."""
        if not self.index:
            logger.error("Index not connected")
            return False
        
        if output_file is None:
            output_file = Path(__file__).parent / f"backup_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Get all vectors with metadata
            import numpy as np
            dummy_vector = np.zeros(3072).tolist()
            
            all_metadata = []
            
            # Query in batches to get all vectors
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            logger.info(f"Backing up metadata for {total_vectors} vectors...")
            
            batch_size = 1000
            for i in tqdm(range(0, total_vectors, batch_size), desc="Backing up"):
                query_response = self.index.query(
                    vector=dummy_vector,
                    top_k=min(batch_size, total_vectors - i),
                    include_metadata=True
                )
                
                for match in query_response.get('matches', []):
                    metadata = match.get('metadata', {})
                    all_metadata.append({
                        "id": match.get('id'),
                        "metadata": metadata
                    })
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "backup_time": datetime.now().isoformat(),
                    "index_name": self.index_name,
                    "total_vectors": len(all_metadata),
                    "vectors": all_metadata
                }, f, indent=2)
            
            logger.info(f"Metadata backup saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating metadata backup: {e}")
            return False
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report for the index."""
        report = {
            "report_time": datetime.now().isoformat(),
            "index_info": self.get_index_info(),
            "integrity_check": self.verify_vector_integrity(),
            "source_files": self.get_unique_source_files()
        }
        
        # Calculate health score
        integrity = report["integrity_check"]
        if "error" not in integrity:
            metadata_completeness = integrity["metadata_completeness"]
            total_sampled = integrity["total_sampled"]
            
            if total_sampled > 0:
                completeness_score = (
                    metadata_completeness["has_context"] +
                    metadata_completeness["has_source_file"] +
                    metadata_completeness["has_tokens"]
                ) / (total_sampled * 3)
                
                report["health_score"] = completeness_score
                report["health_status"] = "healthy" if completeness_score > 0.95 else "warning" if completeness_score > 0.8 else "critical"
        
        return report

def main():
    """Interactive utility for Pinecone management."""
    print("🔧 Pinecone Management Utility")
    print("=" * 40)
    
    try:
        manager = PineconeManager()
        
        while True:
            print("\nAvailable commands:")
            print("1. Get index info")
            print("2. Verify vector integrity")
            print("3. Sample vectors")
            print("4. List source files")
            print("5. Search by source file")
            print("6. Generate health report")
            print("7. Backup metadata")
            print("8. Delete by source file")
            print("9. Exit")
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == "1":
                info = manager.get_index_info()
                print(json.dumps(info, indent=2))
                
            elif choice == "2":
                sample_size = input("Sample size (default 100): ").strip()
                sample_size = int(sample_size) if sample_size.isdigit() else 100
                
                verification = manager.verify_vector_integrity(sample_size)
                print(json.dumps(verification, indent=2))
                
            elif choice == "3":
                num_samples = input("Number of samples (default 5): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 5
                
                samples = manager.sample_vectors(num_samples)
                print(json.dumps(samples, indent=2))
                
            elif choice == "4":
                source_files = manager.get_unique_source_files()
                print(f"Found {len(source_files)} unique source files:")
                for file in source_files:
                    print(f"  - {file}")
                    
            elif choice == "5":
                source_file = input("Enter source file name: ").strip()
                if source_file:
                    results = manager.search_by_source_file(source_file)
                    print(f"Found {len(results)} vectors from {source_file}")
                    if results:
                        print(json.dumps(results[:3], indent=2))  # Show first 3
                        
            elif choice == "6":
                print("Generating health report...")
                report = manager.generate_health_report()
                print(json.dumps(report, indent=2))
                
            elif choice == "7":
                print("Creating metadata backup...")
                success = manager.backup_index_metadata()
                if success:
                    print("✅ Backup completed successfully")
                else:
                    print("❌ Backup failed")
                    
            elif choice == "8":
                source_file = input("Enter source file to delete (WARNING: This cannot be undone): ").strip()
                if source_file:
                    confirm = input(f"Are you sure you want to delete all vectors from '{source_file}'? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        success = manager.delete_by_source_file(source_file)
                        if success:
                            print("✅ Deletion completed")
                        else:
                            print("❌ Deletion failed")
                    else:
                        print("Deletion cancelled")
                        
            elif choice == "9":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


