#!/usr/bin/env python3
"""
Standalone Pinecone Upload Script for Ray Peat Embeddings

Migrates existing numpy embeddings and metadata to Pinecone vector database.
This is a simplified standalone version that doesn't rely on project imports.
"""

import os
import sys
import json
import re
import unicodedata
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from pinecone import Pinecone, ServerlessSpec
    from tqdm.auto import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependencies. Please install:")
    print(f"pip install pinecone-client numpy tqdm python-dotenv")
    print(f"Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_vector_id(raw_id: str, fallback_prefix: str = "vector", index: Optional[int] = None) -> str:
    """Return a Pinecone-safe ASCII ID derived from raw_id.

    - Normalizes to NFKD and strips non-ASCII
    - Replaces whitespace with underscores
    - Restricts to safe characters [A-Za-z0-9._:/+\-=]
    - Truncates to 512 chars, preserving a short hash suffix
    - Falls back to a generated ID if empty after sanitization
    """
    try:
        if not isinstance(raw_id, str):
            raw_id = str(raw_id)

        # Normalize and remove non-ASCII
        normalized = unicodedata.normalize("NFKD", raw_id)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")

        # Replace whitespace with underscores
        ascii_only = re.sub(r"\s+", "_", ascii_only)

        # Keep only allowed characters
        # Allowed: letters, digits, dot, underscore, colon, slash, plus, minus, equals
        ascii_only = re.sub(r"[^A-Za-z0-9\._:/\+\-=]", "-", ascii_only)

        # Remove leading/trailing separators
        ascii_only = ascii_only.strip("-_.:/+=")

        if not ascii_only:
            suffix_src = f"{fallback_prefix}-{index if index is not None else ''}-{datetime.utcnow().timestamp()}"
            suffix = hashlib.sha1(suffix_src.encode("utf-8")).hexdigest()[:8]
            ascii_only = f"{fallback_prefix}_{suffix}"

        # Enforce max length 512 with stable hash suffix
        if len(ascii_only) > 512:
            digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:10]
            # leave room for hyphen and digest
            keep_len = 512 - (1 + len(digest))
            ascii_only = f"{ascii_only[:keep_len]}-{digest}"

        return ascii_only
    except Exception:
        # Last-resort fallback
        suffix = hashlib.sha1(str(raw_id).encode("utf-8", errors="ignore")).hexdigest()[:8]
        base = fallback_prefix
        candidate = f"{base}_{index if index is not None else 'id'}_{suffix}"
        return candidate[:512]

@dataclass
class PineconeConfig:
    """Configuration for Pinecone upload."""
    api_key: str
    index_name: str = "ray-peat-corpus"
    vector_dimension: int = 3072
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    batch_size: int = 128

class RayPeatPineconeUploader:
    """Handles uploading Ray Peat embeddings to Pinecone."""
    
    def __init__(self, config: PineconeConfig):
        self.config = config
        
        # Find vectors directory
        current_dir = Path(__file__).parent
        self.vectors_dir = current_dir.parent / "vectors"
        
        if not self.vectors_dir.exists():
            raise FileNotFoundError(f"Vectors directory not found: {self.vectors_dir}")
        
        self.pc = None
        self.index = None
        self.start_total_count: int = 0
        
        # Initialize Pinecone
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.config.api_key)
            logger.info("Pinecone client initialized successfully")
            
            # Create index if it doesn't exist
            self._create_index_if_needed()
            
            # Connect to index
            self.index = self.pc.Index(self.config.index_name)
            logger.info(f"Connected to index: {self.config.index_name}")
            
            # Show initial stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            self.start_total_count = int(stats.get('total_vector_count', 0))
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _create_index_if_needed(self):
        """Create index if it doesn't exist."""
        existing_indexes = self.pc.list_indexes().names()
        
        if self.config.index_name not in existing_indexes:
            logger.info(f"Creating new index: {self.config.index_name}")
            
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.vector_dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(
                    cloud=self.config.cloud, 
                    region=self.config.region
                )
            )
            logger.info(f"Created index: {self.config.index_name}")
        else:
            logger.info(f"Index {self.config.index_name} already exists")
    
    def load_latest_embeddings(self) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load the latest embeddings and metadata files."""
        try:
            # Find the latest embedding files
            embedding_files = list(self.vectors_dir.glob("embeddings_*.npy"))
            metadata_files = list(self.vectors_dir.glob("metadata_*.json"))
            
            if not embedding_files or not metadata_files:
                raise FileNotFoundError("No embedding files found in vectors directory")
            
            # Get the latest files (by modification time)
            latest_embedding = max(embedding_files, key=lambda x: x.stat().st_mtime)
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            logger.info(f"Loading embeddings from: {latest_embedding.name}")
            logger.info(f"Loading metadata from: {latest_metadata.name}")
            
            # Load embeddings
            vectors = np.load(latest_embedding)
            logger.info(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")
            
            # Load metadata
            with open(latest_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded {len(metadata)} metadata entries")
            
            # Verify dimensions
            if vectors.shape[1] != self.config.vector_dimension:
                raise ValueError(f"Vector dimension mismatch: got {vectors.shape[1]}, expected {self.config.vector_dimension}")
            
            if len(vectors) != len(metadata):
                raise ValueError(f"Vector/metadata count mismatch: {len(vectors)} vectors vs {len(metadata)} metadata entries")
            
            return vectors, metadata
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def prepare_vectors_for_upload(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[tuple]:
        """Prepare vectors in Pinecone format."""
        logger.info("Preparing vectors for upload...")
        
        prepared_vectors = []
        seen_ids = set()
        
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            # Create unique ID and sanitize to ASCII per Pinecone requirements
            raw_id = meta.get("id", f"vector_{i}")
            vector_id = sanitize_vector_id(raw_id, fallback_prefix="vector", index=i)
            # Ensure uniqueness within this preparation run
            if vector_id in seen_ids:
                # Append short hash of raw to disambiguate
                disambig = hashlib.sha1(str(raw_id).encode("utf-8", errors="ignore")).hexdigest()[:6]
                candidate = f"{vector_id}-{disambig}"
                # Truncate to 512 if needed
                if len(candidate) > 512:
                    candidate = candidate[:512]
                vector_id = candidate
            seen_ids.add(vector_id)
            
            # Convert numpy array to list
            vector_values = vector.tolist()
            
            # Create metadata for Pinecone (store searchable fields)
            pinecone_metadata = {
                "context": meta.get("context", "")[:1000],  # Limit context length
                "source_file": meta.get("source_file", ""),
                "tokens": meta.get("tokens", 0),
                "original_id": meta.get("id", ""),
                "sanitized_id": vector_id
            }
            
            # Store the full Ray Peat response in metadata (if it fits)
            ray_peat_response = meta.get("ray_peat_response", "")
            if len(ray_peat_response) <= 40000:  # Pinecone metadata limit
                pinecone_metadata["ray_peat_response"] = ray_peat_response
            else:
                # Store truncated version and note the truncation
                pinecone_metadata["ray_peat_response"] = ray_peat_response[:39000] + "... [truncated]"
                pinecone_metadata["truncated"] = True
            
            prepared_vectors.append((vector_id, vector_values, pinecone_metadata))
        
        logger.info(f"Prepared {len(prepared_vectors)} vectors for upload")
        return prepared_vectors
    
    def upload_vectors(self, prepared_vectors: List[tuple]) -> Dict[str, Any]:
        """Upload vectors to Pinecone in batches."""
        logger.info(f"Starting upload of {len(prepared_vectors)} vectors...")
        
        upload_stats = {
            "total_vectors": len(prepared_vectors),
            "successful_uploads": 0,
            "failed_uploads": 0,
            "start_time": datetime.now(),
            "batch_size": self.config.batch_size
        }
        
        # Upload in batches
        for i in tqdm(range(0, len(prepared_vectors), self.config.batch_size), desc="Uploading batches"):
            batch_end = min(i + self.config.batch_size, len(prepared_vectors))
            batch = prepared_vectors[i:batch_end]
            
            try:
                # Upsert the batch
                self.index.upsert(vectors=batch)
                upload_stats["successful_uploads"] += len(batch)
                
            except Exception as e:
                batch_no = i//self.config.batch_size + 1
                err_msg = str(e)
                logger.error(f"Failed to upload batch {batch_no}: {e}")
                # Attempt a one-time recovery if the error suggests bad IDs
                if "Vector ID must be ASCII" in err_msg or "id" in err_msg.lower():
                    logger.info(f"Attempting to sanitize IDs and retry batch {batch_no} once...")
                    try:
                        recovered_batch = []
                        for j, (vid, vvals, vmeta) in enumerate(batch):
                            safe_id = sanitize_vector_id(vid, fallback_prefix="vector", index=i + j)
                            # ensure metadata carries both original and sanitized ids
                            new_meta = dict(vmeta)
                            new_meta.setdefault("original_id", vmeta.get("original_id", vid))
                            new_meta["sanitized_id"] = safe_id
                            recovered_batch.append((safe_id, vvals, new_meta))
                        self.index.upsert(vectors=recovered_batch)
                        upload_stats["successful_uploads"] += len(batch)
                        logger.info(f"Batch {batch_no} recovered by sanitizing IDs.")
                        continue
                    except Exception as e2:
                        logger.error(f"Recovery for batch {batch_no} failed: {e2}")
                upload_stats["failed_uploads"] += len(batch)
        
        upload_stats["end_time"] = datetime.now()
        upload_stats["duration_minutes"] = (upload_stats["end_time"] - upload_stats["start_time"]).total_seconds() / 60
        
        logger.info(f"Upload completed: {upload_stats['successful_uploads']}/{upload_stats['total_vectors']} vectors uploaded")
        return upload_stats
    
    def verify_upload(self, expected_count: int) -> bool:
        """Verify that all vectors were uploaded successfully."""
        logger.info("Verifying upload...")
        
        try:
            stats = self.index.describe_index_stats()
            end_total_count = int(stats.get('total_vector_count', 0))
            # Prefer delta over absolute when index is pre-populated
            if self.start_total_count:
                delta = end_total_count - self.start_total_count
                logger.info(f"Expected new: {expected_count}, Actual new: {delta} (start={self.start_total_count}, end={end_total_count})")
                ok = (delta == expected_count)
            else:
                logger.info(f"Expected: {expected_count}, Actual: {end_total_count}")
                ok = (end_total_count == expected_count)
            
            if ok:
                logger.info("✅ Verification successful: All vectors uploaded correctly")
                return True
            else:
                if self.start_total_count:
                    logger.warning(f"❌ Verification failed: Expected delta {expected_count}, got {delta}")
                else:
                    logger.warning(f"❌ Verification failed: Expected {expected_count}, got {end_total_count}")
                return False
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False
    
    def save_upload_report(self, upload_stats: Dict[str, Any], vectors_count: int, metadata_count: int):
        """Save detailed upload report."""
        report = {
            "upload_completed": datetime.now().isoformat(),
            "pinecone_config": {
                "index_name": self.config.index_name,
                "vector_dimension": self.config.vector_dimension,
                "metric": self.config.metric,
                "cloud": self.config.cloud,
                "region": self.config.region
            },
            "source_data": {
                "vectors_count": vectors_count,
                "metadata_count": metadata_count,
                "vectors_directory": str(self.vectors_dir)
            },
            "upload_stats": upload_stats,
            "verification_passed": upload_stats["successful_uploads"] == upload_stats["total_vectors"]
        }
        
        report_file = Path(__file__).parent / f"upload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Upload report saved to: {report_file}")

def main():
    """Main upload function."""
    print("🚀 Ray Peat Corpus - Pinecone Migration")
    print("=" * 50)
    
    # Load environment variables
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from: {env_file}")
    else:
        print(f"⚠️ No .env file found at: {env_file}")
        print("Looking for environment variables...")
    
    # Get Pinecone API key from environment
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment variables.")
        print("Please set your Pinecone API key:")
        print("  export PINECONE_API_KEY=your_api_key_here")
        print("Or create a .env file in the project root with:")
        print("  PINECONE_API_KEY=your_api_key_here")
        return
    
    try:
        # Create configuration
        config = PineconeConfig(api_key=api_key)
        
        print(f"📋 Configuration:")
        print(f"   • Index name: {config.index_name}")
        print(f"   • Vector dimension: {config.vector_dimension}")
        print(f"   • Metric: {config.metric}")
        print(f"   • Batch size: {config.batch_size}")
        print(f"   • Cloud: {config.cloud} ({config.region})")
        print()
        
        # Initialize uploader
        uploader = RayPeatPineconeUploader(config)
        
        # Load embeddings
        print("📂 Loading embeddings and metadata...")
        vectors, metadata = uploader.load_latest_embeddings()
        
        print(f"✅ Loaded data:")
        print(f"   • Vectors: {len(vectors):,} ({vectors.shape[1]}D)")
        print(f"   • Metadata entries: {len(metadata):,}")
        print()
        
        # Ask for confirmation
        estimated_cost = len(vectors) * 0.0001  # Rough estimate
        response = input(f"💰 Proceed with upload? (~${estimated_cost:.2f} estimated) (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Upload cancelled.")
            return
        
        # Prepare vectors
        print("🔄 Preparing vectors for Pinecone...")
        prepared_vectors = uploader.prepare_vectors_for_upload(vectors, metadata)
        
        # Upload vectors
        print("⬆️  Uploading to Pinecone...")
        upload_stats = uploader.upload_vectors(prepared_vectors)
        
        # Verify upload
        print("🔍 Verifying upload...")
        verification_passed = uploader.verify_upload(len(vectors))
        
        # Save report
        uploader.save_upload_report(upload_stats, len(vectors), len(metadata))
        
        # Final summary
        print(f"\n✅ Migration completed!")
        print(f"   • Total vectors: {upload_stats['total_vectors']:,}")
        print(f"   • Successfully uploaded: {upload_stats['successful_uploads']:,}")
        print(f"   • Failed uploads: {upload_stats['failed_uploads']:,}")
        print(f"   • Duration: {upload_stats['duration_minutes']:.1f} minutes")
        print(f"   • Verification: {'✅ PASSED' if verification_passed else '❌ FAILED'}")
        print(f"   • Index: {config.index_name}")
        
        if verification_passed:
            print(f"\n🎉 Your embeddings are now available in Pinecone!")
            print(f"You can now use the Pinecone-based RAG system for better performance.")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        print(f"\n❌ Upload failed: {e}")
        return

if __name__ == "__main__":
    main()
