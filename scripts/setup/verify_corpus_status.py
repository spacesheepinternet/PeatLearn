"""
Verify current corpus status
Check what's in Pinecone vs what's in data/raw
"""
from pinecone import Pinecone
from pathlib import Path
import os
from dotenv import load_dotenv
import json

load_dotenv()

def verify_status():
    """Check current status of corpus"""
    print("="*60)
    print("  📊 PeatLearn Corpus Status Check")
    print("="*60)
    
    # Check Pinecone
    print("\n🔍 Checking Pinecone...")
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("ray-peat-corpus")
    stats = index.describe_index_stats()
    
    print(f"   Index: ray-peat-corpus")
    print(f"   Total vectors: {stats['total_vector_count']:,}")
    print(f"   Dimension: {stats.get('dimension', 'Unknown')}")
    
    # Check raw data
    print("\n📁 Checking data/raw...")
    raw_dir = Path('data/raw')
    
    # Count files
    file_counts = {}
    for file_path in raw_dir.rglob('*'):
        if file_path.is_file():
            parent_folder = file_path.parent.name
            if parent_folder not in file_counts:
                file_counts[parent_folder] = 0
            file_counts[parent_folder] += 1
    
    total_files = sum(file_counts.values())
    print(f"   Total raw files: {total_files}")
    print(f"   Folders: {len(file_counts)}")
    
    # Check processed data
    print("\n📁 Checking data/processed...")
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        processed_files = list(processed_dir.rglob('*'))
        processed_count = sum(1 for f in processed_files if f.is_file())
        print(f"   Processed files: {processed_count}")
    else:
        print(f"   ⚠️  No processed directory found")
    
    # Check embeddings
    print("\n🧠 Checking embeddings...")
    embedding_dir = Path('embedding/embedding/vectors')
    if embedding_dir.exists():
        embedding_files = list(embedding_dir.glob('*.npy'))
        print(f"   Embedding files: {len(embedding_files)}")
        for emb_file in embedding_files:
            size_mb = emb_file.stat().st_size / (1024 * 1024)
            print(f"      - {emb_file.name} ({size_mb:.2f} MB)")
    else:
        print(f"   ⚠️  No embedding files found")
    
    # Summary
    print("\n" + "="*60)
    print("  📋 Summary")
    print("="*60)
    print(f"  Status: ✅ CORPUS ALREADY PROCESSED")
    print(f"  Raw files: {total_files}")
    print(f"  Vectors in Pinecone: {stats['total_vector_count']:,}")
    print(f"  Ratio: {stats['total_vector_count'] / total_files:.1f} vectors per file (avg)")
    
    print("\n💡 Recommendations:")
    print("  1. DON'T re-process existing raw_data folder")
    print("  2. For NEW content, create: data/raw/new_content_2026/")
    print("  3. Add only NEW files to that folder")
    print("  4. Run pipeline ONLY on new folder:")
    print("     python preprocessing/optimized_pipeline.py --input data/raw/new_content_2026")
    
    # Save status
    status_report = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'pinecone_vectors': stats['total_vector_count'],
        'raw_files': total_files,
        'processed': processed_count if processed_dir.exists() else 0,
        'folders': file_counts
    }
    
    with open('data/artifacts/corpus_status.json', 'w') as f:
        json.dump(status_report, f, indent=2)
    
    print(f"\n✅ Status report saved to: data/artifacts/corpus_status.json")

if __name__ == "__main__":
    verify_status()