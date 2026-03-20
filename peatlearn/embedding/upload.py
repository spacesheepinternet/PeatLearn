"""
Upload embeddings from local files to Pinecone with robust error handling
"""
import numpy as np
import json
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "ray-peat-corpus"
index = pc.Index(index_name)

print(f"📊 Loading embeddings from disk...")

# Load the data
embeddings_path = "embedding/vectors/embeddings_20250728_221826.npy"
metadata_path = "embedding/vectors/metadata_20250728_221826.json"
# Load embeddings
embeddings = np.load(embeddings_path)
print(f"✅ Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")

# Load metadata
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)
print(f"✅ Loaded {len(metadata_list)} metadata entries")

# Prepare vectors for upload
print(f"📤 Uploading to Pinecone...")

batch_size = 100
total_batches = (len(embeddings) + batch_size - 1) // batch_size

# Track progress
uploaded_batches = set()
if os.path.exists('upload_progress.txt'):
    with open('upload_progress.txt', 'r') as f:
        uploaded_batches = set(int(line.strip()) for line in f)
    print(f"📝 Resuming from batch {max(uploaded_batches) + 1 if uploaded_batches else 0}")

for batch_num in tqdm(range(0, total_batches), total=total_batches):
    if batch_num in uploaded_batches:
        continue  # Skip already uploaded batches
    
    i = batch_num * batch_size
    batch_end = min(i + batch_size, len(embeddings))
    
    vectors = []
    for j in range(i, batch_end):
        vector_id = f"vec_{j}"
        vector_data = embeddings[j].tolist()
        vector_metadata = metadata_list[j] if j < len(metadata_list) else {}
        
        vectors.append({
            "id": vector_id,
            "values": vector_data,
            "metadata": vector_metadata
        })
    
    # Upload with retries
    max_retries = 5
    for retry in range(max_retries):
        try:
            index.upsert(vectors=vectors)
            
            # Save progress
            with open('upload_progress.txt', 'a') as f:
                f.write(f"{batch_num}\n")
            
            time.sleep(1)  # Longer delay between batches
            break  # Success, exit retry loop
            
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 5  # Exponential backoff: 5s, 10s, 15s, 20s, 25s
                print(f"\n⚠️  Batch {batch_num} failed (attempt {retry+1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Batch {batch_num} failed after {max_retries} attempts: {e}")

# Verify upload
stats = index.describe_index_stats()
print(f"\n✅ Upload complete!")
print(f"📊 Total vectors in index: {stats['total_vector_count']}")
print(f"🎉 Your RAG system is now ready!")

# Clean up progress file
if os.path.exists('upload_progress.txt'):
    os.remove('upload_progress.txt')
