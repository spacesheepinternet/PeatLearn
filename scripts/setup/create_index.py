"""
Create or recreate Pinecone index
"""
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

def create_pinecone_index(
    index_name="ray-peat-corpus",
    dimension=3072,
    metric="cosine",
    cloud="aws",
    region="us-east-1"
):
    """Create Pinecone index"""
    print("="*60)
    print("  📦 Pinecone Index Creator")
    print("="*60)
    
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    
    if index_name in existing_indexes:
        print(f"\n⚠️  Index '{index_name}' already exists!")
        response = input("Delete and recreate? (y/n): ")
        
        if response.lower() == 'y':
            print(f"🗑️  Deleting index '{index_name}'...")
            pc.delete_index(index_name)
            import time
            time.sleep(5)  # Wait for deletion
        else:
            print("❌ Cancelled")
            return
    
    # Create new index
    print(f"\n📦 Creating index '{index_name}'...")
    print(f"   Dimension: {dimension}")
    print(f"   Metric: {metric}")
    print(f"   Cloud: {cloud}")
    print(f"   Region: {region}")
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud=cloud,
            region=region
        )
    )
    
    print(f"\n✅ Index '{index_name}' created successfully!")
    print(f"\n📊 Index info:")
    print(f"   Name: {index_name}")
    print(f"   Dimension: {dimension}")
    print(f"   Ready for embeddings!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Pinecone index')
    parser.add_argument('--name', default='ray-peat-corpus', help='Index name')
    parser.add_argument('--dimension', type=int, default=3072, help='Vector dimension')
    parser.add_argument('--metric', default='cosine', help='Distance metric')
    
    args = parser.parse_args()
    
    create_pinecone_index(
        index_name=args.name,
        dimension=args.dimension,
        metric=args.metric
    )