"""
Optimized PeatLearn Pipeline
Integrates: Parallel Processing + Embedding Cache + Checkpoints + Smart Duplicate Detection
"""
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.parallel_processor import ParallelProcessor
from peatlearn.embedding.cache import EmbeddingCache
from preprocessing.checkpoint_system import PipelineCheckpoint
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class OptimizedPipeline:
    def __init__(
        self,
        batch_id: str,
        input_dir: str,
        max_workers: Optional[int] = None
    ):
        """
        Initialize optimized pipeline
        
        Args:
            batch_id: Unique identifier for this batch
            input_dir: Directory containing input documents
            max_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.batch_id = batch_id
        self.input_dir = Path(input_dir)
        
        # Initialize components
        self.parallel_processor = ParallelProcessor(max_workers=max_workers)
        self.embedding_cache = EmbeddingCache()
        self.checkpoint = PipelineCheckpoint(batch_id)
        
        # Results tracking
        self.pipeline_stats = {
            'batch_id': batch_id,
            'started': datetime.now().isoformat(),
            'stages': {}
        }
        
        print("="*60)
        print("  🚀 Optimized PeatLearn Pipeline")
        print("="*60)
        print(f"  Batch ID: {batch_id}")
        print(f"  Input: {input_dir}")
        print(f"  Workers: {self.parallel_processor.max_workers}")
        print("="*60)
    
    def discover_documents(self) -> List[Path]:
        """Find all documents in input directory"""
        print(f"\n📁 Discovering documents in {self.input_dir}...")
        
        # Support multiple formats
        extensions = ['.txt', '.pdf', '.docx', '.md', '.json']
        documents = []
        
        for ext in extensions:
            docs = list(self.input_dir.rglob(f'*{ext}'))
            documents.extend(docs)
        
        print(f"   Found {len(documents)} documents")
        return documents
    
    def stage_1_quality_analysis(self, documents: List[Path]) -> Dict:
        """
        Stage 1: Parallel quality analysis
        
        Args:
            documents: List of document paths
            
        Returns:
            Results from quality analysis
        """
        stage_name = 'quality_analysis'
        print(f"\n{'='*60}")
        print(f"  📊 Stage 1: Quality Analysis")
        print(f"{'='*60}")
        
        # Check checkpoint
        pending = self.checkpoint.get_pending_items(
            stage_name,
            [str(d) for d in documents]
        )
        
        if not pending:
            print("✅ Stage already completed")
            return {'status': 'skipped', 'reason': 'already_completed'}
        
        print(f"⏳ Processing {len(pending)} pending documents...")
        
        # Mark stage start
        self.checkpoint.mark_stage_start(stage_name, len(documents))
        
        # Process in parallel
        def analyze_quality(doc_path):
            """Analyze single document quality"""
            # TODO: Implement actual quality analysis
            # For now, return mock data
            import time
            import random
            time.sleep(0.05)  # Simulate processing
            
            return {
                'path': str(doc_path),
                'quality_score': round(random.uniform(0.6, 0.95), 2),
                'complexity': random.choice(['low', 'medium', 'high']),
                'recommended_tier': random.choice(['tier1', 'tier2'])
            }
        
        results = self.parallel_processor.process_batch(
            [Path(p) for p in pending],
            analyze_quality,
            description="Quality Analysis"
        )
        
        # Update checkpoint
        for item in results['results']:
            self.checkpoint.mark_item_completed(
                stage_name,
                item['item'],
                metadata=item['result']
            )
        
        for error in results['errors']:
            self.checkpoint.mark_item_failed(
                stage_name,
                error['item'],
                error['error']
            )
        
        self.checkpoint.mark_stage_complete(stage_name)
        self.pipeline_stats['stages'][stage_name] = results['stats']
        
        return results
    
    def stage_2_cleaning(self, documents: List[Path], quality_results: Dict) -> Dict:
        """
        Stage 2: Parallel document cleaning
        
        Args:
            documents: List of document paths
            quality_results: Results from quality analysis
            
        Returns:
            Results from cleaning
        """
        stage_name = 'cleaning'
        print(f"\n{'='*60}")
        print(f"  🧹 Stage 2: Document Cleaning")
        print(f"{'='*60}")
        
        # Check checkpoint
        pending = self.checkpoint.get_pending_items(
            stage_name,
            [str(d) for d in documents]
        )
        
        if not pending:
            print("✅ Stage already completed")
            return {'status': 'skipped', 'reason': 'already_completed'}
        
        print(f"⏳ Processing {len(pending)} pending documents...")
        
        # Mark stage start
        self.checkpoint.mark_stage_start(stage_name, len(documents))
        
        # Process in parallel
        def clean_document(doc_path):
            """Clean single document"""
            # TODO: Implement actual cleaning logic
            import time
            import random
            time.sleep(random.uniform(0.1, 0.3))  # Simulate variable processing time
            
            return {
                'path': str(doc_path),
                'cleaned': True,
                'tier_used': random.choice(['tier1', 'tier2']),
                'cleaning_score': round(random.uniform(0.8, 1.0), 2)
            }
        
        results = self.parallel_processor.process_batch(
            [Path(p) for p in pending],
            clean_document,
            description="Document Cleaning"
        )
        
        # Update checkpoint
        for item in results['results']:
            self.checkpoint.mark_item_completed(
                stage_name,
                item['item'],
                metadata=item['result']
            )
        
        for error in results['errors']:
            self.checkpoint.mark_item_failed(
                stage_name,
                error['item'],
                error['error']
            )
        
        self.checkpoint.mark_stage_complete(stage_name)
        self.pipeline_stats['stages'][stage_name] = results['stats']
        
        return results
    
    def stage_3_embedding(self, documents: List[Path]) -> Dict:
        """
        Stage 3: Generate embeddings with caching
        
        Args:
            documents: List of document paths
            
        Returns:
            Results from embedding generation
        """
        stage_name = 'embedding'
        print(f"\n{'='*60}")
        print(f"  🧠 Stage 3: Embedding Generation (with caching)")
        print(f"{'='*60}")
        
        # Check checkpoint
        pending = self.checkpoint.get_pending_items(
            stage_name,
            [str(d) for d in documents]
        )
        
        if not pending:
            print("✅ Stage already completed")
            return {'status': 'skipped', 'reason': 'already_completed'}
        
        print(f"⏳ Processing {len(pending)} pending documents...")
        
        # Mark stage start
        self.checkpoint.mark_stage_start(stage_name, len(documents))
        
        # Check cache statistics
        cache_stats_before = self.embedding_cache.get_stats()
        
        def embed_document(doc_path):
            """Generate or retrieve cached embedding"""
            # Read document content
            try:
                # TODO: Implement proper document reading
                # For now, use placeholder
                content = f"Content of {doc_path}"
                
                # Check cache
                cached_embedding = self.embedding_cache.get(content, str(doc_path))
                
                if cached_embedding is not None:
                    return {
                        'path': str(doc_path),
                        'embedding': cached_embedding,
                        'from_cache': True,
                        'dimension': len(cached_embedding)
                    }
                
                # Generate new embedding
                # TODO: Call actual Gemini API
                import time
                time.sleep(0.1)  # Simulate API call
                embedding = np.random.rand(3072)  # Mock 3072-dim embedding
                
                # Cache it
                self.embedding_cache.set(
                    content,
                    embedding,
                    doc_id=str(doc_path)
                )
                
                return {
                    'path': str(doc_path),
                    'embedding': embedding,
                    'from_cache': False,
                    'dimension': len(embedding)
                }
                
            except Exception as e:
                raise Exception(f"Embedding failed: {e}")
        
        results = self.parallel_processor.process_batch(
            [Path(p) for p in pending],
            embed_document,
            description="Embedding Generation"
        )
        
        # Calculate cache hit rate
        cache_hits = sum(
            1 for r in results['results']
            if r['result'].get('from_cache', False)
        )
        cache_hit_rate = (cache_hits / len(results['results']) * 100) if results['results'] else 0
        
        print(f"\n📊 Cache Performance:")
        print(f"   Cache hits: {cache_hits}/{len(results['results'])} ({cache_hit_rate:.1f}%)")
        print(f"   API calls saved: {cache_hits}")
        
        cache_stats_after = self.embedding_cache.get_stats()
        print(f"   Cache size: {cache_stats_after['cache_size_mb']:.2f} MB")
        
        # Update checkpoint
        for item in results['results']:
            # Don't store the actual embedding in checkpoint (too large)
            metadata = {k: v for k, v in item['result'].items() if k != 'embedding'}
            self.checkpoint.mark_item_completed(
                stage_name,
                item['item'],
                metadata=metadata
            )
        
        for error in results['errors']:
            self.checkpoint.mark_item_failed(
                stage_name,
                error['item'],
                error['error']
            )
        
        self.checkpoint.mark_stage_complete(stage_name)
        
        # Add cache stats to results
        results['stats']['cache_hit_rate'] = cache_hit_rate
        results['stats']['cache_hits'] = cache_hits
        
        self.pipeline_stats['stages'][stage_name] = results['stats']
        
        return results
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            # Discover documents
            documents = self.discover_documents()
            
            if not documents:
                print("❌ No documents found!")
                return
            
            # Stage 1: Quality Analysis
            quality_results = self.stage_1_quality_analysis(documents)
            
            # Stage 2: Cleaning
            cleaning_results = self.stage_2_cleaning(documents, quality_results)
            
            # Stage 3: Embedding
            embedding_results = self.stage_3_embedding(documents)
            
            # Pipeline complete
            self.pipeline_stats['completed'] = datetime.now().isoformat()
            self.pipeline_stats['status'] = 'success'
            
            # Save results
            results_file = f'data/artifacts/pipeline_results_{self.batch_id}.json'
            with open(results_file, 'w') as f:
                json.dump(self.pipeline_stats, f, indent=2)
            
            # Print summary
            self.print_summary()
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📄 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            self.pipeline_stats['status'] = 'failed'
            self.pipeline_stats['error'] = str(e)
            raise
    
    def print_summary(self):
        """Print pipeline execution summary"""
        print(f"\n{'='*60}")
        print("  📊 Pipeline Summary")
        print(f"{'='*60}")
        
        for stage_name, stats in self.pipeline_stats['stages'].items():
            print(f"\n{stage_name.upper()}:")
            print(f"  Items processed: {stats['successful']}/{stats['total_items']}")
            print(f"  Time: {stats['elapsed_seconds']:.2f}s")
            print(f"  Speed: {stats['items_per_second']:.2f} items/sec")
            
            if 'cache_hit_rate' in stats:
                print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run optimized PeatLearn pipeline'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with documents'
    )
    parser.add_argument(
        '--batch-id',
        type=str,
        help='Batch identifier (default: auto-generated)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    # Generate batch ID if not provided
    batch_id = args.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run pipeline
    pipeline = OptimizedPipeline(
        batch_id=batch_id,
        input_dir=args.input,
        max_workers=args.workers
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()