"""
Parallel Processing Engine for PeatLearn Pipeline
Processes multiple documents simultaneously using all CPU cores
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import List, Dict, Callable, Any
import traceback

class ParallelProcessor:
    def __init__(self, max_workers=None, show_progress=True):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Number of parallel workers (default: CPU count)
            show_progress: Show progress bar
        """
        self.max_workers = max_workers or max(1, cpu_count() - 1)  # Leave 1 core free
        self.show_progress = show_progress
        self.results = []
        self.errors = []
        
        print(f"🚀 Initialized parallel processor with {self.max_workers} workers")
    
    def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        description: str = "Processing"
    ) -> Dict:
        """
        Process a batch of items in parallel
        
        Args:
            items: List of items to process
            processor_func: Function to process each item (must be picklable)
            description: Description for progress bar
            
        Returns:
            Dict with 'results', 'errors', and 'stats'
        """
        print(f"\n📊 Processing {len(items)} items in parallel...")
        print(f"⚙️  Workers: {self.max_workers}")
        
        results = []
        errors = []
        
        start_time = time.time()
        
        # Process in parallel with progress bar
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(processor_func, item): item
                for item in items
            }
            
            # Collect results with progress bar
            if self.show_progress:
                futures = tqdm(
                    as_completed(future_to_item),
                    total=len(items),
                    desc=description
                )
            else:
                futures = as_completed(future_to_item)
            
            for future in futures:
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append({
                        'item': item,
                        'result': result,
                        'status': 'success'
                    })
                except Exception as e:
                    error_info = {
                        'item': item,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'status': 'failed'
                    }
                    errors.append(error_info)
                    if self.show_progress:
                        tqdm.write(f"❌ Error processing {item}: {e}")
        
        elapsed = time.time() - start_time
        
        stats = {
            'total_items': len(items),
            'successful': len(results),
            'failed': len(errors),
            'elapsed_seconds': elapsed,
            'items_per_second': len(items) / elapsed if elapsed > 0 else 0,
            'workers_used': self.max_workers
        }
        
        print(f"\n✅ Parallel processing complete!")
        print(f"   Success: {stats['successful']}/{stats['total_items']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Time: {elapsed:.2f}s ({stats['items_per_second']:.2f} items/sec)")
        
        return {
            'results': results,
            'errors': errors,
            'stats': stats
        }


# Example processor functions (must be top-level for pickling)
def process_document_quality(doc_path):
    """Example: Quality analysis for a single document"""
    # Simulate quality analysis
    time.sleep(0.1)  # Replace with actual analysis
    
    return {
        'path': str(doc_path),
        'quality_score': 0.85,
        'complexity': 'medium'
    }

def process_document_cleaning(doc_path):
    """Example: Clean a single document"""
    # Simulate cleaning
    time.sleep(0.5)  # Replace with actual cleaning
    
    return {
        'path': str(doc_path),
        'cleaned': True,
        'tier': 'tier1'
    }


def demo():
    """Demo of parallel processing"""
    # Create dummy document paths
    doc_paths = [f"data/raw/doc_{i}.txt" for i in range(100)]
    
    processor = ParallelProcessor()
    
    # Process quality analysis
    results = processor.process_batch(
        doc_paths,
        process_document_quality,
        description="Quality Analysis"
    )
    
    print(f"\n📊 Results:")
    print(json.dumps(results['stats'], indent=2))


if __name__ == "__main__":
    demo()