"""
Checkpoint & Resume System for PeatLearn Pipeline
Allows recovery from crashes and incremental processing
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
import hashlib

class PipelineCheckpoint:
    def __init__(self, batch_id: str, checkpoint_dir='checkpoints'):
        """
        Initialize checkpoint system
        
        Args:
            batch_id: Unique identifier for this batch
            checkpoint_dir: Directory to store checkpoint files
        """
        self.batch_id = batch_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / f'{batch_id}.json'
        
        # Load or initialize state
        self.state = self._load_state()
        
        print(f"💾 Checkpoint system initialized")
        print(f"   Batch ID: {batch_id}")
        print(f"   Checkpoint: {self.checkpoint_file}")
    
    def _load_state(self) -> Dict:
        """Load checkpoint state from disk"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    state = json.load(f)
                    print(f"   📂 Loaded existing checkpoint")
                    return state
            except Exception as e:
                print(f"   ⚠️  Error loading checkpoint: {e}")
        
        # Initialize new state
        return {
            'batch_id': self.batch_id,
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'stages': {},
            'metadata': {},
            'total_items': 0,
            'completed_items': 0
        }
    
    def _save_state(self):
        """Save checkpoint state to disk"""
        self.state['last_updated'] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _item_hash(self, item: str) -> str:
        """Generate hash for an item"""
        return hashlib.md5(str(item).encode()).hexdigest()[:16]
    
    def mark_stage_start(self, stage_name: str, total_items: int):
        """
        Mark the start of a pipeline stage
        
        Args:
            stage_name: Name of the stage
            total_items: Total number of items in this stage
        """
        if stage_name not in self.state['stages']:
            self.state['stages'][stage_name] = {
                'started': datetime.now().isoformat(),
                'completed': [],
                'failed': [],
                'total_items': total_items,
                'status': 'in_progress'
            }
        
        self._save_state()
        print(f"🎬 Started stage: {stage_name} ({total_items} items)")
    
    def mark_item_completed(self, stage_name: str, item: str, metadata: Dict = None):
        """
        Mark an item as completed in a stage
        
        Args:
            stage_name: Name of the stage
            item: Item identifier
            metadata: Optional metadata about the processing
        """
        if stage_name not in self.state['stages']:
            self.mark_stage_start(stage_name, 1)
        
        item_hash = self._item_hash(item)
        
        if item_hash not in self.state['stages'][stage_name]['completed']:
            self.state['stages'][stage_name]['completed'].append(item_hash)
            self.state['completed_items'] += 1
            
            # Store metadata if provided
            if metadata:
                if 'item_metadata' not in self.state['stages'][stage_name]:
                    self.state['stages'][stage_name]['item_metadata'] = {}
                self.state['stages'][stage_name]['item_metadata'][item_hash] = metadata
            
            self._save_state()
    
    def mark_item_failed(self, stage_name: str, item: str, error: str):
        """
        Mark an item as failed in a stage
        
        Args:
            stage_name: Name of the stage
            item: Item identifier
            error: Error message
        """
        if stage_name not in self.state['stages']:
            self.mark_stage_start(stage_name, 1)
        
        item_hash = self._item_hash(item)
        
        self.state['stages'][stage_name]['failed'].append({
            'item_hash': item_hash,
            'item': str(item),
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_state()
    
    def mark_stage_complete(self, stage_name: str):
        """
        Mark a stage as complete
        
        Args:
            stage_name: Name of the stage
        """
        if stage_name in self.state['stages']:
            self.state['stages'][stage_name]['status'] = 'completed'
            self.state['stages'][stage_name]['completed_at'] = datetime.now().isoformat()
            self._save_state()
            
            print(f"✅ Completed stage: {stage_name}")
    
    def is_item_completed(self, stage_name: str, item: str) -> bool:
        """
        Check if an item has been completed in a stage
        
        Args:
            stage_name: Name of the stage
            item: Item identifier
            
        Returns:
            True if completed, False otherwise
        """
        if stage_name not in self.state['stages']:
            return False
        
        item_hash = self._item_hash(item)
        return item_hash in self.state['stages'][stage_name]['completed']
    
    def get_pending_items(self, stage_name: str, all_items: List[str]) -> List[str]:
        """
        Get list of items that still need processing in a stage
        
        Args:
            stage_name: Name of the stage
            all_items: All items in the stage
            
        Returns:
            List of pending items
        """
        if stage_name not in self.state['stages']:
            return all_items
        
        completed_hashes = set(self.state['stages'][stage_name]['completed'])
        
        return [
            item for item in all_items
            if self._item_hash(item) not in completed_hashes
        ]
    
    def get_stage_progress(self, stage_name: str) -> Dict:
        """
        Get progress information for a stage
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dict with progress stats
        """
        if stage_name not in self.state['stages']:
            return {
                'total': 0,
                'completed': 0,
                'failed': 0,
                'pending': 0,
                'progress_pct': 0.0
            }
        
        stage = self.state['stages'][stage_name]
        total = stage['total_items']
        completed = len(stage['completed'])
        failed = len(stage['failed'])
        pending = total - completed - failed
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'progress_pct': (completed / total * 100) if total > 0 else 0.0,
            'status': stage.get('status', 'unknown')
        }
    
    def get_summary(self) -> Dict:
        """Get overall checkpoint summary"""
        total_completed = sum(
            len(stage['completed'])
            for stage in self.state['stages'].values()
        )
        
        total_failed = sum(
            len(stage['failed'])
            for stage in self.state['stages'].values()
        )
        
        return {
            'batch_id': self.batch_id,
            'created': self.state['created'],
            'last_updated': self.state['last_updated'],
            'stages': len(self.state['stages']),
            'total_items_processed': total_completed,
            'total_failures': total_failed,
            'stage_summaries': {
                name: self.get_stage_progress(name)
                for name in self.state['stages'].keys()
            }
        }
    
    def resume_info(self) -> Dict:
        """Get information about where to resume from"""
        incomplete_stages = [
            name for name, stage in self.state['stages'].items()
            if stage['status'] != 'completed'
        ]
        
        if not incomplete_stages:
            return {
                'can_resume': False,
                'message': 'All stages completed'
            }
        
        next_stage = incomplete_stages[0]
        progress = self.get_stage_progress(next_stage)
        
        return {
            'can_resume': True,
            'next_stage': next_stage,
            'progress': progress,
            'message': f"Resume from stage '{next_stage}' at {progress['progress_pct']:.1f}%"
        }


def demo():
    """Demo of checkpoint system"""
    # Simulate processing a batch
    checkpoint = PipelineCheckpoint('batch_20260117_demo')
    
    # Stage 1: Quality Analysis
    documents = [f'doc_{i}.txt' for i in range(10)]
    checkpoint.mark_stage_start('quality_analysis', len(documents))
    
    for doc in documents[:7]:  # Process first 7
        checkpoint.mark_item_completed('quality_analysis', doc, {'score': 0.85})
    
    # Simulate crash... then resume
    print("\n💥 Simulating crash...\n")
    
    # Later, resume
    checkpoint2 = PipelineCheckpoint('batch_20260117_demo')
    resume_info = checkpoint2.resume_info()
    print(f"📍 Resume Info:")
    print(json.dumps(resume_info, indent=2))
    
    # Get pending items
    pending = checkpoint2.get_pending_items('quality_analysis', documents)
    print(f"\n⏳ Pending items: {pending}")
    
    # Complete remaining
    for doc in pending:
        checkpoint2.mark_item_completed('quality_analysis', doc, {'score': 0.90})
    
    checkpoint2.mark_stage_complete('quality_analysis')
    
    # Summary
    summary = checkpoint2.get_summary()
    print(f"\n📊 Final Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    demo()