"""
Smart Ray Peat Content Cleaner
Removes noise while preserving ALL Ray Peat content.
Uses AI only where rules-based cleaning isn't sufficient.
"""

import os
import json
import time
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from google import genai
from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Smart thresholds
NOISE_THRESHOLD = 0.3  # If >30% noise, use AI to clean
MIN_CONTENT_LENGTH = 200  # Skip very short files

class SmartCleaner:
    """Smart cleaner that preserves content while removing noise."""
    
    def __init__(self):
        """Initialize with API key."""
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self._model_name = 'gemini-2.5-flash-lite'
            logger.info("✅ AI cleaning available")
        else:
            self.client = None
            logger.warning("⚠️ No API key - rules-based cleaning only")
    
    def detect_noise_level(self, content: str) -> float:
        """Detect noise level in content (0.0 = clean, 1.0 = very noisy)."""
        total_chars = len(content)
        if total_chars < 50:
            return 0.0
        
        noise_patterns = [
            r'🎶.*?🎶',  # Music markers
            r'\[.*?\]',   # Transcript markers  
            r'\d{1,2}:\d{2}',  # Timestamps
            r'(?:um|uh|ah|er|you know|like)\b',  # Filler words
            r'(?:hello|hi|hey|good|okay|alright)\s*[.!?]',  # Basic greetings
            r'(?:KMUD|radio|station|commercial|sponsor|advertisement)',  # Radio noise
            r'(?:phone|call|caller|you\'re on|where are you from)',  # Call-in noise
            r'(?:recorded|transcribed|audio|sound quality)',  # Technical noise
        ]
        
        noise_chars = 0
        for pattern in noise_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            noise_chars += sum(len(match) for match in matches)
        
        return min(1.0, noise_chars / total_chars)
    
    def rules_based_clean(self, content: str) -> str:
        """Apply rules-based cleaning to remove obvious noise."""
        cleaned = content
        
        # Remove music markers
        cleaned = re.sub(r'🎶.*?🎶', '', cleaned, flags=re.DOTALL)
        
        # Remove timestamps 
        cleaned = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', cleaned)
        
        # Remove transcript markers
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        
        # Clean filler words with better patterns
        # Remove filler words with surrounding punctuation/spaces
        cleaned = re.sub(r'\b(?:um|uh|ah|er),?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(?:you know|like),?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',\s*(?:um|uh|ah|er)\b,?', '', cleaned, flags=re.IGNORECASE)
        
        # Clean basic conversational noise
        cleaned = re.sub(r'^(?:hello|hi|hey|good|okay|alright)\s*[.!?]\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove radio station references
        cleaned = re.sub(r'(?:KMUD|radio station|commercial break)', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra punctuation and spaces left by removals
        cleaned = re.sub(r',\s*,', ',', cleaned)  # Double commas
        cleaned = re.sub(r'\s*,\s*([.!?])', r'\1', cleaned)  # Comma before punctuation
        cleaned = re.sub(r':\s*,', ':', cleaned)  # Colon followed by comma
        cleaned = re.sub(r'\b,\s*(?=[A-Z])', ' ', cleaned)  # Comma before capital letter
        
        # Normalize whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple blank lines -> double
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces -> single
        cleaned = re.sub(r' +([.!?])', r'\1', cleaned)  # Space before punctuation
        cleaned = cleaned.strip()
        
        return cleaned
    
    def ai_smart_clean(self, content: str) -> str:
        """Use AI to intelligently clean noisy content while preserving all Ray Peat material."""
        if not self.client:
            return content
        
        prompt = f"""
Clean this Ray Peat content by removing noise while preserving ALL educational content.

PRESERVE EVERYTHING FROM RAY PEAT:
- Keep ALL Ray Peat's words, explanations, and ideas
- Maintain complete scientific discussions  
- Preserve all educational value
- Keep host questions that lead to Ray Peat explanations

REMOVE ONLY NOISE:
- Remove greetings, small talk, logistics
- Remove music markers (🎶), timestamps, technical artifacts
- Remove repetitive filler words (um, uh, like)
- Remove advertisement content and radio station references
- Remove "you're on the air" type call-in show logistics

FORMAT IMPROVEMENTS:
- Use clear speaker attribution: "Ray Peat:" and "Host:" 
- Clean up grammar and flow while keeping exact meaning
- Organize into coherent paragraphs

IMPORTANT: Do NOT summarize or shorten Ray Peat's actual content. Transform format, not substance.

Content to clean:
{content[:4000]}

Return the cleaned content:
"""
        
        try:
            response = self.client.models.generate_content(model=self._model_name, contents=prompt)
            return response.text
        except Exception as e:
            logger.error(f"AI cleaning failed: {e}")
            return content
    
    def process_file(self, file_path: Path) -> Dict:
        """Process a single file with smart cleaning."""
        start_time = time.time()
        
        try:
            # Read content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original = f.read()
            
            if len(original) < MIN_CONTENT_LENGTH:
                return {
                    'success': False,
                    'reason': 'too_short',
                    'original_size': len(original),
                    'processed_size': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Detect noise level
            noise_level = self.detect_noise_level(original)
            
            # Always apply rules-based cleaning first
            cleaned = self.rules_based_clean(original)
            
            # Use AI if still noisy after rules-based cleaning
            if noise_level > NOISE_THRESHOLD and self.client:
                logger.info(f"🤖 AI cleaning (noise: {noise_level:.1%}): {file_path.name}")
                final_content = self.ai_smart_clean(cleaned)
                method = 'ai_enhanced'
            else:
                logger.info(f"📝 Rules cleaning (noise: {noise_level:.1%}): {file_path.name}")
                final_content = cleaned
                method = 'rules_only'
            
            # Verify we didn't lose too much content
            size_ratio = len(final_content) / len(original)
            if size_ratio < 0.3:  # Lost more than 70% of content
                logger.warning(f"⚠️ Excessive content loss ({size_ratio:.1%}), using rules-only")
                final_content = cleaned
                method = 'rules_fallback'
            
            return {
                'success': True,
                'method': method,
                'original_size': len(original),
                'processed_size': len(final_content),
                'content': final_content,
                'noise_level': noise_level,
                'size_ratio': size_ratio,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                'success': False,
                'reason': str(e),
                'original_size': 0,
                'processed_size': 0,
                'processing_time': time.time() - start_time
            }
    
    def process_corpus(self, input_dir: str, output_dir: str, limit: Optional[int] = None) -> Dict:
        """Process entire corpus with smart cleaning."""
        logger.info("🚀 Starting smart corpus cleaning")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all text files
        all_files = list(input_path.rglob('*.txt')) + list(input_path.rglob('*.html'))
        if limit:
            all_files = all_files[:limit]
        
        logger.info(f"📁 Processing {len(all_files)} files")
        
        results = []
        stats = {
            'total_files': len(all_files),
            'successful': 0,
            'rules_only': 0,
            'ai_enhanced': 0,
            'failed': 0,
            'total_size_before': 0,
            'total_size_after': 0
        }
        
        for i, file_path in enumerate(all_files, 1):
            result = self.process_file(file_path)
            results.append(result)
            
            if result['success']:
                stats['successful'] += 1
                stats['total_size_before'] += result['original_size']
                stats['total_size_after'] += result['processed_size']
                
                if result['method'] == 'ai_enhanced':
                    stats['ai_enhanced'] += 1
                else:
                    stats['rules_only'] += 1
                
                # Save processed content
                output_file = output_path / f"{file_path.stem}_clean.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['content'])
                
                logger.info(f"✅ {i}/{len(all_files)}: {file_path.name} "
                          f"({result['method']}, {result['size_ratio']:.1%} size)")
            else:
                stats['failed'] += 1
                logger.warning(f"❌ {i}/{len(all_files)}: {file_path.name} failed")
            
            # Progress update
            if i % 25 == 0:
                avg_size_ratio = stats['total_size_after'] / max(stats['total_size_before'], 1)
                logger.info(f"📊 Progress: {i}/{len(all_files)} | "
                          f"Success: {stats['successful']}/{i} | "
                          f"AI: {stats['ai_enhanced']} | "
                          f"Avg size: {avg_size_ratio:.1%}")
        
        # Save summary
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'stats': stats,
            'average_size_retention': stats['total_size_after'] / max(stats['total_size_before'], 1)
        }
        
        with open(output_path / 'cleaning_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("🎉 Smart cleaning complete!")
        logger.info(f"📊 Results: {stats['successful']}/{stats['total_files']} successful")
        logger.info(f"🤖 AI enhanced: {stats['ai_enhanced']} files")
        logger.info(f"📝 Rules only: {stats['rules_only']} files")
        logger.info(f"📏 Content retained: {summary['average_size_retention']:.1%}")
        
        return summary

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Ray Peat Content Cleaner')
    parser.add_argument('--input-dir', required=True, help='Input directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit files for testing')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cleaner = SmartCleaner()
    summary = cleaner.process_corpus(args.input_dir, args.output_dir, args.limit)
    
    print(f"✅ Smart cleaning complete!")
    print(f"📊 Success rate: {summary['stats']['successful']}/{summary['stats']['total_files']}")
    print(f"📏 Content retained: {summary['average_size_retention']:.1%}")

if __name__ == "__main__":
    main() 