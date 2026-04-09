"""
Enhanced AI-powered cleaning functions specifically designed to extract
pure Ray Peat signal from noisy transcripts and documents.

Focus on:
1. Extracting Ray Peat's core bioenergetic insights
2. Clear speaker attribution in conversations  
3. Removing ads, introductions, and off-topic content
4. Creating compendious, pertinent data for RAG
"""

import re
import json
import logging
import os
from pathlib import Path
from google import genai
from typing import Dict, List, Tuple, Optional

_AI_MODEL = 'gemini-2.0-flash-exp'

def _get_client() -> genai.Client:
    return genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced prompts for Ray Peat signal extraction
RAY_PEAT_SIGNAL_EXTRACTION_PROMPT = """
You are an expert at extracting Ray Peat's bioenergetic wisdom from noisy transcripts and documents.

Your task is to extract ONLY the pure signal - Ray Peat's actual teachings, insights, and scientific discussions about bioenergetics, metabolism, hormones, nutrition, and health.

REMOVE COMPLETELY:
- All commercial advertisements and product promotions
- Host introductions and show logistics
- Website mentions and contact information  
- Caller management and technical difficulties
- Social pleasantries and off-topic conversations
- Meandering discussions not related to Ray Peat's teachings
- Repetitive content and filler

PRESERVE AND CLEARLY LABEL:
- Ray Peat's direct quotes and explanations
- Scientific discussions led by Ray Peat
- Q&A where Ray Peat provides substantive answers
- Core bioenergetic principles and mechanisms

For conversation formats, use this labeling:
**RAY PEAT:** [His actual words and teachings]
**HOST/CALLER:** [Only if directly relevant to Ray Peat's response]

OUTPUT FORMAT:
Return a JSON object with:
{
  "extracted_content": "The cleaned, labeled content with pure Ray Peat signal",
  "content_type": "conversation" or "article" or "paper",
  "ray_peat_percentage": estimated percentage of content that is Ray Peat's actual words,
  "key_topics": ["list", "of", "main", "bioenergetic", "topics"],
  "signal_quality": "high/medium/low based on density of Ray Peat insights"
}

Content to process:
{content}
"""

CONVERSATION_SEGMENTATION_PROMPT = """
You are an expert at segmenting Ray Peat conversations into focused, coherent topic-based chunks.

Take this cleaned Ray Peat content and break it into logical segments where each segment:
1. Focuses on a specific bioenergetic topic or concept
2. Contains substantial Ray Peat insights (not just brief answers)
3. Is self-contained and makes sense independently
4. Maintains speaker attribution

SEGMENT CRITERIA:
- Minimum 200 words of substantive content
- Clear topic focus (thyroid, PUFA, sugar metabolism, etc.)
- Contains actionable Ray Peat insights
- Excludes pure social interaction

OUTPUT FORMAT:
Return a JSON object with:
{
  "segments": [
    {
      "segment_id": 1,
      "topic": "Primary topic/concept discussed",
      "content": "The segmented content with speaker labels",
      "key_concepts": ["concept1", "concept2"],
      "ray_peat_insights": ["key insight 1", "key insight 2"],
      "word_count": number
    }
  ],
  "total_segments": number,
  "extraction_summary": "Brief summary of what was extracted"
}

Content to segment:
{content}
"""

def extract_ray_peat_signal(content: str) -> Dict:
    """
    Extract pure Ray Peat signal from noisy content using AI.
    
    Args:
        content (str): Raw content to clean
        model: Gemini model instance
        
    Returns:
        Dict: Extracted signal with metadata
    """
    if not content or len(content.strip()) < 100:
        return {
            "extracted_content": "",
            "content_type": "insufficient",
            "ray_peat_percentage": 0,
            "key_topics": [],
            "signal_quality": "low"
        }
    
    try:
        client = _get_client()
        prompt = RAY_PEAT_SIGNAL_EXTRACTION_PROMPT.format(content=content[:50000])
        response = client.models.generate_content(model=_AI_MODEL, contents=prompt)
        
        # Clean and parse JSON response
        clean_response = response.text.strip()
        if '```json' in clean_response:
            clean_response = clean_response.split('```json')[1].split('```')[0]
        elif '```' in clean_response:
            clean_response = clean_response.split('```')[1].split('```')[0]
            
        result = json.loads(clean_response)
        
        logger.info(f"Extracted Ray Peat signal: {result.get('ray_peat_percentage', 0)}% signal, "
                   f"quality: {result.get('signal_quality', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting Ray Peat signal: {e}")
        return {
            "extracted_content": content,  # Fallback to original
            "content_type": "error",
            "ray_peat_percentage": 0,
            "key_topics": [],
            "signal_quality": "low",
            "error": str(e)
        }

def segment_ray_peat_content(content: str) -> Dict:
    """
    Segment cleaned Ray Peat content into focused topic-based chunks.
    
    Args:
        content (str): Cleaned content to segment
        model: Gemini model instance
        
    Returns:
        Dict: Segmented content with metadata
    """
    if not content or len(content.strip()) < 200:
        return {
            "segments": [],
            "total_segments": 0,
            "extraction_summary": "Content too short for segmentation"
        }
    
    try:
        client = _get_client()
        prompt = CONVERSATION_SEGMENTATION_PROMPT.format(content=content[:50000])
        response = client.models.generate_content(model=_AI_MODEL, contents=prompt)
        
        # Clean and parse JSON response
        clean_response = response.text.strip()
        if '```json' in clean_response:
            clean_response = clean_response.split('```json')[1].split('```')[0]
        elif '```' in clean_response:
            clean_response = clean_response.split('```')[1].split('```')[0]
            
        result = json.loads(clean_response)
        
        logger.info(f"Segmented content into {result.get('total_segments', 0)} focused segments")
        
        return result
        
    except Exception as e:
        logger.error(f"Error segmenting content: {e}")
        return {
            "segments": [{
                "segment_id": 1,
                "topic": "Unknown",
                "content": content,
                "key_concepts": [],
                "ray_peat_insights": [],
                "word_count": len(content.split())
            }],
            "total_segments": 1,
            "extraction_summary": f"Segmentation failed: {str(e)}"
        }

def enhance_tier1_with_signal_extraction(input_dir: str, output_dir: str) -> Dict:
    """
    Re-process Tier 1 files with enhanced signal extraction.
    
    Args:
        input_dir (str): Directory with Tier 1 cleaned files
        output_dir (str): Directory for enhanced output
        model: Gemini model instance
        
    Returns:
        Dict: Processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "files_processed": 0,
        "files_enhanced": 0,
        "total_signal_extracted": 0,
        "high_quality_files": 0,
        "segments_created": 0
    }
    
    for file_path in input_path.glob("*.txt"):
        try:
            logger.info(f"Enhancing signal extraction for: {file_path.name}")
            
            # Read original cleaned content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract pure Ray Peat signal
            signal_result = extract_ray_peat_signal(content)
            
            if signal_result["signal_quality"] in ["high", "medium"] and \
               signal_result["ray_peat_percentage"] >= 20:
                
                # Segment if it's substantial content
                if len(signal_result["extracted_content"]) > 1000:
                    segment_result = segment_ray_peat_content(
                        signal_result["extracted_content"]
                    )
        
                    # Save segmented content
                    for i, segment in enumerate(segment_result["segments"]):
                        segment_filename = f"{file_path.stem}_segment_{i+1}.txt"
                        segment_path = output_path / segment_filename
                        
                        # Create enhanced content with metadata
                        enhanced_content = f"""# Ray Peat Content - {segment['topic']}

**Source:** {file_path.name}
**Signal Quality:** {signal_result['signal_quality']}
**Ray Peat Content:** {signal_result['ray_peat_percentage']}%
**Key Concepts:** {', '.join(segment['key_concepts'])}

---

{segment['content']}

---

**Key Insights:**
{chr(10).join(f"• {insight}" for insight in segment['ray_peat_insights'])}
"""
                        
                        with open(segment_path, 'w', encoding='utf-8') as f:
                            f.write(enhanced_content)
                        
                        stats["segments_created"] += 1
                
                else:
                    # Save as single enhanced file
                    enhanced_filename = f"{file_path.stem}_enhanced.txt"
                    enhanced_path = output_path / enhanced_filename
                    
                    enhanced_content = f"""# Ray Peat Content

**Source:** {file_path.name}
**Signal Quality:** {signal_result['signal_quality']}
**Ray Peat Content:** {signal_result['ray_peat_percentage']}%
**Topics:** {', '.join(signal_result['key_topics'])}

---

{signal_result['extracted_content']}
"""
                    
                    with open(enhanced_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced_content)
                
                stats["files_enhanced"] += 1
                if signal_result["signal_quality"] == "high":
                    stats["high_quality_files"] += 1
                
                stats["total_signal_extracted"] += signal_result["ray_peat_percentage"]
            
            stats["files_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error enhancing {file_path.name}: {e}")
            continue
    
    # Save processing statistics
    with open(output_path / "enhancement_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Enhanced signal extraction complete: {stats}")
    return stats

if __name__ == "__main__":
    try:
        _get_client()
        print("Enhanced AI cleaners module loaded successfully")
        print("Gemini API connection established")
    except Exception as e:
        print(f"Error initializing Enhanced AI cleaners: {e}") 