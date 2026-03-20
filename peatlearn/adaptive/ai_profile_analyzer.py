#!/usr/bin/env python3
"""
AI-Enhanced Profile Analyzer for PeatLearn Adaptive Learning System
Uses small AI models like Gemini 2.5-flash-lite for intelligent mastery assessment
"""

import google.generativeai as genai
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from .profile_analyzer import TopicExtractor, LearnerProfiler

class AIEnhancedProfiler(LearnerProfiler):
    """
    Enhanced profiler that uses AI models for smarter analysis
    """
    
    def __init__(self, profiles_file: str = "data/user_interactions/user_profiles.json"):
        super().__init__(profiles_file)
        
        # Initialize Gemini with API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.ai_enabled = True
        else:
            print("Warning: GEMINI_API_KEY not found. Falling back to rule-based analysis.")
            self.ai_enabled = False
    
    def analyze_mastery_with_ai(self, 
                               user_query: str, 
                               llm_response: str, 
                               user_feedback: Optional[int],
                               topic: str) -> Dict[str, Any]:
        """
        Use AI to analyze mastery level from interaction content
        
        Args:
            user_query: User's question
            llm_response: AI's response
            user_feedback: User feedback (1, -1, or None)
            topic: Topic category
            
        Returns:
            AI assessment of mastery indicators
        """
        
        if not self.ai_enabled:
            return self._fallback_mastery_analysis(user_query, user_feedback, topic)
        
        try:
            prompt = f"""
            Analyze this learning interaction for mastery indicators:
            
            TOPIC: {topic}
            USER QUESTION: "{user_query}"
            AI RESPONSE: "{llm_response[:500]}..."  # Truncate for token limits
            USER FEEDBACK: {user_feedback if user_feedback else "No feedback"}
            
            Based on Ray Peat's bioenergetic concepts, assess:
            
            1. QUESTION COMPLEXITY: Rate the sophistication of the user's question (1-5)
               - 1: Very basic, general questions
               - 3: Shows some understanding, asks for clarification
               - 5: Advanced, specific, shows deep knowledge
            
            2. CONCEPTUAL UNDERSTANDING: Based on the question, does the user show (1-5):
               - 1: No prior knowledge
               - 3: Basic understanding with gaps
               - 5: Strong grasp of interconnected concepts
            
            3. LEARNING INDICATORS: What does this interaction suggest about their learning?
               - struggling: Confused, basic questions, negative feedback
               - learning: Progressing, mixed complexity, generally positive
               - advanced: Sophisticated questions, connecting concepts
            
            4. TOPIC MASTERY ESTIMATE: Overall mastery level (0.0-1.0)
            
            Respond in JSON format:
            {{
                "question_complexity": <1-5>,
                "conceptual_understanding": <1-5>, 
                "learning_stage": "<struggling|learning|advanced>",
                "mastery_estimate": <0.0-1.0>,
                "confidence": <0.0-1.0>,
                "reasoning": "<brief explanation>"
            }}
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response text - remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            result = json.loads(response_text)
            
            # Validate and normalize the response
            result['question_complexity'] = max(1, min(5, result.get('question_complexity', 3)))
            result['conceptual_understanding'] = max(1, min(5, result.get('conceptual_understanding', 3)))
            result['mastery_estimate'] = max(0.0, min(1.0, result.get('mastery_estimate', 0.5)))
            result['confidence'] = max(0.0, min(1.0, result.get('confidence', 0.5)))
            
            return result
            
        except Exception as e:
            print(f"AI analysis failed: {e}. Falling back to rules.")
            return self._fallback_mastery_analysis(user_query, user_feedback, topic)
    
    def _fallback_mastery_analysis(self, 
                                  user_query: str, 
                                  user_feedback: Optional[int], 
                                  topic: str) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when AI is unavailable
        """
        
        # Simple heuristics
        query_length = len(user_query.split())
        has_technical_terms = any(term in user_query.lower() for term in 
                                ['thyroid', 't3', 't4', 'progesterone', 'estrogen', 'cortisol', 
                                 'metabolism', 'mitochondria', 'oxidative', 'cellular'])
        
        # Estimate complexity based on query characteristics
        if query_length > 15 and has_technical_terms:
            complexity = 4
            understanding = 4
        elif query_length > 8 and has_technical_terms:
            complexity = 3
            understanding = 3
        elif has_technical_terms:
            complexity = 3
            understanding = 2
        else:
            complexity = 2
            understanding = 2
        
        # Determine learning stage
        if user_feedback == -1:
            stage = "struggling"
            mastery = 0.3
        elif user_feedback == 1 and complexity >= 4:
            stage = "advanced"
            mastery = 0.8
        else:
            stage = "learning"
            mastery = 0.6
        
        return {
            "question_complexity": complexity,
            "conceptual_understanding": understanding,
            "learning_stage": stage,
            "mastery_estimate": mastery,
            "confidence": 0.6,
            "reasoning": "Rule-based analysis (AI unavailable)"
        }
    
    def analyze_learning_progression(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use AI to analyze overall learning progression patterns
        
        Args:
            interactions: List of user interactions
            
        Returns:
            Learning progression analysis
        """
        
        if not self.ai_enabled or len(interactions) < 3:
            return super().analyze_user_interactions(interactions)
        
        try:
            # Prepare interaction summary for AI
            interaction_summary = []
            for i, interaction in enumerate(interactions[-10:]):  # Last 10 interactions
                summary = {
                    "order": i + 1,
                    "query": interaction.get('user_query', '')[:100],  # Truncate
                    "topic": interaction.get('topic', 'unknown'),
                    "feedback": interaction.get('user_feedback'),
                    "timestamp": interaction.get('timestamp', '')[:10]  # Date only
                }
                interaction_summary.append(summary)
            
            prompt = f"""
            Analyze this user's learning progression over {len(interaction_summary)} recent interactions:
            
            INTERACTIONS:
            {json.dumps(interaction_summary, indent=2)}
            
            Based on Ray Peat's bioenergetic concepts, analyze:
            
            1. LEARNING STYLE:
               - explorer: Asks about many different topics, broad curiosity
               - deep_diver: Focuses intensely on specific topics
               - balanced: Mix of exploration and depth
            
            2. OVERALL PROGRESSION:
               - improving: Questions becoming more sophisticated
               - struggling: Consistent basic questions, negative feedback
               - advanced: Consistently sophisticated questions
               - inconsistent: Mixed patterns
            
            3. TOPIC PREFERENCES: Which Ray Peat topics show most engagement?
            
            4. LEARNING VELOCITY: How quickly are they progressing?
            
            Respond in JSON format:
            {{
                "learning_style": "<explorer|deep_diver|balanced>",
                "overall_progression": "<improving|struggling|advanced|inconsistent>",
                "preferred_topics": ["<topic1>", "<topic2>"],
                "learning_velocity": "<slow|moderate|fast>",
                "confidence": <0.0-1.0>,
                "insights": ["<insight1>", "<insight2>"],
                "recommendations": ["<rec1>", "<rec2>"]
            }}
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response text - remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            ai_analysis = json.loads(response_text)
            
            # Combine AI insights with rule-based analysis
            base_profile = super().analyze_user_interactions(interactions)
            
            # Enhance with AI insights
            base_profile['ai_analysis'] = ai_analysis
            base_profile['learning_style'] = ai_analysis.get('learning_style', base_profile['learning_style'])
            
            # Map AI progression to our states
            progression_map = {
                'struggling': 'struggling',
                'improving': 'learning', 
                'advanced': 'advanced',
                'inconsistent': 'learning'
            }
            ai_state = progression_map.get(ai_analysis.get('overall_progression'), 'learning')
            base_profile['overall_state'] = ai_state
            
            return base_profile
            
        except Exception as e:
            print(f"AI progression analysis failed: {e}. Using rule-based analysis.")
            return super().analyze_user_interactions(interactions)
    
    def generate_personalized_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use AI to generate highly personalized recommendations
        
        Args:
            user_profile: User's learning profile
            
        Returns:
            List of personalized recommendations
        """
        
        if not self.ai_enabled:
            return super().get_recommendations(user_profile)
        
        try:
            # Prepare profile summary
            profile_summary = {
                "overall_state": user_profile.get('overall_state'),
                "learning_style": user_profile.get('learning_style'),
                "total_interactions": user_profile.get('total_interactions'),
                "topic_mastery": {
                    topic: {
                        "state": data.get('state'),
                        "mastery_level": round(data.get('mastery_level', 0), 2)
                    }
                    for topic, data in user_profile.get('topic_mastery', {}).items()
                },
                "ai_insights": user_profile.get('ai_analysis', {}).get('insights', [])
            }
            
            prompt = f"""
            Generate personalized learning recommendations for this Ray Peat student:
            
            PROFILE:
            {json.dumps(profile_summary, indent=2)}
            
            Create 3-5 specific, actionable recommendations based on:
            - Their current mastery levels
            - Learning style preferences  
            - Ray Peat's bioenergetic principles
            - Optimal learning progression
            
            For each recommendation, provide:
            - Type: quiz, topic_exploration, review, challenge, etc.
            - Priority: high, medium, low
            - Specific action they should take
            - Why this helps their learning journey
            
            Respond in JSON format:
            {{
                "recommendations": [
                    {{
                        "type": "<type>",
                        "priority": "<high|medium|low>",
                        "title": "<short title>",
                        "description": "<what to do>",
                        "reasoning": "<why this helps>",
                        "action": "<specific action>"
                    }}
                ]
            }}
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response text - remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            ai_recommendations = json.loads(response_text)
            
            # Combine with rule-based recommendations
            rule_recommendations = super().get_recommendations(user_profile)
            
            # Convert AI recommendations to our format
            enhanced_recommendations = []
            for rec in ai_recommendations.get('recommendations', []):
                enhanced_recommendations.append({
                    'type': rec.get('type', 'general'),
                    'title': rec.get('title', 'Learning Recommendation'),
                    'description': rec.get('description', ''),
                    'priority': rec.get('priority', 'medium'),
                    'reasoning': rec.get('reasoning', ''),
                    'action': rec.get('action', ''),
                    'source': 'ai_generated'
                })
            
            # Add rule-based recommendations
            for rec in rule_recommendations:
                rec['source'] = 'rule_based'
                enhanced_recommendations.append(rec)
            
            return enhanced_recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            print(f"AI recommendation generation failed: {e}. Using rule-based recommendations.")
            return super().get_recommendations(user_profile)
    
    def analyze_interaction_with_ai(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single interaction using AI for enhanced insights
        
        Args:
            interaction: Single interaction dictionary
            
        Returns:
            Enhanced interaction analysis
        """
        
        user_query = interaction.get('user_query', '')
        llm_response = interaction.get('llm_response', '')
        user_feedback = interaction.get('user_feedback')
        topic = interaction.get('topic', 'general')
        
        # Get AI mastery analysis
        ai_analysis = self.analyze_mastery_with_ai(user_query, llm_response, user_feedback, topic)
        
        # Enhance interaction with AI insights
        enhanced_interaction = interaction.copy()
        enhanced_interaction['ai_analysis'] = ai_analysis
        enhanced_interaction['complexity_score'] = ai_analysis.get('question_complexity', 3)
        enhanced_interaction['understanding_level'] = ai_analysis.get('conceptual_understanding', 3)
        enhanced_interaction['ai_mastery_estimate'] = ai_analysis.get('mastery_estimate', 0.5)
        
        return enhanced_interaction
    
    def update_user_profile_with_ai(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update user profile using AI-enhanced analysis
        
        Args:
            user_id: User identifier
            interactions: List of user interactions
            
        Returns:
            AI-enhanced user profile
        """
        
        # Analyze progression with AI
        profile = self.analyze_learning_progression(interactions)
        profile['user_id'] = user_id
        
        # Generate AI recommendations
        ai_recommendations = self.generate_personalized_recommendations(profile)
        profile['recommendations'] = ai_recommendations
        
        # Save enhanced profile
        profiles_data = self.load_profiles()
        profiles_data['profiles'][user_id] = profile
        self.save_profiles(profiles_data)
        
        return profile

# Global instance for easy access
ai_profiler = AIEnhancedProfiler()
