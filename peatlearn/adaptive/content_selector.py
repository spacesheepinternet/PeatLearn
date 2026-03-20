#!/usr/bin/env python3
"""
Content Selector for PeatLearn Adaptive Learning System
Modifies RAG responses and generates adaptive content based on user profiles
"""

import random
from typing import Dict, List, Any, Optional, Tuple
import requests

class ContentSelector:
    """
    Selects and modifies content based on user learning profiles
    """
    
    def __init__(self, profiler=None, rag_api_base: str = "http://localhost:8000"):
        self.rag_api_base = rag_api_base
        self.profiler = profiler
        
        # Prompt templates for different learner states
        self.prompt_templates = {
            'struggling': {
                'system_prompt': """You are Ray Peat AI, but you need to explain things very simply and clearly. 
                The user is struggling with this topic, so:
                - Use simple, everyday language
                - Break down complex concepts into small steps
                - Use analogies and examples
                - Avoid jargon and technical terms
                - Be encouraging and supportive
                - Focus on the most basic, foundational concepts first""",
                
                'response_style': "simple and encouraging"
            },
            
            'learning': {
                'system_prompt': """You are Ray Peat AI providing balanced explanations. 
                The user is actively learning, so:
                - Use clear, informative language
                - Provide good detail but stay accessible
                - Include some technical terms with explanations
                - Give examples to illustrate points
                - Build on fundamental concepts progressively""",
                
                'response_style': "informative and clear"
            },
            
            'advanced': {
                'system_prompt': """You are Ray Peat AI providing detailed, advanced explanations. 
                The user has good understanding, so:
                - Use precise scientific language
                - Include technical details and mechanisms
                - Reference research and studies when relevant
                - Explore nuanced aspects and connections
                - Assume familiarity with basic concepts""",
                
                'response_style': "detailed and scientific"
            }
        }
    
    def get_modified_rag_prompt(self, query: str, user_profile: Dict[str, Any]) -> str:
        """
        Get a modified RAG prompt based on user profile
        
        Args:
            query: Original user query
            user_profile: User's learning profile
            
        Returns:
            Modified prompt for RAG system
        """
        if not user_profile:
            return query
        
        overall_state = user_profile.get('overall_state', 'learning')
        
        # Get appropriate template
        template = self.prompt_templates.get(overall_state, self.prompt_templates['learning'])
        
        # Modify the query with context
        if overall_state == 'struggling':
            modified_query = f"Please explain in simple terms: {query}. Use basic language and avoid complex technical terms."
        elif overall_state == 'advanced':
            modified_query = f"Please provide a detailed explanation with technical details: {query}. Include mechanisms, pathways, and advanced concepts."
        else:  # learning
            modified_query = f"Please provide a clear, informative explanation: {query}. Include some technical details but keep it accessible."
        
        return modified_query
    
    def get_adaptive_rag_response(self, 
                                 user_query: str,
                                 user_profile: Dict[str, Any],
                                 user_id: str) -> Dict[str, Any]:
        """
        Get RAG response adapted to user's learning profile
        
        Args:
            user_query: User's question
            user_profile: User's learning profile
            user_id: User identifier
            
        Returns:
            Adapted RAG response with metadata
        """
        
        # Determine adaptation strategy
        adaptation_info = self._analyze_query_and_profile(user_query, user_profile)
        
        # Modify the query if needed
        modified_query = self._adapt_query(user_query, adaptation_info)
        
        # Get base RAG response
        try:
            base_response = self._call_rag_api(modified_query, user_id)
        except Exception as e:
            return {
                'answer': f"I apologize, but I'm having trouble accessing the knowledge base right now. Error: {str(e)}",
                'adaptation_applied': adaptation_info,
                'error': str(e)
            }
        
        # Adapt the response based on user profile
        adapted_response = self._adapt_response(base_response, adaptation_info, user_profile)
        
        return adapted_response
    
    def _analyze_query_and_profile(self, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query and user profile to determine adaptation strategy
        
        Args:
            query: User's query
            profile: User's profile
            
        Returns:
            Adaptation information dictionary
        """
        
        from .profile_analyzer import TopicExtractor
        
        topic_extractor = TopicExtractor()
        primary_topic = topic_extractor.get_primary_topic(query)
        
        # Get overall learning state
        overall_state = profile.get('overall_state', 'learning')
        learning_style = profile.get('learning_style', 'explorer')
        topic_mastery = profile.get('topic_mastery', {})
        
        # Check topic-specific mastery
        topic_state = None
        if primary_topic and primary_topic in topic_mastery:
            topic_state = topic_mastery[primary_topic]['state']
        
        # Determine adaptation level (topic-specific overrides overall)
        adaptation_level = topic_state if topic_state else overall_state
        
        # Special handling for new users
        if adaptation_level == 'new':
            adaptation_level = 'learning'
        
        return {
            'primary_topic': primary_topic,
            'adaptation_level': adaptation_level,
            'learning_style': learning_style,
            'topic_mastery_level': topic_mastery.get(primary_topic, {}).get('mastery_level', 0.5) if primary_topic else 0.5,
            'total_interactions': profile.get('total_interactions', 0)
        }
    
    def _adapt_query(self, query: str, adaptation_info: Dict[str, Any]) -> str:
        """
        Modify the query based on adaptation strategy
        
        Args:
            query: Original query
            adaptation_info: Adaptation strategy information
            
        Returns:
            Modified query
        """
        
        adaptation_level = adaptation_info['adaptation_level']
        
        # Add context to the query based on user level
        if adaptation_level == 'struggling':
            modified_query = f"Please explain in simple terms: {query}"
        elif adaptation_level == 'advanced':
            modified_query = f"Please provide detailed scientific explanation: {query}"
        else:
            modified_query = query
        
        return modified_query
    
    def _call_rag_api(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Call the RAG API with the query
        
        Args:
            query: Query to send
            user_id: User identifier
            
        Returns:
            RAG API response
        """
        
        try:
            response = requests.get(
                f"{self.rag_api_base}/api/ask",
                params={
                    "q": query,
                    "user_id": user_id,
                    "max_sources": 5,
                    "min_similarity": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"answer": "I apologize, but I'm having trouble accessing the knowledge base right now."}
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"RAG API call failed: {str(e)}")
    
    def _adapt_response(self, 
                       base_response: Dict[str, Any], 
                       adaptation_info: Dict[str, Any],
                       user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt the RAG response based on user profile
        
        Args:
            base_response: Original RAG response
            adaptation_info: Adaptation strategy
            user_profile: User profile
            
        Returns:
            Adapted response
        """
        
        adaptation_level = adaptation_info['adaptation_level']
        original_answer = base_response.get('answer', '')
        
        # Create adapted response
        adapted_response = base_response.copy()
        
        # Add adaptation metadata
        adapted_response['adaptation_applied'] = adaptation_info
        adapted_response['original_answer'] = original_answer
        
        # Add learner-specific modifications
        if adaptation_level == 'struggling':
            adapted_response['answer'] = self._simplify_response(original_answer)
            adapted_response['helpful_tips'] = self._get_struggling_tips(adaptation_info['primary_topic'])
            adapted_response['encouragement'] = self._get_encouragement()
        
        elif adaptation_level == 'advanced':
            adapted_response['answer'] = self._enhance_response(original_answer)
            adapted_response['related_concepts'] = self._get_related_concepts(adaptation_info['primary_topic'])
            adapted_response['research_suggestions'] = self._get_research_suggestions(adaptation_info['primary_topic'])
        
        else:  # learning level
            adapted_response['answer'] = original_answer
            adapted_response['next_steps'] = self._get_learning_next_steps(adaptation_info['primary_topic'])
        
        # Add personalized recommendations
        adapted_response['recommendations'] = self._get_content_recommendations(adaptation_info, user_profile)
        
        return adapted_response
    
    def _simplify_response(self, response: str) -> str:
        """
        Simplify a response for struggling learners
        
        Args:
            response: Original response text
            
        Returns:
            Simplified response
        """
        
        # Add simplification prefix
        simplified = "Let me explain this in simple terms:\n\n"
        simplified += response
        
        # Add clarification offer
        simplified += "\n\n💡 **Need it even simpler?** Feel free to ask me to explain any part of this differently!"
        
        return simplified
    
    def _enhance_response(self, response: str) -> str:
        """
        Enhance a response for advanced learners
        
        Args:
            response: Original response text
            
        Returns:
            Enhanced response
        """
        
        # Add complexity note
        enhanced = response
        enhanced += "\n\n🔬 **Advanced Note:** This connects to broader bioenergetic principles that Ray Peat emphasized throughout his work."
        
        return enhanced
    
    def _get_struggling_tips(self, topic: Optional[str]) -> List[str]:
        """Get helpful tips for struggling learners"""
        
        general_tips = [
            "Don't worry if this seems complex - Ray Peat's ideas take time to understand!",
            "Try focusing on one concept at a time rather than trying to understand everything at once.",
            "It's okay to ask the same question multiple ways until it clicks.",
            "Remember: even Dr. Peat spent decades developing these insights."
        ]
        
        topic_specific_tips = {
            'metabolism': [
                "Think of your metabolism like a car engine - it needs the right fuel and conditions to run well.",
                "Start with understanding that your thyroid is like your body's gas pedal."
            ],
            'hormones': [
                "Hormones are like messengers in your body - they tell different parts what to do.",
                "Progesterone and estrogen are like a balance scale - you want them in harmony."
            ],
            'nutrition': [
                "Focus on foods that give you energy and make you feel good.",
                "Ray Peat emphasized that good nutrition should make you feel warm and energetic."
            ]
        }
        
        tips = general_tips.copy()
        if topic and topic in topic_specific_tips:
            tips.extend(topic_specific_tips[topic])
        
        return random.sample(tips, min(2, len(tips)))
    
    def _get_encouragement(self) -> str:
        """Get encouraging message for struggling learners"""
        
        encouragements = [
            "You're on the right track - keep asking questions! 🌟",
            "Learning Ray Peat's approach takes time, but you're making progress! 💪",
            "Great question! Curiosity is the first step to understanding. 🧠",
            "Don't give up - each question brings you closer to understanding! ✨"
        ]
        
        return random.choice(encouragements)
    
    def _get_related_concepts(self, topic: Optional[str]) -> List[str]:
        """Get related concepts for advanced learners"""
        
        if not topic:
            return []
        
        concept_map = {
            'metabolism': [
                'Oxidative phosphorylation efficiency',
                'Cellular respiration and CO2 production',
                'Thyroid hormone peripheral conversion',
                'Mitochondrial biogenesis'
            ],
            'hormones': [
                'Steroidogenesis pathway',
                'Hormone receptor sensitivity',
                'Feedback loop mechanisms',
                'Circadian rhythm regulation'
            ],
            'nutrition': [
                'Metabolic flexibility',
                'Nutrient timing and absorption',
                'Food-hormone interactions',
                'Digestive enzyme optimization'
            ],
            'stress': [
                'HPA axis dysfunction',
                'Cortisol rhythm disruption',
                'Stress-induced inflammation',
                'Autonomic nervous system balance'
            ]
        }
        
        return concept_map.get(topic, [])
    
    def _get_research_suggestions(self, topic: Optional[str]) -> List[str]:
        """Get research suggestions for advanced learners"""
        
        if not topic:
            return []
        
        suggestions = [
            "Explore Ray Peat's newsletters for deeper insights on this topic",
            "Look into the research papers Ray Peat referenced in his discussions",
            "Consider how this concept relates to other aspects of bioenergetic medicine",
            "Investigate the historical context of these ideas in endocrinology"
        ]
        
        return random.sample(suggestions, min(2, len(suggestions)))
    
    def _get_learning_next_steps(self, topic: Optional[str]) -> List[str]:
        """Get next learning steps for active learners"""
        
        steps = [
            "Try asking a follow-up question about a specific aspect that interests you",
            "Consider how this applies to your own health and well-being",
            "Explore related topics to build a more complete understanding"
        ]
        
        if topic:
            steps.append(f"Learn more about how {topic} connects to other bioenergetic concepts")
        
        return steps
    
    def _get_content_recommendations(self, 
                                   adaptation_info: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get personalized content recommendations
        
        Args:
            adaptation_info: Adaptation strategy information
            user_profile: User profile
            
        Returns:
            List of content recommendations
        """
        
        recommendations = []
        primary_topic = adaptation_info['primary_topic']
        adaptation_level = adaptation_info['adaptation_level']
        learning_style = adaptation_info['learning_style']
        
        # Topic-based recommendations
        if primary_topic:
            if adaptation_level == 'struggling':
                recommendations.append({
                    'type': 'simpler_explanation',
                    'title': f'Basics of {primary_topic.title()}',
                    'description': f'Let\'s start with the fundamentals of {primary_topic}',
                    'action': 'Ask me: "What are the basics of ' + primary_topic + '?"'
                })
            
            elif adaptation_level == 'advanced':
                recommendations.append({
                    'type': 'advanced_topic',
                    'title': f'Advanced {primary_topic.title()} Concepts',
                    'description': f'Explore complex aspects of {primary_topic}',
                    'action': 'Ask me: "What are the advanced concepts in ' + primary_topic + '?"'
                })
            
            else:  # learning
                recommendations.append({
                    'type': 'related_topic',
                    'title': f'Topics Related to {primary_topic.title()}',
                    'description': f'Discover how {primary_topic} connects to other concepts',
                    'action': 'Ask me: "How does ' + primary_topic + ' relate to other bioenergetic concepts?"'
                })
        
        # Learning style recommendations
        if learning_style == 'explorer':
            recommendations.append({
                'type': 'exploration',
                'title': 'Explore New Topics',
                'description': 'Discover different areas of Ray Peat\'s work',
                'action': 'Ask me: "What other topics did Ray Peat write about?"'
            })
        
        elif learning_style == 'deep_diver':
            if primary_topic:
                recommendations.append({
                    'type': 'deep_dive',
                    'title': f'Deep Dive into {primary_topic.title()}',
                    'description': f'Explore {primary_topic} in great detail',
                    'action': 'Ask me: "Tell me everything about ' + primary_topic + '"'
                })
        
        # General recommendations based on interaction count
        total_interactions = user_profile.get('total_interactions', 0)
        
        if total_interactions < 3:
            recommendations.append({
                'type': 'getting_started',
                'title': 'Getting Started with Ray Peat',
                'description': 'Learn about Ray Peat\'s core principles',
                'action': 'Ask me: "What are Ray Peat\'s main ideas?"'
            })
        
        return recommendations[:3]  # Limit to 3 recommendations

# Global instance for easy access
content_selector = ContentSelector()
