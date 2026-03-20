#!/usr/bin/env python3
"""
Profile Analyzer for PeatLearn Adaptive Learning System
Analyzes user interactions to create learner profiles and classify learning states
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import math

class TopicExtractor:
    """
    Extracts topics from user queries using keyword matching and pattern recognition
    """
    
    # Ray Peat topic taxonomy
    TOPIC_KEYWORDS = {
        'metabolism': [
            'thyroid', 't3', 't4', 'tsh', 'metabolic', 'energy', 'mitochondria',
            'cellular respiration', 'oxidative metabolism', 'metabolic rate',
            'metabolism', 'energy production', 'cellular energy'
        ],
        'hormones': [
            'progesterone', 'estrogen', 'cortisol', 'hormone', 'hormonal',
            'testosterone', 'adrenaline', 'insulin', 'growth hormone',
            'hormones', 'endocrine', 'steroid', 'pregnenolone'
        ],
        'nutrition': [
            'sugar', 'diet', 'food', 'eating', 'nutrition', 'nutritional',
            'carbohydrate', 'protein', 'fat', 'vitamin', 'mineral',
            'fructose', 'glucose', 'sucrose', 'milk', 'orange juice',
            'saturated fat', 'coconut oil', 'gelatin'
        ],
        'stress': [
            'stress', 'ptsd', 'anxiety', 'cortisol', 'adrenaline',
            'stress response', 'chronic stress', 'acute stress',
            'psychological stress', 'oxidative stress'
        ],
        'inflammation': [
            'inflammation', 'inflammatory', 'anti-inflammatory',
            'cytokines', 'prostaglandins', 'endotoxin', 'serotonin',
            'histamine', 'nitric oxide', 'free radicals'
        ],
        'reproduction': [
            'fertility', 'pregnancy', 'menstrual', 'reproductive',
            'ovulation', 'menopause', 'pms', 'libido', 'sexual'
        ],
        'aging': [
            'aging', 'longevity', 'lifespan', 'age', 'elderly',
            'anti-aging', 'life extension', 'cellular aging'
        ],
        'disease': [
            'cancer', 'diabetes', 'heart disease', 'arthritis',
            'alzheimer', 'disease', 'illness', 'pathology'
        ]
    }
    
    def extract_topics(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract topics from text with confidence scores
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (topic, confidence_score) tuples, sorted by confidence
        """
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                # Count occurrences of each keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                if count > 0:
                    # Weight by keyword specificity (longer keywords get higher weight)
                    keyword_weight = len(keyword) / 10
                    score += count * keyword_weight
            
            # Normalize score by total keywords in topic
            if total_keywords > 0:
                topic_scores[topic] = score / total_keywords
        
        # Sort by score and return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out topics with very low scores
        return [(topic, score) for topic, score in sorted_topics if score > 0.05]
    
    def get_primary_topic(self, text: str) -> Optional[str]:
        """
        Get the primary topic from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Primary topic name or None if no clear topic
        """
        
        topics = self.extract_topics(text)
        if topics and topics[0][1] > 0.15:  # Lower confidence threshold
            return topics[0][0]
        return None

class LearnerProfiler:
    """
    Analyzes user interactions to create and update learner profiles
    """
    
    def __init__(self, profiles_file: str = "data/user_interactions/user_profiles.json"):
        self.profiles_file = Path(profiles_file)
        self.topic_extractor = TopicExtractor()
        self.profiles_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize profiles file if it doesn't exist
        if not self.profiles_file.exists():
            self._initialize_profiles_file()
    
    def _initialize_profiles_file(self):
        """Initialize the profiles JSON file"""
        initial_data = {
            "profiles": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2)
    
    def load_profiles(self) -> Dict[str, Any]:
        """Load user profiles from JSON file"""
        try:
            with open(self.profiles_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_profiles_file()
            return self.load_profiles()
    
    def save_profiles(self, profiles_data: Dict[str, Any]):
        """Save user profiles to JSON file"""
        profiles_data['metadata']['last_updated'] = datetime.now().isoformat()
        
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, indent=2)
    
    def analyze_user_interactions(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze user interactions to create/update profile
        
        Args:
            interactions: List of user interaction dictionaries
            
        Returns:
            User profile dictionary
        """
        
        if not interactions:
            return self._create_default_profile()
        
        # Extract topics from all interactions
        topic_interactions = defaultdict(list)
        feedback_by_topic = defaultdict(list)
        
        for interaction in interactions:
            # Prefer provided topic (from embedding-based classifier), fallback to keyword extractor
            query = interaction.get('user_query', '')
            primary_topic = interaction.get('topic') or self.topic_extractor.get_primary_topic(query)
            
            if primary_topic:
                topic_interactions[primary_topic].append(interaction)
                
                # Collect feedback for this topic
                feedback = interaction.get('user_feedback')
                if feedback is not None:
                    feedback_by_topic[primary_topic].append(feedback)
        
        # Analyze learning patterns
        profile = {
            'user_id': interactions[0].get('user_id', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_interactions': len(interactions),
            'topic_mastery': {},
            'learning_style': self._determine_learning_style(topic_interactions),
            'session_patterns': self._analyze_session_patterns(interactions),
            'overall_state': 'learning'
        }
        
        # Analyze each topic
        for topic, topic_interactions_list in topic_interactions.items():
            topic_analysis = self._analyze_topic_mastery(topic, topic_interactions_list, feedback_by_topic[topic])
            profile['topic_mastery'][topic] = topic_analysis
        
        # Determine overall learning state
        profile['overall_state'] = self._determine_overall_state(profile['topic_mastery'])
        
        return profile
    
    def _create_default_profile(self) -> Dict[str, Any]:
        """Create a default profile for new users"""
        return {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_interactions': 0,
            'topic_mastery': {},
            'learning_style': 'explorer',
            'session_patterns': {
                'avg_session_length': 0,
                'total_sessions': 0,
                'preferred_interaction_type': 'chat'
            },
            'overall_state': 'new'
        }
    
    def _analyze_topic_mastery(self, topic: str, interactions: List[Dict], feedback: List[int]) -> Dict[str, Any]:
        """
        Analyze mastery level for a specific topic
        
        Args:
            topic: Topic name
            interactions: List of interactions for this topic
            feedback: List of feedback scores for this topic
            
        Returns:
            Topic mastery analysis
        """
        
        total_interactions = len(interactions)
        
        # Calculate normalized rating statistics (supports 1..10 or legacy +/-1)
        total_feedback = len(feedback)
        if total_feedback > 0:
            norm = []
            for f in feedback:
                try:
                    val = int(f)
                    if 1 <= val <= 10:
                        norm.append((val - 1) / 9.0)
                    elif val in (-1, 1):
                        norm.append(1.0 if val > 0 else 0.0)
                except Exception:
                    continue
            avg_norm = sum(norm) / len(norm) if norm else 0.5
        else:
            avg_norm = 0.5
        
        # Additional signals from context (embedding-based):
        # - average jargon_score (0-1)
        # - average similarity_confidence (0-1)
        avg_jargon = 0.0
        avg_sim = 0.0
        cnt_ctx = 0
        for it in interactions:
            ctx = it.get('context') or {}
            try:
                # context may be a JSON string in some pipelines
                if isinstance(ctx, str):
                    import json as _json
                    ctx = _json.loads(ctx)
            except Exception:
                ctx = {}
            if isinstance(ctx, dict):
                if isinstance(ctx.get('jargon_score'), (int, float)):
                    avg_jargon += float(ctx['jargon_score'])
                    cnt_ctx += 1
                if isinstance(ctx.get('similarity_confidence'), (int, float)):
                    avg_sim += float(ctx['similarity_confidence'])
        if cnt_ctx > 0:
            avg_jargon /= cnt_ctx
            avg_sim /= cnt_ctx

        # Determine mastery state based on interaction patterns
        if total_interactions >= 5 and avg_norm >= 0.8:
            state = 'advanced'
            mastery_level = 0.8 + (avg_norm - 0.8) * 0.5  # 0.8 to 0.9
        elif total_interactions >= 3 and avg_norm <= 0.4:
            state = 'struggling'
            mastery_level = 0.2 + avg_norm * 0.3  # 0.2 to 0.32
        else:
            state = 'learning'
            mastery_level = 0.4 + avg_norm * 0.3  # 0.4 to 0.7

        # Apply bounded boosts from embedding-based signals (holistic delta)
        mastery_level = mastery_level + 0.05 * avg_jargon + 0.05 * avg_sim
        mastery_level = max(0.0, min(1.0, mastery_level))
        
        return {
            'state': state,
            'mastery_level': mastery_level,
            'total_interactions': total_interactions,
            # Keep legacy key name for compatibility; now represents avg normalized rating
            'feedback_ratio': avg_norm,
            'last_interaction': interactions[-1]['timestamp'] if interactions else None,
            'confidence': min(total_interactions / 5.0, 1.0)  # Confidence increases with more data
        }
    
    def _determine_learning_style(self, topic_interactions: Dict[str, List]) -> str:
        """
        Determine learning style based on topic exploration patterns
        
        Args:
            topic_interactions: Dictionary of topic -> interactions
            
        Returns:
            Learning style: 'explorer' or 'deep_diver'
        """
        
        total_topics = len(topic_interactions)
        
        if total_topics == 0:
            return 'explorer'
        
        # Calculate topic distribution
        interactions_per_topic = [len(interactions) for interactions in topic_interactions.values()]
        avg_interactions_per_topic = sum(interactions_per_topic) / len(interactions_per_topic)
        max_interactions_per_topic = max(interactions_per_topic)
        
        # Deep diver: focuses heavily on few topics
        # Explorer: asks about many different topics
        
        if total_topics >= 4 and max_interactions_per_topic <= avg_interactions_per_topic * 1.5:
            return 'explorer'
        elif total_topics <= 2 and max_interactions_per_topic >= 5:
            return 'deep_diver'
        else:
            return 'balanced'
    
    def _analyze_session_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze session patterns and preferences
        
        Args:
            interactions: List of all user interactions
            
        Returns:
            Session patterns analysis
        """
        
        # Group by session
        sessions = defaultdict(list)
        for interaction in interactions:
            session_id = interaction.get('session_id', 'unknown')
            sessions[session_id].append(interaction)
        
        # Calculate session statistics
        session_lengths = [len(session) for session in sessions.values()]
        avg_session_length = sum(session_lengths) / len(session_lengths) if session_lengths else 0
        
        # Analyze interaction types
        interaction_types = [i.get('interaction_type', 'chat') for i in interactions]
        most_common_type = Counter(interaction_types).most_common(1)
        preferred_type = most_common_type[0][0] if most_common_type else 'chat'
        
        return {
            'avg_session_length': avg_session_length,
            'total_sessions': len(sessions),
            'preferred_interaction_type': preferred_type,
            'interaction_type_distribution': dict(Counter(interaction_types))
        }
    
    def _determine_overall_state(self, topic_mastery: Dict[str, Dict]) -> str:
        """
        Determine overall learning state based on topic masteries
        
        Args:
            topic_mastery: Dictionary of topic mastery analyses
            
        Returns:
            Overall state: 'new', 'struggling', 'learning', 'advanced'
        """
        
        if not topic_mastery:
            return 'new'
        
        states = [analysis['state'] for analysis in topic_mastery.values()]
        state_counts = Counter(states)
        
        # Determine dominant state
        if state_counts['struggling'] >= len(states) * 0.6:
            return 'struggling'
        elif state_counts['advanced'] >= len(states) * 0.6:
            return 'advanced'
        else:
            return 'learning'
    
    def update_user_profile(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update or create user profile based on interactions
        
        Args:
            user_id: User identifier
            interactions: List of user interactions
            
        Returns:
            Updated user profile
        """
        
        # Load existing profiles
        profiles_data = self.load_profiles()
        
        # Analyze interactions to create/update profile
        new_profile = self.analyze_user_interactions(interactions)
        new_profile['user_id'] = user_id
        
        # Update profiles data
        profiles_data['profiles'][user_id] = new_profile
        
        # Save updated profiles
        self.save_profiles(profiles_data)
        
        return new_profile
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get existing user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile or None if not found
        """
        
        profiles_data = self.load_profiles()
        return profiles_data['profiles'].get(user_id)
    
    def get_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on user profile
        
        Args:
            user_profile: User profile dictionary
            
        Returns:
            List of recommendation dictionaries
        """
        
        recommendations = []
        overall_state = user_profile.get('overall_state', 'learning')
        topic_mastery = user_profile.get('topic_mastery', {})
        learning_style = user_profile.get('learning_style', 'explorer')
        
        # Recommendations based on overall state
        if overall_state == 'struggling':
            recommendations.append({
                'type': 'simplification',
                'title': 'Try Simpler Explanations',
                'description': 'Click the "Simpler Explanation" button for easier-to-understand answers.',
                'priority': 'high'
            })
            
            # Recommend prerequisite topics
            struggling_topics = [topic for topic, analysis in topic_mastery.items() 
                               if analysis['state'] == 'struggling']
            
            for topic in struggling_topics:
                recommendations.append({
                    'type': 'prerequisite',
                    'title': f'Review {topic.title()} Basics',
                    'description': f'Let\'s start with fundamental concepts in {topic} before diving deeper.',
                    'priority': 'medium',
                    'topic': topic
                })
        
        elif overall_state == 'advanced':
            recommendations.append({
                'type': 'challenge',
                'title': 'Try Advanced Topics',
                'description': 'You\'re doing great! Ready for more complex bioenergetic concepts?',
                'priority': 'medium'
            })
            
            # Recommend related advanced topics
            mastered_topics = [topic for topic, analysis in topic_mastery.items() 
                             if analysis['state'] == 'advanced']
            
            for topic in mastered_topics:
                recommendations.append({
                    'type': 'related_advanced',
                    'title': f'Advanced {topic.title()} Concepts',
                    'description': f'Explore deeper aspects of {topic} and its connections to other systems.',
                    'priority': 'low',
                    'topic': topic
                })
        
        # Recommendations based on learning style
        if learning_style == 'explorer':
            recommendations.append({
                'type': 'exploration',
                'title': 'Explore Related Topics',
                'description': 'Discover how different bioenergetic concepts connect to each other.',
                'priority': 'medium'
            })
        
        elif learning_style == 'deep_diver':
            # Find their favorite topic
            if topic_mastery:
                favorite_topic = max(topic_mastery.items(), 
                                   key=lambda x: x[1]['total_interactions'])[0]
                
                recommendations.append({
                    'type': 'deep_dive',
                    'title': f'Master {favorite_topic.title()}',
                    'description': f'Continue exploring {favorite_topic} with specialized content and quizzes.',
                    'priority': 'medium',
                    'topic': favorite_topic
                })
        
        return recommendations

# Global instance for easy access
profiler = LearnerProfiler()
