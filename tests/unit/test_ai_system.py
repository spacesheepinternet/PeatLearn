#!/usr/bin/env python3
"""
Test script for AI-Enhanced PeatLearn Adaptive Learning System
Tests the intelligent profiling and mastery assessment
"""

import sys
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from peatlearn.adaptive.profile_analyzer import TopicExtractor
from peatlearn.adaptive.ai_profile_analyzer import AIEnhancedProfiler

def test_topic_extraction():
    """Test AI-enhanced topic extraction"""
    print("🔍 Testing AI-Enhanced Topic Extraction...")
    
    extractor = TopicExtractor()
    
    test_queries = [
        "What does Ray Peat say about thyroid T3 conversion and mitochondrial function?",
        "How does progesterone balance estrogen dominance in PCOS?", 
        "What's the relationship between sugar metabolism and cellular energy production?",
        "How does chronic stress affect cortisol and lead to inflammation?",
        "Can you explain the connection between calcium, vitamin D, and hormonal health?"
    ]
    
    for query in test_queries:
        topics = extractor.extract_topics(query)
        primary = extractor.get_primary_topic(query)
        print(f"  Query: {query}")
        print(f"  Primary Topic: {primary}")
        print(f"  All Topics: {topics[:3]}")  # Top 3
        print()
    
    print("✅ Topic Extraction test completed!\n")

def test_ai_mastery_assessment():
    """Test AI-powered mastery assessment"""
    print("🧠 Testing AI Mastery Assessment...")
    
    # Check if AI is available
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("  ⚠️  No GOOGLE_API_KEY found - testing fallback mode")
    else:
        print("  ✅ GOOGLE_API_KEY found - testing AI mode")
    
    profiler = AIEnhancedProfiler()
    
    # Test different complexity levels
    test_interactions = [
        {
            "user_query": "What is thyroid?",
            "llm_response": "The thyroid is a gland that produces hormones...",
            "user_feedback": -1,  # Struggling
            "topic": "metabolism"
        },
        {
            "user_query": "How does T3 conversion from T4 work in peripheral tissues, and what factors influence this process?",
            "llm_response": "T3 conversion from T4 occurs primarily in peripheral tissues through deiodinase enzymes...",
            "user_feedback": 1,  # Advanced understanding
            "topic": "metabolism"
        },
        {
            "user_query": "What's the relationship between progesterone, pregnenolone steal, and chronic stress?",
            "llm_response": "During chronic stress, the body prioritizes cortisol production...",
            "user_feedback": 1,  # Advanced question
            "topic": "hormones"
        }
    ]
    
    for i, interaction in enumerate(test_interactions):
        print(f"  Testing Interaction {i+1}:")
        print(f"    Query: {interaction['user_query']}")
        
        # Analyze with AI
        analysis = profiler.analyze_mastery_with_ai(
            interaction['user_query'],
            interaction['llm_response'],
            interaction['user_feedback'],
            interaction['topic']
        )
        
        print(f"    Complexity: {analysis['question_complexity']}/5")
        print(f"    Understanding: {analysis['conceptual_understanding']}/5")
        print(f"    Learning Stage: {analysis['learning_stage']}")
        print(f"    Mastery Estimate: {analysis['mastery_estimate']:.2f}")
        print(f"    Reasoning: {analysis['reasoning']}")
        print()
    
    print("✅ AI Mastery Assessment test completed!\n")

def test_learning_progression():
    """Test AI learning progression analysis"""
    print("📈 Testing Learning Progression Analysis...")
    
    profiler = AIEnhancedProfiler()
    
    # Simulate a learning journey
    learning_journey = [
        # Week 1: Basic questions
        {"user_query": "What is metabolism?", "topic": "metabolism", "user_feedback": 1, "timestamp": "2024-01-01T10:00:00Z"},
        {"user_query": "What does Ray Peat say about thyroid?", "topic": "metabolism", "user_feedback": 1, "timestamp": "2024-01-01T10:30:00Z"},
        
        # Week 2: Getting more specific
        {"user_query": "How does T3 work differently from T4?", "topic": "metabolism", "user_feedback": 1, "timestamp": "2024-01-08T10:00:00Z"},
        {"user_query": "What about progesterone?", "topic": "hormones", "user_feedback": -1, "timestamp": "2024-01-08T10:30:00Z"},
        
        # Week 3: Advanced questions
        {"user_query": "How does progesterone synthesis relate to cholesterol metabolism?", "topic": "hormones", "user_feedback": 1, "timestamp": "2024-01-15T10:00:00Z"},
        {"user_query": "What's the connection between oxidative metabolism and CO2 production?", "topic": "metabolism", "user_feedback": 1, "timestamp": "2024-01-15T10:30:00Z"},
        
        # Week 4: Expert level
        {"user_query": "How do cytochrome oxidase efficiency and cellular respiration relate to Ray Peat's bioenergetic view?", "topic": "metabolism", "user_feedback": 1, "timestamp": "2024-01-22T10:00:00Z"},
    ]
    
    # Add required fields for each interaction
    for interaction in learning_journey:
        interaction.update({
            "llm_response": f"Response about {interaction['topic']}...",
            "interaction_type": "chat",
            "user_id": "test_user",
            "session_id": "test_session"
        })
    
    print("  Analyzing learning progression...")
    profile = profiler.analyze_learning_progression(learning_journey)
    
    print(f"  Overall State: {profile.get('overall_state')}")
    print(f"  Learning Style: {profile.get('learning_style')}")
    print(f"  Total Interactions: {profile.get('total_interactions')}")
    
    print("  Topic Mastery:")
    for topic, mastery in profile.get('topic_mastery', {}).items():
        print(f"    {topic}: {mastery['state']} (level: {mastery['mastery_level']:.2f})")
    
    # Show AI insights if available
    ai_analysis = profile.get('ai_analysis', {})
    if ai_analysis:
        print("  AI Insights:")
        print(f"    Learning Velocity: {ai_analysis.get('learning_velocity')}")
        print(f"    Preferred Topics: {ai_analysis.get('preferred_topics')}")
        insights = ai_analysis.get('insights', [])
        for insight in insights[:2]:  # Top 2 insights
            print(f"    • {insight}")
    
    print("✅ Learning Progression test completed!\n")
    return profile

def test_ai_recommendations(profile):
    """Test AI-generated recommendations"""
    print("💡 Testing AI Recommendations...")
    
    profiler = AIEnhancedProfiler()
    
    # Generate recommendations
    recommendations = profiler.generate_personalized_recommendations(profile)
    
    print(f"  Generated {len(recommendations)} recommendations:")
    
    for i, rec in enumerate(recommendations):
        print(f"    {i+1}. {rec['title']} (Priority: {rec['priority']})")
        print(f"       {rec['description']}")
        
        # Show AI reasoning if available
        if 'reasoning' in rec:
            print(f"       Reasoning: {rec['reasoning']}")
        
        print(f"       Source: {rec.get('source', 'unknown')}")
        print()
    
    print("✅ AI Recommendations test completed!\n")

def test_full_ai_workflow():
    """Test complete AI-enhanced workflow"""
    print("🔄 Testing Full AI Workflow...")
    
    profiler = AIEnhancedProfiler()
    
    # Simulate real usage
    user_id = "ai_test_user"
    
    # Simulate interactions with different complexity levels
    interactions = []
    
    # Add some basic interactions
    basic_interactions = [
        ("What is Ray Peat's view on metabolism?", "metabolism", 1),
        ("Tell me about thyroid function", "metabolism", 1),
        ("What does he say about stress?", "stress", -1),  # Struggling
    ]
    
    for query, topic, feedback in basic_interactions:
        interaction = {
            "user_query": query,
            "llm_response": f"Ray Peat's response about {topic}...",
            "topic": topic,
            "user_feedback": feedback,
            "interaction_type": "chat",
            "user_id": user_id,
            "session_id": "workflow_test",
            "timestamp": datetime.now().isoformat()
        }
        interactions.append(interaction)
        
        # Analyze individual interaction
        enhanced = profiler.analyze_interaction_with_ai(interaction)
        print(f"  Interaction: '{query}'")
        print(f"    AI Complexity Score: {enhanced.get('complexity_score', 'N/A')}")
        print(f"    Understanding Level: {enhanced.get('understanding_level', 'N/A')}")
        print(f"    AI Mastery Estimate: {enhanced.get('ai_mastery_estimate', 'N/A')}")
        print()
    
    # Update profile with all interactions
    print("  Updating complete user profile...")
    final_profile = profiler.update_user_profile_with_ai(user_id, interactions)
    
    print(f"  Final Profile Summary:")
    print(f"    Overall State: {final_profile.get('overall_state')}")
    print(f"    Learning Style: {final_profile.get('learning_style')}")
    
    # Show AI recommendations
    recommendations = final_profile.get('recommendations', [])
    ai_recommendations = [r for r in recommendations if r.get('source') == 'ai_generated']
    print(f"    AI-Generated Recommendations: {len(ai_recommendations)}")
    
    for rec in ai_recommendations[:2]:  # Show top 2
        print(f"      • {rec['title']}: {rec['description']}")
    
    print("✅ Full AI Workflow test completed!\n")

def main():
    """Run all AI system tests"""
    print("🚀 Starting AI-Enhanced PeatLearn Tests\n")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"✅ Google API Key found: {api_key[:10]}...")
    else:
        print("⚠️  Google API Key not found - testing fallback mode")
    print()
    
    # Run tests
    test_topic_extraction()
    test_ai_mastery_assessment()
    profile = test_learning_progression()
    test_ai_recommendations(profile)
    test_full_ai_workflow()
    
    print("=" * 60)
    print("🎉 All AI system tests completed!")
    print("\nNext Steps:")
    print("1. Set GOOGLE_API_KEY environment variable for full AI features")
    print("2. Integrate with Streamlit dashboard")
    print("3. Test with real user interactions")
    print("4. Monitor AI analysis accuracy and adjust prompts")

if __name__ == "__main__":
    main()
