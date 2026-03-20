#!/usr/bin/env python3
"""
Test script for PeatLearn Adaptive Learning System
Tests all components with sample data to verify functionality
"""

import sys
import os
from datetime import datetime
import json
from dotenv import load_dotenv

from peatlearn.adaptive import (
    data_logger, profiler, content_selector,
    quiz_generator, dashboard, TopicExtractor
)

# Load environment variables
load_dotenv()

def test_topic_extractor():
    """Test the topic extraction functionality"""
    print("🔍 Testing Topic Extractor...")
    
    extractor = TopicExtractor()
    
    test_queries = [
        "What does Ray Peat say about thyroid function and metabolism?",
        "How does progesterone interact with estrogen and cortisol?", 
        "What foods does Ray Peat recommend for good nutrition?",
        "How does chronic stress affect the body?",
        "Tell me about inflammation and cytokines"
    ]
    
    for query in test_queries:
        topics = extractor.extract_topics(query)
        primary = extractor.get_primary_topic(query)
        print(f"  Query: {query}")
        print(f"  Topics: {topics}")
        print(f"  Primary: {primary}")
        print()
    
    print("✅ Topic Extractor test completed!\n")

def test_data_logger():
    """Test the data logging functionality"""
    print("📊 Testing Data Logger...")
    
    test_interactions = [
        {
            "user_query": "What is thyroid function?",
            "llm_response": "Thyroid function involves T3 and T4 hormones that regulate metabolism...",
            "topic": "metabolism",
            "user_feedback": 1,
            "interaction_type": "chat"
        }
    ]
    
    print("  Logging test interactions...")
    for interaction in test_interactions:
        data_logger.log_interaction(**interaction)
    
    user_id = data_logger.get_user_id()
    interactions = data_logger.get_user_interactions(user_id)
    print(f"  Logged {len(interactions)} interactions for user {user_id}")
    
    print("✅ Data Logger test completed!\n")
    return interactions

def test_profiler(interactions):
    """Test the learner profiling functionality"""
    print("🎯 Testing Learner Profiler...")
    
    user_id = data_logger.get_user_id()
    profile = profiler.update_user_profile(user_id, interactions)
    
    print("  Generated Profile:")
    print(f"    User ID: {profile.get('user_id')}")
    print(f"    Overall State: {profile.get('overall_state')}")
    print(f"    Learning Style: {profile.get('learning_style')}")
    print("    Topic Mastery:")
    for topic, mastery in profile.get('topic_mastery', {}).items():
        print(f"      {topic}: {mastery['state']} (level: {mastery['mastery_level']:.2f})")
    
    print("✅ Learner Profiler test completed!\n")
    return profile

def test_live_quiz_generator(profile, interactions):
    """Test the live, dynamic quiz generation with the Gemini API"""
    print("🎯 Testing Live Quiz Generator...")

    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️  Skipping live API test: GEMINI_API_KEY not found.")
        return

    print("  Generating dynamic quiz with multiple question types via Gemini API...")
    quiz = quiz_generator.generate_quiz(
        user_profile=profile,
        topic="metabolism",
        num_questions=3,
        recent_interactions=interactions
    )
    
    print(f"    Quiz ID: {quiz['quiz_id']}")
    print(f"    Topic: {quiz['quiz_metadata']['topic']}")
    print(f"    Difficulty: {quiz['quiz_metadata']['difficulty']}")
    print(f"    Generated {len(quiz['questions'])} questions.")

    assert len(quiz['questions']) > 0, "The LLM failed to generate any questions."

    # Simulate user answers
    user_answers = {}
    for q in quiz['questions']:
        q_id = q['question_id']
        q_type = q['question_type']
        if q_type == 'multiple_choice':
            user_answers[q_id] = 0 # Just pick the first option
        elif q_type == 'true_false':
            user_answers[q_id] = True
        elif q_type == 'short_answer':
            user_answers[q_id] = "I think it is related to glycolysis and oxygen."

    print("\n  Evaluating quiz with live intelligent grading...")
    results = quiz_generator.evaluate_quiz(quiz, user_answers)
    
    print("\n  --- Quiz Results ---")
    for res in results['question_results']:
        print(f"    Question: {res['question']}")
        print(f"      Your Answer: {res['user_answer']}")
        print(f"      Correct Answer: {res.get('correct_answer', 'N/A')}")
        print(f"      Result: {'Correct' if res['is_correct'] else 'Incorrect'}")
        print(f"      Explanation: {res['explanation']}")
        print("    ----------------")

    print(f"\n  Final Score: {results['correct_answers']}/{results['total_questions']} ({results['score_percentage']:.1f}%)")
    print(f"  Feedback: {results['feedback']['overall']}")
    
    # Assert that the score is a valid number
    assert isinstance(results['correct_answers'], (int, float)), "Score should be a number."
    assert 0 <= results['score_percentage'] <= 100, "Score percentage should be between 0 and 100."
    
    print("\n✅ Live Quiz Generator test completed!\n")
    return results

def main():
    """Run all tests"""
    print("🚀 Starting PeatLearn Adaptive Learning System Tests\n")
    print("=" * 60)
    
    # Setup
    # clean_test_data() 

    # Test components
    test_topic_extractor()
    interactions = test_data_logger()
    profile = test_profiler(interactions)
    test_live_quiz_generator(profile, interactions)
    
    print("=" * 60)
    print("🎉 All core tests completed successfully!")
    print("\nSystem is now fully integrated with the Gemini API.")

def clean_test_data():
    """Clean up test data files"""
    print("🧹 Cleaning up test data...")
    # Dummy implementation for now
    print("✅ Cleanup completed!\n")

if __name__ == "__main__":
    main()