import unittest
import sys
from pathlib import Path

from peatlearn.personalization.engine import PersonalizationEngine, QuizPerformance

class TestPersonalizationEngine(unittest.TestCase):
    def test_engine_initialization(self):
        engine = PersonalizationEngine()
        self.assertIsNotNone(engine.quiz_recommender)

    def test_quiz_recommendation_placeholder(self):
        engine = PersonalizationEngine()
        user_id = "test_user"
        performance_history = [
            QuizPerformance(user_id="test_user", topic="hormones", score=0.7, time_taken=10.0, difficulty_level="easy")
        ]
        recommendation = engine.get_quiz_recommendation(user_id, performance_history)
        self.assertIn("topic", recommendation)
        self.assertIn("difficulty", recommendation)

if __name__ == '__main__':
    unittest.main()