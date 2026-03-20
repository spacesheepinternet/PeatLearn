#!/usr/bin/env python3
"""
Quiz Generator for PeatLearn Adaptive Learning System
Creates personalized quizzes based on user profiles and recent topics
"""

import json
import random
import uuid
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import google.generativeai as genai
import requests
from dotenv import load_dotenv

from .profile_analyzer import TopicExtractor

# Load environment variables from .env file
load_dotenv()

QUIZ_LLM_PROVIDER = "gemini"

# Configure the Gemini API (optional)
MODEL = None
try:
    if QUIZ_LLM_PROVIDER in ("gemini", "auto"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            GENERATION_CONFIG = {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 32,
                "max_output_tokens": 800,
                "response_mime_type": "application/json",
            }
            MODEL = genai.GenerativeModel(model_name="gemini-2.5-flash-lite", generation_config=GENERATION_CONFIG)
            print("Gemini API configured successfully.")
except Exception as e:
    print(f"Gemini configuration failed: {e}")

def call_llm_api(prompt: str) -> str:
    """
    Calls the Gemini LLM API and returns the JSON response as a string.
    """
    # Gemini only, with short backoff on 429
    if MODEL is None:
        print("Error: Gemini model not configured.")
        return "{}"
    attempts = 0
    last_error = None
    while attempts < 2:
        attempts += 1
        try:
            response = MODEL.generate_content(prompt)
            # Prefer safe extraction from candidates
            try:
                if getattr(response, "candidates", None):
                    for cand in response.candidates:
                        fr = getattr(cand, "finish_reason", None)
                        if fr is not None and str(fr).lower() not in ("stop", "finish_reason.stop"):
                            continue
                        parts = getattr(cand, "content", None)
                        if parts and getattr(parts, "parts", None):
                            texts = []
                            for p in parts.parts:
                                t = getattr(p, "text", None)
                                if t:
                                    texts.append(t)
                            if texts:
                                return "\n".join(texts)
            except Exception:
                pass
            try:
                return response.text
            except Exception:
                pass
            return "{}"
        except Exception as e:
            last_error = str(e)
            # Minimal respect of retry delay if present in message
            import re, time
            m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", last_error)
            delay = int(m.group(1)) if m else 5
            delay = min(delay, 8)
            if attempts < 2:
                time.sleep(delay)
            else:
                print(f"Error calling Gemini API: {e}")
                break
    return "{}"

class QuizGenerator:
    """
    Generates adaptive quizzes based on user learning profiles and topics
    """
    
    def __init__(self, profiler=None):
        self.topic_extractor = TopicExtractor()
        self.profiler = profiler
        self.available_topics = ['metabolism', 'hormones', 'nutrition', 'stress']
    
    def generate_quiz(self, 
                     user_profile: Dict[str, Any],
                     topic: Optional[str] = None,
                     num_questions: int = 5,
                     recent_interactions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a personalized quiz based on user profile
        """
        
        quiz_topic, difficulty_level = self._determine_quiz_parameters(
            user_profile, topic, recent_interactions
        )
        
        questions = self._generate_questions_with_llm(
            quiz_topic, difficulty_level, num_questions, recent_interactions
        )
        
        quiz_id = str(uuid.uuid4())
        quiz_metadata = {
            'quiz_id': quiz_id,
            'topic': quiz_topic,
            'difficulty': difficulty_level,
            'num_questions': len(questions),
            'created_at': datetime.now().isoformat(),
            'user_id': user_profile.get('user_id', 'unknown'),
            'adaptation_info': {
                'user_state': user_profile.get('overall_state', 'learning'),
                'topic_mastery': user_profile.get('topic_mastery', {}).get(quiz_topic, {}),
                'learning_style': user_profile.get('learning_style', 'explorer')
            }
        }
        
        return {
            'quiz_id': quiz_id,
            'quiz_metadata': quiz_metadata,
            'questions': questions,
            'instructions': self._get_quiz_instructions(difficulty_level, user_profile),
            'estimated_time_minutes': len(questions) * 1.5
        }
    
    def _determine_quiz_parameters(self, 
                                  user_profile: Dict[str, Any],
                                  requested_topic: Optional[str],
                                  recent_interactions: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Determine quiz topic and difficulty based on user profile
        """
        
        if requested_topic and requested_topic in self.available_topics:
            topic = requested_topic
        else:
            topic = self._choose_topic_from_profile(user_profile, recent_interactions)
        
        difficulty = self._determine_difficulty_level(user_profile, topic)
        
        return topic, difficulty
    
    def _choose_topic_from_profile(self, 
                                  user_profile: Dict[str, Any],
                                  recent_interactions: List[Dict[str, Any]]) -> str:
        """
        Choose quiz topic based on user profile and recent activity
        """
        
        topic_mastery = user_profile.get('topic_mastery', {})
        learning_style = user_profile.get('learning_style', 'explorer')
        
        recent_topics = []
        if recent_interactions:
            for interaction in recent_interactions[-10:]:
                query = interaction.get('user_query', '')
                primary_topic = self.topic_extractor.get_primary_topic(query)
                if primary_topic:
                    recent_topics.append(primary_topic)
        
        if learning_style == 'deep_diver' and recent_topics:
            most_recent_topic = recent_topics[-1] if recent_topics else None
            if most_recent_topic and most_recent_topic in self.available_topics:
                return most_recent_topic
        
        elif learning_style == 'explorer':
            for topic in self.available_topics:
                if topic not in topic_mastery or topic_mastery[topic]['state'] != 'advanced':
                    return topic
        
        if topic_mastery:
            mastery_for_available_topics = {t: topic_mastery.get(t, {'mastery_level': 0}) for t in self.available_topics}
            lowest_mastery_topic = min(
                mastery_for_available_topics.items(),
                key=lambda x: x[1].get('mastery_level', 0)
            )[0]
            return lowest_mastery_topic
        
        return random.choice(self.available_topics)
    
    def _determine_difficulty_level(self, user_profile: Dict[str, Any], topic: str) -> str:
        """
        Determine appropriate difficulty level for the quiz
        """
        
        overall_state = user_profile.get('overall_state', 'learning')
        topic_mastery = user_profile.get('topic_mastery', {})
        total_interactions = user_profile.get('total_interactions', 0)
        
        if topic in topic_mastery:
            topic_state = topic_mastery[topic]['state']
            mastery_level = topic_mastery[topic]['mastery_level']
            
            if topic_state == 'struggling' or mastery_level < 0.4:
                return 'beginner'
            elif topic_state == 'advanced' or mastery_level > 0.8:
                return 'advanced'
            else:
                return 'intermediate'
        
        if overall_state == 'struggling' or total_interactions < 5:
            return 'beginner'
        elif overall_state == 'advanced' and total_interactions > 15:
            return 'advanced'
        else:
            return 'intermediate'

    def _generate_questions_with_llm(self, topic: str, difficulty: str, num_questions: int, recent_interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates quiz questions by calling the Gemini LLM.
        """
        context_summary = "User has been asking about: " + ", ".join(
            [inter['user_query'] for inter in recent_interactions[-3:]]
        ) if recent_interactions else "No recent context."

        prompt = f"""
        You are an expert in bioenergetic medicine and Ray Peat's work. Generate a quiz with {num_questions} questions on the topic of '{topic}'.
        The target difficulty is '{difficulty}'.
        The user's recent interactions are: {context_summary}

        Generate a mix of question types: multiple_choice, short_answer, and true_false.

        Format the output as a single JSON object with a key "questions", which is a list of question objects.
        Each question object must have:
        - "question_type": "multiple_choice", "short_answer", or "true_false"
        - "question": The question text.
        - "explanation": A brief explanation of the correct answer.

        For "multiple_choice":
        - "options": A list of 4 strings.
        - "correct": The 0-based index of the correct option.

        For "short_answer":
        - "correct_answer": A string representing the ideal answer.

        For "true_false":
        - "correct": A boolean value (true or false).
        """
        
        llm_response_str = call_llm_api(prompt)
        
        try:
            response_json = json.loads(llm_response_str)
            questions = response_json.get('questions', [])
            
            for i, q in enumerate(questions):
                q['question_id'] = f"{topic}_{difficulty}_{i+1}_{random.randint(1000, 9999)}"
                q['topic'] = topic
                q['difficulty_level'] = difficulty

            return questions
        except json.JSONDecodeError:
            print(f"Error: Failed to decode LLM response: {llm_response_str}")
            return []

    def _get_quiz_instructions(self, difficulty: str, user_profile: Dict[str, Any]) -> str:
        """
        Get personalized quiz instructions
        """
        
        base_instructions = "Welcome to your new dynamically generated Ray Peat quiz! "
        
        if difficulty == 'beginner':
            base_instructions += "This quiz covers fundamental concepts. Take your time!"
        elif difficulty == 'intermediate':
            base_instructions += "This quiz explores Ray Peat's ideas in more depth."
        elif difficulty == 'advanced':
            base_instructions += "This is an advanced quiz covering complex concepts. Good luck!"
        
        learning_style = user_profile.get('learning_style', 'explorer')
        if learning_style == 'deep_diver':
            base_instructions += " Since you like to dive deep, these questions will help reinforce your understanding."
        elif learning_style == 'explorer':
            base_instructions += " These questions will help you connect different concepts."
        
        base_instructions += "\n\nRemember: This quiz is a learning tool. Each question includes an explanation!"
        
        return base_instructions

    def _evaluate_short_answer_with_llm(self, user_answer: str, correct_answer: str) -> Dict[str, Any]:
        """
        Evaluates a short answer using the Gemini LLM.
        """
        prompt = f"""
        You are a teaching assistant. Evaluate the user's answer to a quiz question.

        Question's Correct Answer: "{correct_answer}"
        User's Answer: "{user_answer}"

        Evaluate the user's answer based on the correct answer. Determine if it is 'correct', 'partially_correct', or 'incorrect'.

        Provide brief feedback for the user, explaining why their answer is correct or incorrect.

        Format the output as a single JSON object with keys:
        - "correctness": A string ('correct', 'partially_correct', or 'incorrect').
        - "feedback": A string of personalized feedback.
        - "score": A float from 0.0 to 1.0 (0 for incorrect, 0.5 for partial, 1.0 for correct).
        """
        llm_response_str = call_llm_api(prompt)
        try:
            response_json = json.loads(llm_response_str)
            return response_json.get('evaluation', {})
        except json.JSONDecodeError:
            print(f"Error: Failed to decode LLM evaluation response: {llm_response_str}")
            return {"correctness": "error", "feedback": "Could not evaluate answer.", "score": 0}

    def evaluate_quiz(self, quiz_data: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate quiz results and provide feedback
        """
        
        questions = quiz_data['questions']
        quiz_metadata = quiz_data['quiz_metadata']
        
        total_questions = len(questions)
        correct_answers_score = 0
        question_results = []
        
        for question in questions:
            question_id = question['question_id']
            user_answer = user_answers.get(question_id)
            q_type = question.get('question_type')
            
            is_correct = False
            feedback_text = question.get('explanation', 'No explanation provided.')

            if q_type == 'multiple_choice':
                correct_index = question['correct']
                is_correct = user_answer == correct_index
                if is_correct:
                    correct_answers_score += 1
                
                question_results.append({
                    'question_id': question_id,
                    'question': question['question'],
                    'user_answer': question['options'][user_answer] if isinstance(user_answer, int) and 0 <= user_answer < len(question['options']) else 'No answer',
                    'correct_answer': question['options'][correct_index],
                    'is_correct': is_correct,
                    'explanation': feedback_text
                })

            elif q_type == 'true_false':
                correct_bool = question['correct']
                is_correct = user_answer == correct_bool
                if is_correct:
                    correct_answers_score += 1

                question_results.append({
                    'question_id': question_id,
                    'question': question['question'],
                    'user_answer': 'True' if user_answer else 'False',
                    'correct_answer': 'True' if correct_bool else 'False',
                    'is_correct': is_correct,
                    'explanation': feedback_text
                })

            elif q_type == 'short_answer':
                evaluation = self._evaluate_short_answer_with_llm(user_answer, question['correct_answer'])
                correctness = evaluation.get('correctness')
                score = evaluation.get('score', 0)
                feedback_text = evaluation.get('feedback', feedback_text)

                is_correct = correctness == 'correct'
                correct_answers_score += score

                question_results.append({
                    'question_id': question_id,
                    'question': question['question'],
                    'user_answer': user_answer,
                    'correct_answer': question['correct_answer'],
                    'is_correct': is_correct,
                    'explanation': feedback_text
                })

        score_percentage = (correct_answers_score / total_questions) * 100 if total_questions > 0 else 0
        
        feedback = self._generate_quiz_feedback(score_percentage, quiz_metadata, question_results)
        
        return {
            'quiz_id': quiz_data['quiz_id'],
            'total_questions': total_questions,
            'correct_answers': correct_answers_score, # This is now a score, not a count
            'score_percentage': score_percentage,
            'feedback': feedback,
            'question_results': question_results,
            'completed_at': datetime.now().isoformat()
        }
    
    def _generate_quiz_feedback(self, 
                               score_percentage: float,
                               quiz_metadata: Dict[str, Any],
                               question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate personalized feedback based on quiz performance
        """
        
        feedback = {}
        
        if score_percentage >= 90:
            feedback['overall'] = "Excellent work! You have a strong understanding of these concepts. 🌟"
            feedback['level'] = 'excellent'
        elif score_percentage >= 70:
            feedback['overall'] = "Good job! You're building a solid foundation. 👍"
            feedback['level'] = 'good'
        elif score_percentage >= 50:
            feedback['overall'] = "You're making progress! Keep reviewing the explanations. 📚"
            feedback['level'] = 'developing'
        else:
            feedback['overall'] = "This is challenging material. Don't be discouraged! The goal is to learn. 💪"
            feedback['level'] = 'needs_work'
        
        difficulty = quiz_metadata.get('difficulty', 'intermediate')
        topic = quiz_metadata.get('topic', 'general')
        
        recommendations = []
        
        if feedback['level'] in ['needs_work', 'developing']:
            recommendations.append(f"Focus on the fundamentals of {topic}. Ask the chat for simpler explanations.")
        
        elif feedback['level'] == 'good':
            recommendations.append(f"You are ready to tackle more advanced concepts in {topic}.")
        
        elif feedback['level'] == 'excellent':
            if difficulty != 'advanced':
                recommendations.append("Consider trying a quiz at a higher difficulty level.")
            recommendations.append(f"Explore how {topic} connects to other areas of health.")
        
        feedback['recommendations'] = recommendations
        
        incorrect_questions = [q for q in question_results if not q['is_correct']]
        if incorrect_questions:
            feedback['areas_for_improvement'] = [
                f"Review: {q['question']}" for q in incorrect_questions[:3]
            ]
        
        return feedback

# Global instance for easy access
quiz_generator = QuizGenerator()
