#!/usr/bin/env python3
"""
PeatLearn Adaptive Learning System

This package provides adaptive learning capabilities for the PeatLearn platform,
including user profiling, content adaptation, quiz generation, and progress tracking.

Main Components:
- DataLogger: Tracks user interactions and feedback
- LearnerProfiler: Analyzes user behavior and creates learning profiles
- ContentSelector: Modifies RAG responses based on user profiles
- QuizGenerator: Creates personalized quizzes
- Dashboard: Visualizes learning progress and insights
"""

__version__ = "1.0.0"
__author__ = "PeatLearn Development Team"

# Import main classes and instances for easy access
from .data_logger import data_logger
from .profile_analyzer import profiler, TopicExtractor
from .content_selector import content_selector
from .quiz_generator import quiz_generator
from .dashboard import dashboard

__all__ = [
    'data_logger',
    'profiler', 
    'TopicExtractor',
    'content_selector',
    'quiz_generator',
    'dashboard'
]
