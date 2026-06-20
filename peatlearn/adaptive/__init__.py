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

# Lazy re-exports (PEP 562). Importing a submodule such as
# `peatlearn.adaptive.rag_system` should NOT drag in the whole adaptive stack
# (pandas, sklearn, plotly via data_logger/quiz_generator/dashboard). Those
# heavy modules load only when these names are actually accessed, which keeps
# the slim production API image (app/web_api.py) free of that dependency tree.
# `from peatlearn.adaptive import data_logger` still works exactly as before.
_LAZY_EXPORTS = {
    "data_logger": ".data_logger",
    "profiler": ".profile_analyzer",
    "TopicExtractor": ".profile_analyzer",
    "content_selector": ".content_selector",
    "quiz_generator": ".quiz_generator",
    "dashboard": ".dashboard",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    import importlib

    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
