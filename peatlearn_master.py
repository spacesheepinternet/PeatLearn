"""Backward-compatible launcher. Real entry point: app/dashboard.py"""
import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("app.dashboard", run_name="__main__", alter_sys=True)
