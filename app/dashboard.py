#!/usr/bin/env python3
"""
PeatLearn Master Dashboard - AI-Enhanced Adaptive Learning System
Full integration of all adaptive learning features with live AI profiling
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path regardless of how Streamlit is launched
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import subprocess
import signal
import time
import requests
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
import html
import re
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import argparse

# Load environment variables
load_dotenv()

# Development mode detection
def is_development_mode():
    """Check if development mode is enabled via environment variable or command line"""
    # Check environment variable
    if os.getenv('PEATLEARN_DEV_MODE', '').lower() in ['true', '1', 'yes', 'on']:
        return True
    
    # Check command line arguments (for when launched directly)
    if '--dev' in sys.argv or '--development' in sys.argv:
        return True
        
    # Check if running under Streamlit with dev flag
    if os.getenv('STREAMLIT_DEV_MODE', '').lower() in ['true', '1']:
        return True
        
    return False

DEVELOPMENT_MODE = is_development_mode()

# Auto-refresh functionality
class AutoRefreshHandler(FileSystemEventHandler):
    """Handle file changes and trigger Streamlit refresh"""
    
    def __init__(self, watched_files=None):
        self.watched_files = watched_files or []
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Only refresh for specific file types or watched files
        if (file_path.suffix in ['.py', '.json', '.yaml', '.yml', '.env'] or 
            str(file_path) in self.watched_files):
            
            # Debounce: only refresh if file hasn't been modified in last 2 seconds
            current_time = time.time()
            if (file_path not in self.last_modified or 
                current_time - self.last_modified[file_path] > 2):
                
                self.last_modified[file_path] = current_time
                st.rerun()

def setup_auto_refresh(watch_dirs=None, watch_files=None):
    """Setup file watching for auto-refresh (development mode only)"""
    if not DEVELOPMENT_MODE:
        st.error("🔒 Auto-refresh is disabled in production mode")
        return
        
    if 'auto_refresh_setup' in st.session_state:
        return
        
    watch_dirs = watch_dirs or ['.', 'src', 'inference', 'data']
    watch_files = watch_files or ['peatlearn_master.py', '.env']
    
    try:
        handler = AutoRefreshHandler(watch_files)
        observer = Observer()
        
        for watch_dir in watch_dirs:
            if Path(watch_dir).exists():
                observer.schedule(handler, watch_dir, recursive=True)
        
        observer.start()
        st.session_state.auto_refresh_setup = True
        st.session_state.file_observer = observer
        
    except Exception as e:
        st.warning(f"Auto-refresh setup failed: {e}")

# Periodic refresh options
def setup_periodic_refresh(interval_seconds=30):
    """Setup periodic refresh for data updates"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh > interval_seconds:
        st.session_state.last_refresh = current_time
        st.rerun()

# Lightweight cached helpers to reduce backend chatter
@st.cache_data(ttl=60)
def _fetch_recommendations_cached(user_id: str, topic_filter: list | None, num_recommendations: int = 8):
    payload = {
        "user_id": user_id,
        "num_recommendations": num_recommendations,
        "exclude_seen": True,
        "topic_filter": topic_filter,
    }
    r = requests.post("http://localhost:8001/api/recommendations", json=payload, timeout=6)
    r.raise_for_status()
    return r.json().get("recommendations", [])

@st.cache_data(ttl=30)
def _adv_health_cached() -> bool:
    try:
        resp = requests.get("http://localhost:8001/api/health", timeout=2)
        return resp.status_code == 200 and resp.json().get('status') == 'healthy'
    except Exception:
        return False

# --- Orchestrator: run backend servers + Streamlit together when invoked via `python peatlearn_master.py` ---
def _wait_for_health(url: str, timeout_seconds: int = 90) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def _launch_all():
    print("🚀 Launching PeatLearn: backends + Streamlit...")
    env = os.environ.copy()
    # Mark child Streamlit process to avoid re-launch recursion
    env["RUNNING_UNDER_STREAMLIT"] = "1"
    
    # Pass through development mode flags
    if DEVELOPMENT_MODE:
        env["STREAMLIT_DEV_MODE"] = "true"
        print("🔧 Development mode enabled")

    procs = []
    try:
        # Start API backends
        api_cmd = [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
        adv_cmd = [sys.executable, "-m", "uvicorn", "app.advanced_api:app", "--host", "0.0.0.0", "--port", "8001"]
        procs.append(subprocess.Popen(api_cmd, env=env))
        procs.append(subprocess.Popen(adv_cmd, env=env))

        # Wait for health
        ok_api = _wait_for_health("http://localhost:8000/api/health", 90)
        print(f"{'✅' if ok_api else '⚠️'} API 8000 health: {'OK' if ok_api else 'not ready'}")
        ok_adv = _wait_for_health("http://localhost:8001/api/health", 90)
        print(f"{'✅' if ok_adv else '⚠️'} Advanced API 8001 health: {'OK' if ok_adv else 'not ready'}")

        # Launch Streamlit for this same script
        st_cmd = ["streamlit", "run", os.path.abspath(__file__)]
        streamlit_proc = subprocess.Popen(st_cmd, env=env)

        # Wait until streamlit exits
        exit_code = streamlit_proc.wait()
        return exit_code
    finally:
        # Cleanup child processes
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
            except Exception:
                pass

# If run directly via `python app/dashboard.py`, act as a launcher.
# When Streamlit exec's this file, get_script_run_ctx() returns a valid context, so we skip.
_in_streamlit = False
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _gctx
    _in_streamlit = _gctx() is not None
except Exception:
    pass

if __name__ == "__main__" and not _in_streamlit and os.environ.get("RUNNING_UNDER_STREAMLIT") != "1":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PeatLearn Master Dashboard')
    parser.add_argument('--dev', '--development', action='store_true',
                       help='Enable development mode with auto-refresh features')
    parser.add_argument('--port', type=int, default=8501,
                       help='Streamlit port (default: 8501)')
    args, unknown = parser.parse_known_args()
    if args.dev:
        os.environ['PEATLEARN_DEV_MODE'] = 'true'
    sys.exit(_launch_all())

# Import our AI-enhanced adaptive learning system
from peatlearn.adaptive.data_logger import DataLogger
from peatlearn.adaptive.ai_profile_analyzer import AIEnhancedProfiler
from peatlearn.adaptive.content_selector import ContentSelector
from peatlearn.adaptive.quiz_generator import QuizGenerator
from peatlearn.adaptive.dashboard import Dashboard
from peatlearn.adaptive.rag_system import RayPeatRAG
from peatlearn.adaptive.topic_model import CorpusTopicModel

# Page configuration
st.set_page_config(
    page_title="PeatLearn - AI-Enhanced Adaptive Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');

    /* === Frutiger Aero — Color system === */
    :root {
        --primary:       #0EA5E9;
        --primary-light: #7DD3FC;
        --primary-dark:  #0369A1;
        --accent:        #34D399;
        --accent-warm:   #F4A261;
        --green:         #22C55E;
        --green-light:   #86EFAC;
        --success:       #2EC4B6;
        --danger:        #EF6461;
        --warning:       #FCA311;
        --text:          #E8F4FD;
        --surface:       rgba(186, 230, 253, 0.10);
        --surface-hover: rgba(186, 230, 253, 0.16);
        --surface-glass: rgba(255, 255, 255, 0.14);
        --border:        rgba(186, 230, 253, 0.22);
        --border-top:    rgba(255, 255, 255, 0.45);
        --text-muted:    rgba(224, 242, 254, 0.65);
        --glow-primary:  rgba(14, 165, 233, 0.35);
        --glow-teal:     rgba(46, 196, 182, 0.30);
        --glow-green:    rgba(34, 197, 94, 0.28);
    }

    /* === Global text visibility fix === */
    body, .stApp, p, span, li, td, th, label,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    [data-testid="stText"], .stChatMessage,
    [data-baseweb="select"] span,
    .stSlider label, .stSelectbox label, .stTextInput label,
    .stRadio label, .stCheckbox label, .element-container {
        color: var(--text) !important;
    }

    h1, h2, h3, h4, h5, h6 { color: #F0F9FF !important; }

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
    }

    /* === Frutiger Aero — Bokeh page background === */
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 70% 55% at 8%  12%, rgba(56,  189, 248, 0.26) 0%, transparent 70%),
            radial-gradient(ellipse 55% 50% at 82% 30%, rgba(34,  211, 238, 0.20) 0%, transparent 65%),
            radial-gradient(ellipse 50% 45% at 70% 75%, rgba(34,  197,  94, 0.22) 0%, transparent 62%),
            radial-gradient(ellipse 40% 40% at 25% 60%, rgba(134, 239, 172, 0.16) 0%, transparent 58%),
            radial-gradient(ellipse 40% 35% at 88% 82%, rgba(99,  220, 249, 0.16) 0%, transparent 60%),
            radial-gradient(ellipse 90% 80% at 50% 50%, rgba(2,    78, 121, 0.50) 0%, transparent 100%),
            linear-gradient(160deg, #021F3D 0%, #032B50 35%, #031A2A 65%, #011810 100%);
        background-attachment: fixed;
    }

    [data-testid="stAppViewContainer"] > .main > .block-container {
        background: transparent !important;
    }

    /* === Frutiger Aero — Glass card base (shared) === */
    .stat-card,
    .metric-card,
    .insight-card,
    .sq-item,
    .empty-state-card,
    .profile-card,
    .welcome-features {
        background: var(--surface-glass) !important;
        border: 1px solid var(--border) !important;
        border-top-color: var(--border-top) !important;
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        box-shadow: 0 4px 24px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.22) !important;
    }

    /* === Main header — Aero sky gloss === */
    .main-header {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.06) 52%, transparent 52%),
            linear-gradient(135deg, #0C4A6E 0%, #0EA5E9 45%, #38BDF8 80%, #7DD3FC 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow:
            0 8px 32px rgba(14, 165, 233, 0.45),
            0 0  60px rgba(14, 165, 233, 0.20),
            inset 0 1px 0 rgba(255,255,255,0.30);
        border: 1px solid rgba(125, 211, 252, 0.35);
        border-top-color: rgba(255,255,255,0.45);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(4px);
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -40%;
        left: 10%;
        width: 80%;
        height: 90%;
        background: radial-gradient(ellipse, rgba(255,255,255,0.14) 0%, transparent 70%);
        pointer-events: none;
    }

    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 0.5rem;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.82;
        margin: 0;
        font-weight: 300;
    }

    /* === Metric card === */
    .metric-card {
        background: var(--surface);
        padding: 1.1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        border-color: rgba(255,255,255,0.18);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    /* === Chat messages === */
    .chat-message {
        padding: 1rem 1.3rem;
        margin: 0.8rem 0;
        border-radius: 16px;
        animation: fadeInUp 0.22s ease forwards;
        position: relative;
        line-height: 1.65;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0.05) 50%, transparent 50%),
            linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%);
        color: white;
        margin-left: 8%;
        border-bottom-right-radius: 4px;
        border: 1px solid rgba(125, 211, 252, 0.30);
        border-top-color: rgba(255,255,255,0.38);
        box-shadow: 0 3px 14px rgba(14, 165, 233, 0.35), inset 0 1px 0 rgba(255,255,255,0.20);
    }

    .assistant-message {
        background: rgba(2, 40, 80, 0.55);
        border: 1px solid var(--border);
        border-top-color: rgba(255,255,255,0.18);
        margin-right: 4%;
        border-bottom-left-radius: 4px;
        backdrop-filter: blur(20px) saturate(140%);
        -webkit-backdrop-filter: blur(20px) saturate(140%);
        box-shadow: 0 4px 20px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.10);
    }

    .message-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.45rem;
    }

    .avatar {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    .avatar-user {
        background: rgba(255,255,255,0.22);
        color: white;
    }

    .avatar-ai {
        background: linear-gradient(135deg, #2EC4B6, #06B6D4);
        color: white;
        box-shadow: 0 0 8px rgba(6, 182, 212, 0.55);
    }

    .msg-name {
        font-weight: 600;
        font-size: 0.82rem;
        opacity: 0.9;
    }

    .msg-time {
        font-size: 0.7rem;
        opacity: 0.45;
        margin-left: auto;
    }

    /* === RAG answer === */
    .rag-answer h3 {
        font-size: 1.05em;
        font-weight: 600;
        margin: 0.9rem 0 0.4rem;
    }

    .rag-answer p {
        margin: 0.3rem 0;
        line-height: 1.7;
    }

    .rag-answer ul {
        padding-left: 1.4rem;
        margin: 0.4rem 0;
    }

    .rag-answer li { margin-bottom: 0.35rem; }

    /* === Sources === */
    .sources-container {
        margin-top: 1.2rem;
        border-top: 1px solid var(--border);
        padding-top: 0.8rem;
    }

    .sources-toggle {
        cursor: pointer;
        color: var(--primary-light);
        font-weight: 500;
        font-size: 0.83rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.28rem 0.6rem;
        border-radius: 6px;
        transition: background 0.2s;
    }

    .sources-toggle:hover { background: var(--surface-hover); }

    .sources-content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.35s ease-out;
        margin-top: 0.5rem;
    }

    .sources-content ul { padding-left: 1.2rem; margin: 0.5rem 0; }

    .sources-content li {
        margin-bottom: 0.4rem;
        font-size: 0.84em;
        opacity: 0.65;
    }

    .sources-container:hover .sources-content {
        max-height: 500px;
        transition: max-height 0.5s ease-in;
    }

    /* === Recommendation card — amber glass === */
    .recommendation-card {
        background: linear-gradient(135deg, rgba(251,191,36,0.10) 0%, rgba(251,191,36,0.04) 100%);
        padding: 1.1rem 1.2rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        border: 1px solid rgba(251,191,36,0.20);
        border-top-color: rgba(255,255,255,0.18);
        border-left: 4px solid var(--accent-warm);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
        transition: transform 0.15s, box-shadow 0.15s;
    }

    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(251,191,36,0.18), inset 0 1px 0 rgba(255,255,255,0.12);
    }

    .recommendation-card h4 {
        color: var(--accent-warm);
        margin: 0 0 0.4rem;
        font-size: 0.94rem;
        font-weight: 600;
    }

    /* === Profile card — sky-teal glass === */
    .profile-card {
        background: linear-gradient(135deg, rgba(46,196,182,0.09) 0%, rgba(14,165,233,0.09) 100%);
        padding: 1.1rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        border: 1px solid rgba(14,165,233,0.18);
    }

    /* === Mastery badges — glass shimmer === */
    .mastery-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.75rem;
        border-radius: 20px;
        font-size: 0.77rem;
        font-weight: 600;
        margin: 0.2rem 0.25rem 0.2rem 0;
        letter-spacing: 0.01em;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
    }

    .struggling { background: rgba(239,100,97,0.16);  color: #FCA5A5; border: 1px solid rgba(239,100,97,0.30); border-top-color: rgba(255,255,255,0.22); }
    .learning   { background: rgba(252,163,17,0.16);  color: #FCD34D; border: 1px solid rgba(252,163,17,0.30); border-top-color: rgba(255,255,255,0.22); }
    .advanced   { background: rgba(46,196,182,0.16);  color: #5EEAD4; border: 1px solid rgba(46,196,182,0.30); border-top-color: rgba(255,255,255,0.22); }

    /* === Welcome card — Aero glass === */
    .welcome-card {
        max-width: 540px;
        margin: 2.5rem auto 1rem;
        text-align: center;
        padding: 3rem 2.5rem 2.5rem;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0.04) 50%, transparent 50%),
            linear-gradient(145deg, rgba(14,165,233,0.18) 0%, rgba(6,182,212,0.10) 50%, rgba(46,196,182,0.10) 100%);
        border: 1px solid rgba(125,211,252,0.28);
        border-top-color: rgba(255,255,255,0.42);
        border-radius: 24px;
        backdrop-filter: blur(24px) saturate(150%);
        -webkit-backdrop-filter: blur(24px) saturate(150%);
        box-shadow:
            0 20px 60px rgba(0,0,0,0.35),
            0 0   40px rgba(14,165,233,0.15),
            inset 0 1px 0 rgba(255,255,255,0.28);
    }

    .welcome-card .wc-icon {
        font-size: 3.2rem;
        line-height: 1;
        margin-bottom: 1rem;
        display: block;
    }

    .welcome-card h2 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.5rem;
        background: linear-gradient(135deg, #7DD3FC, #2EC4B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .welcome-card .wc-tagline {
        opacity: 0.62;
        margin: 0 0 1.8rem;
        font-size: 0.97rem;
        line-height: 1.6;
    }

    .welcome-features {
        text-align: left;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        line-height: 2;
        opacity: 0.85;
    }

    .wc-cta {
        font-size: 0.88rem;
        opacity: 0.5;
        font-style: italic;
    }

    /* === Sidebar — Aero frosted panel === */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(ellipse 80% 60% at 50% 0%, rgba(14,165,233,0.14) 0%, transparent 70%),
            rgba(2, 25, 55, 0.75);
        border-right: 1px solid rgba(125, 211, 252, 0.18);
        backdrop-filter: blur(24px) saturate(140%);
        -webkit-backdrop-filter: blur(24px) saturate(140%);
        box-shadow: 2px 0 20px rgba(0,0,0,0.28);
    }

    /* === Gel buttons === */
    .stButton > button {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 600;
        border-radius: 20px;
        border: 1px solid rgba(125,211,252,0.40);
        border-top-color: rgba(255,255,255,0.52);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.26) 0%, rgba(255,255,255,0.05) 50%, transparent 50%),
            linear-gradient(180deg, #0284C7 0%, #0369A1 100%);
        color: #E0F2FE !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.40);
        box-shadow:
            0 4px 14px rgba(14,165,233,0.40),
            inset 0 1px 0 rgba(255,255,255,0.30),
            inset 0 -1px 0 rgba(0,0,0,0.20);
        transition: all 0.18s ease;
        padding: 0.45rem 1.3rem;
    }

    .stButton > button:hover {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.32) 0%, rgba(255,255,255,0.08) 50%, transparent 50%),
            linear-gradient(180deg, #0EA5E9 0%, #0284C7 100%);
        box-shadow:
            0 6px 20px rgba(14,165,233,0.55),
            inset 0 1px 0 rgba(255,255,255,0.38);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(14,165,233,0.30);
    }

    /* === Sidebar user card === */
    .sidebar-user-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem 0.9rem;
        margin: 0.4rem 0 0.9rem;
    }

    .sidebar-user-card .user-name {
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .session-chip {
        display: inline-block;
        background: rgba(14,165,233,0.16);
        border: 1px solid rgba(14,165,233,0.28);
        border-top-color: rgba(255,255,255,0.28);
        color: var(--primary-light);
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.7rem;
        font-family: 'SF Mono', 'Fira Code', monospace;
        margin-bottom: 0.75rem;
        backdrop-filter: blur(8px);
    }

    .status-row { display: flex; gap: 0.4rem; flex-wrap: wrap; }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.2rem 0.55rem;
        border-radius: 20px;
        font-size: 0.71rem;
        font-weight: 500;
    }

    .status-pill.ok   { background: rgba(46,196,182,0.18); color: #5EEAD4; border: 1px solid rgba(46,196,182,0.32); border-top-color: rgba(255,255,255,0.25); }
    .status-pill.warn { background: rgba(252,163,17,0.18); color: #FCD34D; border: 1px solid rgba(252,163,17,0.32); border-top-color: rgba(255,255,255,0.20); }

    /* === Section header === */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.3rem;
    }

    .section-header h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }

    /* === Quiz question card — sky glass === */
    .quiz-question-card {
        background: linear-gradient(135deg, rgba(14,165,233,0.12) 0%, rgba(14,165,233,0.04) 100%);
        border: 1px solid rgba(14,165,233,0.22);
        border-top-color: rgba(255,255,255,0.22);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        font-size: 1.02rem;
        font-weight: 500;
        line-height: 1.65;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
    }

    /* === Passage excerpt === */
    .passage-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 0.4rem;
        font-size: 0.88rem;
        line-height: 1.75;
        font-style: italic;
        opacity: 0.82;
    }

    .passage-source {
        font-size: 0.73rem;
        opacity: 0.42;
        margin-top: 0.4rem;
        font-style: normal;
    }

    /* === Stat cards (profile + analytics) === */
    .stat-cards-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .stat-cards-row.three-col { grid-template-columns: repeat(3, 1fr); }

    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }

    .stat-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.22); }

    .stat-card .sc-icon { font-size: 1.4rem; display: block; margin-bottom: 0.4rem; line-height: 1; }
    .stat-card .sc-value { font-size: 1.45rem; font-weight: 700; color: var(--primary-light); line-height: 1.1; margin-bottom: 0.25rem; }
    .stat-card .sc-label { font-size: 0.7rem; opacity: 0.48; text-transform: uppercase; letter-spacing: 0.06em; }

    /* === Chat empty state === */
    .chat-empty-state {
        text-align: center;
        padding: 2.5rem 1.5rem 1.5rem;
    }

    .chat-empty-state .ces-icon { font-size: 2.8rem; display: block; margin-bottom: 0.8rem; }
    .chat-empty-state h4 { font-size: 1rem; font-weight: 600; margin: 0 0 0.4rem; opacity: 0.85; }
    .chat-empty-state p { font-size: 0.85rem; opacity: 0.45; margin: 0 0 1.4rem; }

    .suggested-questions {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        max-width: 500px;
        margin: 0 auto;
        text-align: left;
    }

    .sq-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.65rem 0.95rem;
        font-size: 0.84rem;
        opacity: 0.78;
        transition: border-color 0.18s, opacity 0.18s;
        cursor: default;
    }

    .sq-item:hover { border-color: rgba(14,165,233,0.45); opacity: 1; }

    /* === Insight cards === */
    .insight-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        line-height: 1.65;
    }

    .insight-card .ic-icon { flex-shrink: 0; font-size: 0.95rem; opacity: 0.65; margin-top: 0.08rem; }

    /* === Badges container === */
    .badges-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.75rem; margin-bottom: 0.5rem; }

    /* === Empty state card === */
    .empty-state-card {
        text-align: center;
        padding: 2.2rem 1.5rem;
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 14px;
        opacity: 0.68;
        margin: 0.8rem 0;
    }

    .empty-state-card .esc-icon { font-size: 2rem; display: block; margin-bottom: 0.6rem; }
    .empty-state-card p { font-size: 0.88rem; opacity: 0.62; margin: 0; }

    /* === Sidebar footer === */
    .sidebar-footer {
        margin-top: 2rem;
        padding-top: 0.75rem;
        border-top: 1px solid var(--border);
        text-align: center;
        font-size: 0.69rem;
        opacity: 0.32;
        line-height: 1.7;
    }

    /* === Quiz progress label === */
    .quiz-progress-label {
        font-size: 0.76rem;
        opacity: 0.48;
        text-align: right;
        margin-bottom: 0.6rem;
        letter-spacing: 0.02em;
    }

    /* === Quiz result card === */
    .quiz-result-card {
        background: linear-gradient(135deg, rgba(46,196,182,0.1) 0%, rgba(46,196,182,0.04) 100%);
        border: 1px solid rgba(46,196,182,0.22);
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        text-align: center;
        margin: 0.8rem 0 1.2rem;
    }

    .quiz-result-card .qr-score {
        font-size: 2rem;
        font-weight: 700;
        color: #2EC4B6;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .quiz-result-card .qr-label {
        font-size: 0.85rem;
        opacity: 0.55;
    }

    /* === Chat input — Aero glass === */
    [data-testid="stChatInput"] {
        background: rgba(2, 40, 80, 0.65) !important;
        border: 1px solid rgba(125, 211, 252, 0.25) !important;
        border-top-color: rgba(255,255,255,0.20) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
    }

    /* === Tabs — Aero pill bar === */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(2, 30, 60, 0.60);
        border-radius: 14px;
        padding: 0.25rem;
        border: 1px solid var(--border);
        border-top-color: rgba(255,255,255,0.15);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        gap: 0.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted);
        border-radius: 10px;
        transition: all 0.2s;
        font-weight: 600;
        font-family: 'Nunito', sans-serif;
        padding: 0.4rem 1rem;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.05) 50%, transparent 50%),
            linear-gradient(180deg, #0284C7, #0369A1) !important;
        color: white !important;
        box-shadow:
            0 2px 10px rgba(14, 165, 233, 0.40),
            inset 0 1px 0 rgba(255,255,255,0.28) !important;
        border: 1px solid rgba(125,211,252,0.30) !important;
        border-top-color: rgba(255,255,255,0.42) !important;
    }

    /* === Sidebar user card — Aero glass === */
    .sidebar-user-card {
        background: rgba(255,255,255,0.09) !important;
        border: 1px solid var(--border) !important;
        border-top-color: rgba(255,255,255,0.32) !important;
        border-radius: 14px;
        padding: 1rem 1.1rem 0.9rem;
        margin: 0.4rem 0 0.9rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.18) !important;
    }

    /* === Landing page full-screen === */
    .landing-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
        text-align: center;
        padding: 2rem 1rem;
    }

    .landing-icon {
        font-size: 4.5rem;
        line-height: 1;
        margin-bottom: 1.2rem;
        display: block;
        filter: drop-shadow(0 0 24px rgba(34,197,94,0.55));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-8px); }
    }

    .landing-hello {
        font-size: 3rem;
        font-weight: 700;
        margin: 0 0 0.4rem;
        background: linear-gradient(135deg, #86EFAC, #38BDF8, #7DD3FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }

    .landing-sub {
        font-size: 1.05rem;
        color: rgba(224,242,254,0.65) !important;
        margin: 0 0 2.5rem;
        max-width: 420px;
        line-height: 1.6;
        -webkit-text-fill-color: rgba(224,242,254,0.65);
    }

    .landing-form-wrap {
        width: 100%;
        max-width: 380px;
        margin: 0 auto;
    }

    .landing-features {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 2.5rem;
        max-width: 460px;
    }

    .landing-feature-pill {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        border-top-color: rgba(255,255,255,0.28);
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        color: rgba(224,242,254,0.72) !important;
        -webkit-text-fill-color: rgba(224,242,254,0.72);
        backdrop-filter: blur(8px);
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def _split_answer_and_sources(raw: str) -> tuple[str, list[str]]:
    """Split body and sources list from the model output without escaping.

    Returns (body_markdown, sources_lines)
    """
    if not raw:
        return "", []
    m = re.search(r"(?:^|\n)\s*(?:Source mapping:|📚\s*Sources[^\n]*:)\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return raw, []
    body = raw[:m.start()].rstrip()
    tail = m.group(1)
    sources = [l.strip(" -*\t") for l in tail.splitlines() if l.strip()]
    # Deduplicate by filename — keep first occurrence (highest relevance), renumber
    seen_files: set[str] = set()
    deduped: list[str] = []
    for s in sources:
        # Extract filename before the "(relevance:" part
        fname = re.sub(r"\d+\.\s*", "", s).split("(relevance")[0].strip()
        if fname not in seen_files:
            seen_files.add(fname)
            # Renumber: replace leading "N. " with new sequential number
            renumbered = re.sub(r"^\d+\.\s*", f"{len(deduped) + 1}. ", s)
            deduped.append(renumbered)
    return body, deduped

# Initialize components
@st.cache_resource
def init_adaptive_system():
    """Initialize the adaptive learning system components"""
    data_logger = DataLogger()
    ai_profiler = AIEnhancedProfiler()
    content_selector = ContentSelector(ai_profiler)
    quiz_generator = QuizGenerator(ai_profiler)
    dashboard = Dashboard()
    rag_system = RayPeatRAG()
    # Load topic model if available
    try:
        topic_model = CorpusTopicModel(model_dir="data/models/topics")
        topic_model.load()
    except Exception:
        topic_model = None
    
    return data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    # Quiz state
    if 'quiz_active' not in st.session_state:
        st.session_state.quiz_active = False
    if 'quiz_payload' not in st.session_state:
        st.session_state.quiz_payload = None
    if 'quiz_index' not in st.session_state:
        st.session_state.quiz_index = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_result' not in st.session_state:
        st.session_state.quiz_result = None


@st.cache_resource
def _get_rag_system():
    """Create RayPeatRAG once and cache it for the lifetime of the Streamlit process."""
    return RayPeatRAG()

_GREETING_PREFIXES = ("hello", "hi ", "hi!", "hey", "howdy", "yo ", "yo!", "hiya", "good morning", "good afternoon", "good evening", "what's up", "whats up", "sup")
_GREETING_ROOTS = {"hello", "helo", "hi", "hey", "hiya", "howdy", "yo", "sup", "wassup"}
_WHO_AM_I_PREFIXES = ("who are you", "what are you", "what can you do", "what do you do", "tell me about yourself", "describe yourself")
_THANKS_PREFIXES = ("thank", "thx", "cheers", "appreciate")
_BYE_PREFIXES = ("bye", "goodbye", "see you", "cya", "take care", "later")
_SLANG = {"wyd", "wbu", "wym", "ngl", "imo", "imho", "tbh", "brb", "gtg", "afk", "gg", "fr", "irl", "idk", "smh", "rn", "btw", "fyi", "omg", "wtf", "lmk", "np", "ty", "nm", "wb"}
_FILLER = {"ok", "okay", "cool", "nice", "great", "awesome", "sure", "alright", "got it", "yep", "yup", "nope", "no", "yes", "interesting", "fascinating", "wow", "really", "huh", "hmm", "hm", "ah", "oh", "lol", "haha", "lmao", "k"}
_REACTION_PREFIXES = ("that's crazy", "thats crazy", "that's insane", "thats insane", "that's wild", "thats wild", "that's weird", "thats weird", "i did not know", "i didn't know", "no way", "no wonder", "makes sense", "that makes sense", "good to know", "interesting,", "wow,", "crazy,", "thats very", "that's very", "thats so", "that's so", "very interesting", "so interesting", "pretty interesting", "that's quite", "thats quite")
_SOCIAL_PREFIXES = ("im good", "i'm good", "im fine", "i'm fine", "im doing", "i'm doing", "im great", "i'm great", "doing well", "not bad", "how are you", "how r u", "how ru ", "hows it", "how's it", "im okay", "i'm okay", "im alright", "i'm alright")
_SOCIAL_FEEDBACK = {"rude", "mean", "wrong", "boring", "weird", "strange", "odd", "dumb", "stupid", "offensive", "unhelpful"}

def _is_greeting_variant(q: str) -> bool:
    """Catch greeting typos/variants like 'herro', 'helo', 'heyyyy' (single short word)."""
    if len(q.split()) > 2 or len(q) > 12:
        return False
    norm = re.sub(r'(.)\1+', r'\1', q)  # collapse repeated chars: "heyyyy" → "hey"
    norm = norm.replace('r', 'l')        # r→l swap catches "herro" → "hello"
    return norm in _GREETING_ROOTS

def _is_gibberish(q: str) -> bool:
    """True if the query looks like gibberish, random chars, or a greeting variant with repeated letters."""
    # Long run of repeated chars (e.g. "hellllll", "aaaaa")
    if re.search(r'(.)\1{3,}', q):
        return True
    letters = re.sub(r'[^a-z]', '', q)
    # No vowels at all in a word-length string
    if len(letters) >= 4 and not any(c in letters for c in 'aeiou'):
        return True
    # 5+ consecutive consonant letters — real English words top out at 4 (e.g. "ngth").
    # Check each word individually to avoid cross-word false positives
    # (e.g. "cortisol and stress" → "ndstr" when concatenated).
    # Explicitly enumerate consonants so y is treated as a vowel (avoids "trypt" false-positive).
    for w in q.split():
        w_letters = re.sub(r'[^a-z]', '', w)
        if re.search(r'[b-df-hj-np-tv-xz]{5,}', w_letters):
            return True
    # Every long word lacks vowels (multi-word gibberish like "sdfds qwrtz")
    words = q.split()
    if all(len(w) >= 4 and not any(c in w for c in 'aeiou') for w in words if w.isalpha()):
        return True
    return False

def _is_small_talk(query: str) -> str | None:
    """Return a canned response for greetings/small talk, or None if it's a real question."""
    q = query.strip().lower().rstrip("!?.")
    # Gibberish / random chars
    if _is_gibberish(q):
        return "That doesn't look like a question I can help with. Ask me anything about Ray Peat's work — metabolism, thyroid, hormones, nutrition."
    # Pure filler, slang, or very short
    if q in _FILLER or q in _SLANG or len(q) <= 2:
        return "What would you like to know about Ray Peat's work?"
    # Single-word social feedback (no '?')
    words = q.split()
    if len(words) == 1 and q in _SOCIAL_FEEDBACK and "?" not in query:
        return "Sorry if that wasn't helpful! Ask me anything about Ray Peat's work — metabolism, thyroid, hormones, nutrition."
    # Conversational reactions — no question mark
    if "?" not in query and any(q.startswith(p) for p in _REACTION_PREFIXES):
        return "Glad that's useful! What else would you like to dig into?"
    # Social self-reports ("im good hbu", "doing well thanks") — not questions about Peat
    if "?" not in query and any(q.startswith(p) for p in _SOCIAL_PREFIXES):
        return "Good to hear! What would you like to know about Ray Peat's work?"
    if any(q == g or q.startswith(g) for g in _GREETING_PREFIXES) or _is_greeting_variant(q):
        return "Hey! I'm Ray Peat AI — ask me anything about bioenergetics, metabolism, hormones, thyroid, nutrition, or anything else from Ray Peat's work. What are you curious about?"
    if any(q.startswith(p) for p in _WHO_AM_I_PREFIXES):
        return "I'm Ray Peat AI — a chatbot trained on Ray Peat's writings, interviews, and newsletters. I can answer questions about his views on metabolism, thyroid, hormones, nutrition, CO2, and bioenergetics. What do you want to explore?"
    if any(q.startswith(p) for p in _THANKS_PREFIXES):
        return "Happy to help! What else would you like to know about Ray Peat's work?"
    if any(q.startswith(p) for p in _BYE_PREFIXES):
        return "Take care! Come back anytime you want to explore Ray Peat's ideas."
    return None

def render_user_setup():
    """Render user identification setup"""
    st.markdown("<div class='main-header'><h1>🧠 PeatLearn AI - Adaptive Learning</h1><p>Your Personal Ray Peat Bioenergetics Tutor</p></div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<p style="font-size:0.8rem;opacity:0.4;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">🧠 PeatLearn</p>', unsafe_allow_html=True)
        
        # Get user ID
        user_id = st.text_input("Enter your name or ID:", value=st.session_state.get('user_id', ''))
        
        if user_id and user_id != st.session_state.user_id:
            st.session_state.user_id = user_id
            
            # Initialize session for this user
            data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()
            st.session_state.session_id = data_logger.get_session_id()
            
            # Load existing profile if available
            st.session_state.user_profile = ai_profiler.get_user_profile(user_id)
            
            st.success(f"Welcome, {user_id}! 🎉")
            st.rerun()
        
        if st.session_state.user_id:
            # Ensure session_id is initialized
            if not st.session_state.get('session_id'):
                try:
                    DataLogger().get_session_id()
                except Exception:
                    pass
            api_key = os.getenv('GEMINI_API_KEY')
            adv_ok = _adv_health_cached()
            _sid = st.session_state.session_id[:8] + '…' if st.session_state.session_id else 'initializing'
            _uid_display = html.escape(st.session_state.user_id)
            _ai_pill = '<span class="status-pill ok">🤖 AI Active</span>' if api_key else '<span class="status-pill warn">🤖 Fallback</span>'
            _api_pill = '<span class="status-pill ok">🧩 ML Online</span>' if adv_ok else '<span class="status-pill warn">🧩 ML Offline</span>'
            st.markdown(f"""
            <div class="sidebar-user-card">
                <div class="user-name">👤 {_uid_display}</div>
                <div class="session-chip">session {_sid}</div>
                <div class="status-row">{_ai_pill}{_api_pill}</div>
            </div>
            """, unsafe_allow_html=True)
            
             # Development mode indicator and controls
            if DEVELOPMENT_MODE:
                st.markdown("---")
                st.subheader("🔄 Development Mode")
                st.info("🚀 Development mode is active!")
                
                auto_refresh_enabled = st.toggle(
                    "Enable file watching", 
                    value=st.session_state.get('auto_refresh_enabled', False),
                    help="Automatically refresh when code files change"
                )
                
                if auto_refresh_enabled != st.session_state.get('auto_refresh_enabled', False):
                    st.session_state.auto_refresh_enabled = auto_refresh_enabled
                    if auto_refresh_enabled:
                        setup_auto_refresh()
                        st.success("🔄 Auto-refresh enabled!")
                    else:
                        # Stop file observer
                        if 'file_observer' in st.session_state:
                            try:
                                st.session_state.file_observer.stop()
                                del st.session_state.file_observer
                                del st.session_state.auto_refresh_setup
                            except Exception:
                                pass
                        st.info("🔄 Auto-refresh disabled")
                    st.rerun()
                
                # Periodic refresh for data updates
                periodic_refresh_enabled = st.toggle(
                    "Periodic data refresh", 
                    value=st.session_state.get('periodic_refresh_enabled', False),
                    help="Refresh analytics/data every 30 seconds"
                )
                
                if periodic_refresh_enabled:
                    st.session_state.periodic_refresh_enabled = True
                    refresh_interval = st.slider("Refresh interval (seconds)", 10, 120, 30)
                    setup_periodic_refresh(refresh_interval)
                else:
                    st.session_state.periodic_refresh_enabled = False
                
                # Manual refresh button
                if st.button("🔄 Manual Refresh", help="Force refresh the app"):
                    st.rerun()
                    
                # Status indicators
                if st.session_state.get('auto_refresh_enabled', False):
                    st.success("🟢 File watching active")
                if st.session_state.get('periodic_refresh_enabled', False):
                    st.info("🔄 Periodic refresh active")
            else:
                # Production mode - show minimal refresh options
                st.markdown("---")
                st.subheader("🔄 Refresh")
                
                # Only manual refresh in production
                if st.button("🔄 Refresh Data", help="Refresh analytics and data"):
                    st.rerun()
                
                # Show how to enable dev mode
                with st.expander("💡 Enable Development Mode"):
                    st.markdown("""
                    **To enable development features:**
                    
                    **Method 1: Environment Variable**
                    ```bash
                    export PEATLEARN_DEV_MODE=true
                    python peatlearn_master.py
                    ```
                    
                    **Method 2: Command Line Flag**
                    ```bash
                    python peatlearn_master.py --dev
                    ```
                    
                    **Method 3: Via Streamlit**
                    ```bash
                    STREAMLIT_DEV_MODE=true streamlit run peatlearn_master.py
                    ```
                    """)
                st.caption("🔒 Production mode active - development features disabled")

            st.markdown("""
            <div class="sidebar-footer">
                PeatLearn v1.0 · Powered by Gemini 2.5<br>
                552 Sources · Ray Peat Corpus
            </div>
            """, unsafe_allow_html=True)

def render_user_profile():
    """Render user profile and learning analytics"""
    if not st.session_state.user_profile:
        st.info("Start chatting to build your learning profile! 📈")
        return
    
    profile = st.session_state.user_profile
    
    st.markdown('<div class="section-header"><h3>📊 Your Learning Profile</h3></div>', unsafe_allow_html=True)
    
    # Overall stats
    _state = profile.get('overall_state', 'new').title()
    _style = profile.get('learning_style', 'balanced').replace('_', ' ').title()
    _interactions = profile.get('total_interactions', 0)
    _avg_fb = profile.get('average_feedback', 0)
    _fb_display = f"{_avg_fb:.1f}" if _avg_fb else "—"
    st.markdown(f"""
    <div class="stat-cards-row">
        <div class="stat-card">
            <span class="sc-icon">🎓</span>
            <div class="sc-value">{_state}</div>
            <div class="sc-label">Learning State</div>
        </div>
        <div class="stat-card">
            <span class="sc-icon">🧭</span>
            <div class="sc-value">{_style}</div>
            <div class="sc-label">Learning Style</div>
        </div>
        <div class="stat-card">
            <span class="sc-icon">💬</span>
            <div class="sc-value">{_interactions}</div>
            <div class="sc-label">Total Interactions</div>
        </div>
        <div class="stat-card">
            <span class="sc-icon">⭐</span>
            <div class="sc-value">{_fb_display}</div>
            <div class="sc-label">Avg Feedback</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Topic mastery
    topic_mastery = profile.get('topic_mastery', {})
    if topic_mastery:
        st.markdown('<div class="section-header" style="margin-top:1rem"><h3>🎯 Topic Mastery</h3></div>', unsafe_allow_html=True)
        
        mastery_data = []
        for topic, data in topic_mastery.items():
            mastery_data.append({
                'Topic': topic.title(),
                'State': data.get('state', 'unknown'),
                'Mastery Level': data.get('mastery_level', 0),
                'Interactions': data.get('total_interactions', 0)
            })
        
        df = pd.DataFrame(mastery_data)
        
        # Create mastery chart
        fig = px.bar(df, x='Topic', y='Mastery Level',
                    color='State',
                    title="Topic Mastery Overview",
                    color_discrete_map={
                        'struggling': '#F87171',
                        'learning':   '#FBBF24',
                        'advanced':   '#34D399'
                    },
                    template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Nunito',
            title_font_size=15,
            legend_title_text='State',
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show mastery badges
        badges_html = "".join(
            f'<span class="mastery-badge {row["State"]}">{row["Topic"]}: {row["State"].title()} ({row["Mastery Level"]:.1f})</span>'
            for _, row in df.iterrows()
        )
        st.markdown(f'<div class="badges-row">{badges_html}</div>', unsafe_allow_html=True)
    
    # AI insights if available
    ai_analysis = profile.get('ai_analysis', {})
    if ai_analysis:
        st.markdown('<div class="section-header" style="margin-top:1rem"><h3>🤖 AI Insights</h3></div>', unsafe_allow_html=True)
        insights = ai_analysis.get('insights', [])
        for insight in insights:
            st.markdown(f'<div class="insight-card"><span class="ic-icon">💡</span>{html.escape(str(insight))}</div>', unsafe_allow_html=True)
        learning_velocity = ai_analysis.get('learning_velocity')
        if learning_velocity:
            st.markdown(f'<div class="insight-card"><span class="ic-icon">📈</span><strong>Learning Velocity:</strong>&nbsp;{html.escape(learning_velocity.title())}</div>', unsafe_allow_html=True)

def render_recommendations():
    """Render personalized recommendations via backend."""
    if not st.session_state.get('user_id'):
        return
    st.markdown('<div class="section-header"><h3>💡 Personalized Recommendations</h3></div>', unsafe_allow_html=True)
    try:
        topic_mastery = (st.session_state.user_profile or {}).get('topic_mastery', {})
        topic_filter = list(topic_mastery.keys())[:5] if topic_mastery else None
        recs = _fetch_recommendations_cached(st.session_state.user_id, topic_filter, 8)
        if True:
            if not recs:
                st.markdown("""
                <div class="empty-state-card">
                    <span class="esc-icon">🔮</span>
                    <p>No recommendations yet — interact with the chat to personalize your feed.</p>
                </div>
                """, unsafe_allow_html=True)
                return
            for rec in recs:
                title = html.escape(str(rec.get('title') or rec.get('content_id', '')))
                reason = html.escape(str(rec.get('recommendation_reason', '')))
                snippet = html.escape(str(rec.get('snippet', '')))
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{title}</h4>
                        <p>{snippet}</p>
                        {f"<small><i>{reason}</i></small>" if reason else ""}
                    </div>
                """, unsafe_allow_html=True)
        
    except Exception:
        if not _adv_health_cached():
            st.caption("_Recommendations load once the ML server (port 8001) is running._")
        else:
            st.warning("Recommendations temporarily unavailable. Try refreshing.")

def render_chat_interface():
    """Render the main chat interface with AI profiling"""
    data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()

    st.markdown('<div class="section-header"><h3>💬 Chat with Ray Peat AI</h3></div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-empty-state">
            <span class="ces-icon">🌿</span>
            <h4>Ask Ray Peat AI anything</h4>
            <p>Start with one of these questions or type your own below</p>
            <div class="suggested-questions">
                <div class="sq-item">💡 What does Ray Peat say about thyroid and metabolism?</div>
                <div class="sq-item">🧬 How does progesterone protect against estrogen dominance?</div>
                <div class="sq-item">⚡ Why does Ray Peat recommend sugar over starch?</div>
                <div class="sq-item">🫁 What is the role of CO₂ in oxygen delivery?</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            _ts = message.get('timestamp', '')
            _ts_fmt = ''
            if _ts:
                try:
                    _ts_fmt = datetime.fromisoformat(_ts).strftime('%H:%M')
                except Exception:
                    pass
            _uid = st.session_state.get('user_id', 'U')
            _initials = (_uid[:2].upper()) if _uid else 'U'
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">
                        <span class="avatar avatar-user">{_initials}</span>
                        <span class="msg-name">You</span>
                        <span class="msg-time">{_ts_fmt}</span>
                    </div>
                    {html.escape(message['content'])}
                </div>
            """, unsafe_allow_html=True)
        else:
            body_md, sources = _split_answer_and_sources(message['content'])
            _ts = message.get('timestamp', '')
            _ts_fmt = ''
            if _ts:
                try:
                    _ts_fmt = datetime.fromisoformat(_ts).strftime('%H:%M')
                except Exception:
                    pass
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-header">
                        <span class="avatar avatar-ai">🌿</span>
                        <span class="msg-name">Ray Peat AI</span>
                        <span class="msg-time">{_ts_fmt}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            # Render markdown body first (allows headings/lists)
            st.markdown(body_md)
            # Render sources with hover UI if present
            if sources:
                sources_list = "".join(f"<li>{html.escape(it)}</li>" for it in sources[:12])
                st.markdown(f"""
                    <div class="rag-answer">
                        <div class="sources-container">
                            <div class="sources-toggle">📚 Sources</div>
                            <div class="sources-content">
                                <ul>{sources_list}</ul>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show rating slider for assistant messages (1-10)
            if 'feedback' not in message:
                col1, col2 = st.columns([3, 1])
                with col1:
                    rating = st.slider(
                        "Rate your understanding (1=poor, 10=mastered)", 1, 10, 7,
                        key=f"rate_{i}_{message.get('timestamp','')}"
                    )
                with col2:
                    if st.button("Submit", key=f"rate_submit_{i}_{message.get('timestamp','')}"):
                        handle_feedback(message, int(rating), data_logger, ai_profiler)
    
    # Chat input
    if prompt := st.chat_input("Ask Ray Peat about bioenergetics, metabolism, hormones..."):
        # Add user message
        user_message = {
            'role': 'user',
            'content': prompt,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Small talk check runs on the ORIGINAL prompt before any context resolution,
        # so prepended context like "thank you, bye —" can't accidentally trigger canned replies.
        canned = _is_small_talk(prompt)
        if canned:
            response = canned
        else:
            # Resolve ambiguous pronouns using recent chat history
            resolved_query = prompt
            ambiguous = any(w in prompt.lower().split() for w in ["it", "this", "that", "they", "them", "its", "more", "elaborate"])
            if ambiguous and len(st.session_state.chat_history) >= 2:
                last_user = next(
                    (m["content"] for m in reversed(st.session_state.chat_history[:-1]) if m["role"] == "user"),
                    None
                )
                if last_user:
                    resolved_query = f"{last_user} — {prompt}"

            # Generate AI response using real RAG with Gemini
            # Pass the last 3 turns (6 messages) as context so the model can
            # reference the conversation without relying solely on pronoun resolution.
            recent_history = st.session_state.chat_history[:-1][-6:] if len(st.session_state.chat_history) > 1 else []
            with st.spinner("Ray Peat AI is thinking..."):
                response = _get_rag_system().get_rag_response(
                    resolved_query,
                    st.session_state.user_profile,
                    chat_history=recent_history,
                )
        
        # Add assistant message (attach parsed sources)
        body_md_tmp, sources_tmp = _split_answer_and_sources(response)
        assistant_message = {
            'role': 'assistant', 
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'user_query': prompt,
            'sources': sources_tmp
        }
        st.session_state.chat_history.append(assistant_message)
        
        st.rerun()

def handle_feedback(message, feedback_value, data_logger, ai_profiler):
    """Handle user feedback and update profile"""
    if not st.session_state.get('user_id'):
        st.warning("Set a user ID first in the sidebar.")
        return
    # Ensure session id exists
    if not st.session_state.get('session_id'):
        try:
            DataLogger().get_session_id()
        except Exception:
            pass
    
    # Extract topic (hybrid): prefer RAG sources vote, fallback to centroid similarity
    assigned_topic = 'general'
    similarity_conf = 0.0
    jargon = 0.0
    try:
        tm = CorpusTopicModel(model_dir="data/models/topics")
        tm.load()
        # Source vote
        srcs = message.get('sources', []) if isinstance(message, dict) else []
        files = []
        import re as _re
        for s in srcs:
            m = _re.search(r"\d+\.\s*([^\(\n]+)", s)
            if m:
                files.append(m.group(1).strip())
        cluster = tm.assign_topic_from_rag_sources(files) if files else None
        # Filter meta-like clusters from source vote
        meta_terms = ["host", "author", "dr ", "dr.", "yeah", "uh", "context", "asks"]
        def is_meta(lbl: str) -> bool:
            l = lbl.lower()
            return any(t in l for t in meta_terms)
        q = message.get('user_query', '')
        if cluster and is_meta(cluster.label):
            cluster = None
        if not cluster:
            cluster = tm.assign_topic_from_text(q)
        if cluster:
            assigned_topic = cluster.label.split(',')[0].strip().lower().replace(' ', '_') or 'general'
            # Compute both metrics regardless of path
            similarity_conf = tm.similarity_to_cluster(q, cluster)
            jargon = tm.jargon_score(q, cluster, top_n=12)
    except Exception:
        from peatlearn.adaptive.profile_analyzer import TopicExtractor
        topic_extractor = TopicExtractor()
        assigned_topic = topic_extractor.get_primary_topic(message.get('user_query', '')) or 'general'
    topic = assigned_topic
    
    # Log the interaction (include sources in context if present)
    sources_list = message.get('sources', []) if isinstance(message, dict) else []
    # Use a fresh logger to avoid any stale state
    _logger = DataLogger()
    _logger.log_interaction(
        user_query=message.get('user_query', ''),
        llm_response=message.get('content', ''),
        topic=topic,
        user_feedback=feedback_value,
        interaction_type='chat',
        context={'sources': sources_list, 'jargon_score': jargon, 'similarity_confidence': similarity_conf}
    )
    try:
        import os as _os
        csv_path = str(_logger.interactions_file)
        size = _os.path.getsize(csv_path)
        st.toast(f"Interaction logged → {csv_path} ({size} bytes)")
    except Exception:
        st.toast("Interaction logged.")

    # Forward interaction to personalization backend to update state
    try:
        perf = 0.9 if feedback_value == 1 else (0.1 if feedback_value == -1 else 0.5)
        first_source = ''
        if sources_list:
            # Try to extract a filename from "1. filename (relevance: x)"
            import re as _re
            m = _re.search(r"\d+\.\s*([^\(\n]+)", sources_list[0])
            if m:
                first_source = m.group(1).strip()
        payload = {
            'user_id': st.session_state.user_id,
            'content_id': first_source or f"chat_{datetime.now().timestamp():.0f}",
            'interaction_type': 'chat',
            'performance_score': perf,
            'time_spent': 0.0,
            'difficulty_level': 0.5,
            'topic_tags': [topic] if topic else [],
            'context': {'sources': sources_list}
        }
        requests.post("http://localhost:8001/api/interactions", json=payload, timeout=3)
    except Exception:
        pass
    
    # Update user profile with AI analysis
    all_interactions = data_logger._load_interactions()
    user_interactions = all_interactions[all_interactions['user_id'] == st.session_state.user_id].to_dict(orient='records')
    
    # Update profile using AI
    updated_profile = ai_profiler.update_user_profile_with_ai(st.session_state.user_id, user_interactions)
    st.session_state.user_profile = updated_profile
    
    # Mark message as having feedback
    message['feedback'] = feedback_value
    
    # Show success message
    feedback_text = "positive" if feedback_value > 0 else "negative"
    st.success(f"Thanks for the {feedback_text} feedback! Your profile has been updated. 📊")
    
    st.rerun()

def render_quiz_interface():
    """Render personalized quiz interface (one question at a time via backend)."""
    if not st.session_state.get('user_id'):
        st.info("Enter your user ID first.")
        return
    st.markdown('<div class="section-header"><h3>🎯 Personalized Quiz</h3></div>', unsafe_allow_html=True)

    # Debug toggle
    debug_mode = st.toggle("Show adaptive debug info", value=False, help="Displays ability, item difficulty and target anchors.")

    # Session-based quiz flow using new endpoints
    if st.session_state.get('quiz_active') and st.session_state.get('quiz_session_id'):
        _answered = st.session_state.get('quiz_answered', 0)
        _total_q = st.session_state.get('quiz_num_questions', 5)
        _progress = min(_answered / _total_q, 1.0)
        st.markdown(f'<div class="quiz-progress-label">Question {_answered + 1} of {_total_q}</div>', unsafe_allow_html=True)
        st.progress(_progress)
        session_id = st.session_state.quiz_session_id
        # Fetch next item if we don't have a current one
        if 'quiz_current_item' not in st.session_state or st.session_state.quiz_current_item is None:
            try:
                params = {"session_id": session_id}
                if st.session_state.get('user_id'):
                    params["user_id"] = st.session_state.user_id
                r = requests.get("http://localhost:8001/api/quiz/next", params=params, timeout=10)
                data = r.json()
                if data.get('done'):
                    # Finish session
                    fr = requests.post("http://localhost:8001/api/quiz/finish", params={"session_id": session_id}, timeout=10)
                    if fr.status_code == 200:
                        st.session_state.quiz_result = fr.json()
                        st.success(f"Quiz complete: {st.session_state.quiz_result.get('correct',0)}/{st.session_state.quiz_result.get('total',0)}")
                    st.session_state.quiz_active = False
                    st.session_state.quiz_session_id = None
                    st.session_state.quiz_current_item = None
                    st.rerun()
                else:
                    st.session_state.quiz_current_item = data
            except Exception as e:
                st.error(f"Quiz service error: {e}")
                st.session_state.quiz_active = False
                st.session_state.quiz_session_id = None
                return
        item = st.session_state.quiz_current_item
        st.markdown(f"""
        <div class="quiz-question-card">
            {html.escape(item.get('stem', ''))}
        </div>
        """, unsafe_allow_html=True)
        if debug_mode:
            cols_dbg = st.columns(3)
            cols_dbg[0].metric("Item difficulty (b)", f"{item.get('difficulty_b', 0.5):.2f}")
            if item.get('ability_topic') is not None:
                cols_dbg[1].metric("Ability (θ topic)", f"{item.get('ability_topic'):.2f}")
            if item.get('target_anchor') is not None:
                cols_dbg[2].metric("Target anchor", f"{item.get('target_anchor'):.2f}")
        # Passage and source
        ctx = item.get('passage_excerpt') or ''
        src = item.get('source_file') or ''
        if ctx:
            _src_line = f'<div class="passage-source">📄 {html.escape(src)}</div>' if src else ''
            st.markdown(f"""
            <div class="passage-card">
                {html.escape(ctx)}
                {_src_line}
            </div>
            """, unsafe_allow_html=True)
        options = item.get('options', [])
        choice = st.radio("Choose an option:", options=[f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)], key=f"quiz_choice_{item.get('item_id')}")
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Submit Answer", key=f"submit_{item.get('item_id')}"):
                try:
                    selected_idx = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)].index(choice)
                except Exception:
                    selected_idx = 0
                with st.spinner("Checking..."):
                    r = requests.post("http://localhost:8001/api/quiz/answer", json={
                        "session_id": session_id,
                        "item_id": item.get('item_id'),
                        "chosen_index": selected_idx,
                        "time_ms": 0,
                        "user_id": st.session_state.user_id,
                    }, timeout=10)
                if r.status_code == 200:
                    res = r.json()
                    if res.get('correct'):
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect. Correct answer: {chr(65 + int(res.get('correct_index',0)))}")
                st.session_state.quiz_current_item = None
                st.session_state.quiz_answered = st.session_state.get('quiz_answered', 0) + 1
                st.rerun()
        with colB:
            if st.button("Cancel Quiz"):
                st.session_state.quiz_active = False
                st.session_state.quiz_session_id = None
                st.session_state.quiz_current_item = None
                st.rerun()
        return

    # Config to start a new quiz
    topic_mastery = (st.session_state.user_profile or {}).get('topic_mastery', {})
    topics = list(topic_mastery.keys()) if topic_mastery else [
        "thyroid function and metabolism",
        "progesterone and estrogen balance",
        "sugar and cellular energy",
        "carbon dioxide and metabolism",
    ]
    selected_topic = st.selectbox("Choose a topic for your quiz (optional):", [""] + topics)
    num_q = st.slider("Number of questions", 3, 10, 5)
    # Show last result if available
    if st.session_state.get('quiz_result'):
        res = st.session_state.quiz_result
        _correct = res.get('correct', 0)
        _total = res.get('total', 0)
        _pct = res.get('score_percentage', 0)
        st.markdown(f"""
        <div class="quiz-result-card">
            <div class="qr-score">{_correct}/{_total}</div>
            <div class="qr-label">Last quiz — {_pct:.1f}% correct</div>
        </div>
        """, unsafe_allow_html=True)
    if st.button("Start Quiz", type="primary"):
        try:
            payload = {"user_id": st.session_state.user_id, "num_questions": num_q}
            if selected_topic:
                payload["topics"] = [selected_topic]
            with st.spinner("Starting your quiz session..."):
                r = requests.post("http://localhost:8001/api/quiz/session/start", json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                st.session_state.quiz_session_id = data.get('session_id')
                st.session_state.quiz_active = True
                st.session_state.quiz_current_item = None
                st.session_state.quiz_num_questions = num_q
                st.session_state.quiz_answered = 0
                st.rerun()
            else:
                st.error(f"Failed to generate quiz: {r.text}")
        except Exception as e:
            st.error(f"Quiz service unavailable: {e}")

    # Ability history debug view
    if debug_mode and st.session_state.get('user_id'):
        try:
            # Read local ability history from DB if present
            import sqlite3 as _sql
            from pathlib import Path as _Path
            dbp = _Path("data/user_interactions/interactions.db")
            if dbp.exists():
                conn = _sql.connect(str(dbp))
                df_hist = pd.read_sql_query(
                    "SELECT topic, ability, updated_at FROM user_ability_history WHERE user_id = ? ORDER BY updated_at ASC",
                    conn,
                    params=(st.session_state.user_id,),
                )
                conn.close()
                if not df_hist.empty:
                    df_hist['updated_at'] = pd.to_datetime(df_hist['updated_at'])
                    for topic in sorted(df_hist['topic'].unique()):
                        seg = df_hist[df_hist['topic'] == topic]
                        st.line_chart(seg.set_index('updated_at')['ability'], height=140)
        except Exception:
            pass

def render_memorial():
    """Render an in-app memorial page for Dr. Ray Peat with technical details."""
    st.header("🕯️ In Memoriam: Dr. Raymond Peat (1936–2022)")
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            img_path = Path("data/assets/ray_peat.jpg")
            if img_path.exists():
                st.image(str(img_path), caption="Dr. Ray Peat")
            else:
                st.image("https://upload.wikimedia.org/wikipedia/commons/6/65/Placeholder_Person.jpg", caption="Dr. Ray Peat")
        except Exception:
            pass
    with col2:
        st.markdown(
            """
            Dr. Ray Peat advanced a bioenergetic view of biology: energy and structure are interdependent at every level.
            PeatLearn is dedicated to preserving his corpus and helping learners progress with adaptive AI.
            """
        )

    st.subheader("Bioenergetics (Primer)")
    st.markdown("""
    - Energy as a central variable: oxidative metabolism supports structure and resilience
    - Thyroid hormones (T3/T4) sustain respiration, temperature, and CO₂ production
    - Protective factors (progesterone, adequate carbs, calcium, saturated fats) support oxidative metabolism
    - Stress mediators (excess estrogen, serotonin, nitric oxide, endotoxin, PUFA) push toward stress metabolism
    - CO₂ improves oxygen delivery (Bohr effect) and stabilizes enzymes and membranes
    """)

    st.subheader("How PeatLearn Works (User)")
    st.markdown("- Ask questions and browse sources\n- Get personalized recommendations\n- Take short adaptive quizzes calibrated to your level\n- Improve over time as difficulty adjusts")

    st.subheader("Architecture (Technical)")
    st.markdown("- RAG over Pinecone index of Ray Peat’s corpus\n- Gemini 2.5 Flash Lite to synthesize grounded items/answers\n- Adaptive updates per answer: ability θ(user, topic) and item difficulty b(item)\n- FastAPI services (8000 basic, 8001 advanced) and SQLite for quiz/session state")
    st.markdown("""
```mermaid
flowchart TD
  A[Streamlit UI] -->|Ask| B(Advanced API 8001)
  B -->|Search| C[Pinecone]
  C --> B
  B -->|LLM\n(Gemini 2.5 Flash Lite)| D[Question & Answer]
  D --> B
  B -->|Return\nAnswer+Sources| A
  A -->|Start Quiz| B
  B -->|Seed items| C
  B -->|Sessions & Stats| E[(SQLite)]
  A -->|Answer| B
  B -->|Update θ,b| E
```
""")

    st.subheader("Adaptive Model Details")
    st.code("""
Ability update:  θ_new = θ + Kθ · (observed − expected)
Item update:     b_new = b + Kb · (expected − observed)
Expected prob:   expected = σ(1.7 · (θ − b))
""", language="text")

    st.subheader("Project Links")
    st.markdown("- `docs/RAY_Peat_IN_MEMORIAM.md` (full memorial page)\n- README for architecture and endpoints")

def render_landing_page():
    """Full-screen centered onboarding — no sidebar, just a greeting + name input."""
    # Hide sidebar entirely on landing
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="stAppViewContainer"] > .main > .block-container {
        max-width: 560px !important;
        margin: 0 auto !important;
        padding-top: 8vh !important;
        padding-bottom: 4rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-wrap">
        <span class="landing-icon">🌿</span>
        <div class="landing-hello">Hello there 👋</div>
        <p class="landing-sub">
            Welcome to PeatLearn — your AI-powered guide to Dr. Ray Peat's
            bioenergetic philosophy. What's your name?
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 4, 1])
    with col_c:
        with st.form("landing_form", clear_on_submit=False):
            name = st.text_input(
                "Your name",
                placeholder="Enter your name...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "Start Learning →",
                use_container_width=True,
                type="primary",
            )
            if submitted:
                if name.strip():
                    data_logger, ai_profiler, _, _, _, _, _ = init_adaptive_system()
                    st.session_state.user_id = name.strip()
                    st.session_state.session_id = data_logger.get_session_id()
                    st.session_state.user_profile = ai_profiler.get_user_profile(name.strip())
                    st.rerun()
                else:
                    st.error("Please enter your name to continue.")

    st.markdown("""
    <div class="landing-features">
        <span class="landing-feature-pill">🔬 552+ Source Documents</span>
        <span class="landing-feature-pill">🤖 Gemini 2.5 AI</span>
        <span class="landing-feature-pill">🎯 Adaptive Quizzes</span>
        <span class="landing-feature-pill">📊 Learning Profile</span>
        <span class="landing-feature-pill">💡 Personalized Recs</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    init_session_state()
    
    # Setup cleanup on app exit
    def cleanup_observers():
        if 'file_observer' in st.session_state:
            try:
                st.session_state.file_observer.stop()
                st.session_state.file_observer.join()
            except Exception:
                pass
    
    import atexit
    atexit.register(cleanup_observers)
    
    # Check if user is set up — show full-screen landing if not
    if not st.session_state.user_id:
        render_landing_page()
        st.stop()
    
    # Sidebar user info and navigation
    render_user_setup()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📊 Profile", "🎯 Quiz", "📈 Analytics", "🕯️ Memorial"])
    
    with tab1:
        render_chat_interface()
        render_recommendations()
    
    with tab2:
        render_user_profile()
    
    with tab3:
        render_quiz_interface()
    
    with tab4:
        st.markdown('<div class="section-header"><h3>📈 Learning Analytics</h3></div>', unsafe_allow_html=True)
        if not st.session_state.user_id:
            st.info("Enter your user ID to view analytics.")
        else:
            # --- Quiz history (always available, reads directly from SQLite) ---
            qh_ok = False
            qh_data = {}
            try:
                qh_r = requests.get(f"http://localhost:8001/api/analytics/quiz-history/{st.session_state.user_id}", timeout=8)
                if qh_r.status_code == 200:
                    qh_ok = True
                    qh_data = qh_r.json()
            except Exception:
                pass

            if qh_ok and qh_data.get('sessions'):
                sessions = qh_data['sessions']
                total_sessions = qh_data.get('total_sessions', 0)
                avg_score = sum(s['score_pct'] for s in sessions) / len(sessions) if sessions else 0
                last_score = sessions[-1]['score_pct'] if sessions else 0

                st.markdown(f"""
                <div class="stat-cards-row three-col">
                    <div class="stat-card">
                        <span class="sc-icon">🎯</span>
                        <div class="sc-value">{total_sessions}</div>
                        <div class="sc-label">Quizzes Taken</div>
                    </div>
                    <div class="stat-card">
                        <span class="sc-icon">📊</span>
                        <div class="sc-value">{avg_score:.0f}%</div>
                        <div class="sc-label">Avg Score</div>
                    </div>
                    <div class="stat-card">
                        <span class="sc-icon">⚡</span>
                        <div class="sc-value">{last_score:.0f}%</div>
                        <div class="sc-label">Last Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Score over time chart
                df_sessions = pd.DataFrame(sessions)
                fig_scores = px.line(
                    df_sessions, x='session_num', y='score_pct',
                    title="Quiz Score Over Time",
                    markers=True,
                    labels={'session_num': 'Quiz #', 'score_pct': 'Score (%)'},
                    color_discrete_sequence=['#0EA5E9'],
                    template='plotly_dark',
                )
                fig_scores.add_hline(y=avg_score, line_dash='dot', line_color='#34D399',
                                     annotation_text=f"avg {avg_score:.0f}%",
                                     annotation_font_color='#34D399')
                fig_scores.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_family='Nunito',
                    title_font_size=15,
                    yaxis_range=[0, 105],
                )
                fig_scores.update_xaxes(gridcolor='rgba(255,255,255,0.06)', dtick=1)
                fig_scores.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
                st.plotly_chart(fig_scores, use_container_width=True)

                # Topic mastery bar chart
                if qh_data.get('mastery'):
                    df_mastery = pd.DataFrame(qh_data['mastery'])
                    df_mastery.columns = ['Topic', 'Ability']
                    fig_mastery = px.bar(
                        df_mastery.head(12), x='Topic', y='Ability',
                        title="Topic Ability (IRT θ)",
                        color_discrete_sequence=['#34D399'],
                        template='plotly_dark',
                    )
                    fig_mastery.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_family='Nunito',
                        title_font_size=15,
                    )
                    fig_mastery.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
                    fig_mastery.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
                    st.plotly_chart(fig_mastery, use_container_width=True)
            else:
                st.info("No quiz sessions recorded yet. Complete a quiz to see your analytics.")

            # --- Advanced ML analytics (optional, in-memory personalization engine) ---
            adv_ok = False
            try:
                r = requests.get(f"http://localhost:8001/api/analytics/user/{st.session_state.user_id}", timeout=8)
                adv_ok = r.status_code == 200
            except Exception:
                pass
            if adv_ok:
                data = r.json().get('user_analytics', {})
                if data and 'error' not in data:
                    with st.expander("Advanced ML Profile", expanded=False):
                        top_topics = data.get('top_topics', [])
                        if top_topics:
                            if isinstance(top_topics[0], dict):
                                df_top = pd.DataFrame(top_topics)
                                if 'name' in df_top.columns and 'importance' in df_top.columns:
                                    df_top.rename(columns={'name': 'Topic', 'importance': 'Mastery'}, inplace=True)
                            else:
                                df_top = pd.DataFrame(top_topics, columns=["Topic", "Mastery"])
                            fig_top = px.bar(df_top, x=df_top.columns[0], y=df_top.columns[1],
                                             title="Top Topics by Mastery",
                                             color_discrete_sequence=['#0EA5E9'],
                                             template='plotly_dark')
                            fig_top.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_family='Nunito',
                                title_font_size=15,
                            )
                            fig_top.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
                            fig_top.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
                            st.plotly_chart(fig_top, use_container_width=True)

    with tab5:
        # Import and use the enhanced memorial
        from peatlearn.adaptive.enhanced_memorial import render_enhanced_memorial
        render_enhanced_memorial()

if __name__ == "__main__":
    main()
