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
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }
    
    .chat-message:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #667eea;
        color: white;
        margin-left: 2rem;
        border-bottom-right-radius: 0;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        margin-right: 2rem;
        border-bottom-left-radius: 0;
    }
    
    .rag-answer h3 {
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.2em;
    }
    
    .rag-answer p {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
        line-height: 1.6;
    }
    
    .rag-answer ul {
        padding-left: 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .rag-answer li {
        margin-bottom: 0.3rem;
    }
    
    /* Sources styling with hover effect */
    .sources-container {
        margin-top: 1.5rem;
        border-top: 1px solid #eaeaea;
        padding-top: 1rem;
    }
    
    .sources-toggle {
        cursor: pointer;
        color: #667eea;
        font-weight: bold;
        display: inline-block;
        padding: 0.3rem 0.5rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    
    .sources-toggle:hover {
        background-color: #f0f2f5;
    }
    
    .sources-content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
        margin-top: 0.5rem;
    }
    
    .sources-content ul {
        padding-left: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .sources-content li {
        margin-bottom: 0.4rem;
        font-size: 0.9em;
        color: #555;
    }
    
    /* When sources are expanded */
    .sources-container:hover .sources-content {
        max-height: 500px;
        transition: max-height 0.5s ease-in;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .profile-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .mastery-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .struggling { background-color: #ffebee; color: #c62828; }
    .learning { background-color: #fff3e0; color: #ef6c00; }
    .advanced { background-color: #e8f5e8; color: #2e7d32; }
    
    /* Feedback buttons */
    .stButton button {
        padding: 0.2rem 0.5rem;
        font-size: 1rem;
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
    return body, sources

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
    # Run on letter-only string to avoid punctuation (dots, dashes) padding the count.
    # Explicitly enumerate consonants so y is treated as a vowel (avoids "trypt" false-positive).
    if re.search(r'[b-df-hj-np-tv-xz]{5,}', letters):
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
        st.header("👤 User Setup")
        
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
            st.write(f"**Current User:** {st.session_state.user_id}")
            # Ensure session_id is initialized
            if not st.session_state.get('session_id'):
                try:
                    # Lazily initialize session id if missing
                    DataLogger().get_session_id()
                except Exception:
                    pass
            if st.session_state.session_id:
                st.write(f"**Session:** {st.session_state.session_id[:8]}...")
            else:
                st.write("**Session:** Not initialized")
            
            # Show AI and personalization backend status
            api_key = os.getenv('GEMINI_API_KEY')
            adv_ok = False
            adv_ok = _adv_health_cached()
            cols = st.columns(2)
            with cols[0]:
                if api_key:
                    st.success("🤖 AI Analysis: Enabled")
                else:
                    st.warning("🤖 AI Analysis: Using Fallback Mode")
            with cols[1]:
                if adv_ok:
                    st.success("🧩 Personalization API: Connected")
                else:
                    st.warning("🧩 Personalization API: Basic Mode")
            
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

def render_user_profile():
    """Render user profile and learning analytics"""
    if not st.session_state.user_profile:
        st.info("Start chatting to build your learning profile! 📈")
        return
    
    profile = st.session_state.user_profile
    
    st.subheader("📊 Your Learning Profile")
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Learning State", profile.get('overall_state', 'new').title())
    
    with col2:
        st.metric("Learning Style", profile.get('learning_style', 'balanced').replace('_', ' ').title())
    
    with col3:
        st.metric("Total Interactions", profile.get('total_interactions', 0))
    
    with col4:
        avg_feedback = profile.get('average_feedback', 0)
        st.metric("Avg Feedback", f"{avg_feedback:.1f}" if avg_feedback else "N/A")
    
    # Topic mastery
    topic_mastery = profile.get('topic_mastery', {})
    if topic_mastery:
        st.subheader("🎯 Topic Mastery")
        
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
                    title="Mastery Levels by Topic",
                    color_discrete_map={
                        'struggling': '#ff6b6b',
                        'learning': '#feca57', 
                        'advanced': '#48db71'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Show mastery badges
        for _, row in df.iterrows():
            badge_class = row['State']
            st.markdown(f"""
                <span class="mastery-badge {badge_class}">
                    {row['Topic']}: {row['State'].title()} ({row['Mastery Level']:.1f})
                </span>
            """, unsafe_allow_html=True)
    
    # AI insights if available
    ai_analysis = profile.get('ai_analysis', {})
    if ai_analysis:
        st.subheader("🤖 AI Insights")
        
        insights = ai_analysis.get('insights', [])
        for insight in insights:
            st.write(f"• {insight}")
        
        learning_velocity = ai_analysis.get('learning_velocity')
        if learning_velocity:
            st.info(f"**Learning Velocity:** {learning_velocity.title()}")

def render_recommendations():
    """Render personalized recommendations via backend."""
    if not st.session_state.get('user_id'):
        return
    st.subheader("💡 Personalized Recommendations")
    try:
        topic_mastery = (st.session_state.user_profile or {}).get('topic_mastery', {})
        topic_filter = list(topic_mastery.keys())[:5] if topic_mastery else None
        recs = _fetch_recommendations_cached(st.session_state.user_id, topic_filter, 8)
        if True:
            if not recs:
                st.info("No recommendations yet. Start interacting to personalize.")
                return
            for rec in recs:
                title = rec.get('title') or rec.get('content_id')
                reason = rec.get('recommendation_reason', '')
                snippet = rec.get('snippet', '')
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4 style="color:#ff6b6b;">{title}</h4>
                        <p>{snippet}</p>
                        {f"<small><i>{reason}</i></small>" if reason else ""}
                    </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"Recommendations unavailable: {e}")

def render_chat_interface():
    """Render the main chat interface with AI profiling"""
    data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()
    
    st.subheader("💬 Chat with Ray Peat AI")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            body_md, sources = _split_answer_and_sources(message['content'])
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Ray Peat AI:</strong>
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
            with st.spinner("Ray Peat AI is thinking..."):
                response = _get_rag_system().get_rag_response(resolved_query, st.session_state.user_profile)
        
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
    st.subheader("🎯 Personalized Quiz")

    # Debug toggle
    debug_mode = st.toggle("Show adaptive debug info", value=False, help="Displays ability, item difficulty and target anchors.")

    # Session-based quiz flow using new endpoints
    if st.session_state.get('quiz_active') and st.session_state.get('quiz_session_id'):
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
        st.write(f"**Question:** {item.get('stem','')}")
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
            st.markdown(f"<div class='metric-card'><small><strong>Passage</strong></small><br/>{html.escape(ctx)}</div>", unsafe_allow_html=True)
        if src:
            st.caption(f"Source: {src}")
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
        st.info(f"Last quiz: {res.get('correct',0)}/{res.get('total',0)} correct ({res.get('score_percentage',0):.1f}%)")
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
    
    # Check if user is set up
    if not st.session_state.user_id:
        render_user_setup()
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
        st.subheader("📈 Learning Analytics")
        if not st.session_state.user_id:
            st.info("Enter your user ID to view analytics.")
        else:
            ok = False
            try:
                r = requests.get(f"http://localhost:8001/api/analytics/user/{st.session_state.user_id}", timeout=8)
                ok = r.status_code == 200
            except Exception:
                ok = False
            if ok:
                data = r.json().get('user_analytics', {})
                if 'error' in data:
                    st.info("No analytics yet. Interact more to build your profile.")
                else:
                    cols = st.columns(3)
                    cols[0].metric("Avg Mastery", f"{data.get('average_mastery',0):.2f}")
                    cols[1].metric("Learning Velocity", f"{data.get('learning_velocity',0):.2f}")
                    cols[2].metric("Preferred Difficulty", f"{data.get('preferred_difficulty',0):.2f}")
                    top_topics = data.get('top_topics', [])
                    if top_topics:
                        if isinstance(top_topics[0], dict):
                            df_top = pd.DataFrame(top_topics)
                            if 'name' in df_top.columns and 'importance' in df_top.columns:
                                df_top.rename(columns={'name':'Topic','importance':'Mastery'}, inplace=True)
                        else:
                            df_top = pd.DataFrame(top_topics, columns=["Topic","Mastery"]) 
                        st.bar_chart(df_top.set_index(df_top.columns[0]))
            else:
                st.info("Analytics service unavailable. Showing local session stats.")
                data_logger, ai_profiler, content_selector, quiz_generator, dashboard, rag_system, topic_model = init_adaptive_system()
                all_interactions = data_logger._load_interactions()
                user_interactions = all_interactions.loc[all_interactions['user_id'] == st.session_state.user_id].copy()
                if not user_interactions.empty:
                    user_interactions['timestamp'] = pd.to_datetime(user_interactions['timestamp'])
                    daily = user_interactions.groupby(user_interactions['timestamp'].dt.date).size()
                    st.line_chart(daily)

    with tab5:
        # Import and use the enhanced memorial
        from peatlearn.adaptive.enhanced_memorial import render_enhanced_memorial
        render_enhanced_memorial()

if __name__ == "__main__":
    main()
