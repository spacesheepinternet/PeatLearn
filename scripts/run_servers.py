#!/usr/bin/env python3
"""
PeatLearn Development Server Launcher
Starts both the backend API and Streamlit frontend in the virtual environment
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import signal

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

def find_venv_python():
    """Find the Python executable in the virtual environment."""
    venv_path = PROJECT_ROOT / "venv"

    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    if python_exe.exists():
        return str(python_exe)
    else:
        print("ERROR: Virtual environment not found. Please run: python -m venv venv")
        sys.exit(1)

def check_embeddings():
    """Check if embeddings are available, download if needed."""
    emb_file = PROJECT_ROOT / "data" / "embeddings" / "vectors" / "embeddings_20250728_221826.npy"

    if not emb_file.exists():
        print("Downloading embeddings from Hugging Face (one-time ~700MB)...")
        try:
            python_exe = find_venv_python()
            subprocess.run([
                python_exe, "peatlearn/embedding/hf_download.py"
            ], check=True)
            print("Embeddings downloaded successfully!")
        except subprocess.CalledProcessError:
            print("WARNING: Failed to download embeddings. Backend may start without them.")
    else:
        print("Embeddings already available.")

def main():
    print("PeatLearn Development Server")
    print("=" * 50)

    python_exe = find_venv_python()
    print(f"Using Python: {python_exe}")

    # Check embeddings
    check_embeddings()

    processes = []

    try:
        # Start basic RAG backend
        print("Starting RAG backend server (port 8000)...")
        backend_env = os.environ.copy()
        backend_env["PYTHONIOENCODING"] = "utf-8"
        backend_env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")

        backend_process = subprocess.Popen([
            python_exe, "-m", "uvicorn",
            "app.api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], env=backend_env, cwd=PROJECT_ROOT)
        processes.append(backend_process)

        # Start advanced ML backend
        print("Starting Advanced ML backend server (port 8001)...")
        advanced_process = subprocess.Popen([
            python_exe, "-m", "uvicorn",
            "app.advanced_api:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        ], env=backend_env, cwd=PROJECT_ROOT)
        processes.append(advanced_process)

        # Wait a moment for backends to start
        time.sleep(3)

        # Start Streamlit frontend
        print("Starting Streamlit frontend (port 8501)...")
        streamlit_env = backend_env.copy()
        streamlit_env["PYTHONPATH"] = str(PROJECT_ROOT)

        streamlit_process = subprocess.Popen([
            python_exe, "-m", "streamlit", "run",
            "app/dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.fileWatcherType", "none"
        ], env=streamlit_env, cwd=PROJECT_ROOT)
        processes.append(streamlit_process)

        print("\nDevelopment servers are running:")
        print("  RAG API:       http://localhost:8000")
        print("  RAG API Docs:  http://localhost:8000/docs")
        print("  Advanced ML:   http://localhost:8001")
        print("  ML API Docs:   http://localhost:8001/docs")
        print("  Dashboard:     http://localhost:8501")
        print("\nPress Ctrl+C to stop all servers")

        # Wait for processes
        while True:
            time.sleep(1)
            for process in processes:
                if process.poll() is not None:
                    print(f"WARNING: Process {process.pid} exited")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nShutting down servers...")

        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        print("All servers stopped.")

if __name__ == "__main__":
    main()
