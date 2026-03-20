"""
PeatLearn Diagnostic Tool
Run this to check your setup and identify issues
"""
import os
import sys
import subprocess
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_python():
    print_section("Python Environment")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
def check_venv():
    print_section("Virtual Environment")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"In virtual environment: {in_venv}")
    if in_venv:
        print(f"✅ Virtual environment active")
    else:
        print(f"⚠️  Not in virtual environment - this might cause issues")

def check_env_file():
    print_section(".env File")
    env_path = Path('.env')
    if env_path.exists():
        print("✅ .env file exists")
        with open(env_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '=' in line and not line.strip().startswith('#'):
                    key = line.split('=')[0].strip()
                    value = line.split('=')[1].strip()
                    if value and value != 'your_key_here':
                        print(f"  ✅ {key}: Set")
                    else:
                        print(f"  ❌ {key}: NOT SET")
    else:
        print("❌ .env file NOT FOUND")
        print("  Create .env from .env.template or .env.example")

def check_packages():
    print_section("Required Packages")
    required = [
        'streamlit',
        'fastapi',
        'uvicorn',
        'pinecone-client',
        'google-generativeai',
        'python-dotenv',
        'pandas',
        'numpy'
    ]
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")

def check_ports():
    print_section("Port Availability")
    import socket
    
    ports = {8000: "FastAPI Backend", 8501: "Streamlit", 8001: "Advanced ML Service"}
    
    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"  ⚠️  Port {port} ({service}) - IN USE")
        else:
            print(f"  ✅ Port {port} ({service}) - Available")

def check_directories():
    print_section("Project Structure")
    required_dirs = [
        'data',
        'inference/backend',
        'embedding',
        'preprocessing',
        'web_ui/frontend',
        'config'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")

def check_key_files():
    print_section("Key Files")
    required_files = [
        'requirements.txt',
        'peatlearn_master.py',
        'inference/backend/app.py',
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")

def check_api_connectivity():
    print_section("API Connectivity")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Gemini
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            print("  ✅ Gemini API: Key configured")
        except Exception as e:
            print(f"  ❌ Gemini API: Error - {str(e)[:50]}")
    else:
        print("  ❌ Gemini API: Key not found")
    
    # Check Pinecone
    pinecone_key = os.getenv('PINECONE_API_KEY')
    if pinecone_key:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_key)
            print("  ✅ Pinecone API: Key configured")
        except Exception as e:
            print(f"  ❌ Pinecone API: Error - {str(e)[:50]}")
    else:
        print("  ❌ Pinecone API: Key not found")

def main():
    print("\n" + "="*60)
    print("  🔍 PeatLearn Diagnostic Tool")
    print("="*60)
    
    try:
        check_python()
        check_venv()
        check_env_file()
        check_packages()
        check_ports()
        check_directories()
        check_key_files()
        check_api_connectivity()
        
        print("\n" + "="*60)
        print("  Diagnostic Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Fix any ❌ issues above")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check your .env file has valid API keys")
        print("4. Try running: streamlit run peatlearn_master.py")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
