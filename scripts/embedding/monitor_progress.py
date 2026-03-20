#!/usr/bin/env python3
"""
Ray Peat Legacy - Embedding Progress Monitor

This script monitors the embedding generation process and provides real-time status updates.
"""

import os
import sys
import time
import psutil
import json
from pathlib import Path
from datetime import datetime, timedelta

def find_embedding_process():
    """Find the running embedding process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'embed_corpus.py' in ' '.join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def check_output_files():
    """Check for any output files created by the embedding process."""
    embedding_dir = Path("embedding")
    vectors_dir = embedding_dir / "vectors"
    
    files_found = {}
    
    # Check for any files in embedding directory
    for file_path in embedding_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.pkl', '.npy', '.json', '.log']:
            stat = file_path.stat()
            files_found[str(file_path)] = {
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime)
            }
    
    return files_found

def check_network_activity():
    """Check if there's network activity indicating API calls."""
    try:
        proc = find_embedding_process()
        if proc:
            connections = proc.connections()
            active_connections = [conn for conn in connections if conn.status == 'ESTABLISHED']
            return len(active_connections)
    except:
        pass
    return 0

def monitor_logs():
    """Monitor recent log output."""
    # Check if there are any recent log files
    log_files = []
    for log_path in Path(".").rglob("*.log"):
        if log_path.stat().st_mtime > time.time() - 3600:  # Modified in last hour
            log_files.append(log_path)
    
    return log_files

def main():
    """Main monitoring function."""
    print("🔍 Ray Peat Embedding Process Monitor")
    print("=" * 50)
    
    # Check if process is running
    proc = find_embedding_process()
    if proc:
        print(f"✅ Embedding process found (PID: {proc.pid})")
        
        # Get process info
        try:
            cpu_percent = proc.cpu_percent()
            memory_mb = proc.memory_info().rss / 1024 / 1024
            create_time = datetime.fromtimestamp(proc.create_time())
            runtime = datetime.now() - create_time
            
            print(f"   📊 CPU Usage: {cpu_percent:.1f}%")
            print(f"   💾 Memory: {memory_mb:.1f} MB")
            print(f"   ⏱️  Runtime: {runtime}")
            
        except Exception as e:
            print(f"   ⚠️  Could not get process stats: {e}")
    else:
        print("❌ No embedding process found")
        return
    
    print("\n" + "=" * 50)
    
    # Check output files
    print("📁 Checking output files...")
    output_files = check_output_files()
    
    if output_files:
        print(f"✅ Found {len(output_files)} output files:")
        for file_path, info in output_files.items():
            size_kb = info['size'] / 1024
            age = datetime.now() - info['modified']
            print(f"   📄 {file_path}")
            print(f"      Size: {size_kb:.1f} KB")
            print(f"      Last modified: {age} ago")
    else:
        print("⚠️  No output files found yet")
    
    print("\n" + "=" * 50)
    
    # Check network activity
    print("🌐 Checking network activity...")
    connections = check_network_activity()
    if connections > 0:
        print(f"✅ {connections} active network connections (API calls)")
    else:
        print("⚠️  No active network connections detected")
    
    print("\n" + "=" * 50)
    
    # Recommendations
    print("💡 Status Summary:")
    
    if proc and connections > 0:
        print("✅ Process appears to be working correctly")
        print("   - Embedding script is running")
        print("   - Making API calls to Gemini")
        print("   - Monitor for output files to appear")
    elif proc and connections == 0:
        print("⚠️  Process running but no network activity")
        print("   - May be processing locally or rate-limited")
        print("   - Check if API key is valid")
    else:
        print("❌ Process not running or has issues")
    
    print("\n📋 To manually check progress:")
    print("   tail -f /path/to/log/file  # if log files exist")
    print("   ps aux | grep embed_corpus")
    print("   ls -la embedding/vectors/")

if __name__ == "__main__":
    main() 