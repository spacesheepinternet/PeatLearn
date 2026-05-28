#!/bin/bash
# Launch PeatLearn in Production Mode
# This script runs without development features for security

echo "🚀 Launching PeatLearn in Production Mode..."
echo "🔒 Development features disabled for security"
echo ""

# Ensure no development environment variables are set
unset PEATLEARN_DEV_MODE
unset STREAMLIT_DEV_MODE

# Launch in production mode (default)
streamlit run app/dashboard.py
