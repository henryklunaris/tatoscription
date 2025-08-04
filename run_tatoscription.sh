#!/bin/bash

# Tatoscription Voice Dictation Launcher
# This script can be run from anywhere to start the voice dictation tool

echo "🎙️ Starting Tatoscription Voice Dictation Tool..."

# Navigate to the project directory
cd "/Users/henryk/Cursor Projects/tatoscription/tatoscription"

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found! Please make sure you have set up your OpenAI API key."
    echo "Create a .env file with: OPENAI_API_KEY=your-api-key-here"
    exit 1
fi

# Run the application
python start.py 