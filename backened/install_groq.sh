#!/bin/bash

echo "🚀 Installing Groq dependencies..."

# Activate virtual environment
source venv/bin/activate

# Install/upgrade required packages
pip install --upgrade langchain-groq groq

echo "✅ Groq dependencies installed!"
echo "🔧 Testing Groq connection..."

# Test Groq API
python -c "
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
try:
    llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name='mixtral-8x7b-32768')
    response = llm.invoke('Hello, this is a test.')
    print('✅ Groq API working!')
    print(f'Response: {response.content[:100]}...')
except Exception as e:
    print(f'❌ Groq API error: {e}')
"

echo "🚀 Starting server..."
python main.py 