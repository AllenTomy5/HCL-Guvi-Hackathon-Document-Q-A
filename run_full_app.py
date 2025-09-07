#!/usr/bin/env python3
"""
Script to run DocuMind AI with full functionality
Loads environment variables and starts the Streamlit app
"""

import os
import subprocess
import sys
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Environment variables loaded from .env file")
    else:
        print("⚠️ No .env file found. Please create one with your API keys.")

def check_api_keys():
    """Check if API keys are available"""
    openai_key = os.getenv('OPENAI_API_KEY')
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    
    if openai_key and openai_key != 'your_openai_api_key_here':
        print("✅ OpenAI API key found")
        return True
    elif hf_key and hf_key != 'your_huggingface_api_key_here':
        print("✅ Hugging Face API key found")
        return True
    else:
        print("⚠️ No valid API keys found. App will run in demo mode.")
        print("Please add your API keys to the .env file:")
        print("  - OPENAI_API_KEY=your_actual_key_here")
        print("  - HUGGINGFACE_API_KEY=your_actual_key_here")
        return False

def main():
    """Main function to run the app"""
    print("🚀 Starting DocuMind AI with full functionality...")
    
    # Load environment variables
    load_env_file()
    
    # Check API keys
    has_keys = check_api_keys()
    
    if has_keys:
        print("🎉 Running with full AI functionality!")
    else:
        print("📝 Running in demo mode. Add API keys for full functionality.")
    
    print("🌐 Starting Streamlit app...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the app")
    print("-" * 50)
    
    # Run Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
