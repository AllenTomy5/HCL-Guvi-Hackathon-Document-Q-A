"""
LLM interface module for cloud deployment.
Handles calls to cloud-based LLM APIs (OpenAI, Hugging Face, etc.).
"""

import os
import requests
import streamlit as st

def ask_llm(prompt, model="gpt-3.5-turbo"):
    """
    Call a cloud-based LLM API with the given prompt and return the response.
    Args:
        prompt (str): The prompt to send to the LLM
        model (str): The model to use (default: gpt-3.5-turbo)
    Returns:
        str: LLM-generated answer
    """
    
    # Try OpenAI first
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return ask_openai(prompt, model, openai_api_key)
    
    # Try Hugging Face
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if hf_api_key:
        return ask_huggingface(prompt, hf_api_key)
    
    # Fallback to demo mode
    return f"[Demo Mode] Here's a sample response to your question: {prompt[:100]}..."

def ask_openai(prompt, model, api_key):
    """Call OpenAI API"""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"OpenAI API error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"

def ask_huggingface(prompt, api_key):
    """Call Hugging Face API"""
    try:
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"inputs": prompt}
        
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            return str(result)
        else:
            return f"Hugging Face API error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Error calling Hugging Face: {str(e)}"

def check_api_availability():
    """Check which APIs are available"""
    available = []
    
    if os.getenv("OPENAI_API_KEY"):
        available.append("OpenAI")
    if os.getenv("HUGGINGFACE_API_KEY"):
        available.append("Hugging Face")
    
    return available
