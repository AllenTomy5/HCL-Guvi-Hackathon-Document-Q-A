"""
LLM interface module.
Handles calls to local LLMs via Ollama REST API.
"""

import requests

def ask_llm(prompt, model="llama3.2:3b"):
    """
    Call the local Ollama LLM API with the given prompt and return the response.
    Args:
        prompt (str): The prompt to send to the LLM
        model (str): The model to use (default: llama3.2:3b)
    Returns:
        str: LLM-generated answer
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code} {response.text}")
    result = response.json()
    return result.get("response", "[Error: No response from LLM]")
