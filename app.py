import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from ingestion import parse_document
from chunking import chunk_text
from embedding import embed_chunks, embed_query
from vector_store import VectorStore
from retrieval import retrieve_relevant_chunks
from rag_pipeline import generate_answer
import requests
import os

def ask_llm_cloud(prompt, model="gpt-3.5-turbo"):
    """
    Call cloud-based LLM APIs with smart fallback
    """
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if openai_key:
        try:
            return ask_openai(prompt, model, openai_key)
        except:
            pass
    
    # Try Hugging Face
    hf_key = os.getenv("HUGGINGFACE_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY")
    if hf_key:
        try:
            return ask_huggingface(prompt, hf_key)
        except:
            pass
    
    # Try local Ollama
    try:
        return ask_ollama(prompt)
    except:
        pass
    
    # Fallback to demo response
    return f"🤖 [Demo Mode] Here's a sample response to your question: {prompt[:100]}...\n\nNote: Add API keys (OpenAI or Hugging Face) for full AI functionality, or run locally with Ollama."

def ask_openai(prompt, model, api_key):
    """Call OpenAI API"""
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
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenAI API error: {response.status_code}")

def ask_huggingface(prompt, api_key):
    """Call Hugging Face API"""
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"inputs": prompt}
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No response generated")
        return str(result)
    else:
        raise Exception(f"Hugging Face API error: {response.status_code}")

def ask_ollama(prompt):
    """Call local Ollama API"""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data, timeout=60)
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No response generated")
    else:
        raise Exception(f"Ollama API error: {response.status_code}")

def check_llm_availability():
    """Check which LLM services are available"""
    available = []
    
    if os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY"):
        available.append("OpenAI")
    if os.getenv("HUGGINGFACE_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY"):
        available.append("Hugging Face")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            available.append("Ollama (Local)")
    except:
        pass
    
    return available

# Page configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_modern_css(theme='light'):
    if theme == 'dark':
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            .stApp {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                font-family: 'Inter', sans-serif;
                color: #ffffff;
            }
            
            .stApp > header { display: none; }
            .stApp > div[data-testid="stToolbar"] { display: none; }
            .stApp > div[data-testid="stDecoration"] { display: none; }
            
            .css-1d391kg {
                background-color: #0f0f23 !important;
                border-right: 1px solid #333 !important;
            }
            
            .css-1d391kg .stMarkdown {
                color: #ffffff !important;
            }
            
            .css-1d391kg .stSelectbox label {
                color: #ffffff !important;
            }
            
            .css-1d391kg .stSlider label {
                color: #ffffff !important;
            }
            
            .css-1d391kg .stButton button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
            }
            
            .main .block-container {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stTextInput > div > div > input {
                background-color: rgba(255, 255, 255, 0.1) !important;
                color: #ffffff !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #667eea !important;
                background-color: rgba(255, 255, 255, 0.15) !important;
            }
            
            .stFileUploader > div > div > div > div > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 2rem !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            
            .stFileUploader > div > div > div > div > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
            }
            
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
            }
            
            .ai-message {
                background: rgba(255, 255, 255, 0.1) !important;
                color: #ffffff !important;
                border-left: 4px solid #667eea !important;
            }
            
            .stSuccess {
                background-color: rgba(34, 197, 94, 0.1) !important;
                border: 1px solid rgba(34, 197, 94, 0.3) !important;
                color: #22c55e !important;
            }
            
            .stError {
                background-color: rgba(239, 68, 68, 0.1) !important;
                border: 1px solid rgba(239, 68, 68, 0.3) !important;
                color: #ef4444 !important;
            }
            
            .stInfo {
                background-color: rgba(59, 130, 246, 0.1) !important;
                border: 1px solid rgba(59, 130, 246, 0.3) !important;
                color: #3b82f6 !important;
            }
            
            .stWarning {
                background-color: rgba(245, 158, 11, 0.1) !important;
                border: 1px solid rgba(245, 158, 11, 0.3) !important;
                color: #f59e0b !important;
            }
            
            .stExpander {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }
            
            .stExpander > div > div > div {
                color: white !important;
            }
            
            .stSelectbox > div > div > select {
                background-color: rgba(255, 255, 255, 0.1) !important;
                color: #ffffff !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
            }
            
            .stSlider > div > div > div > div {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            }
            
            .metric-container {
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 12px !important;
                padding: 1rem !important;
            }
            
            .upload-area {
                border: 2px dashed #667eea !important;
                border-radius: 16px !important;
                padding: 3rem 2rem !important;
                text-align: center !important;
                background: rgba(102, 126, 234, 0.05) !important;
                transition: all 0.3s ease !important;
                margin: 2rem 0 !important;
            }
            
            .upload-area:hover {
                border-color: #764ba2 !important;
                background: rgba(118, 75, 162, 0.1) !important;
            }
            
            .question-form {
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 16px !important;
                padding: 2rem !important;
                margin: 2rem 0 !important;
            }
            
            .answer-container {
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 16px !important;
                padding: 2rem !important;
                margin: 2rem 0 !important;
            }
            
            .stats-container {
                display: flex !important;
                justify-content: space-around !important;
                margin: 2rem 0 !important;
            }
            
            .stat-item {
                text-align: center !important;
                padding: 1rem !important;
            }
            
            .stat-number {
                font-size: 2rem !important;
                font-weight: 700 !important;
                color: #667eea !important;
            }
            
            .stat-label {
                font-size: 0.9rem !important;
                color: #a0a0a0 !important;
                margin-top: 0.5rem !important;
            }
        </style>
        """
    else:
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            .stApp {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                font-family: 'Inter', sans-serif;
                color: #2c3e50;
            }
            
            .stApp > header { display: none; }
            .stApp > div[data-testid="stToolbar"] { display: none; }
            .stApp > div[data-testid="stDecoration"] { display: none; }
            
            .css-1d391kg {
                background-color: #ffffff !important;
                border-right: 1px solid #dee2e6 !important;
            }
            
            .css-1d391kg .stMarkdown {
                color: #2c3e50 !important;
            }
            
            .css-1d391kg .stSelectbox label {
                color: #2c3e50 !important;
            }
            
            .css-1d391kg .stSlider label {
                color: #2c3e50 !important;
            }
            
            .css-1d391kg .stButton button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
            }
            
            .main .block-container {
                background-color: transparent !important;
                color: #2c3e50 !important;
            }
            
            .stTextInput > div > div > input {
                background-color: rgba(255, 255, 255, 0.8) !important;
                color: #2c3e50 !important;
                border: 2px solid rgba(102, 126, 234, 0.3) !important;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #667eea !important;
                background-color: rgba(255, 255, 255, 0.9) !important;
            }
            
            .stFileUploader > div > div > div > div > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 0.75rem 2rem !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            
            .stFileUploader > div > div > div > div > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
            }
            
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
            }
            
            .ai-message {
                background: rgba(255, 255, 255, 0.8) !important;
                color: #2c3e50 !important;
                border-left: 4px solid #667eea !important;
            }
            
            .stSuccess {
                background-color: rgba(34, 197, 94, 0.1) !important;
                border: 1px solid rgba(34, 197, 94, 0.3) !important;
                color: #22c55e !important;
            }
            
            .stError {
                background-color: rgba(239, 68, 68, 0.1) !important;
                border: 1px solid rgba(239, 68, 68, 0.3) !important;
                color: #ef4444 !important;
            }
            
            .stInfo {
                background-color: rgba(59, 130, 246, 0.1) !important;
                border: 1px solid rgba(59, 130, 246, 0.3) !important;
                color: #3b82f6 !important;
            }
            
            .stWarning {
                background-color: rgba(245, 158, 11, 0.1) !important;
                border: 1px solid rgba(245, 158, 11, 0.3) !important;
                color: #f59e0b !important;
            }
            
            .stExpander {
                background-color: rgba(255, 255, 255, 0.8) !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
            }
            
            .stExpander > div > div > div {
                color: #2c3e50 !important;
            }
            
            .stSelectbox > div > div > select {
                background-color: rgba(255, 255, 255, 0.8) !important;
                color: #2c3e50 !important;
                border: 1px solid rgba(0, 0, 0, 0.2) !important;
            }
            
            .stSlider > div > div > div > div {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            }
            
            .metric-container {
                background: rgba(255, 255, 255, 0.8) !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                border-radius: 12px !important;
                padding: 1rem !important;
            }
            
            .upload-area {
                border: 2px dashed #667eea !important;
                border-radius: 16px !important;
                padding: 3rem 2rem !important;
                text-align: center !important;
                background: rgba(102, 126, 234, 0.05) !important;
                transition: all 0.3s ease !important;
                margin: 2rem 0 !important;
            }
            
            .upload-area:hover {
                border-color: #764ba2 !important;
                background: rgba(118, 75, 162, 0.1) !important;
            }
            
            .question-form {
                background: rgba(255, 255, 255, 0.8) !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                border-radius: 16px !important;
                padding: 2rem !important;
                margin: 2rem 0 !important;
            }
            
            .answer-container {
                background: rgba(255, 255, 255, 0.8) !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                border-radius: 16px !important;
                padding: 2rem !important;
                margin: 2rem 0 !important;
            }
            
            .stats-container {
                display: flex !important;
                justify-content: space-around !important;
                margin: 2rem 0 !important;
            }
            
            .stat-item {
                text-align: center !important;
                padding: 1rem !important;
            }
            
            .stat-number {
                font-size: 2rem !important;
                font-weight: 700 !important;
                color: #667eea !important;
            }
            
            .stat-label {
                font-size: 0.9rem !important;
                color: #6c757d !important;
                margin-top: 0.5rem !important;
            }
        </style>
        """

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_document' not in st.session_state:
    st.session_state.uploaded_document = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Apply CSS
st.markdown(get_modern_css(st.session_state.theme), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 DocuMind AI")
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📋 Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()
    
    if st.session_state.uploaded_document:
        if st.button("📄 Document View", use_container_width=True):
            st.session_state.current_page = 'document'
            st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    # LLM Status
    available_llms = check_llm_availability()
    if available_llms:
        st.success(f"✅ Connected to: {', '.join(available_llms)}")
    else:
        st.warning("⚠️ Running in demo mode. Add API keys for full functionality.")
    
    # AI Model selection
    model_options = []
    if "OpenAI" in available_llms:
        model_options.extend(["gpt-3.5-turbo", "gpt-4"])
    if "Hugging Face" in available_llms:
        model_options.append("huggingface")
    if "Ollama (Local)" in available_llms:
        model_options.extend(["llama3.2:3b", "llama3.2:1b", "phi3:mini"])
    
    if not model_options:
        model_options = ["demo-mode"]
    
    model_choice = st.selectbox(
        "AI Model",
        model_options,
        help="Select the AI model for generating answers"
    )
    
    # Advanced settings
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, help="Size of text chunks for processing")
    overlap = st.slider("Chunk Overlap", 0, 200, 50, help="Overlap between chunks")
    top_k = st.slider("Top K Results", 1, 10, 3, help="Number of relevant chunks to retrieve")

def home_page():
    """Display the home page"""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.5rem;">🧠 DocuMind AI</h1>
        <p style="font-size: 1.2rem; color: #6c757d; font-weight: 400;">Intelligent Document Analysis & Q&A Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("### 🚀 Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
            <h3 style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Smart Document Upload</h3>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.9rem;">Upload PDFs, text files, and more. Our AI automatically processes and understands your content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <h3 style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Analytics & Insights</h3>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.9rem;">Get detailed analytics about your document structure, content, and usage patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
            <h3 style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Intelligent Q&A</h3>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.9rem;">Ask questions in natural language and get accurate answers based on your document content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
            <h3 style="font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Advanced Search</h3>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.9rem;">Find specific information quickly with semantic search and keyword matching.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload section with proper file uploader
    st.markdown("### 📁 Upload Your Document")
    
    # Create a custom upload area
    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=['pdf', 'txt', 'md', 'html'],
        help="Supported formats: PDF, TXT, Markdown, HTML",
        label_visibility="visible"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        with st.spinner("🔄 Processing your document..."):
            # Parse document
            text = parse_document(uploaded_file)
            
            # Chunk text with custom settings
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            # Generate embeddings
            embeddings = embed_chunks(chunks)
            
            # Add to vector store
            st.session_state.vector_store.add_embeddings(chunks, embeddings)
            
            # Store document info
            st.session_state.uploaded_document = {
                'name': uploaded_file.name,
                'size': len(text),
                'chunks': len(chunks),
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        st.success("✅ Document processed successfully!")
        
        # Show document analysis
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Chunks", len(chunks))
        with col2:
            st.metric("📏 Characters", f"{len(text):,}")
        with col3:
            st.metric("📊 Avg Chunk Size", f"{len(text) // len(chunks) if chunks else 0:,}")
        with col4:
            st.metric("⏱️ Processing Time", "< 1s")
        
        # Auto-navigate to document view
        st.session_state.current_page = 'document'
        st.rerun()

def document_page():
    """Display the document analysis page"""
    if not st.session_state.uploaded_document:
        st.warning("⚠️ Please upload a document first!")
        st.session_state.current_page = 'home'
        st.rerun()
        return
    
    st.markdown("## 📄 Document Analysis")
    
    # Document info
    doc = st.session_state.uploaded_document
    st.info(f"📄 **Document:** {doc['name']} | 📝 **Chunks:** {doc['chunks']} | 📏 **Size:** {doc['size']:,} chars")
    
    # Quick question buttons
    st.markdown("### 🚀 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Summarize Document", use_container_width=True):
            ask_question("Please provide a comprehensive summary of this document.")
    
    with col2:
        if st.button("🔍 Key Points", use_container_width=True):
            ask_question("What are the main key points in this document?")
    
    with col3:
        if st.button("❓ Explain Concepts", use_container_width=True):
            ask_question("Explain the main concepts discussed in this document.")
    
    # Custom question input
    st.markdown("### 💬 Ask a Custom Question")
    question = st.text_input(
        "",
        placeholder="Ask anything about your document...",
        key="question_input",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Get Answer", use_container_width=True):
            if question:
                ask_question(question)
            else:
                st.warning("Please enter a question!")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 💭 Chat History")
        for entry in reversed(st.session_state.chat_history[-5:]):  # Show last 5
            st.markdown(f"""
            <div class="user-message" style="padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>👤 You ({entry['timestamp']}):</strong> {entry['question']}
            </div>
            <div class="ai-message" style="padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <strong>🤖 AI:</strong> {entry['answer'][:200]}{'...' if len(entry['answer']) > 200 else ''}
            </div>
            """, unsafe_allow_html=True)

def ask_question(question):
    """Ask a question and get an answer"""
    with st.spinner("🧠 AI is thinking..."):
        # Generate query embedding
        query_embedding = embed_query(question)
        
        # Retrieve relevant chunks
        top_chunks = retrieve_relevant_chunks(query_embedding, st.session_state.vector_store, top_k=top_k)
        
        # Create context from chunks
        context = "\n\n".join(top_chunks) if top_chunks else "No relevant context found."
        
        # Create enhanced prompt
        enhanced_prompt = f"""
Based on the following document context, please answer the question: {question}

Document Context:
{context}

Please provide a comprehensive and accurate answer based on the document content. If the answer cannot be found in the context, please state that clearly.
"""
        
        # Generate answer using cloud LLM
        start_time = time.time()
        answer = ask_llm_cloud(enhanced_prompt, model_choice)
        response_time = time.time() - start_time
        
        # Store in chat history
        chat_entry = {
            'question': question,
            'answer': answer,
            'references': top_chunks,
            'response_time': response_time,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.chat_history.append(chat_entry)
        
        st.rerun()

# Main app logic
if st.session_state.current_page == 'home':
    home_page()
else:
    document_page()