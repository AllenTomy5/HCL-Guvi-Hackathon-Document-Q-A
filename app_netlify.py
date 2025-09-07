import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
from ingestion import parse_document
from chunking import chunk_text
from embedding import embed_chunks, embed_query
from vector_store import VectorStore
from retrieval import retrieve_relevant_chunks
from rag_pipeline import generate_answer
from llm_interface_cloud import ask_llm, check_api_availability

# Page configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_dark_css():
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
        
        .css-1d391kg .stTextInput label {
            color: #ffffff !important;
        }
        
        .css-1d391kg .stTextArea label {
            color: #ffffff !important;
        }
        
        .app-title {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            margin-bottom: 0.5rem !important;
            text-align: center !important;
        }
        
        .app-subtitle {
            font-size: 1.2rem !important;
            color: #a0a0a0 !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            margin: 1rem 0 !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.3s ease !important;
        }
        
        .feature-card:hover {
            transform: translateY(-5px) !important;
            border-color: #667eea !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3) !important;
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
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        .stTextArea > div > div > textarea {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        .stSelectbox > div > div > select {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: white !important;
        }
        
        .stSlider > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        .stExpander {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
        }
        
        .stExpander > div > div > div {
            color: white !important;
        }
        
        .stSuccess {
            background-color: rgba(34, 197, 94, 0.1) !important;
            border: 1px solid rgba(34, 197, 94, 0.3) !important;
            border-radius: 12px !important;
            color: #22c55e !important;
        }
        
        .stError {
            background-color: rgba(239, 68, 68, 0.1) !important;
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
            border-radius: 12px !important;
            color: #ef4444 !important;
        }
        
        .stInfo {
            background-color: rgba(59, 130, 246, 0.1) !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
            border-radius: 12px !important;
            color: #3b82f6 !important;
        }
        
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1) !important;
            border: 1px solid rgba(245, 158, 11, 0.3) !important;
            border-radius: 12px !important;
            color: #f59e0b !important;
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

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'document_name' not in st.session_state:
        st.session_state.document_name = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'api_status' not in st.session_state:
        st.session_state.api_status = check_api_availability()

def home_page():
    """Display the home page"""
    st.markdown(get_dark_css(), unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0;">
        <h1 class="app-title">🧠 DocuMind AI</h1>
        <p class="app-subtitle">Intelligent Document Analysis & Question Answering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status
    if st.session_state.api_status:
        st.success(f"✅ Connected to: {', '.join(st.session_state.api_status)}")
    else:
        st.warning("⚠️ Running in demo mode. Add API keys for full functionality.")
    
    # Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📄 Document Processing</h3>
            <p>Upload and analyze PDFs, text files, and more with advanced parsing capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Smart Retrieval</h3>
            <p>Advanced semantic search and retrieval system for accurate document understanding.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 AI Q&A</h3>
            <p>Ask questions about your documents and get intelligent, context-aware answers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Started Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "document"
            st.rerun()

def document_page():
    """Display the document processing page"""
    st.markdown(get_dark_css(), unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="app-title">📄 Document Analysis</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model Settings
        st.markdown("#### 🤖 AI Model")
        model_type = st.selectbox(
            "Model Type",
            ["OpenAI GPT-3.5", "OpenAI GPT-4", "Hugging Face", "Demo Mode"],
            index=0
        )
        
        # Chunking Settings
        st.markdown("#### 📝 Chunking")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100)
        
        # Document Info
        if st.session_state.document_processed:
            st.markdown("#### 📊 Document Info")
            st.info(f"**File:** {st.session_state.document_name}")
            st.info(f"**Chunks:** {len(st.session_state.chunks)}")
    
    # Main Content
    if not st.session_state.document_processed:
        # File Upload Section
        st.markdown("""
        <div class="upload-area">
            <h3>📁 Upload Your Document</h3>
            <p>Drag and drop your file here or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md', 'html'],
            help="Supported formats: PDF, TXT, MD, HTML",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                try:
                    # Parse document
                    text = parse_document(uploaded_file)
                    
                    # Chunk text
                    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                    
                    # Generate embeddings
                    embeddings = embed_chunks(chunks)
                    
                    # Create vector store
                    vector_store = VectorStore()
                    vector_store.add_chunks(chunks, embeddings)
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.document_processed = True
                    st.session_state.document_name = uploaded_file.name
                    
                    st.success("✅ Document processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
    
    else:
        # Question Answering Section
        st.markdown("""
        <div class="question-form">
            <h3>💬 Ask a Question</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area(
                "Enter your question about the document:",
                placeholder="What is the main topic of this document?",
                height=100
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("🔍 Get Answer", use_container_width=True)
            
            if submitted and question:
                with st.spinner("Generating answer..."):
                    try:
                        # Get answer using RAG pipeline
                        answer = generate_answer(question, st.session_state.vector_store)
                        
                        # Display answer
                        st.markdown("""
                        <div class="answer-container">
                            <h4>🤖 AI Answer:</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(answer)
                        
                        # Show stats
                        st.markdown("""
                        <div class="stats-container">
                            <div class="stat-item">
                                <div class="stat-number">{}</div>
                                <div class="stat-label">Chunks Retrieved</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-number">{}</div>
                                <div class="stat-label">Model Used</div>
                            </div>
                        </div>
                        """.format(
                            min(5, len(st.session_state.chunks)),
                            model_type
                        ), unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
        
        # Reset button
        if st.button("🔄 Process New Document", use_container_width=True):
            st.session_state.document_processed = False
            st.session_state.vector_store = None
            st.session_state.chunks = []
            st.session_state.document_name = None
            st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "document":
        document_page()

if __name__ == "__main__":
    main()
