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
        
        .stExpander > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        .stExpander > div > div {
            background: rgba(255, 255, 255, 0.05) !important;
        }
        
        .app-title {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            margin-bottom: 0.5rem !important;
        }
        
        .app-subtitle {
            font-size: 1.2rem !important;
            color: #ffffff !important;
            font-weight: 400 !important;
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

# Apply dark mode CSS
st.markdown(get_dark_css(), unsafe_allow_html=True)

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
    
    # Model selection
    model = st.selectbox(
        "🤖 AI Model",
        ["llama3.2:3b", "mistral", "codellama"],
        index=0
    )
    
    # Chunk size
    chunk_size = st.slider(
        "📏 Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )
    
    # Top K results
    top_k = st.slider(
        "🔍 Top K Results",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )
    
    st.markdown("---")
    
    # Document info
    if st.session_state.uploaded_document:
        st.markdown("### 📊 Document Info")
        doc = st.session_state.uploaded_document
        st.info(f"""
        **📄 {doc['name']}**
        
        📝 {doc['chunks']} chunks
        📏 {doc['size']:,} characters
        ⏰ {doc['upload_time']}
        """)
        
        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.uploaded_document = None
            st.session_state.vector_store = VectorStore()
            st.session_state.chat_history = []
            st.session_state.current_page = 'home'
            st.rerun()

def home_page():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; padding: 2rem 0;">
        <h1 class="app-title">DocuMind AI</h1>
        <p class="app-subtitle">Intelligent Document Analysis & Q&A Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features using Streamlit columns - Compact version
    st.markdown("### 🚀 Features")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1); text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Smart Upload</h4>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.85rem; margin: 0;">Upload & process document</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1); text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Smart Q&A</h4>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.85rem; margin: 0;">Ask questions in natural language</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1); text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Analytics</h4>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.85rem; margin: 0;">Detailed insights & statistics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid rgba(102, 126, 234, 0.1); text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Search</h4>
            <p style="color: #6c757d; line-height: 1.4; font-size: 0.85rem; margin: 0;">Semantic & keyword search</p>
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
            text = parse_document(uploaded_file)
            chunks = chunk_text(text, chunk_size=1000, overlap=200)
            embeddings = embed_chunks(chunks)
            st.session_state.vector_store.add_embeddings(chunks, embeddings)
            
            st.session_state.uploaded_document = {
                'name': uploaded_file.name,
                'size': len(text),
                'chunks': len(chunks),
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': text
            }
            
            st.session_state.current_page = 'document'
            st.rerun()
    else:
        st.info("👆 Please upload a document to get started")

def document_page():
    doc = st.session_state.uploaded_document
    
    # Document info
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 16px; padding: 2rem; margin: 2rem 0;">
        <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem;">📄 {doc['name']}</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 12px; text-align: center; backdrop-filter: blur(10px);">
                <span style="font-size: 1.5rem; font-weight: 700; display: block;">{doc['chunks']}</span>
                <div style="font-size: 0.9rem; opacity: 0.9;">Chunks</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 12px; text-align: center; backdrop-filter: blur(10px);">
                <span style="font-size: 1.5rem; font-weight: 700; display: block;">{doc['size']:,}</span>
                <div style="font-size: 0.9rem; opacity: 0.9;">Characters</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); padding: 1rem; border-radius: 12px; text-align: center; backdrop-filter: blur(10px);">
                <span style="font-size: 1.5rem; font-weight: 700; display: block;">{doc['upload_time']}</span>
                <div style="font-size: 0.9rem; opacity: 0.9;">Uploaded</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Summarize", use_container_width=True):
            question = "Provide a comprehensive summary of this document."
            ask_question(question)
    
    with col2:
        if st.button("🔍 Key Points", use_container_width=True):
            question = "What are the main key points and important information in this document?"
            ask_question(question)
    
    with col3:
        if st.button("❓ Explain", use_container_width=True):
            question = "Explain the main concepts and ideas discussed in this document."
            ask_question(question)
    
    with col4:
        if st.button("📊 Analyze", use_container_width=True):
            question = "Analyze the structure and content of this document."
            ask_question(question)
    
    # Custom question input
    st.markdown("### 💬 Ask a Custom Question")
    
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Ask Question", use_container_width=True)
            if submitted and question:
                ask_question(question)
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 16px; margin: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin-left: 20%; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
                <strong>👤 You ({entry['timestamp']}):</strong><br>
                {entry['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="padding: 1.5rem; border-radius: 16px; margin: 1rem 0; background: #f8f9fa; color: #2c3e50; margin-right: 20%; border-left: 4px solid #667eea; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);">
                <strong>🤖 AI Response:</strong><br>
                {entry['answer']}<br>
                <small>⏱️ Response time: {entry['response_time']:.2f}s</small>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📚 References & Sources"):
                for i, ref in enumerate(entry['references'], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(ref[:300] + "..." if len(ref) > 300 else ref)
    
    # Analytics section
    st.markdown("### 📊 Document Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        words = doc['text'].split()
        word_freq = pd.Series(words).value_counts().head(10)
        
        fig = px.bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            title="Top 10 Most Frequent Words",
            color=word_freq.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        stats_data = {
            'Metric': ['Total Words', 'Total Characters', 'Chunks', 'Avg Chunk Size'],
            'Value': [len(words), doc['size'], doc['chunks'], doc['size'] // doc['chunks']]
        }
        
        fig = go.Figure(data=[go.Bar(
            x=stats_data['Metric'],
            y=stats_data['Value'],
            marker_color=['#667eea', '#764ba2', '#667eea', '#764ba2']
        )])
        fig.update_layout(title="Document Statistics", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Back to home button
    if st.button("🏠 Back to Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.session_state.uploaded_document = None
        st.session_state.vector_store = VectorStore()
        st.session_state.chat_history = []
        st.rerun()

def ask_question(question):
    with st.spinner("🧠 AI is thinking..."):
        query_embedding = embed_query(question)
        top_chunks = retrieve_relevant_chunks(query_embedding, st.session_state.vector_store, top_k=5)
        
        start_time = time.time()
        answer, references = generate_answer(question, top_chunks)
        response_time = time.time() - start_time
        
        chat_entry = {
            'question': question,
            'answer': answer,
            'references': references,
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
