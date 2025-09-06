"""
Main entry point for the Document Q&A system.
Handles the UI and orchestrates the workflow.
"""

# Example: Streamlit UI (can be replaced with Gradio/Flask)
import streamlit as st
from ingestion import parse_document
from chunking import chunk_text
from embedding import embed_chunks, embed_query
from vector_store import VectorStore
from retrieval import retrieve_relevant_chunks
from rag_pipeline import generate_answer

# Initialize vector store (e.g., FAISS/ChromaDB)
vector_store = VectorStore()

st.title("Document Q&A System (RAG)")

uploaded_file = st.file_uploader("Upload a document (PDF, Markdown, HTML)")

if uploaded_file:
    text = parse_document(uploaded_file)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    vector_store.add_embeddings(chunks, embeddings)
    st.success("Document processed and indexed!")

question = st.text_input("Ask a question about your document:")

if question:
    query_embedding = embed_query(question)
    top_chunks = retrieve_relevant_chunks(query_embedding, vector_store)
    answer, references = generate_answer(question, top_chunks)
    st.markdown(f"**Answer:** {answer}")
    st.markdown("**References:**")
    for ref in references:
        st.markdown(f"- {ref}")
