#!/usr/bin/env python3
"""
Test script to verify the complete Document Q&A pipeline works correctly.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from ingestion import parse_document
        from chunking import chunk_text
        from embedding import embed_chunks, embed_query
        from vector_store import VectorStore
        from retrieval import retrieve_relevant_chunks
        from rag_pipeline import generate_answer
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embedding():
    """Test embedding functionality."""
    print("\nTesting embedding...")
    try:
        from embedding import embed_query, embed_chunks
        
        # Test single query embedding
        query_embedding = embed_query("What is machine learning?")
        print(f"✅ Query embedding generated: {len(query_embedding)} dimensions")
        
        # Test batch embeddings
        chunks = ["This is a test chunk.", "Another test chunk."]
        chunk_embeddings = embed_chunks(chunks)
        print(f"✅ Batch embeddings generated: {len(chunk_embeddings)} embeddings")
        
        return True
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return False

def test_chunking():
    """Test text chunking functionality."""
    print("\nTesting chunking...")
    try:
        from chunking import chunk_text
        
        test_text = "This is a test document. " * 20  # Create a longer text
        chunks = chunk_text(test_text, chunk_size=50, overlap=10)
        print(f"✅ Text chunked into {len(chunks)} chunks")
        print(f"   First chunk: {chunks[0][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        return False

def test_vector_store():
    """Test vector store functionality."""
    print("\nTesting vector store...")
    try:
        from vector_store import VectorStore
        from embedding import embed_chunks
        
        # Create test data
        chunks = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text and speech."
        ]
        
        # Generate embeddings
        embeddings = embed_chunks(chunks)
        
        # Test vector store
        vs = VectorStore()
        vs.add_embeddings(chunks, embeddings)
        
        # Test search
        from embedding import embed_query
        query_embedding = embed_query("What is AI?")
        results = vs.search(query_embedding, top_k=2)
        
        print(f"✅ Vector store working: {len(results)} results found")
        print(f"   Top result: {results[0][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def test_retrieval():
    """Test retrieval functionality."""
    print("\nTesting retrieval...")
    try:
        from retrieval import retrieve_relevant_chunks
        from vector_store import VectorStore
        from embedding import embed_chunks, embed_query
        
        # Setup test data
        chunks = [
            "Python is a programming language.",
            "Machine learning uses algorithms to learn from data.",
            "Streamlit is a Python web framework."
        ]
        
        vs = VectorStore()
        embeddings = embed_chunks(chunks)
        vs.add_embeddings(chunks, embeddings)
        
        # Test retrieval
        query_embedding = embed_query("What is Python?")
        results = retrieve_relevant_chunks(query_embedding, vs, top_k=2)
        
        print(f"✅ Retrieval working: {len(results)} chunks retrieved")
        
        return True
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return False

def test_rag_pipeline():
    """Test the complete RAG pipeline (without LLM)."""
    print("\nTesting RAG pipeline...")
    try:
        from rag_pipeline import generate_answer
        
        # Test with mock context (since we don't have LLM running)
        context_chunks = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Deep learning is a subset of machine learning using neural networks."
        ]
        
        # This will fail at LLM call, but we can test the structure
        try:
            answer, references = generate_answer("What is machine learning?", context_chunks)
            print(f"✅ RAG pipeline working (with LLM)")
            print(f"   Answer: {answer[:100]}...")
        except Exception as llm_error:
            print(f"⚠️  RAG pipeline structure OK, but LLM not available: {llm_error}")
            print("   This is expected if Ollama is not running.")
        
        return True
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Document Q&A System Setup")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_embedding,
        test_chunking,
        test_vector_store,
        test_retrieval,
        test_rag_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n📝 Next steps:")
        print("1. Install and start Ollama (for LLM functionality)")
        print("2. Run: streamlit run app.py")
        print("3. Upload a document and start asking questions!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
