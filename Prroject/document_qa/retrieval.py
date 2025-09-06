"""
Retrieval module.
Handles retrieval of relevant chunks from the vector store.
"""

def retrieve_relevant_chunks(query_embedding, vector_store, top_k=5):
    """
    Retrieve top-k relevant chunks from the vector store.
    Args:
        query_embedding (List[float]): Embedding of the user query
        vector_store (VectorStore): The vector store instance
        top_k (int): Number of chunks to retrieve
    Returns:
        List[str]: Top relevant text chunks
    """
    # Call the vector store's search method to get top-k relevant chunks
    results = vector_store.search(query_embedding, top_k=top_k)
    return results
