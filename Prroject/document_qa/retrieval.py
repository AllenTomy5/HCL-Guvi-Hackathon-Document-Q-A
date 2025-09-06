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
    # TODO: Call vector_store.search and return results
    pass
