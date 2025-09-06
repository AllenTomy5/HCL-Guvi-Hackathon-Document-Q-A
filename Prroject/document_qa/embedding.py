"""
Embedding module.
Handles embedding generation for text chunks and queries.
"""


from sentence_transformers import SentenceTransformer

# Load the model once (you can change the model name as needed)
_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    """
    Generate embeddings for a list of text chunks.
    Args:
        chunks (List[str]): List of text chunks
    Returns:
        List[List[float]]: List of embeddings
    """
    return _model.encode(chunks, show_progress_bar=False).tolist()


def embed_query(query):
    """
    Generate embedding for a user query.
    Args:
        query (str): User's question
    Returns:
        List[float]: Query embedding
    """
    return _model.encode([query])[0].tolist()
