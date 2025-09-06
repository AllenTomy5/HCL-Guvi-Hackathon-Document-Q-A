"""
Text chunking module.
Splits extracted text into manageable chunks for embedding.
"""

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks.
    Args:
        text (str): The input text
        chunk_size (int): Number of tokens/words per chunk
        overlap (int): Overlap between chunks
    Returns:
        List[str]: List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks
