"""
Vector store module.
Handles storage and retrieval of embeddings using FAISS/ChromaDB/etc.
"""


import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        """
        Initialize the vector store (in-memory FAISS index and chunk mapping).
        """
        self.embeddings = None  # Will be a numpy array
        self.chunks = []        # List of text chunks
        self.index = None       # FAISS index

    def add_embeddings(self, chunks, embeddings):
        """
        Add text chunks and their embeddings to the store.
        """
        embeddings = np.array(embeddings).astype('float32')
        if self.embeddings is None:
            self.embeddings = embeddings
            self.chunks = list(chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)
            self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Retrieve top-k most similar chunks for a query embedding.
        """
        if self.index is None or self.embeddings is None or len(self.chunks) == 0:
            return []
        query = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results
