"""
🔍 Advanced retrieval system with improved ranking and filtering.
"""

import numpy as np
from typing import List, Tuple
import re

def retrieve_relevant_chunks(query_embedding, vector_store, top_k=3, min_similarity=0.3):
    """
    Retrieve the most relevant chunks with advanced filtering and ranking.
    Args:
        query_embedding: The query embedding vector
        vector_store: The vector store containing document chunks
        top_k: Number of chunks to retrieve
        min_similarity: Minimum similarity threshold
    Returns:
        List[str]: Most relevant chunks
    """
    if not hasattr(vector_store, 'embeddings') or vector_store.embeddings is None or len(vector_store.embeddings) == 0:
        return []
    
    # Get similarity scores
    similarities = vector_store.search(query_embedding, top_k=top_k * 2)  # Get more for filtering
    
    # Filter by similarity threshold
    filtered_chunks = []
    for chunk, score in similarities:
        if score >= min_similarity:
            filtered_chunks.append((chunk, score))
    
    # If we don't have enough chunks above threshold, lower the threshold
    if len(filtered_chunks) < top_k:
        filtered_chunks = similarities[:top_k]
    
    # Advanced ranking: combine similarity with content quality
    ranked_chunks = advanced_ranking(filtered_chunks, query_embedding)
    
    # Return top chunks
    return [chunk for chunk, score in ranked_chunks[:top_k]]

def advanced_ranking(chunks_with_scores, query_embedding):
    """
    Advanced ranking that combines similarity with content quality metrics.
    """
    ranked = []
    
    for chunk, similarity_score in chunks_with_scores:
        # Calculate content quality score
        quality_score = calculate_content_quality(chunk)
        
        # Calculate diversity score (avoid redundant chunks)
        diversity_score = calculate_diversity_score(chunk, [c for c, _ in ranked])
        
        # Combine scores with weights
        final_score = (
            similarity_score * 0.5 +      # 50% similarity
            quality_score * 0.3 +         # 30% content quality
            diversity_score * 0.2         # 20% diversity
        )
        
        ranked.append((chunk, final_score))
    
    # Sort by final score
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def calculate_content_quality(chunk):
    """
    Calculate content quality score based on various factors.
    """
    if not chunk or len(chunk.strip()) < 10:
        return 0.0
    
    score = 0.0
    
    # Length factor (prefer medium-length chunks)
    length = len(chunk)
    if 100 <= length <= 500:
        score += 0.3
    elif 50 <= length <= 800:
        score += 0.2
    else:
        score += 0.1
    
    # Sentence structure (prefer well-formed sentences)
    sentences = re.split(r'[.!?]+', chunk)
    if len(sentences) >= 2:
        score += 0.2
    
    # Information density (prefer chunks with numbers, specific terms)
    if re.search(r'\d+', chunk):  # Contains numbers
        score += 0.1
    
    if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', chunk):  # Contains proper nouns
        score += 0.1
    
    # Avoid chunks that are mostly whitespace or special characters
    if len(chunk.strip()) / len(chunk) < 0.7:
        score -= 0.2
    
    return min(score, 1.0)

def calculate_diversity_score(chunk, existing_chunks):
    """
    Calculate diversity score to avoid redundant information.
    """
    if not existing_chunks:
        return 1.0
    
    chunk_words = set(chunk.lower().split())
    max_overlap = 0.0
    
    for existing_chunk in existing_chunks:
        existing_words = set(existing_chunk.lower().split())
        overlap = len(chunk_words.intersection(existing_words))
        total_words = len(chunk_words.union(existing_words))
        
        if total_words > 0:
            overlap_ratio = overlap / total_words
            max_overlap = max(max_overlap, overlap_ratio)
    
    # Higher diversity score for less overlap
    return 1.0 - max_overlap

def retrieve_by_keywords(query, vector_store, top_k=3):
    """
    Alternative retrieval method using keyword matching.
    """
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in vector_store.chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words.intersection(chunk_words))
        
        if overlap > 0:
            score = overlap / len(query_words)
            scored_chunks.append((chunk, score))
    
    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

def hybrid_retrieval(query, query_embedding, vector_store, top_k=3):
    """
    Hybrid retrieval combining semantic and keyword-based search.
    """
    # Semantic retrieval
    semantic_chunks = retrieve_relevant_chunks(query_embedding, vector_store, top_k=top_k)
    
    # Keyword retrieval
    keyword_chunks = retrieve_by_keywords(query, vector_store, top_k=top_k)
    
    # Combine and deduplicate
    all_chunks = {}
    
    # Add semantic chunks with higher weight
    for i, chunk in enumerate(semantic_chunks):
        all_chunks[chunk] = (len(semantic_chunks) - i) * 2
    
    # Add keyword chunks
    for i, chunk in enumerate(keyword_chunks):
        if chunk in all_chunks:
            all_chunks[chunk] += (len(keyword_chunks) - i)
        else:
            all_chunks[chunk] = (len(keyword_chunks) - i)
    
    # Sort by combined score
    sorted_chunks = sorted(all_chunks.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in sorted_chunks[:top_k]]