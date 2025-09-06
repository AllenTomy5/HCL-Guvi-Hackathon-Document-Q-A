"""
RAG pipeline module.
Combines retrieval and LLM to generate answers with references.
"""
from llm_interface import ask_llm

def generate_answer(question, context_chunks):
    """
    Generate an answer using the LLM, given the question and retrieved context.
    Args:
        question (str): User's question
        context_chunks (List[str]): Retrieved relevant chunks
    Returns:
        Tuple[str, List[str]]: (Answer, List of references)
    """
    # TODO: Format prompt, call LLM, extract references
    pass
