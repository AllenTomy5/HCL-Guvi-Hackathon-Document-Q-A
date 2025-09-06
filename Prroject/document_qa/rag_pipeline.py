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
    if not context_chunks:
        return "Sorry, I couldn't find relevant information in the document.", []

    # Combine the top context chunks into a single context string
    context = "\n\n".join(context_chunks)

    # Format the prompt for the LLM
    prompt = (
        "You are an expert assistant. Use ONLY the context below to answer the user's question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Call the LLM
    answer = ask_llm(prompt)

    # Optionally, provide references (here, just the first 2 chunks as a simple example)
    references = [chunk[:200] + "..." if len(chunk) > 200 else chunk for chunk in context_chunks[:2]]

    return answer, references
