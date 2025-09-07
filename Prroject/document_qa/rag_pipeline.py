"""
🚀 Advanced RAG pipeline with answer quality improvements.
Combines retrieval and LLM to generate high-quality answers with references.
"""

from llm_interface import ask_llm
import re

def generate_answer(question, context_chunks, answer_style="comprehensive"):
    """
    Generate a high-quality answer using advanced prompting techniques.
    Args:
        question (str): User's question
        context_chunks (List[str]): Retrieved relevant chunks
        answer_style (str): Style of answer ("comprehensive", "concise", "detailed")
    Returns:
        Tuple[str, List[str]]: (Answer, List of references)
    """
    if not context_chunks:
        return "I apologize, but I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or check if the document contains the information you're looking for.", []

    # Combine the top context chunks into a single context string
    context = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

    # Advanced prompting based on question type and style
    prompt = create_advanced_prompt(question, context, answer_style)
    
    # Generate initial answer
    answer = ask_llm(prompt)
    
    # Post-process and refine the answer
    refined_answer = refine_answer(answer, question, context_chunks)
    
    # Create enhanced references
    references = create_enhanced_references(context_chunks, question)
    
    return refined_answer, references

def create_advanced_prompt(question, context, answer_style):
    """Create an advanced prompt based on question type and desired style."""
    
    # Detect question type
    question_type = detect_question_type(question)
    
    # Base system prompt
    system_prompt = """You are DocuMind AI, an expert document analysis assistant. You excel at:
- Providing accurate, well-structured answers based on document content
- Citing specific sources and evidence
- Maintaining professional yet engaging communication
- Being honest when information is not available in the provided context

IMPORTANT: Only use information from the provided context. If the answer isn't in the context, clearly state this."""

    # Style-specific instructions
    style_instructions = {
        "comprehensive": "Provide a thorough, well-structured answer with clear explanations and examples.",
        "concise": "Give a direct, to-the-point answer while maintaining accuracy.",
        "detailed": "Provide an in-depth analysis with multiple perspectives and detailed explanations."
    }
    
    # Question type-specific instructions
    type_instructions = {
        "summary": "Focus on key points, main themes, and overall structure. Organize information logically.",
        "explanation": "Break down complex concepts into understandable parts. Use clear examples.",
        "comparison": "Highlight similarities, differences, and relationships between concepts.",
        "factual": "Provide specific facts, numbers, dates, and concrete information.",
        "analytical": "Analyze patterns, trends, and implications. Provide insights and interpretations.",
        "procedural": "Provide step-by-step instructions or processes in logical order."
    }
    
    # Build the complete prompt
    prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer Style: {style_instructions.get(answer_style, style_instructions["comprehensive"])}
- Question Type: {type_instructions.get(question_type, "Provide a clear, informative answer")}
- Always cite your sources using [Source X] format
- If information is incomplete, mention what additional context would be helpful
- Structure your answer with clear paragraphs and logical flow

ANSWER:"""
    
    return prompt

def detect_question_type(question):
    """Detect the type of question to tailor the response."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["summarize", "summary", "overview", "main points"]):
        return "summary"
    elif any(word in question_lower for word in ["explain", "what is", "how does", "why"]):
        return "explanation"
    elif any(word in question_lower for word in ["compare", "difference", "similar", "versus", "vs"]):
        return "comparison"
    elif any(word in question_lower for word in ["when", "where", "who", "how many", "how much"]):
        return "factual"
    elif any(word in question_lower for word in ["analyze", "analysis", "trend", "pattern", "implication"]):
        return "analytical"
    elif any(word in question_lower for word in ["how to", "steps", "process", "procedure"]):
        return "procedural"
    else:
        return "general"

def refine_answer(answer, question, context_chunks):
    """Post-process and refine the generated answer."""
    
    # Remove any hallucinated citations that don't exist
    refined = remove_invalid_citations(answer, len(context_chunks))
    
    # Ensure proper formatting
    refined = format_answer(refined)
    
    # Add confidence indicators if appropriate
    refined = add_confidence_indicators(refined, context_chunks)
    
    return refined

def remove_invalid_citations(answer, max_sources):
    """Remove citations that reference non-existent sources."""
    # Find all [Source X] patterns
    citations = re.findall(r'\[Source (\d+)\]', answer)
    
    for citation in citations:
        source_num = int(citation)
        if source_num > max_sources:
            # Replace invalid citation with generic reference
            answer = answer.replace(f'[Source {source_num}]', '[Document]')
    
    return answer

def format_answer(answer):
    """Format the answer for better readability."""
    # Ensure proper paragraph breaks
    answer = re.sub(r'\n\s*\n', '\n\n', answer)
    
    # Clean up any formatting issues
    answer = answer.strip()
    
    return answer

def add_confidence_indicators(answer, context_chunks):
    """Add confidence indicators based on context quality."""
    if len(context_chunks) >= 3:
        confidence_note = "\n\n💡 *This answer is based on multiple relevant sources from the document.*"
    elif len(context_chunks) == 2:
        confidence_note = "\n\n💡 *This answer is based on relevant information found in the document.*"
    else:
        confidence_note = "\n\n⚠️ *This answer is based on limited information from the document. Consider asking for more specific details.*"
    
    return answer + confidence_note

def create_enhanced_references(context_chunks, question):
    """Create enhanced references with relevance scoring."""
    references = []
    
    for i, chunk in enumerate(context_chunks, 1):
        # Truncate long chunks for display
        display_chunk = chunk[:300] + "..." if len(chunk) > 300 else chunk
        
        # Add relevance indicator
        relevance_score = calculate_relevance_score(chunk, question)
        relevance_indicator = "🔥" if relevance_score > 0.8 else "📄" if relevance_score > 0.6 else "📝"
        
        references.append(f"{relevance_indicator} **Source {i}** (Relevance: {relevance_score:.1%}):\n{display_chunk}")
    
    return references

def calculate_relevance_score(chunk, question):
    """Calculate a simple relevance score based on keyword overlap."""
    question_words = set(question.lower().split())
    chunk_words = set(chunk.lower().split())
    
    # Simple overlap calculation
    overlap = len(question_words.intersection(chunk_words))
    total_question_words = len(question_words)
    
    if total_question_words == 0:
        return 0.5  # Default score
    
    return min(overlap / total_question_words * 2, 1.0)  # Scale up and cap at 1.0

def generate_summary(document_chunks, max_length=500):
    """Generate a document summary using the RAG pipeline."""
    if not document_chunks:
        return "No content available for summarization."
    
    # Create a summary prompt
    context = "\n\n".join(document_chunks[:10])  # Use first 10 chunks for summary
    prompt = f"""Please provide a comprehensive summary of the following document content. 
    Focus on the main themes, key points, and important information.
    
    Document Content:
    {context}
    
    Summary (max {max_length} words):"""
    
    summary = ask_llm(prompt)
    return summary[:max_length] + "..." if len(summary) > max_length else summary

def generate_key_points(document_chunks, num_points=5):
    """Extract key points from the document."""
    if not document_chunks:
        return []
    
    context = "\n\n".join(document_chunks[:15])  # Use first 15 chunks
    prompt = f"""Extract the {num_points} most important key points from the following document content.
    Present each point as a clear, concise statement.
    
    Document Content:
    {context}
    
    Key Points:"""
    
    response = ask_llm(prompt)
    
    # Parse the response into individual points
    points = []
    for line in response.split('\n'):
        line = line.strip()
        if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•', '*')) or 
                    any(char.isdigit() for char in line[:3])):
            # Clean up the point
            point = re.sub(r'^\d+\.\s*', '', line)
            point = re.sub(r'^[-•*]\s*', '', point)
            if point:
                points.append(point)
    
    return points[:num_points] if points else [response]