"""
Document ingestion and parsing module.
Supports PDF, Markdown, and HTML extraction.
"""


import io
import pdfplumber
import markdown as md
from bs4 import BeautifulSoup

def parse_document(file):
    """
    Detects file type and extracts clean text from PDF, Markdown, or HTML.
    Args:
        file: Uploaded file object
    Returns:
        str: Extracted text
    """
    filename = file.name.lower()
    if filename.endswith('.pdf'):
        # PDF extraction
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        return text
    elif filename.endswith('.md') or filename.endswith('.markdown'):
        # Markdown extraction
        text = file.read().decode('utf-8')
        # Optionally convert markdown to plain text
        html = md.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator=' ')
    elif filename.endswith('.html') or filename.endswith('.htm'):
        # HTML extraction
        text = file.read().decode('utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ')
    else:
        # Try to guess by content if extension is missing
        try:
            # Try PDF
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            return text
        except Exception:
            file.seek(0)
            text = file.read().decode('utf-8')
            # Try HTML
            if '<html' in text.lower():
                soup = BeautifulSoup(text, 'html.parser')
                return soup.get_text(separator=' ')
            # Try Markdown
            html = md.markdown(text)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator=' ')
