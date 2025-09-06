"""
Utility functions for the Document Q&A system.
"""

import re

def clean_text(text):
	"""
	Basic text cleaning: removes extra whitespace and non-printable characters.
	Args:
		text (str): Input text
	Returns:
		str: Cleaned text
	"""
	text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
	text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove non-printable chars
	return text.strip()
