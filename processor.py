import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        return text.strip()
    
    def process_file(self, file_content: str) -> List[str]:
        """Process uploaded file and return chunks"""
        cleaned_text = self.clean_text(file_content)
        chunks = self.text_splitter.split_text(cleaned_text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess user query"""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return list(set(words))