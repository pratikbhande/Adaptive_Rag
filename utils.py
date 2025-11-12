"""Utility functions for the Self-Adaptive RAG system"""

import os
import json
from typing import Dict, Any
from datetime import datetime

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """Save data to JSON file"""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str, default: Dict = None) -> Dict:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        return default if default is not None else {}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def format_timestamp(dt: datetime = None) -> str:
    """Format datetime to ISO string"""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()

def calculate_word_overlap(text1: str, text2: str) -> float:
    """Calculate word overlap similarity between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."