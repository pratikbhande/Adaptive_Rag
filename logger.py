"""Logging utility for tracking system events"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

class SystemLogger:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = f"{self.log_dir}/system_{user_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"RAG_System_{user_id}")
    
    def log_query(self, query: str, strategy: str, complexity: str) -> None:
        """Log user query with metadata"""
        self.logger.info(
            f"QUERY | Query='{query[:50]}...' | Strategy={strategy} | Complexity={complexity}"
        )
    
    def log_feedback(self, query: str, feedback: int, strategy: str) -> None:
        """Log user feedback"""
        feedback_type = "POSITIVE" if feedback > 0 else "NEGATIVE"
        self.logger.info(
            f"FEEDBACK | Type={feedback_type} | Strategy={strategy} | Query='{query[:30]}...'"
        )
    
    def log_retrieval(self, query: str, num_docs: int, top_k: int) -> None:
        """Log document retrieval"""
        self.logger.info(
            f"RETRIEVAL | Query='{query[:30]}...' | Retrieved={num_docs} | TopK={top_k}"
        )
    
    def log_indexing(self, num_chunks: int) -> None:
        """Log document indexing"""
        self.logger.info(f"INDEXING | Chunks={num_chunks} | User={self.user_id}")
    
    def log_strategy_update(self, strategy: str, new_stats: Dict[str, Any]) -> None:
        """Log strategy performance update"""
        self.logger.info(
            f"STRATEGY_UPDATE | Strategy={strategy} | Stats={new_stats}"
        )
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log errors"""
        self.logger.error(f"ERROR | Context='{context}' | Error={str(error)}")
    
    def log_session_start(self) -> None:
        """Log session start"""
        self.logger.info(f"SESSION_START | User={self.user_id} | Time={datetime.now()}")
    
    def log_session_end(self) -> None:
        """Log session end"""
        self.logger.info(f"SESSION_END | User={self.user_id} | Time={datetime.now()}")