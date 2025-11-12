"""Analytics module for tracking and visualizing RL performance"""

import json
import os
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

class Analytics:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_dir = "./rl_data"
        self.feedback_file = f"{self.data_dir}/feedback_{user_id}.json"
    
    def load_feedback_history(self) -> List[Dict]:
        """Load feedback history"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_temporal_performance(self, hours: int = 24) -> Dict:
        """Get performance metrics over time"""
        feedback_history = self.load_feedback_history()
        
        if not feedback_history:
            return {"error": "No data available"}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_feedback = [
            f for f in feedback_history 
            if datetime.fromisoformat(f['timestamp']) > cutoff_time
        ]
        
        if not recent_feedback:
            recent_feedback = feedback_history[-20:]
        
        total = len(recent_feedback)
        positive = sum(1 for f in recent_feedback if f['reward'] > 0)
        
        return {
            "total_queries": total,
            "positive_rate": positive / total if total > 0 else 0,
            "time_window": f"Last {hours} hours"
        }
    
    def get_query_patterns(self) -> Dict:
        """Analyze query patterns and topics"""
        feedback_history = self.load_feedback_history()
        
        if not feedback_history:
            return {"error": "No data available"}
        
        # Get most common words in queries
        word_freq = defaultdict(int)
        for entry in feedback_history:
            words = entry['query'].lower().split()
            for word in words:
                if len(word) > 3:
                    word_freq[word] += 1
        
        # Get query length distribution
        query_lengths = [len(entry['query'].split()) for entry in feedback_history]
        avg_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        
        # Get most successful query patterns
        positive_queries = [
            entry['query'] for entry in feedback_history 
            if entry['reward'] > 0
        ]
        
        return {
            "total_unique_queries": len(set(e['query'] for e in feedback_history)),
            "avg_query_length": round(avg_length, 1),
            "most_common_topics": dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]),
            "positive_query_count": len(positive_queries)
        }
    
    def get_learning_curve(self) -> List[Dict]:
        """Calculate learning curve showing improvement over time"""
        feedback_history = self.load_feedback_history()
        
        if len(feedback_history) < 5:
            return []
        
        window_size = max(5, len(feedback_history) // 10)
        learning_curve = []
        
        for i in range(0, len(feedback_history), window_size):
            window = feedback_history[i:i+window_size]
            if window:
                positive = sum(1 for f in window if f['reward'] > 0)
                learning_curve.append({
                    "batch": i // window_size + 1,
                    "queries": len(window),
                    "success_rate": positive / len(window)
                })
        
        return learning_curve
    
    def get_strategy_evolution(self) -> Dict:
        """Track how strategy selection evolved over time"""
        feedback_history = self.load_feedback_history()
        
        if not feedback_history:
            return {"error": "No data available"}
        
        strategy_timeline = defaultdict(list)
        
        for entry in feedback_history:
            strategy = entry['strategy']
            strategy_timeline[strategy].append({
                "timestamp": entry['timestamp'],
                "reward": entry['reward']
            })
        
        evolution = {}
        for strategy, timeline in strategy_timeline.items():
            if timeline:
                positive = sum(1 for t in timeline if t['reward'] > 0)
                evolution[strategy] = {
                    "usage_count": len(timeline),
                    "success_rate": positive / len(timeline),
                    "first_used": timeline[0]['timestamp'],
                    "last_used": timeline[-1]['timestamp']
                }
        
        return evolution
    
    def export_analytics_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        return {
            "user_id": self.user_id,
            "generated_at": datetime.now().isoformat(),
            "temporal_performance": self.get_temporal_performance(),
            "query_patterns": self.get_query_patterns(),
            "learning_curve": self.get_learning_curve(),
            "strategy_evolution": self.get_strategy_evolution()
        }