import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

class ReinforcementLearner:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_dir = "./rl_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.feedback_file = f"{self.data_dir}/feedback_{user_id}.json"
        self.strategy_file = f"{self.data_dir}/strategy_{user_id}.json"
        
        self.strategies = ["concise", "detailed", "structured", "example_driven", "analytical"]
        self.strategy_stats = self._load_strategy_stats()
        self.feedback_history = self._load_feedback_history()
        
        self.epsilon = 0.2
    
    def _load_strategy_stats(self) -> Dict:
        """Load strategy performance statistics"""
        if os.path.exists(self.strategy_file):
            with open(self.strategy_file, 'r') as f:
                return json.load(f)
        return {s: {"wins": 0, "total": 0, "reward_sum": 0.0} for s in self.strategies}
    
    def _save_strategy_stats(self) -> None:
        """Save strategy performance statistics"""
        with open(self.strategy_file, 'w') as f:
            json.dump(self.strategy_stats, f, indent=2)
    
    def _load_feedback_history(self) -> List[Dict]:
        """Load feedback history"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback_history(self) -> None:
        """Save feedback history"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def select_strategy(self, query: str, query_complexity: str = "moderate", 
                       cluster_best_strategy: Optional[str] = None) -> Tuple[str, int]:
        """
        Select retrieval strategy using epsilon-greedy with UCB
        If cluster has a proven best strategy, bias towards it
        """
        # If cluster has a proven strategy with good performance, use it more often
        if cluster_best_strategy and cluster_best_strategy in self.strategies:
            cluster_stats = self.strategy_stats[cluster_best_strategy]
            if cluster_stats['total'] >= 3:  # At least 3 uses
                avg_reward = cluster_stats['reward_sum'] / cluster_stats['total']
                if avg_reward > 0.3:  # Positive performance
                    # 70% chance to use cluster's best strategy
                    if np.random.random() < 0.7:
                        strategy = cluster_best_strategy
                        k_values = {
                            "concise": 3, "detailed": 5, "structured": 4,
                            "example_driven": 4, "analytical": 6
                        }
                        top_k = k_values.get(strategy, 4)
                        return strategy, top_k
        
        # Regular epsilon-greedy with UCB
        if np.random.random() < self.epsilon:
            strategy = np.random.choice(self.strategies)
            strategy_idx = self.strategies.index(strategy)
        else:
            ucb_scores = []
            total_pulls = sum(self.strategy_stats[s]["total"] for s in self.strategies)
            
            for strategy in self.strategies:
                stats = self.strategy_stats[strategy]
                n = stats["total"]
                
                if n == 0:
                    ucb_scores.append(float('inf'))
                else:
                    avg_reward = stats["reward_sum"] / n
                    exploration_bonus = np.sqrt(2 * np.log(total_pulls + 1) / n)
                    ucb_scores.append(avg_reward + exploration_bonus)
            
            strategy_idx = np.argmax(ucb_scores)
            strategy = self.strategies[strategy_idx]
        
        k_values = {
            "concise": 3, "detailed": 5, "structured": 4,
            "example_driven": 4, "analytical": 6
        }
        top_k = k_values.get(strategy, 4)
        
        return strategy, top_k
    
    def record_feedback(self, query: str, strategy: str, response: str, 
                       feedback: int, retrieved_docs: List[str], cluster_name: str = None) -> None:
        """Record user feedback and update strategy weights"""
        reward = 1.0 if feedback > 0 else -1.0
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "strategy": strategy,
            "response": response[:200],
            "feedback": feedback,
            "reward": reward,
            "retrieved_docs": retrieved_docs[:2],
            "cluster": cluster_name
        }
        
        self.feedback_history.append(feedback_entry)
        
        self.strategy_stats[strategy]["total"] += 1
        self.strategy_stats[strategy]["reward_sum"] += reward
        if reward > 0:
            self.strategy_stats[strategy]["wins"] += 1
        
        self._save_feedback_history()
        self._save_strategy_stats()
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for dashboard"""
        metrics = {
            "total_interactions": len(self.feedback_history),
            "positive_feedback": sum(1 for f in self.feedback_history if f["reward"] > 0),
            "negative_feedback": sum(1 for f in self.feedback_history if f["reward"] < 0),
            "strategy_performance": {}
        }
        
        for strategy in self.strategies:
            stats = self.strategy_stats[strategy]
            total = stats["total"]
            if total > 0:
                win_rate = stats["wins"] / total
                avg_reward = stats["reward_sum"] / total
            else:
                win_rate = 0.0
                avg_reward = 0.0
            
            metrics["strategy_performance"][strategy] = {
                "total_uses": total,
                "win_rate": round(win_rate, 3),
                "avg_reward": round(avg_reward, 3)
            }
        
        return metrics
    
    def get_query_improvement(self, current_query: str) -> Dict:
        """Check if similar query was asked before and show improvement"""
        similar_queries = []
        
        for entry in self.feedback_history[-20:]:
            past_query = entry["query"].lower()
            if self._calculate_similarity(current_query.lower(), past_query) > 0.6:
                similar_queries.append({
                    "query": entry["query"],
                    "strategy": entry["strategy"],
                    "feedback": entry["feedback"],
                    "timestamp": entry["timestamp"]
                })
        
        return {
            "has_similar": len(similar_queries) > 0,
            "similar_queries": similar_queries[-3:],
            "learning_active": len(similar_queries) > 0
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0