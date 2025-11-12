"""Query clustering module for grouping semantically similar queries"""

import json
import os
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from prompt_template.templates import QUERY_CLUSTERING_PROMPT, SIMILARITY_EVALUATION_PROMPT
from collections import defaultdict

class QueryClusterer:
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.data_dir = "./rl_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.clusters_file = f"{self.data_dir}/query_clusters_{user_id}.json"
        self.clusters = self._load_clusters()
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
    
    def _load_clusters(self) -> Dict:
        """Load existing query clusters"""
        if os.path.exists(self.clusters_file):
            with open(self.clusters_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_clusters(self) -> None:
        """Save query clusters"""
        with open(self.clusters_file, 'w') as f:
            json.dump(self.clusters, f, indent=2)
    
    def _get_existing_groups_summary(self) -> str:
        """Get summary of existing groups for LLM context"""
        if not self.clusters:
            return "No existing groups yet."
        
        summary_parts = []
        for group_name, data in list(self.clusters.items())[:10]:  # Show max 10 groups
            example_queries = data['queries'][:2]  # Show 2 example queries
            summary_parts.append(f"- {group_name}: {', '.join(example_queries)}")
        
        return "\n".join(summary_parts)
    
    def assign_cluster(self, query: str) -> Tuple[str, bool]:
        """
        Assign query to a cluster using LLM
        Returns: (cluster_name, is_new_cluster)
        """
        existing_summary = self._get_existing_groups_summary()
        
        prompt = PromptTemplate.from_template(QUERY_CLUSTERING_PROMPT)
        formatted_prompt = prompt.format(
            query=query,
            existing_groups=existing_summary
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Parse response
            lines = content.split('\n')
            group_name = None
            
            for line in lines:
                if line.startswith('GROUP:'):
                    group_name = line.replace('GROUP:', '').strip()
                    break
            
            if not group_name:
                group_name = f"cluster_{len(self.clusters)}"
            
            # Clean group name
            group_name = group_name.lower().replace(' ', '_')
            
            is_new = group_name not in self.clusters
            
            # Add to cluster
            if is_new:
                self.clusters[group_name] = {
                    'queries': [query],
                    'strategy_performance': {}
                }
            else:
                if query not in self.clusters[group_name]['queries']:
                    self.clusters[group_name]['queries'].append(query)
            
            self._save_clusters()
            
            return group_name, is_new
        
        except Exception as e:
            # Fallback to simple clustering
            return self._fallback_cluster(query)
    
    def _fallback_cluster(self, query: str) -> Tuple[str, bool]:
        """Fallback clustering using keyword matching"""
        query_words = set(query.lower().split())
        
        best_match = None
        best_score = 0
        
        for group_name, data in self.clusters.items():
            for existing_query in data['queries']:
                existing_words = set(existing_query.lower().split())
                overlap = len(query_words & existing_words)
                
                if overlap > best_score:
                    best_score = overlap
                    best_match = group_name
        
        if best_score >= 2:  # At least 2 words in common
            self.clusters[best_match]['queries'].append(query)
            self._save_clusters()
            return best_match, False
        else:
            new_group = f"cluster_{len(self.clusters)}"
            self.clusters[new_group] = {
                'queries': [query],
                'strategy_performance': {}
            }
            self._save_clusters()
            return new_group, True
    
    def record_strategy_performance(self, cluster_name: str, strategy: str, reward: float) -> None:
        """Record how a strategy performed for a cluster"""
        if cluster_name not in self.clusters:
            return
        
        perf = self.clusters[cluster_name]['strategy_performance']
        
        if strategy not in perf:
            perf[strategy] = {'total': 0, 'reward_sum': 0.0}
        
        perf[strategy]['total'] += 1
        perf[strategy]['reward_sum'] += reward
        
        self._save_clusters()
    
    def get_best_strategy_for_cluster(self, cluster_name: str) -> Optional[str]:
        """Get the best performing strategy for a cluster"""
        if cluster_name not in self.clusters:
            return None
        
        perf = self.clusters[cluster_name]['strategy_performance']
        
        if not perf:
            return None
        
        best_strategy = None
        best_avg_reward = float('-inf')
        
        for strategy, stats in perf.items():
            if stats['total'] > 0:
                avg_reward = stats['reward_sum'] / stats['total']
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_strategy = strategy
        
        return best_strategy if best_avg_reward > 0 else None
    
    def get_cluster_info(self, cluster_name: str) -> Dict:
        """Get detailed information about a cluster"""
        if cluster_name not in self.clusters:
            return {}
        
        cluster_data = self.clusters[cluster_name]
        
        # Calculate strategy stats
        strategy_stats = {}
        for strategy, stats in cluster_data['strategy_performance'].items():
            if stats['total'] > 0:
                strategy_stats[strategy] = {
                    'uses': stats['total'],
                    'avg_reward': round(stats['reward_sum'] / stats['total'], 3)
                }
        
        return {
            'name': cluster_name,
            'query_count': len(cluster_data['queries']),
            'example_queries': cluster_data['queries'][:3],
            'strategy_performance': strategy_stats,
            'best_strategy': self.get_best_strategy_for_cluster(cluster_name)
        }
    
    def get_all_clusters_summary(self) -> List[Dict]:
        """Get summary of all clusters"""
        summary = []
        
        for cluster_name in self.clusters:
            info = self.get_cluster_info(cluster_name)
            if info:
                summary.append(info)
        
        # Sort by query count
        summary.sort(key=lambda x: x['query_count'], reverse=True)
        
        return summary
    
    def is_similar_to_cluster(self, query: str, cluster_name: str) -> bool:
        """Check if query is similar to queries in a cluster using LLM"""
        if cluster_name not in self.clusters:
            return False
        
        cluster_queries = self.clusters[cluster_name]['queries']
        
        if not cluster_queries:
            return False
        
        # Compare with most recent query from cluster
        representative_query = cluster_queries[-1]
        
        prompt = PromptTemplate.from_template(SIMILARITY_EVALUATION_PROMPT)
        formatted_prompt = prompt.format(
            query1=query,
            query2=representative_query
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            result = response.content.strip().upper()
            return 'SIMILAR' in result
        except:
            # Fallback to word overlap
            words1 = set(query.lower().split())
            words2 = set(representative_query.lower().split())
            overlap = len(words1 & words2)
            return overlap >= 2