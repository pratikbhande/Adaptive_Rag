from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Tuple
from prompt_template.templates import STRATEGY_PROMPTS, QUERY_ANALYSIS_PROMPT
from indexing import VectorStore
from reinforcement_learning import ReinforcementLearner
from processor import TextProcessor
from logger import SystemLogger
from query_clustering import QueryClusterer

class AdaptiveRAG:
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.openai_api_key = openai_api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        self.analyzer_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        self.vector_store = VectorStore(user_id, openai_api_key)
        self.rl_agent = ReinforcementLearner(user_id)
        self.processor = TextProcessor()
        self.logger = SystemLogger(user_id)
        self.clusterer = QueryClusterer(user_id, openai_api_key)
        
        self.logger.log_session_start()
    
    def index_document(self, file_content: str) -> int:
        """Process and index document"""
        try:
            chunks = self.processor.process_file(file_content)
            self.vector_store.add_documents(chunks)
            self.logger.log_indexing(len(chunks))
            return len(chunks)
        except Exception as e:
            self.logger.log_error(e, "index_document")
            raise
    
    def analyze_query_complexity(self, query: str) -> str:
        """Analyze query to determine complexity"""
        prompt = PromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        formatted_prompt = prompt.format(query=query)
        
        try:
            response = self.analyzer_llm.invoke(formatted_prompt)
            complexity = response.content.strip().lower()
            
            if "complex" in complexity:
                return "complex"
            elif "simple" in complexity:
                return "simple"
            else:
                return "moderate"
        except:
            return "moderate"
    
    def query(self, user_query: str) -> Tuple[str, Dict]:
        """Process query and generate response with adaptive strategy"""
        try:
            processed_query = self.processor.preprocess_query(user_query)
            
            # Assign query to a cluster
            cluster_name, is_new_cluster = self.clusterer.assign_cluster(processed_query)
            
            query_complexity = self.analyze_query_complexity(processed_query)
            
            # Get best strategy for this cluster (if exists)
            cluster_best_strategy = self.clusterer.get_best_strategy_for_cluster(cluster_name)
            
            # Select strategy (biased towards cluster's best if available)
            strategy, top_k = self.rl_agent.select_strategy(
                processed_query, 
                query_complexity,
                cluster_best_strategy
            )
            
            self.logger.log_query(processed_query, strategy, query_complexity)
            
            improvement_info = self.rl_agent.get_query_improvement(processed_query)
            
            # Add cluster information
            cluster_info = self.clusterer.get_cluster_info(cluster_name)
            
            retrieved_docs = self.vector_store.search(processed_query, top_k=top_k)
            
            self.logger.log_retrieval(processed_query, len(retrieved_docs), top_k)
            
            if not retrieved_docs:
                return "I couldn't find relevant information to answer your question.", {
                    "strategy": strategy,
                    "retrieved_docs": [],
                    "complexity": query_complexity,
                    "improvement_info": improvement_info,
                    "cluster_name": cluster_name,
                    "cluster_info": cluster_info,
                    "is_new_cluster": is_new_cluster
                }
            
            context = "\n\n".join([doc['content'] for doc in retrieved_docs])
            
            prompt_template = STRATEGY_PROMPTS[strategy]
            prompt = PromptTemplate.from_template(prompt_template)
            formatted_prompt = prompt.format(context=context, question=processed_query)
            
            response = self.llm.invoke(formatted_prompt)
            answer = response.content
            
            metadata = {
                "strategy": strategy,
                "top_k": top_k,
                "retrieved_docs": [doc['content'][:100] for doc in retrieved_docs],
                "complexity": query_complexity,
                "improvement_info": improvement_info,
                "cluster_name": cluster_name,
                "cluster_info": cluster_info,
                "is_new_cluster": is_new_cluster,
                "used_cluster_strategy": strategy == cluster_best_strategy
            }
            
            return answer, metadata
        
        except Exception as e:
            self.logger.log_error(e, "query")
            raise
    
    def submit_feedback(self, query: str, strategy: str, response: str, 
                       feedback: int, retrieved_docs: List[str], cluster_name: str = None) -> None:
        """Submit user feedback to RL agent and clusterer"""
        reward = 1.0 if feedback > 0 else -1.0
        
        # Record in RL agent
        self.rl_agent.record_feedback(query, strategy, response, feedback, retrieved_docs, cluster_name)
        
        # Record in clusterer for cluster-specific strategy learning
        if cluster_name:
            self.clusterer.record_strategy_performance(cluster_name, strategy, reward)
        
        self.logger.log_feedback(query, feedback, strategy)
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        base_metrics = self.rl_agent.get_performance_metrics()
        
        # Add cluster information
        clusters_summary = self.clusterer.get_all_clusters_summary()
        
        base_metrics['clusters'] = clusters_summary
        base_metrics['total_clusters'] = len(clusters_summary)
        
        return base_metrics
    
    def clear_documents(self) -> None:
        """Clear all indexed documents"""
        self.vector_store.clear_collection()