import streamlit as st
import uuid
import os
from rag import AdaptiveRAG

st.set_page_config(page_title="Self-Adaptive RAG", layout="wide", page_icon="🤖")

def initialize_session():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'document_indexed' not in st.session_state:
        st.session_state.document_indexed = False
    
    if 'pending_feedback' not in st.session_state:
        st.session_state.pending_feedback = {}

def handle_feedback(message_idx: int, feedback_type: str):
    """Handle thumbs up/down feedback"""
    if message_idx in st.session_state.pending_feedback:
        msg_data = st.session_state.pending_feedback[message_idx]
        
        feedback_value = 1 if feedback_type == "up" else -1
        
        st.session_state.rag_system.submit_feedback(
            query=msg_data['query'],
            strategy=msg_data['strategy'],
            response=msg_data['response'],
            feedback=feedback_value,
            retrieved_docs=msg_data['retrieved_docs'],
            cluster_name=msg_data.get('cluster_name')
        )
        
        st.session_state.chat_history[message_idx]['feedback'] = feedback_type
        del st.session_state.pending_feedback[message_idx]
        
        st.success(f"✅ Feedback recorded! System is learning from your input.")
        st.rerun()

def main():
    initialize_session()
    
    st.title("🤖 Self-Adaptive RAG System")
    st.markdown("*Learns from your feedback to improve responses*")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.info(f"**User ID:** `{st.session_state.user_id}`")
        
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to continue")
            return
        
        st.divider()
        
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        
        if uploaded_file:
            if st.button("Index Document", type="primary"):
                with st.spinner("Processing and indexing document..."):
                    if st.session_state.rag_system is None:
                        st.session_state.rag_system = AdaptiveRAG(
                            st.session_state.user_id,
                            api_key
                        )
                    
                    file_content = uploaded_file.read().decode('utf-8')
                    num_chunks = st.session_state.rag_system.index_document(file_content)
                    
                    st.session_state.document_indexed = True
                    st.success(f"✅ Indexed {num_chunks} chunks")
        
        if st.session_state.document_indexed:
            if st.button("Clear Documents"):
                st.session_state.rag_system.clear_documents()
                st.session_state.document_indexed = False
                st.session_state.chat_history = []
                st.success("Documents cleared")
                st.rerun()
        
        st.divider()
        
        if st.session_state.rag_system and st.session_state.document_indexed:
            st.header("📊 Learning Metrics")
            metrics = st.session_state.rag_system.get_metrics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", metrics['total_interactions'])
            with col2:
                if metrics['total_interactions'] > 0:
                    satisfaction = metrics['positive_feedback'] / metrics['total_interactions'] * 100
                    st.metric("Satisfaction", f"{satisfaction:.1f}%")
                else:
                    st.metric("Satisfaction", "N/A")
            
            st.subheader("Strategy Performance")
            for strategy, perf in metrics['strategy_performance'].items():
                if perf['total_uses'] > 0:
                    st.write(f"**{strategy.replace('_', ' ').title()}**")
                    st.write(f"Uses: {perf['total_uses']} | Win Rate: {perf['win_rate']:.2%}")
            
            if metrics.get('total_clusters', 0) > 0:
                st.divider()
                st.subheader("Query Clusters")
                st.metric("Total Clusters", metrics['total_clusters'])
                
                with st.expander("View Cluster Details"):
                    for cluster in metrics['clusters'][:5]:
                        st.write(f"**{cluster['name']}** ({cluster['query_count']} queries)")
                        if cluster.get('best_strategy'):
                            st.write(f"  Best Strategy: {cluster['best_strategy'].replace('_', ' ').title()}")
                        st.write(f"  Examples: {', '.join(cluster['example_queries'][:2])}")
    
    if not st.session_state.document_indexed:
        st.info("👈 Please upload and index a document to start chatting")
        return
    
    tab1, tab2 = st.tabs(["💬 Chat", "🧠 Learning Dashboard"])
    
    with tab1:
        st.header("Chat with your Document")
        
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(message['query'])
            
            with st.chat_message("assistant"):
                st.write(message['response'])
                
                # Show cluster information
                if 'cluster_info' in message and message['cluster_info']:
                    cluster_info = message['cluster_info']
                    with st.expander("🔍 Query Cluster Analysis"):
                        st.write(f"**Cluster:** {message.get('cluster_name', 'N/A')}")
                        if message.get('is_new_cluster'):
                            st.info("This is a new type of query!")
                        else:
                            st.write(f"Similar queries in cluster: {cluster_info.get('query_count', 0)}")
                            if cluster_info.get('best_strategy'):
                                st.write(f"**Best performing strategy for this type:** {cluster_info['best_strategy'].replace('_', ' ').title()}")
                                if message.get('used_cluster_strategy'):
                                    st.success("✓ Used cluster's best strategy")
                
                if 'improvement_info' in message and message['improvement_info']['has_similar']:
                    with st.expander("🔄 Learning Applied"):
                        st.write(f"**Strategy Used:** {message['strategy'].replace('_', ' ').title()}")
                        st.write("Similar queries found in history:")
                        for sq in message['improvement_info']['similar_queries']:
                            feedback_emoji = "👍" if sq['feedback'] > 0 else "👎"
                            st.write(f"- {sq['query']} ({sq['strategy']}) {feedback_emoji}")
                
                if 'feedback' not in message:
                    col1, col2, col3 = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"up_{idx}"):
                            handle_feedback(idx, "up")
                    with col2:
                        if st.button("👎", key=f"down_{idx}"):
                            handle_feedback(idx, "down")
                else:
                    feedback_emoji = "👍" if message['feedback'] == "up" else "👎"
                    st.caption(f"Feedback: {feedback_emoji}")
        
        query = st.chat_input("Ask a question about your document...")
        
        if query:
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, metadata = st.session_state.rag_system.query(query)
                    
                    st.write(response)
                    
                    # Show cluster information
                    if metadata.get('cluster_info'):
                        cluster_info = metadata['cluster_info']
                        with st.expander("🔍 Query Cluster Analysis"):
                            st.write(f"**Cluster:** {metadata.get('cluster_name', 'N/A')}")
                            if metadata.get('is_new_cluster'):
                                st.info("This is a new type of query!")
                            else:
                                st.write(f"Similar queries in cluster: {cluster_info.get('query_count', 0)}")
                                if cluster_info.get('best_strategy'):
                                    st.write(f"**Best performing strategy for this type:** {cluster_info['best_strategy'].replace('_', ' ').title()}")
                                    if metadata.get('used_cluster_strategy'):
                                        st.success("✓ Used cluster's best strategy")
                    
                    if metadata['improvement_info']['has_similar']:
                        with st.expander("🔄 Learning Applied"):
                            st.write(f"**Strategy Used:** {metadata['strategy'].replace('_', ' ').title()}")
                            st.write("Similar queries found in history:")
                            for sq in metadata['improvement_info']['similar_queries']:
                                feedback_emoji = "👍" if sq['feedback'] > 0 else "👎"
                                st.write(f"- {sq['query']} ({sq['strategy']}) {feedback_emoji}")
                    
                    message_idx = len(st.session_state.chat_history)
                    
                    st.session_state.chat_history.append({
                        'query': query,
                        'response': response,
                        'strategy': metadata['strategy'],
                        'improvement_info': metadata['improvement_info'],
                        'cluster_name': metadata.get('cluster_name'),
                        'cluster_info': metadata.get('cluster_info'),
                        'is_new_cluster': metadata.get('is_new_cluster'),
                        'used_cluster_strategy': metadata.get('used_cluster_strategy')
                    })
                    
                    st.session_state.pending_feedback[message_idx] = {
                        'query': query,
                        'response': response,
                        'strategy': metadata['strategy'],
                        'retrieved_docs': metadata['retrieved_docs'],
                        'cluster_name': metadata.get('cluster_name')
                    }
                    
                    st.rerun()
    
    with tab2:
        st.header("🧠 Reinforcement Learning Dashboard")
        
        if st.session_state.rag_system:
            metrics = st.session_state.rag_system.get_metrics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Interactions", metrics['total_interactions'])
            with col2:
                st.metric("Positive Feedback", metrics['positive_feedback'])
            with col3:
                st.metric("Negative Feedback", metrics['negative_feedback'])
            
            if metrics['total_interactions'] > 0:
                satisfaction_rate = metrics['positive_feedback'] / metrics['total_interactions']
                st.progress(satisfaction_rate, text=f"Overall Satisfaction: {satisfaction_rate:.1%}")
            
            st.divider()
            
            st.subheader("Strategy Performance Comparison")
            
            strategy_data = []
            for strategy, perf in metrics['strategy_performance'].items():
                strategy_data.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Total Uses': perf['total_uses'],
                    'Win Rate': perf['win_rate'],
                    'Avg Reward': perf['avg_reward']
                })
            
            if strategy_data:
                import pandas as pd
                df = pd.DataFrame(strategy_data)
                st.dataframe(df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Win Rate by Strategy")
                    st.bar_chart(df.set_index('Strategy')['Win Rate'])
                with col2:
                    st.subheader("Usage Distribution")
                    st.bar_chart(df.set_index('Strategy')['Total Uses'])
            
            st.divider()
            
            # Cluster Analysis
            if metrics.get('total_clusters', 0) > 0:
                st.subheader("Query Clustering Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Query Clusters", metrics['total_clusters'])
                with col2:
                    clusters_with_best = sum(1 for c in metrics['clusters'] if c.get('best_strategy'))
                    st.metric("Clusters with Best Strategy", clusters_with_best)
                
                st.write("**Top Query Clusters:**")
                for cluster in metrics['clusters'][:8]:
                    with st.expander(f"📊 {cluster['name']} ({cluster['query_count']} queries)"):
                        st.write("**Example queries:**")
                        for ex in cluster['example_queries']:
                            st.write(f"- {ex}")
                        
                        if cluster.get('best_strategy'):
                            st.success(f"**Best Strategy:** {cluster['best_strategy'].replace('_', ' ').title()}")
                        
                        if cluster.get('strategy_performance'):
                            st.write("**Strategy Performance in this cluster:**")
                            for strat, perf in cluster['strategy_performance'].items():
                                st.write(f"- {strat.replace('_', ' ').title()}: {perf['uses']} uses, "
                                       f"avg reward: {perf['avg_reward']:.2f}")
            
            st.divider()
            
            st.subheader("Recent Query History")
            recent_feedback = st.session_state.rag_system.rl_agent.feedback_history[-10:]
            if recent_feedback:
                for i, entry in enumerate(reversed(recent_feedback), 1):
                    feedback_icon = "👍" if entry['reward'] > 0 else "👎"
                    with st.expander(f"{feedback_icon} {entry['query'][:60]}..."):
                        st.write(f"**Strategy:** {entry['strategy']}")
                        st.write(f"**Response:** {entry['response']}...")
                        st.write(f"**Timestamp:** {entry['timestamp']}")
            
            st.divider()
            
            st.subheader("How Advanced RL Works")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Learning Algorithm:**
                - **Method**: Contextual Multi-Armed Bandit with Clustering
                - **Exploration Rate**: 15% (tries new strategies)
                - **Exploitation**: 85% (uses proven strategies)
                - **UCB Selection**: Balances reward & uncertainty
                - **Cluster Bias**: 70% preference for cluster's best strategy
                """)
            
            with col2:
                st.markdown("""
                **Feedback & Clustering:**
                - 👍 = +1.0 reward (strategy wins)
                - 👎 = -1.0 reward (strategy loses)
                - LLM groups similar queries semantically
                - Tracks best strategy per query type
                - Adapts based on query patterns
                """)
            
            st.info("""
            **🎯 Advanced Learning Features:**
            
            1. **Query Clustering**: LLM analyzes each query and groups semantically similar questions together
            
            2. **Cluster-Specific Learning**: System learns which strategies work best for each type of query
            
            3. **Smart Strategy Selection**: When you ask a query similar to previous ones, the system:
               - Identifies the query cluster
               - Checks if that cluster has a proven best strategy
               - Uses that strategy 70% of the time (if performance is good)
               - Still explores 15% to discover better approaches
            
            4. **5 Response Strategies**:
               - **Concise**: Brief, direct answers
               - **Detailed**: Comprehensive explanations
               - **Structured**: Organized with clear sections
               - **Example-Driven**: Focuses on concrete examples
               - **Analytical**: Deep analysis from multiple angles
            
            5. **Continuous Improvement**: Every feedback helps the system learn:
               - Which strategies work globally
               - Which strategies work for specific query types
               - Which types of questions you ask most often
            """)
        else:
            st.info("Start chatting to see learning metrics")

if __name__ == "__main__":
    main()