"""Configuration settings for the Self-Adaptive RAG system"""

# Text Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector Store
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Models
CHAT_MODEL = "gpt-4o-mini"
ANALYZER_MODEL = "gpt-4o-mini"
CHAT_TEMPERATURE = 0.7
ANALYZER_TEMPERATURE = 0.0

# Reinforcement Learning
RL_DATA_PATH = "./rl_data"
EPSILON = 0.2 # Reduced exploration rate with 5 strategies
STRATEGIES = ["concise", "detailed", "structured", "example_driven", "analytical"]
STRATEGY_K_VALUES = {
    "concise": 3,
    "detailed": 5,
    "structured": 4,
    "example_driven": 4,
    "analytical": 6
}
CLUSTER_STRATEGY_BIAS = 0.8  # 70% chance to use cluster's best strategy

# Query Similarity
SIMILARITY_THRESHOLD = 0.6

# Feedback
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -1.0