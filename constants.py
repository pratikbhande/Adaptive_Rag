"""Constants for the Self-Adaptive RAG system"""

# Directory paths
CHROMA_DB_DIR = "./chroma_db"
RL_DATA_DIR = "./rl_data"
LOGS_DIR = "./logs"
PROMPTS_DIR = "./prompts"

# Text processing
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Models
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TEMPERATURE = 0.7
ANALYZER_TEMPERATURE = 0.0

# Retrieval strategies
STRATEGIES = ["concise", "detailed", "structured", "example_driven", "analytical"]
STRATEGY_TOP_K = {
    "concise": 3,
    "detailed": 5,
    "structured": 4,
    "example_driven": 4,
    "analytical": 6
}

# Reinforcement learning
EPSILON = 0.2  # Exploration rate
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -1.0
UCB_CONFIDENCE = 2.0

# Query similarity
SIMILARITY_THRESHOLD = 0.6
KEYWORD_MIN_LENGTH = 4

# Feedback
FEEDBACK_POSITIVE = 1
FEEDBACK_NEGATIVE = -1

# UI
APP_TITLE = "Self-Adaptive RAG System"
APP_ICON = "🤖"
MAX_DISPLAY_TEXT = 100

# File upload
ALLOWED_FILE_TYPES = ["txt"]
MAX_FILE_SIZE_MB = 10