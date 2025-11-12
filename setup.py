"""Setup script for Self-Adaptive RAG system"""

from setuptools import setup, find_packages

setup(
    name="self-adaptive-rag",
    version="1.0.0",
    description="Self-Adaptive RAG system with reinforcement learning",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "langchain",
        "langchain-openai",
        "langchain-community",
        "chromadb",
        "openai",
        "numpy",
        "pandas",
        "tiktoken",
    ],
    python_requires=">=3.9",
)