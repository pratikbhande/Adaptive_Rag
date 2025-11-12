import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import os

class VectorStore:
    def __init__(self, user_id: str, openai_api_key: str):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        os.makedirs("./chroma_db", exist_ok=True)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        self.collection_name = f"docs_{user_id}"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, texts: List[str]) -> None:
        """Add document chunks to vector store"""
        embeddings = self.embeddings.embed_documents(texts)
        
        ids = [f"doc_{i}" for i in range(len(texts))]
        metadatas = [{"chunk_id": i, "user_id": self.user_id} for i in range(len(texts))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })
        
        return retrieved_docs
    
    def clear_collection(self) -> None:
        """Clear all documents for this user"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            pass