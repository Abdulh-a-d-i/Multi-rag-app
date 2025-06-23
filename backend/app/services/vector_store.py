import chromadb
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./data/vector_db"):
        """
        Initialize the ChromaDB vector store with persistent storage.
        
        Args:
            persist_directory: Directory to store the vector database
        """
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            self.collection = self.client.get_or_create_collection(
                name="rag_docs",
                metadata={"hnsw:space": "cosine"}  # Using cosine similarity
            )
        except Exception as e:
            logger.error(f"Collection initialization failed: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dictionaries containing 'text' and 'metadata'
            
        Returns:
            bool: True if successful
        """
        try:
            # Generate unique IDs for each document
            ids = [f"doc_{i}_{doc.get('metadata', {}).get('page', 0)}" 
                  for i, doc in enumerate(documents)]
            
            texts = [doc["text"] for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            return True
        except Exception as e:
            logger.error(f"Document addition failed: {str(e)}")
            raise

    def query(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: The embedding vector to query with
            n_results: Number of results to return
            
        Returns:
            List of dictionaries containing matched documents and metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )
            
            return [
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
                for i in range(len(results["documents"][0]))
            ]
        except Exception as e:
            logger.error(f"Vector query failed: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the collection"""
        return {
            "count": self.collection.count(),
            "name": self.collection.name,
            "metadata": self.collection.metadata
        }

# Initialize singleton instance
vector_store = VectorStore()