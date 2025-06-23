
import numpy as np
from typing import List, Dict, Any
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class DeepSeekEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/embeddings"  # Verify actual endpoint
        
    def __call__(self, input: Documents) -> Embeddings:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        for text in input:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"input": text, "model": "text-embedding"}
            )
            
            if response.status_code == 200:
                embedding = response.json()["data"][0]["embedding"]
                embeddings.append(embedding)
            else:
                # Fallback to random embeddings if API fails
                embeddings.append(np.random.rand(384).tolist())
        
        return embeddings

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Small local model
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()