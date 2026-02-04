import numpy as np
import pickle
import os
from typing import List, Dict, Any

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.documents = [] # Stores text content
        self.metadatas = [] # Stores metadata (source, page, etc.)

    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None):
        """
        Adds documents and their embeddings to the store.
        """
        if not texts or not embeddings:
            return

        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match.")

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts.")

        self.documents.extend(texts)
        self.vectors.extend(embeddings)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))

    def search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches for the k most similar documents to the query_embedding.
        Returns a list of dictionaries containing 'text', 'metadata', and 'score'.
        """
        if not self.vectors:
            return []

        query_vec = np.array(query_embedding)
        # Normalize query vector
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return []
        query_vec = query_vec / norm_q

        # Convert simple list to numpy array for efficient calculation
        # We need to ensure all vectors are of the same dimension and valid
        doc_vectors = np.array(self.vectors)
        
        # Calculate Cosine Similarity: (A . B) / (||A|| * ||B||)
        # Assuming embeddings are already normalized from the API, but normalizing again to be safe
        norms_d = np.linalg.norm(doc_vectors, axis=1)
        # Avoid division by zero
        norms_d[norms_d == 0] = 1e-10
        
        similarities = np.dot(doc_vectors, query_vec) / norms_d
        
        # Get top k indices
        # argsort sorts in ascending order, so we take the last k and reverse them
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(similarities[idx])
            })
            
        return results

    def save_to_disk(self, path: str):
        """Saves the store to a pickle file."""
        data = {
            "vectors": self.vectors,
            "documents": self.documents,
            "metadatas": self.metadatas
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Vector store saved to {path}")

    @classmethod
    def load_from_disk(cls, path: str):
        """Loads the store from a pickle file."""
        if not os.path.exists(path):
            return cls() # Return empty store if no file exists
            
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        store = cls()
        store.vectors = data["vectors"]
        store.documents = data["documents"]
        store.metadatas = data.get("metadatas", [{}] * len(store.documents)) # Handle legacy saves if any
        print(f"Vector store loaded from {path} with {len(store.documents)} documents.")
        return store
