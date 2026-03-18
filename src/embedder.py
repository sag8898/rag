"""
Embedder Module
Generates vector embeddings for text chunks using sentence-transformers.
Uses the all-MiniLM-L6-v2 model (384-dimensional embeddings).
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional


# Default model - lightweight, fast, and produces good quality embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Text embedding generator using sentence-transformers.
    
    Attributes:
        model_name: Name of the sentence-transformer model
        model: Loaded SentenceTransformer model instance
        embedding_dim: Dimensionality of the output embeddings
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedder with a sentence-transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        self.model_name = model_name
        print(f"[Embedder] Loading model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Generate embeddings for a batch of text strings.
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embedding vectors
        """
        print(f"[Embedder] Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        print(f"[Embedder] Generated {len(embeddings)} embeddings "
              f"(dim={self.embedding_dim})")
        return embeddings.tolist()

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """
        Generate embeddings for a list of chunk dictionaries.
        
        Args:
            chunks: List of chunk dicts (must have 'text' key)
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk["text"] for chunk in chunks]
        return self.embed_texts(texts)

    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length,
        }


if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    print(f"\nModel info: {embedder.get_model_info()}")

    test_texts = [
        "The user must agree to the terms of service.",
        "Privacy policy explains how we handle your data.",
        "The weather today is sunny and warm.",
    ]

    embeddings = embedder.embed_texts(test_texts)
    print(f"\nEmbedding shapes: {len(embeddings)} x {len(embeddings[0])}")

    # Show similarity between texts
    from numpy import dot
    from numpy.linalg import norm

    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            a, b = np.array(embeddings[i]), np.array(embeddings[j])
            similarity = dot(a, b) / (norm(a) * norm(b))
            print(f"Similarity '{test_texts[i][:40]}...' <-> "
                  f"'{test_texts[j][:40]}...': {similarity:.4f}")
