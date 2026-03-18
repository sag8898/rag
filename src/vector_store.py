"""
Vector Store Module
ChromaDB-based vector database for storing and querying document embeddings.
Supports persistent storage and metadata-based filtering.
"""

import os
import chromadb
from chromadb.config import Settings
from typing import Optional


# Default paths
DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectordb")
COLLECTION_NAME = "document_chunks"


class VectorStore:
    """
    ChromaDB vector database wrapper for document chunk storage and retrieval.
    
    Attributes:
        db_dir: Directory for persistent ChromaDB storage
        client: ChromaDB client instance
        collection: ChromaDB collection for document chunks
    """

    def __init__(self, db_dir: str = DEFAULT_DB_DIR, collection_name: str = COLLECTION_NAME):
        """
        Initialize the vector store with persistent ChromaDB.
        
        Args:
            db_dir: Directory path for persistent database storage
            collection_name: Name of the ChromaDB collection
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        os.makedirs(db_dir, exist_ok=True)

        print(f"[VectorStore] Initializing ChromaDB at '{db_dir}'...")
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"[VectorStore] Collection '{collection_name}' ready "
              f"({self.collection.count()} existing documents)")

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'text', and metadata
            embeddings: Corresponding embedding vectors for each chunk
        """
        if not chunks:
            print("[VectorStore] No chunks to add.")
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.get("source_file", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "word_count": chunk.get("word_count", 0),
            }
            for chunk in chunks
        ]

        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"[VectorStore] Added {len(chunks)} chunks to collection "
              f"(total: {self.collection.count()})")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with 'ids', 'documents', 'metadatas', and 'distances'
        """
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)
        return results

    def get_collection_stats(self) -> dict:
        """Return statistics about the vector store collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "db_directory": self.db_dir,
        }

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Collection '{self.collection_name}' cleared.")

    def collection_exists(self) -> bool:
        """Check if the collection has any documents."""
        return self.collection.count() > 0


if __name__ == "__main__":
    # Quick test
    store = VectorStore()
    print(f"\nCollection stats: {store.get_collection_stats()}")
