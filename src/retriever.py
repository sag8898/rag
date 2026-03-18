"""
Retriever Module
Performs semantic search over indexed document chunks using the vector database.
Combines embedding generation and vector store querying.
"""

from src.embedder import Embedder
from src.vector_store import VectorStore
from typing import Optional


class Retriever:
    """
    Semantic search retriever that finds relevant document chunks
    for a given query using embedding similarity.
    
    Attributes:
        embedder: Embedder instance for generating query embeddings
        vector_store: VectorStore instance for similarity search
        top_k: Default number of results to retrieve
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance for query embedding
            vector_store: VectorStore instance for searching
            top_k: Default number of chunks to retrieve
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: User's natural language query
            top_k: Number of chunks to retrieve (overrides default)
            
        Returns:
            List of dictionaries containing retrieved chunks with:
            - text: The chunk text
            - source_file: Origin document name
            - chunk_index: Position in the original document
            - distance: Similarity distance (lower = more similar)
            - relevance_score: Normalized relevance score (0-1, higher = better)
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
        )

        # Format results
        retrieved_chunks = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert cosine distance to similarity score (1 - distance for cosine)
                relevance_score = max(0, 1 - distance)

                chunk = {
                    "text": results["documents"][0][i],
                    "source_file": results["metadatas"][0][i].get("source_file", "unknown"),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    "word_count": results["metadatas"][0][i].get("word_count", 0),
                    "distance": distance,
                    "relevance_score": round(relevance_score, 4),
                }
                retrieved_chunks.append(chunk)

        return retrieved_chunks

    def format_context(self, chunks: list[dict], max_chunks: int = 5) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.
        
        Args:
            chunks: List of retrieved chunk dictionaries
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted context string with source references
        """
        if not chunks:
            return "No relevant information found in the documents."

        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            source = chunk.get("source_file", "unknown")
            score = chunk.get("relevance_score", 0)
            context_parts.append(
                f"[Source: {source} | Chunk {chunk['chunk_index']} | "
                f"Relevance: {score:.2f}]\n{chunk['text']}"
            )

        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    print("Retriever module loaded. Use with Embedder and VectorStore instances.")
