"""
RAG Pipeline Module
End-to-end Retrieval-Augmented Generation pipeline orchestrator.
Connects the retriever and generator for seamless query processing.
"""

import os
import sys
from typing import Generator, Optional

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator as LLMGenerator
from src.document_loader import load_document, load_all_documents
from src.chunker import create_chunks, save_chunks, print_chunk_stats


class RAGPipeline:
    """
    End-to-end RAG pipeline that orchestrates document ingestion,
    retrieval, and response generation.
    
    Attributes:
        embedder: Embedder instance
        vector_store: VectorStore instance
        retriever: Retriever instance
        generator: LLMGenerator instance
        top_k: Number of chunks to retrieve per query
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        db_dir: str = None,
        top_k: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """
        Initialize all components of the RAG pipeline.
        
        Args:
            embedding_model: Sentence-transformer model for embeddings
            llm_model: Groq model for text generation
            db_dir: Directory for persistent vector database
            top_k: Number of chunks to retrieve per query
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens in LLM response
        """
        if db_dir is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectordb")

        self.top_k = top_k

        print("\n" + "=" * 60)
        print("  INITIALIZING RAG PIPELINE")
        print("=" * 60)

        # Initialize components
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(db_dir=db_dir)
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=top_k,
        )
        self.generator = LLMGenerator(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print("=" * 60)
        print("  RAG PIPELINE READY")
        print("=" * 60 + "\n")

    def ingest_documents(
        self,
        data_dir: str,
        chunks_dir: str = None,
        min_chunk_words: int = 100,
        max_chunk_words: int = 300,
        overlap_words: int = 30,
    ) -> list[dict]:
        """
        Process and ingest documents into the vector database.
        
        Args:
            data_dir: Directory containing source documents
            chunks_dir: Directory to save processed chunks
            min_chunk_words: Minimum words per chunk
            max_chunk_words: Maximum words per chunk
            overlap_words: Overlap between consecutive chunks
            
        Returns:
            List of all generated chunks
        """
        if chunks_dir is None:
            chunks_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "chunks"
            )

        print("\n[Pipeline] Starting document ingestion...")

        # Step 1: Load documents
        documents = load_all_documents(data_dir)
        if not documents:
            print("[Pipeline] No documents found to ingest!")
            return []

        # Step 2: Chunk documents
        all_chunks = []
        for filename, text in documents.items():
            chunks = create_chunks(
                text=text,
                min_chunk_words=min_chunk_words,
                max_chunk_words=max_chunk_words,
                overlap_words=overlap_words,
                source_file=filename,
            )
            all_chunks.extend(chunks)

        print_chunk_stats(all_chunks)

        # Step 3: Save chunks
        save_chunks(all_chunks, chunks_dir)

        # Step 4: Generate embeddings
        embeddings = self.embedder.embed_chunks(all_chunks)

        # Step 5: Clear existing data and store new embeddings
        self.vector_store.clear_collection()
        self.vector_store.add_chunks(all_chunks, embeddings)

        print(f"\n[Pipeline] Ingestion complete! "
              f"{len(all_chunks)} chunks indexed.")

        return all_chunks

    def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Process a query through the full RAG pipeline (non-streaming).
        
        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with 'answer', 'sources', and 'metadata'
        """
        k = top_k or self.top_k

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=k)

        # Step 2: Format context
        context = self.retriever.format_context(retrieved_chunks)

        # Step 3: Generate response
        answer = self.generator.generate(context, question)

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "metadata": {
                "model": self.generator.model,
                "chunks_retrieved": len(retrieved_chunks),
                "top_k": k,
            },
        }

    def query_stream(
        self, question: str, top_k: Optional[int] = None
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """
        Process a query with streaming response.
        
        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (token generator, retrieved source chunks)
        """
        k = top_k or self.top_k

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=k)

        # Step 2: Format context
        context = self.retriever.format_context(retrieved_chunks)

        # Step 3: Return streaming generator and sources
        token_stream = self.generator.generate_stream(context, question)

        return token_stream, retrieved_chunks

    def get_pipeline_info(self) -> dict:
        """Return information about all pipeline components."""
        return {
            "embedder": self.embedder.get_model_info(),
            "vector_store": self.vector_store.get_collection_stats(),
            "generator": self.generator.get_model_info(),
            "retriever": {"top_k": self.top_k},
        }

    def is_ready(self) -> bool:
        """Check if the pipeline has indexed documents and is ready for queries."""
        return self.vector_store.collection_exists()


if __name__ == "__main__":
    # Quick test - ingest and query
    pipeline = RAGPipeline()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if os.path.exists(data_dir) and os.listdir(data_dir):
        pipeline.ingest_documents(data_dir)

        if pipeline.is_ready():
            result = pipeline.query("What are the main terms of service?")
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources used: {len(result['sources'])}")
    else:
        print(f"\nNo documents found in '{data_dir}'. "
              "Please add documents before running the pipeline.")
