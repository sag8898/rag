# RAG Chatbot - Source Package
"""
This package contains the core modules for the RAG (Retrieval-Augmented Generation) pipeline:
- document_loader: Load and clean documents (PDF, TXT, DOCX)
- chunker: Sentence-aware text chunking
- embedder: Generate embeddings using sentence-transformers
- vector_store: ChromaDB vector database operations
- retriever: Semantic search over indexed documents
- generator: LLM response generation with streaming (via Groq)
- rag_pipeline: End-to-end RAG pipeline orchestrator
"""
