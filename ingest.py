"""
Document Ingestion Script
Run this script to process documents and build the vector database.

Usage:
    python ingest.py
    python ingest.py --data-dir ./data --chunks-dir ./chunks
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.document_loader import load_all_documents
from src.chunker import create_chunks, save_chunks, print_chunk_stats
from src.embedder import Embedder
from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database")
    parser.add_argument("--data-dir", default="./data", help="Directory containing documents")
    parser.add_argument("--chunks-dir", default="./chunks", help="Directory to save chunks")
    parser.add_argument("--db-dir", default="./vectordb", help="Directory for vector database")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum words per chunk")
    parser.add_argument("--max-words", type=int, default=300, help="Maximum words per chunk")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap words between chunks")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    # Validate data directory
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
        print(f"\n[ERROR] Data directory '{args.data_dir}' was empty.")
        print("Please place your documents (PDF, TXT, DOCX) in the 'data/' folder.")
        print("Then run this script again.")
        sys.exit(1)

    # Step 1: Load documents
    print("\n📄 Step 1/4: Loading documents...")
    documents = load_all_documents(args.data_dir)
    if not documents:
        print("\n[ERROR] No supported documents found in the data directory.")
        print("Supported formats: .pdf, .txt, .docx")
        sys.exit(1)

    # Step 2: Chunk documents
    print("\n✂️  Step 2/4: Chunking documents...")
    all_chunks = []
    for filename, text in documents.items():
        chunks = create_chunks(
            text=text,
            min_chunk_words=args.min_words,
            max_chunk_words=args.max_words,
            overlap_words=args.overlap,
            source_file=filename,
        )
        all_chunks.extend(chunks)
        print(f"  → {filename}: {len(chunks)} chunks")

    print_chunk_stats(all_chunks)

    # Save chunks to file
    save_chunks(all_chunks, args.chunks_dir)

    # Step 3: Generate embeddings
    print("\n🧮 Step 3/4: Generating embeddings...")
    embedder = Embedder(model_name=args.embedding_model)
    embeddings = embedder.embed_chunks(all_chunks)

    # Step 4: Store in vector database
    print("\n💾 Step 4/4: Storing in vector database...")
    vector_store = VectorStore(db_dir=args.db_dir)
    vector_store.clear_collection()
    vector_store.add_chunks(all_chunks, embeddings)

    # Summary
    stats = vector_store.get_collection_stats()
    print("\n" + "=" * 60)
    print("  ✅ INGESTION COMPLETE!")
    print("=" * 60)
    print(f"  Documents processed:  {len(documents)}")
    print(f"  Total chunks created: {len(all_chunks)}")
    print(f"  Embeddings generated: {len(embeddings)}")
    print(f"  Vector DB location:   {args.db_dir}")
    print(f"  Chunks saved to:      {args.chunks_dir}")
    print(f"  Embedding model:      {args.embedding_model}")
    print(f"  Embedding dimension:  {embedder.embedding_dim}")
    print("=" * 60)
    print("\n🚀 You can now run the chatbot with: streamlit run app.py\n")


if __name__ == "__main__":
    main()
