"""
Text Chunker Module
Implements sentence-aware text chunking with configurable chunk size and overlap.
Ensures chunks are between 100-300 words while respecting sentence boundaries.
"""

import re
import json
import os
from pathlib import Path
from typing import Optional


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex-based sentence boundary detection.
    Handles common abbreviations and edge cases.
    """
    # Common abbreviations that shouldn't trigger sentence splits
    abbreviations = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "Inc", "Ltd",
                     "Corp", "vs", "etc", "viz", "al", "dept", "est", "vol",
                     "Fig", "fig", "No", "no", "Sec", "sec", "Art", "Ch"}

    # First pass: split on sentence-ending punctuation followed by whitespace
    # and an uppercase letter or quote
    raw_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)

    # Second pass: re-join false splits caused by abbreviations
    sentences = []
    buffer = ""
    for segment in raw_splits:
        if buffer:
            # Check if the previous segment ended with an abbreviation
            buffer_stripped = buffer.rstrip()
            # Get the last word before the period
            last_word_match = re.search(r'(\w+)\.$', buffer_stripped)
            if last_word_match and last_word_match.group(1) in abbreviations:
                buffer = buffer + " " + segment
                continue
            else:
                sentences.append(buffer.strip())
                buffer = segment
        else:
            buffer = segment

    if buffer:
        sentences.append(buffer.strip())

    # Clean up: remove empty strings
    return [s for s in sentences if s]


def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


def create_chunks(
    text: str,
    min_chunk_words: int = 100,
    max_chunk_words: int = 300,
    overlap_words: int = 30,
    source_file: str = "unknown"
) -> list[dict]:
    """
    Create sentence-aware text chunks with overlap.
    
    Each chunk contains between min_chunk_words and max_chunk_words words,
    split at sentence boundaries. Adjacent chunks share overlap_words words
    for context continuity.
    
    Args:
        text: The full text to chunk
        min_chunk_words: Minimum words per chunk (default: 100)
        max_chunk_words: Maximum words per chunk (default: 300)
        overlap_words: Number of overlapping words between consecutive chunks (default: 30)
        source_file: Name of the source document for metadata
        
    Returns:
        List of chunk dictionaries with text, metadata, and word count
    """
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_words = count_words(sentence)

        # If adding this sentence exceeds max, finalize the current chunk
        if current_word_count + sentence_words > max_chunk_words and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"{source_file}_chunk_{chunk_index}",
                "text": chunk_text,
                "word_count": count_words(chunk_text),
                "chunk_index": chunk_index,
                "source_file": source_file,
            })
            chunk_index += 1

            # Create overlap by keeping some sentences from the end
            overlap_sentence_list = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_words = count_words(s)
                if overlap_count + s_words <= overlap_words:
                    overlap_sentence_list.insert(0, s)
                    overlap_count += s_words
                else:
                    break

            current_sentences = overlap_sentence_list
            current_word_count = overlap_count

        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        # If the last chunk is too small, merge with previous if possible
        if count_words(chunk_text) < min_chunk_words and chunks:
            prev_chunk = chunks[-1]
            merged_text = prev_chunk["text"] + " " + chunk_text
            chunks[-1] = {
                "chunk_id": prev_chunk["chunk_id"],
                "text": merged_text,
                "word_count": count_words(merged_text),
                "chunk_index": prev_chunk["chunk_index"],
                "source_file": source_file,
            }
        else:
            chunks.append({
                "chunk_id": f"{source_file}_chunk_{chunk_index}",
                "text": chunk_text,
                "word_count": count_words(chunk_text),
                "chunk_index": chunk_index,
                "source_file": source_file,
            })

    return chunks


def save_chunks(chunks: list[dict], output_dir: str) -> str:
    """
    Save chunks to a JSON file in the output directory.
    
    Args:
        chunks: List of chunk dictionaries
        output_dir: Directory to save the chunks file
        
    Returns:
        Path to the saved chunks file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[Chunker] Saved {len(chunks)} chunks to '{output_path}'")
    return output_path


def load_chunks(chunks_dir: str) -> list[dict]:
    """Load chunks from a previously saved JSON file."""
    chunks_path = os.path.join(chunks_dir, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_chunk_stats(chunks: list[dict]) -> None:
    """Print statistics about the generated chunks."""
    if not chunks:
        print("[Chunker] No chunks to analyze.")
        return

    word_counts = [c["word_count"] for c in chunks]
    print(f"\n{'='*50}")
    print(f"  CHUNK STATISTICS")
    print(f"{'='*50}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total words:  {sum(word_counts):,}")
    print(f"  Min words:    {min(word_counts)}")
    print(f"  Max words:    {max(word_counts)}")
    print(f"  Avg words:    {sum(word_counts) / len(word_counts):.0f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Quick test with sample text
    sample_text = """
    This is the first sentence of the document. It contains important information about the terms of service.
    The user must agree to all terms before using the platform. Violation of these terms may result in account suspension.
    
    Privacy is important to us. We collect minimal personal data necessary for service operation. 
    Your data is encrypted and stored securely. We do not sell your personal information to third parties.
    You have the right to request deletion of your data at any time. We will comply with all applicable privacy laws.
    
    Payment terms are as follows. All fees are non-refundable unless otherwise stated. 
    Subscription renewals occur automatically unless cancelled 24 hours before the renewal date.
    We reserve the right to change pricing with 30 days notice to existing subscribers.
    """

    chunks = create_chunks(sample_text, source_file="test_document.txt")
    print_chunk_stats(chunks)
    for chunk in chunks:
        print(f"\n--- Chunk {chunk['chunk_index']} ({chunk['word_count']} words) ---")
        print(chunk["text"][:200] + "...")
