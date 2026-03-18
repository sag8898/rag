"""
Document Loader Module
Handles loading and cleaning of documents in various formats (PDF, TXT, DOCX).
Removes headers, footers, HTML artifacts, and normalizes whitespace.
"""

import os
import re
from pathlib import Path
from typing import Optional


def load_pdf(file_path: str) -> str:
    """Load text content from a PDF file."""
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)


def load_txt(file_path: str) -> str:
    """Load text content from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_docx(file_path: str) -> str:
    """Load text content from a DOCX file."""
    from docx import Document

    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    """
    Clean and normalize the extracted text.
    - Remove HTML tags
    - Remove excessive whitespace
    - Remove headers/footers patterns
    - Normalize unicode characters
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)

    # Remove email addresses (but keep the context)
    text = re.sub(r"\S+@\S+\.\S+", "[email]", text)

    # Remove page numbers (common patterns)
    text = re.sub(r"\n\s*Page \d+ of \d+\s*\n", "\n", text)
    text = re.sub(r"\n\s*-\s*\d+\s*-\s*\n", "\n", text)

    # Normalize whitespace - collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start and end
    text = text.strip()

    return text


def load_document(file_path: str) -> str:
    """
    Load a document from the given file path.
    Supports PDF, TXT, and DOCX formats.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Cleaned text content of the document
    """
    file_path = str(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_txt,
        ".docx": load_docx,
        ".doc": load_docx,
    }

    if ext not in loaders:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported formats: {', '.join(loaders.keys())}"
        )

    raw_text = loaders[ext](file_path)
    cleaned_text = clean_text(raw_text)

    print(f"[DocumentLoader] Loaded '{os.path.basename(file_path)}' "
          f"({len(cleaned_text):,} characters, ~{len(cleaned_text.split()):,} words)")

    return cleaned_text


def load_all_documents(data_dir: str) -> dict[str, str]:
    """
    Load all supported documents from a directory.
    
    Args:
        data_dir: Path to the directory containing documents
        
    Returns:
        Dictionary mapping filename to cleaned text content
    """
    supported_extensions = {".pdf", ".txt", ".docx", ".doc"}
    documents = {}

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in sorted(data_path.iterdir()):
        if file_path.suffix.lower() in supported_extensions:
            try:
                documents[file_path.name] = load_document(str(file_path))
            except Exception as e:
                print(f"[DocumentLoader] Error loading '{file_path.name}': {e}")

    if not documents:
        print(f"[DocumentLoader] No supported documents found in '{data_dir}'")
    else:
        total_words = sum(len(text.split()) for text in documents.values())
        print(f"[DocumentLoader] Loaded {len(documents)} document(s), "
              f"~{total_words:,} total words")

    return documents


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        text = load_document(sys.argv[1])
        print(f"\nFirst 500 characters:\n{text[:500]}")
    else:
        print("Usage: python document_loader.py <file_path>")
