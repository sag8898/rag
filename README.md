# 🤖 RAG Document Chatbot — Fine-Tuned RAG with Streaming Responses

An AI-powered chatbot that answers user queries based on provided documents (Terms & Conditions, Privacy Policies, Legal Contracts) using a **Retrieval-Augmented Generation (RAG)** pipeline with **real-time streaming responses**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple.svg)

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Pipeline](#-running-the-pipeline)
- [Chatbot Usage](#-chatbot-usage)
- [Model & Embedding Choices](#-model--embedding-choices)
- [Demo](#-demo)
- [Sample Queries](#-sample-queries)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    Streamlit Chat Interface                      │
│               (Real-time Streaming Responses)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG PIPELINE                              │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   Retriever   │──▶│  Context     │──▶│    Generator       │  │
│  │  (Semantic    │   │  Formatter   │   │  (Groq LLaMA 3.3) │  │
│  │   Search)     │   │              │   │   + Streaming      │  │
│  └──────┬───────┘   └──────────────┘   └────────────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   Embedder    │   │  ChromaDB    │   │   Document Loader  │  │
│  │ (MiniLM-L6)  │──▶│ Vector Store │◀──│   + Chunker        │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Document Ingestion**: Documents are loaded, cleaned, and split into 100-300 word chunks using sentence-aware splitting.
2. **Embedding Generation**: Each chunk is embedded using `all-MiniLM-L6-v2` (384-dimensional vectors).
3. **Vector Storage**: Embeddings are stored in ChromaDB with cosine similarity indexing.
4. **Query Processing**: User queries are embedded and matched against stored chunks via semantic search.
5. **Response Generation**: Top-k relevant chunks are injected into a prompt template, and the LLM generates a grounded, streaming response.

---

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | LLaMA 3.3 70B (via Groq) | Fast instruction-following text generation |
| **Embeddings** | all-MiniLM-L6-v2 | Lightweight sentence embeddings (384-dim) |
| **Vector DB** | ChromaDB | Persistent vector storage with cosine similarity |
| **Frontend** | Streamlit | Interactive chat UI with streaming support |
| **API Provider** | Groq | Ultra-fast LLM inference (~500 tokens/sec) |

---

## 📁 Project Structure

```
rag/
├── data/                  # 📄 Source document files (PDF, TXT, DOCX)
├── chunks/                # ✂️  Processed text chunks (JSON)
├── vectordb/              # 💾 ChromaDB persistent storage
├── notebooks/             # 📓 Jupyter notebooks for analysis
├── src/                   # 🧩 Core pipeline modules
│   ├── __init__.py
│   ├── document_loader.py # Document loading & text cleaning
│   ├── chunker.py         # Sentence-aware text chunking
│   ├── embedder.py        # Embedding generation (sentence-transformers)
│   ├── vector_store.py    # ChromaDB vector database operations
│   ├── retriever.py       # Semantic search retriever
│   ├── generator.py       # LLM response generation + streaming
│   └── rag_pipeline.py    # End-to-end pipeline orchestrator
├── app.py                 # 🚀 Streamlit chatbot application
├── ingest.py              # 📥 Document ingestion CLI script
├── requirements.txt       # 📦 Python dependencies
├── .env.example           # 🔑 Environment variable template
└── README.md              # 📖 This file
```

---

## ⚡ Setup & Installation

### Prerequisites

- Python 3.9 or higher
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_key_here
```

---

## 🚀 Running the Pipeline

### Step 1: Add Documents

Place your document files (PDF, TXT, or DOCX) in the `data/` folder.

### Step 2: Ingest Documents

Run the ingestion script to process documents and build the vector database:

```bash
python ingest.py
```

**Options:**
```bash
python ingest.py --data-dir ./data --chunks-dir ./chunks --db-dir ./vectordb
python ingest.py --min-words 100 --max-words 300 --overlap 30
```

### Step 3: Launch the Chatbot

```bash
streamlit run app.py
```

The chatbot will open at `http://localhost:8501`.

> **Alternative**: You can also ingest documents directly from the Streamlit sidebar by clicking the "🔄 Ingest Documents" button.

---

## 💬 Chatbot Usage

1. **Type your question** in the chat input box at the bottom
2. **View streaming response** — the answer appears token-by-token in real-time
3. **Check source references** — expand the "📚 Source References" section to see which document chunks were used
4. **Clear chat** — use the sidebar button to reset the conversation

### Features

- ✅ Real-time streaming responses (token-by-token)
- ✅ Source chunk display with relevance scores
- ✅ Chat history persistence within session
- ✅ Sidebar with model info and database stats
- ✅ Document ingestion from the UI
- ✅ Clear chat / reset functionality

---

## 🧠 Model & Embedding Choices

### Embedding Model: `all-MiniLM-L6-v2`

- **Why**: Lightweight (80MB), fast inference, excellent for semantic similarity
- **Dimensions**: 384-dimensional vectors
- **Performance**: Good balance of quality and speed for document retrieval
- **Max Sequence Length**: 256 tokens

### LLM: `LLaMA 3.3 70B Versatile` (via Groq)

- **Why**: State-of-the-art instruction following, excellent for RAG tasks
- **Provider**: Groq provides ultra-fast inference (~500+ tokens/sec)
- **Temperature**: 0.1 (low for factual, grounded responses)
- **Context Window**: Supports large context for multiple retrieved chunks

### Vector Database: ChromaDB

- **Why**: Easy to set up, persistent storage, built-in cosine similarity
- **Index**: HNSW (Hierarchical Navigable Small World) for fast approximate search
- **Distance Metric**: Cosine similarity

### Chunking Strategy

- **Method**: Sentence-aware splitting (respects sentence boundaries)
- **Chunk Size**: 100-300 words per chunk
- **Overlap**: 30 words overlap between consecutive chunks
- **Rationale**: Preserves semantic coherence while keeping chunks manageable for the embedding model

---

## 🎬 Demo

> **TODO**: Add GIF/video link showing the chatbot streaming responses

<!-- ![Demo GIF](demo.gif) -->

---

## 📝 Sample Queries

### Query 1: ✅ Success Case
**Q**: "What personal data does the company collect?"  
**A**: Based on the document, the company collects... [detailed answer from document]

### Query 2: ✅ Success Case
**Q**: "What are the user's rights regarding their data?"  
**A**: According to the privacy policy... [detailed answer from document]

### Query 3: ⚠️ Partial Success
**Q**: "How does the company compare to competitors?"  
**A**: "Based on the available documents, I don't have enough information to answer this question as the documents don't contain competitive analysis."

> 📄 See the full **PDF Report** for detailed analysis of 5+ example queries with success and failure cases.

---

## ⚠️ Known Limitations

- **Embedding model context**: `all-MiniLM-L6-v2` truncates text beyond 256 tokens, which may affect very long chunks
- **API dependency**: Requires internet connection for Groq API calls
- **Single document focus**: Currently optimized for single-document Q&A
- **Language**: English-only support

## Demo Video
- https://drive.google.com/file/d/1iJ_tUId9sfq5NNE_CV4kYlERwCsV6JSE/view?usp=sharing

## Live Link
-[ https://drive.google.com/file/d/1iJ_tUId9sfq5NNE_CV4kYlERwCsV6JSE/view?usp=sharing](https://rag-amlgolabs.streamlit.app/)
