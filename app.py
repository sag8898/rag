"""
RAG Chatbot - Streamlit Application
AI-powered chatbot with real-time streaming responses for document Q&A.

Run with: streamlit run app.py
"""

import os
import sys
import time
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for Premium UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts: Inter for text, Outfit for headings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;500;600;700&display=swap');

    /* ----- Typography ----- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* ----- Main Header ----- */
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #4f46e5, #10b981);
    }

    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        margin: 0 0 0.5rem 0;
    }

    .main-header p {
        color: #94a3b8 !important;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    /* ----- Chat Message Tweaks ----- */
    /* We avoid overriding background completely to respect Light/Dark mode, 
       just adding some radius and subtle shadow. */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        margin-bottom: 0.8rem !important;
    }

    /* ----- Source Chunk Cards (Glassmorphism) ----- */
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        line-height: 1.6;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .source-card:hover {
        border-color: rgba(16, 185, 129, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
        background: rgba(15, 23, 42, 0.8);
    }
    
    .source-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .source-text {
        color: #cbd5e1;
        font-weight: 300;
    }

    /* Relevance Badge */
    .relevance-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .relevance-high { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .relevance-medium { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .relevance-low { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* ----- Sidebar Styling ----- */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .sidebar-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        transition: all 0.2s;
    }
    
    .sidebar-card:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    .sidebar-stat:last-child { border-bottom: none; padding-bottom: 0; }
    .stat-label { color: #94a3b8; font-size: 0.8rem; font-weight: 500; }
    .stat-value { color: #f8fafc; font-weight: 600; font-size: 0.85rem; background: rgba(255,255,255,0.05); padding: 0.1rem 0.5rem; border-radius: 6px; }

    /* ----- Info Banner ----- */
    .info-banner {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .info-banner strong {
        color: #10b981;
    }

    /* ----- Expanders / Accordions ----- */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary {
        background: rgba(30, 41, 59, 0.5) !important;
        padding: 1rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #4f46e5 !important;
    }

    /* ----- Buttons ----- */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%) !important;
    }

    /* Status indicator */
    .status-container {
        display: inline-flex;
        align-items: center;
        background: rgba(0, 0, 0, 0.2);
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px currentColor;
    }
    .status-online { color: #10b981; background: #10b981; animation: pulse 2s infinite; }
    .status-offline { color: #ef4444; background: #ef4444; }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(71, 85, 105, 0.8);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 116, 139, 1);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Initialize Pipeline (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline():
    """Initialize the RAG pipeline (cached across sessions)."""
    try:
        pipeline = RAGPipeline(
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama-3.3-70b-versatile",
            db_dir="./vectordb",
            top_k=5,
            temperature=0.1,
            max_tokens=1024,
        )
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
def render_sidebar(pipeline):
    """Render the sidebar with pipeline information and controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        if pipeline:
            info = pipeline.get_pipeline_info()
            is_ready = pipeline.is_ready()

            # Status
            status_class = "status-online" if is_ready else "status-offline"
            status_text = "Ready" if is_ready else "No Documents Indexed"
            st.markdown(
                f'<div class="status-container" style="margin-bottom:1rem;">'
                f'<span class="status-dot {status_class}"></span>'
                f'<span style="color:rgba(255,255,255,0.9); font-size:0.85rem; font-weight: 500;">{status_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Model Info Card
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("**🧠 LLM Model**")
            gen_info = info["generator"]
            st.markdown(f"""
                <div class="sidebar-stat"><span class="stat-label">Model</span><span class="stat-value">{gen_info['model']}</span></div>
                <div class="sidebar-stat"><span class="stat-label">Provider</span><span class="stat-value">{gen_info['provider']}</span></div>
                <div class="sidebar-stat"><span class="stat-label">Temperature</span><span class="stat-value">{gen_info['temperature']}</span></div>
                <div class="sidebar-stat"><span class="stat-label">Max Tokens</span><span class="stat-value">{gen_info['max_tokens']}</span></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Embedder Info Card
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("**📐 Embedding Model**")
            emb_info = info["embedder"]
            st.markdown(f"""
                <div class="sidebar-stat"><span class="stat-label">Model</span><span class="stat-value">{emb_info['model_name']}</span></div>
                <div class="sidebar-stat"><span class="stat-label">Dimensions</span><span class="stat-value">{emb_info['embedding_dimension']}</span></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Vector DB Info Card
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.markdown("**🗄️ Vector Database**")
            vs_info = info["vector_store"]
            st.markdown(f"""
                <div class="sidebar-stat"><span class="stat-label">Database</span><span class="stat-value">ChromaDB</span></div>
                <div class="sidebar-stat"><span class="stat-label">Indexed Chunks</span><span class="stat-value">{vs_info['total_documents']}</span></div>
                <div class="sidebar-stat"><span class="stat-label">Top-K Retrieval</span><span class="stat-value">{info['retriever']['top_k']}</span></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Document Ingestion Section
        st.markdown("## 📥 Ingest Documents")
        st.caption("Place documents in the `data/` folder and click below.")

        if st.button("🔄 Ingest Documents", use_container_width=True, type="primary"):
            if pipeline:
                data_dir = os.path.join(os.path.dirname(__file__), "data")
                if os.path.exists(data_dir) and os.listdir(data_dir):
                    with st.spinner("Processing documents..."):
                        try:
                            chunks = pipeline.ingest_documents(data_dir)
                            st.success(f"✅ Ingested {len(chunks)} chunks!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("No documents found in `data/` folder.")

        st.markdown("---")

        # Clear Chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Footer
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; color:rgba(255,255,255,0.35); font-size:0.75rem;">'
            'Built with ❤️ using Streamlit + Groq + ChromaDB'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# Source Chunks Display
# ─────────────────────────────────────────────────────────────
def render_sources(sources: list[dict]):
    """Render retrieved source chunks in expandable cards."""
    if not sources:
        return

    with st.expander(f"📚 Source References ({len(sources)} chunks retrieved)", expanded=False):
        for i, source in enumerate(sources):
            score = source.get("relevance_score", 0)

            # Determine relevance level
            if score >= 0.7:
                badge_class = "relevance-high"
                badge_text = f"High ({score:.0%})"
            elif score >= 0.4:
                badge_class = "relevance-medium"
                badge_text = f"Medium ({score:.0%})"
            else:
                badge_class = "relevance-low"
                badge_text = f"Low ({score:.0%})"

            st.markdown(f"""
            <div class="source-card">
                <div class="source-header">
                    📄 {source.get('source_file', 'Unknown')} — Chunk {source.get('chunk_index', '?')}
                    <span class="relevance-badge {badge_class}">{badge_text}</span>
                </div>
                <div class="source-text">{source['text'][:400]}{'...' if len(source['text']) > 400 else ''}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Document Chatbot</h1>
        <p>Ask questions about your documents — powered by Retrieval-Augmented Generation</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize pipeline
    with st.spinner("🔧 Initializing RAG pipeline... (this may take a moment on first load)"):
        pipeline = init_pipeline()

    # Render sidebar
    render_sidebar(pipeline)

    if not pipeline:
        st.error("❌ Pipeline failed to initialize. Check your API key and dependencies.")
        return

    # Check if documents are indexed
    if not pipeline.is_ready():
        st.markdown("""
        <div class="info-banner">
            <strong>👋 Welcome!</strong> To get started:<br>
            1. Place your document(s) in the <code>data/</code> folder (PDF, TXT, or DOCX)<br>
            2. Click <strong>"🔄 Ingest Documents"</strong> in the sidebar, or run <code>python ingest.py</code><br>
            3. Start asking questions!
        </div>
        """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check readiness
        if not pipeline.is_ready():
            with st.chat_message("assistant"):
                msg = ("⚠️ No documents have been indexed yet. "
                       "Please add documents to the `data/` folder and click "
                       "**'🔄 Ingest Documents'** in the sidebar.")
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        # Generate streaming response
        with st.chat_message("assistant"):
            try:
                # Start timing
                start_time = time.time()

                # Get streaming response and sources
                token_stream, sources = pipeline.query_stream(prompt)

                # Stream the response
                full_response = st.write_stream(token_stream)

                # Calculate response time
                elapsed = time.time() - start_time

                # Show response time
                st.caption(f"⏱️ Response generated in {elapsed:.1f}s")

                # Show source references
                render_sources(sources)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


if __name__ == "__main__":
    main()
