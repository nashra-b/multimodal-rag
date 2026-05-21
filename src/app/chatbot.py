"""
src/app/chatbot.py
-------------------
Streamlit application entry point.

Layout:
    ┌──────────┬─────────────────────────┬──────────────┐
    │ Sidebar  │     Chat Window         │ Source Panel │
    │ (upload  │  (conversation + input) │ (citations)  │
    │  config) │                         │              │
    └──────────┴─────────────────────────┴──────────────┘

Run:
    streamlit run src/app/chatbot.py
"""

import os
import sys
import logging
from src.pipeline import rag_chain  # keep only this
from pathlib import Path

# ── Ensure project root is on the path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline          import RAGChain
from src.vectorstore       import PineconeClient, HybridRetriever
from src.app.components    import render_sidebar, render_chat_window, render_source_viewer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title    = "Multimodal PDF RAG",
    page_icon     = "🏦",
    layout        = "wide",
    initial_sidebar_state = "expanded",
    menu_items    = {
        "Get Help":    "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About":       "Multimodal RAG powered by unstructured.io + Pinecone + LangChain",
    },
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
    <style>
        /* ── Global ── */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
            max-width: 100%;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9ff 0%, #f0f2ff 100%);
            border-right: 1px solid #e0e4f0;
        }
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 8px;
            font-size: 0.85rem;
        }

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.5rem;
        }

        /* ── Chat input ── */
        [data-testid="stChatInput"] > div {
            border-radius: 12px;
            border: 1.5px solid #4A90D9;
        }

        /* ── Source panel ── */
        .source-panel {
            border-left: 1px solid #e8e8e8;
            padding-left: 1rem;
        }

        /* ── Metrics ── */
        [data-testid="stMetric"] {
            background: #f8f9ff;
            border-radius: 8px;
            padding: 6px;
        }

        /* ── Expanders ── */
        [data-testid="stExpander"] {
            border: 1px solid #e8eaf0;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }

        /* ── Hide default Streamlit top padding ── */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ── Env validation ─────────────────────────────────────────────────────────────
def _check_env() -> bool:
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing  = [k for k in required if not os.getenv(k)]

    if missing:
        st.error(
            f"**Missing environment variables:** {', '.join(missing)}\n\n"
            "Copy `.env.example` → `.env` and fill in the values, then restart."
        )
        st.stop()
        return False
    return True


# ── RAG chain (cached — one instance per session) ─────────────────────────────
@st.cache_resource(show_spinner="Connecting to Pinecone and loading RAG chain …")
def _load_rag_chain() -> RAGChain:
    """
    Initialise and cache the RAG chain.
    st.cache_resource means this runs once per server session, not per page reload.
    """
    pinecone_client  = PineconeClient()
    hybrid_retriever = HybridRetriever(
        pinecone_index = pinecone_client.index,
        all_docs       = [],
    )
    chain = RAGChain(
        retriever    = hybrid_retriever.retriever,
        model        = os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature  = float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        memory_window = int(os.getenv("MEMORY_WINDOW", "5")),
        streaming    = True,
    )
    logger.info("RAGChain loaded and cached.")
    return chain


# ── Header banner ──────────────────────────────────────────────────────────────
def _render_header() -> None:
    st.markdown("""
        <div style="
            background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 0.8rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        ">
            <span style="font-size:1.8rem;">🏦</span>
            <div>
                <h1 style="
                    margin:0; color:white; font-size:1.3rem; font-weight:700;
                ">Multimodal PDF RAG</h1>
                <p style="
                    margin:0; color:#a0aec0; font-size:0.78rem;
                ">
                    unstructured.io · Pinecone Namespaces · GPT-4o Vision ·
                    Hybrid MMR+BM25 Retrieval · LangChain
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()

    if not _check_env():
        return

    rag_chain = _load_rag_chain()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    render_sidebar(rag_chain)

    # ── Main content area ──────────────────────────────────────────────────────
    _render_header()

    # 2-column layout: chat (wider) | sources
    chat_col, source_col = st.columns([2.2, 1], gap="medium")

    with chat_col:
        render_chat_window(rag_chain)

    with source_col:
        st.markdown('<div class="source-panel">', unsafe_allow_html=True)
        render_source_viewer()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
