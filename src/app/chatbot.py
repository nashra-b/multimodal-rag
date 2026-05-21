"""
src/app/chatbot.py
-------------------
Streamlit application entry point — enterprise UI.

Run:
    streamlit run src/app/chatbot.py
"""

import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline        import RAGChain
from src.vectorstore     import PineconeClient, HybridRetriever
from src.app.components_old  import render_sidebar, render_chat_window, render_source_viewer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="PDF Intelligence | Enterprise RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_global_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main {
        background-color: #0F0F0F !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 1px solid #2A2A2A !important;
    }
    [data-testid="stSidebar"] * {
        color: #E0DDD8 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        font-size: 0.78rem !important;
        color: #888 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1A1A1A !important;
        border: 1.5px dashed #333 !important;
        border-radius: 10px !important;
        padding: 0.25rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #E8622A !important;
    }
    [data-testid="stFileUploader"] * { color: #888 !important; }
    [data-testid="stFileUploader"] span { color: #E8622A !important; }

    /* Buttons */
    .stButton > button {
        background: #1E1E1E !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: #CCC !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        border-color: #E8622A !important;
        color: #E8622A !important;
        background: #1E1611 !important;
    }
    [data-testid="baseButton-primary"] {
        background: #E8622A !important;
        border: none !important;
        color: #fff !important;
    }
    [data-testid="baseButton-primary"]:hover {
        background: #D45520 !important;
        color: #fff !important;
    }

    /* Toggles */
    [data-testid="stToggle"] label { font-size: 0.78rem !important; }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #1A1A1A !important;
        border: 1.5px solid #2A2A2A !important;
        border-radius: 12px !important;
        color: #E0DDD8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #E8622A !important;
        box-shadow: 0 0 0 3px rgba(232,98,42,0.12) !important;
    }
    [data-testid="stChatInputSubmitButton"] svg { color: #E8622A !important; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #181818 !important;
        border: 1px solid #252525 !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        padding: 0.9rem 1rem !important;
    }
    [data-testid="stChatMessage"] p {
        color: #D8D5CF !important;
        font-size: 0.9rem !important;
        line-height: 1.65 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #181818 !important;
        border: 1px solid #252525 !important;
        border-radius: 10px !important;
        margin-bottom: 0.4rem !important;
    }
    [data-testid="stExpander"] summary {
        color: #CCC !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #181818 !important;
        border: 1px solid #252525 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stMetricLabel"] { color: #666 !important; font-size: 0.68rem !important; }
    [data-testid="stMetricValue"] { color: #E0DDD8 !important; font-size: 1.5rem !important; font-weight: 700 !important; }

    /* Dividers */
    hr { border-color: #252525 !important; opacity: 1 !important; }

    /* Main content block */
    [data-testid="stMain"] .block-container {
        padding: 1.25rem 1.75rem 1rem !important;
        max-width: 100% !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 99px; }

    /* Starter buttons grid */
    .starter-grid .stButton > button {
        text-align: left !important;
        padding: 0.65rem 0.85rem !important;
        height: auto !important;
        white-space: normal !important;
        line-height: 1.4 !important;
        border-color: #252525 !important;
        color: #AAA !important;
        font-size: 0.78rem !important;
    }
    .starter-grid .stButton > button:hover {
        border-color: #E8622A !important;
        color: #E8622A !important;
    }

    /* Section labels */
    .section-lbl {
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.6rem;
        font-family: 'DM Mono', monospace;
    }

    /* Badge pills */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        padding: 2px 8px;
        border-radius: 99px;
        font-size: 0.68rem;
        font-weight: 600;
        margin-right: 3px;
        font-family: 'DM Sans', sans-serif;
    }
    .pill-text  { background: #1A2535; color: #60A5FA; }
    .pill-table { background: #152518; color: #4ADE80; }
    .pill-image { background: #271E10; color: #FBB24A; }

    /* Source panel snippet */
    .snippet-block {
        font-size: 0.78rem;
        line-height: 1.55;
        color: #888;
        border-left: 2.5px solid #333;
        padding-left: 0.6rem;
        margin-top: 0.4rem;
    }

    /* Empty state */
    .empty-card {
        text-align: center;
        padding: 2.5rem 1.5rem;
        border: 1px dashed #252525;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .empty-card .icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .empty-card h4 { color: #888; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.25rem; }
    .empty-card p  { color: #555; font-size: 0.78rem; line-height: 1.55; }
    </style>
    """, unsafe_allow_html=True)


def _check_env() -> bool:
    missing = [k for k in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
               if not os.getenv(k)]
    if missing:
        st.error(f"**Missing env vars:** {', '.join(missing)}  —  copy `.env.example` → `.env`")
        return False
    return True


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def _load_rag_chain() -> RAGChain:
    pc   = PineconeClient()
    ret  = HybridRetriever(pinecone_index=pc.index, all_docs=[])
    return RAGChain(
        retriever     = ret.retriever,
        model         = os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature   = float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        memory_window = int(os.getenv("MEMORY_WINDOW", "5")),
        streaming     = True,
    )


def _render_topbar() -> None:
    st.markdown("""
    <div style="
      display:flex; align-items:center; justify-content:space-between;
      padding:0.7rem 1.1rem;
      background:#141414;
      border:1px solid #252525;
      border-radius:12px;
      margin-bottom:1.1rem;
    ">
      <div style="display:flex; align-items:center; gap:11px;">
        <div style="
          width:36px; height:36px; border-radius:9px;
          background:#E8622A;
          display:flex; align-items:center; justify-content:center;
          font-size:0.72rem; font-weight:700; color:#fff; letter-spacing:0.02em;
        ">PDF</div>
        <div>
          <div style="font-size:0.95rem; font-weight:700; color:#E0DDD8; line-height:1.2;">
            PDF Intelligence
          </div>
          <div style="font-size:0.68rem; color:#555; letter-spacing:0.04em; font-family:'DM Mono',monospace;">
            unstructured.io &nbsp;·&nbsp; Pinecone &nbsp;·&nbsp; GPT-4o &nbsp;·&nbsp; LangChain
          </div>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="
          padding:3px 10px; border-radius:99px;
          background:#152518; color:#4ADE80;
          font-size:0.68rem; font-weight:600; letter-spacing:0.04em;
          font-family:'DM Mono',monospace;
        ">● LIVE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    _inject_global_css()

    if not _check_env():
        return

    rag_chain = _load_rag_chain()

    render_sidebar(rag_chain)

    _render_topbar()

    chat_col, source_col = st.columns([2.2, 1], gap="medium")
    with chat_col:
        render_chat_window(rag_chain)
    with source_col:
        render_source_viewer()


if __name__ == "__main__":
    main()
