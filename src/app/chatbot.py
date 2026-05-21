"""
src/app/chatbot.py — clean light enterprise theme
Run: streamlit run src/app/chatbot.py
"""
import os, sys, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.pipeline       import RAGChain
from src.vectorstore    import PineconeClient, HybridRetriever
from src.app.components import render_sidebar, render_chat_window, render_source_viewer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="PDF Intelligence | Enterprise RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Reset app background to clean white ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main,
    .main .block-container {
        background-color: #F5F4F0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .block-container {
        padding: 1.25rem 1.5rem 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E3DC !important;
    }
    [data-testid="stSidebar"] > div {
        padding: 1rem 0.75rem !important;
    }

    /* ── Fix sidebar toggle label overlap ── */
    [data-testid="stSidebar"] [data-testid="stToggle"] {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stToggle"] label {
        font-size: 0.78rem !important;
        color: #6B6860 !important;
        white-space: nowrap !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border: 1px solid #E5E3DC !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
        color: #3D3C38 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.15s !important;
        white-space: normal !important;
        text-align: left !important;
        height: auto !important;
        line-height: 1.4 !important;
    }
    .stButton > button:hover {
        border-color: #C95D2A !important;
        color: #C95D2A !important;
        background: #FDF1EB !important;
    }
    [data-testid="baseButton-primary"],
    button[kind="primary"] {
        background: #C95D2A !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
    }
    [data-testid="baseButton-primary"]:hover {
        background: #B54E22 !important;
        color: #FFFFFF !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #FAFAF7 !important;
        border: 1.5px dashed #D5D2C8 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #C95D2A !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 1.5px solid #E5E3DC !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #C95D2A !important;
        box-shadow: 0 0 0 3px rgba(201,93,42,0.1) !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        border: 1px solid #ECEAE3 !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #ECEAE3 !important;
        border-radius: 10px !important;
        margin-bottom: 0.4rem !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #ECEAE3 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #9B9890 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #1C1B18 !important;
    }

    /* ── Dividers ── */
    hr { border-color: #ECEAE3 !important; opacity: 1 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #D5D2C8; border-radius: 99px; }

    /* ── Section labels ── */
    .sec-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9B9890;
        margin-bottom: 0.5rem;
        font-family: 'DM Mono', monospace;
    }

    /* ── Badge pills ── */
    .pill { display:inline-flex; align-items:center; gap:3px; padding:2px 8px;
            border-radius:99px; font-size:0.7rem; font-weight:600; margin-right:3px; }
    .pill-text  { background:#EFF6FF; color:#2563EB; }
    .pill-table { background:#F0FDF4; color:#16A34A; }
    .pill-image { background:#FFFBEB; color:#D97706; }

    /* ── Empty state ── */
    .empty-card {
        text-align:center; padding:2.5rem 1.5rem;
        border:1px dashed #D5D2C8; border-radius:12px; margin-bottom:1rem;
    }
    .empty-card h4 { color:#6B6860; font-size:0.9rem; font-weight:600; margin:0.5rem 0 0.25rem; }
    .empty-card p  { color:#9B9890; font-size:0.78rem; line-height:1.55; margin:0; }
    </style>
    """, unsafe_allow_html=True)


def _check_env():
    missing = [k for k in ["OPENAI_API_KEY","PINECONE_API_KEY","PINECONE_INDEX_NAME"] if not os.getenv(k)]
    if missing:
        st.error(f"Missing env vars: {', '.join(missing)} — copy .env.example → .env")
        return False
    return True


@st.cache_resource(show_spinner="Connecting to Pinecone…")
def _load_rag_chain():
    pc  = PineconeClient()
    ret = HybridRetriever(pinecone_index=pc.index, all_docs=[])
    return RAGChain(
        retriever     = ret.retriever,
        model         = os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature   = float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        memory_window = int(os.getenv("MEMORY_WINDOW", "5")),
        streaming     = True,
    )


def _render_topbar():
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between;
                padding:0.7rem 1.1rem; background:#FFFFFF;
                border:1px solid #E5E3DC; border-radius:12px;
                margin-bottom:1.1rem; box-shadow:0 1px 4px rgba(0,0,0,0.05);">
      <div style="display:flex; align-items:center; gap:11px;">
        <div style="width:36px; height:36px; border-radius:9px; background:#C95D2A;
                    display:flex; align-items:center; justify-content:center;
                    font-size:0.72rem; font-weight:700; color:#fff;">PDF</div>
        <div>
          <div style="font-size:0.95rem; font-weight:700; color:#1C1B18; line-height:1.2;">
            PDF Intelligence</div>
          <div style="font-size:0.68rem; color:#9B9890; font-family:'DM Mono',monospace;">
            unstructured.io &nbsp;·&nbsp; Pinecone &nbsp;·&nbsp; GPT-4o &nbsp;·&nbsp; LangChain
          </div>
        </div>
      </div>
      <span style="padding:3px 12px; border-radius:99px; background:#F0FDF4;
                   color:#16A34A; font-size:0.68rem; font-weight:700;
                   font-family:'DM Mono',monospace; letter-spacing:0.04em;">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    _inject_css()
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
