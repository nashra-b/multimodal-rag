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
from src.app.components  import render_sidebar, render_chat_window, render_source_viewer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Intelligence | Enterprise RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ─────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
    <style>
      /* ── Google Fonts ── */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

      /* ── Root palette ── */
      :root {
        --bg:           #F7F6F3;
        --surface:      #FFFFFF;
        --border:       #E4E2DC;
        --ink:          #1C1B18;
        --ink-soft:     #4A4740;
        --ink-mute:     #8C897F;
        --accent:       #C95D2A;
        --accent-light: #F5E6DC;
        --blue:         #2563EB;
        --blue-light:   #DBEAFE;
        --green:        #16A34A;
        --green-light:  #DCFCE7;
        --amber:        #D97706;
        --amber-light:  #FEF3C7;
        --radius:       10px;
        --shadow:       0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
        --shadow-md:    0 4px 12px rgba(0,0,0,.08);
      }

      /* ── App shell ── */
      html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        font-family: 'Inter', system-ui, sans-serif;
      }

      /* ── Sidebar ── */
      [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
      }
      [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
      }

      /* ── Main area ── */
      [data-testid="stMain"] .block-container {
        padding: 1.5rem 2rem 1rem !important;
        max-width: 100% !important;
      }

      /* ── Chat messages ── */
      [data-testid="stChatMessage"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.6rem !important;
        box-shadow: var(--shadow) !important;
      }

      /* User message: slightly tinted */
      [data-testid="stChatMessage"][data-testid*="user"],
      .stChatMessage:has([aria-label="user avatar"]) {
        background: var(--accent-light) !important;
        border-color: #E8C9B5 !important;
      }

      /* ── Chat input ── */
      [data-testid="stChatInput"] {
        border: 1.5px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow-md) !important;
      }
      [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(201,93,42,0.12) !important;
      }

      /* ── Buttons ── */
      .stButton > button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--ink-soft) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
      }
      .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-light) !important;
      }

      /* Primary button */
      .stButton > button[kind="primary"] {
        background: var(--ink) !important;
        color: #F7F6F3 !important;
        border: none !important;
      }
      .stButton > button[kind="primary"]:hover {
        background: var(--accent) !important;
        color: white !important;
      }

      /* ── File uploader ── */
      [data-testid="stFileUploader"] {
        border: 1.5px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--bg) !important;
        padding: 0.5rem !important;
      }
      [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: var(--accent-light) !important;
      }

      /* ── Metrics ── */
      [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.75rem 1rem !important;
        box-shadow: var(--shadow) !important;
      }
      [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--ink-mute) !important;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
      }

      /* ── Expander (sources) ── */
      [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--surface) !important;
        margin-bottom: 0.5rem !important;
      }
      [data-testid="stExpander"] summary {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--ink-soft) !important;
      }

      /* ── Dividers ── */
      hr { border-color: var(--border) !important; }

      /* ── Scrollbar ── */
      ::-webkit-scrollbar { width: 5px; height: 5px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
      ::-webkit-scrollbar-thumb:hover { background: var(--ink-mute); }

      /* ── Source badge pills ── */
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 9px;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-right: 4px;
      }
      .badge-text   { background: var(--blue-light);  color: var(--blue);  }
      .badge-table  { background: var(--green-light); color: var(--green); }
      .badge-image  { background: var(--amber-light); color: var(--amber); }

      /* ── Monospace accents ── */
      code, .mono {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82em !important;
        background: var(--bg) !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
        border: 1px solid var(--border) !important;
      }

      /* ── Section headers ── */
      .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-mute);
        margin-bottom: 0.5rem;
      }

      /* ── Empty state ── */
      .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        color: var(--ink-mute);
      }
      .empty-state h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--ink-soft);
        margin-bottom: 0.35rem;
      }
      .empty-state p {
        font-size: 0.82rem;
        line-height: 1.55;
      }
    </style>
    """, unsafe_allow_html=True)


# ── Environment check ──────────────────────────────────────────────────────────
def _check_env() -> bool:
    missing = [k for k in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
               if not os.getenv(k)]
    if missing:
        st.error(f"**Missing environment variables:** {', '.join(missing)}")
        st.info("Copy `.env.example` → `.env` and fill in your API keys, then restart.")
        return False
    return True


# ── RAG chain (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Pinecone & loading RAG chain…")
def _load_rag_chain() -> RAGChain:
    pinecone_client  = PineconeClient()
    hybrid_retriever = HybridRetriever(
        pinecone_index=pinecone_client.index,
        all_docs=[],
    )
    chain = RAGChain(
        retriever     = hybrid_retriever.retriever,
        model         = os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature   = float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        memory_window = int(os.getenv("MEMORY_WINDOW", "5")),
        streaming     = True,
    )
    logger.info("RAGChain loaded and cached.")
    return chain


# ── Top header bar ─────────────────────────────────────────────────────────────
def _render_header() -> None:
    st.markdown("""
    <div style="
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1.25rem;
      background: #FFFFFF;
      border: 1px solid #E4E2DC;
      border-radius: 12px;
      margin-bottom: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.05);
    ">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="
          width:38px; height:38px; border-radius:9px;
          background:#1C1B18; color:#F7F6F3;
          display:flex; align-items:center; justify-content:center;
          font-size:1.1rem; font-weight:700; letter-spacing:-0.02em;
        ">PDF</div>
        <div>
          <div style="font-size:1rem; font-weight:700; color:#1C1B18; line-height:1.2;">
            PDF Intelligence
          </div>
          <div style="font-size:0.72rem; color:#8C897F; letter-spacing:0.02em;">
            Multimodal RAG · unstructured.io · Pinecone · GPT-4o
          </div>
        </div>
      </div>
      <div style="display:flex; gap:6px; align-items:center;">
        <span style="
          padding:3px 10px; border-radius:99px;
          background:#DCFCE7; color:#16A34A;
          font-size:0.7rem; font-weight:600;
        ">● Live</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()

    if not _check_env():
        return

    rag_chain = _load_rag_chain()

    render_sidebar(rag_chain)

    _render_header()

    chat_col, source_col = st.columns([2.2, 1], gap="medium")

    with chat_col:
        render_chat_window(rag_chain)

    with source_col:
        render_source_viewer()


if __name__ == "__main__":
    main()
