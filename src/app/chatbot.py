"""
src/app/chatbot.py
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

# Minimal CSS — only things config.toml can't handle
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'DM Sans', sans-serif !important; }
.block-container { padding: 1.25rem 1.5rem 1rem !important; max-width: 100% !important; }
[data-testid="stChatMessage"] { border-radius: 10px !important; margin-bottom: 0.5rem !important; }
.pill { display:inline-flex; align-items:center; padding:2px 8px; border-radius:99px;
        font-size:0.7rem; font-weight:600; margin-right:4px; }
.pill-text  { background:#EFF6FF; color:#2563EB; }
.pill-table { background:#F0FDF4; color:#16A34A; }
.pill-image { background:#FFFBEB; color:#D97706; }
.empty-card { text-align:center; padding:2rem 1rem; border:1px dashed #D5D2C8;
              border-radius:12px; margin-bottom:1rem; color:#6B6860; }
.sec-label  { font-size:0.65rem; font-weight:700; letter-spacing:0.1em;
              text-transform:uppercase; color:#9B9890; margin-bottom:0.4rem; }
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


def main():
    if not _check_env():
        return

    rag_chain = _load_rag_chain()
    render_sidebar(rag_chain)

    # Top header
    col_logo, col_status = st.columns([6, 1])
    with col_logo:
        st.markdown("## 📊 PDF Intelligence")
        st.caption("Multimodal RAG · unstructured.io · Pinecone · GPT-4o · LangChain")
    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("● Live")

    st.divider()

    chat_col, source_col = st.columns([2.2, 1], gap="large")
    with chat_col:
        render_chat_window(rag_chain)
    with source_col:
        render_source_viewer()


if __name__ == "__main__":
    main()
