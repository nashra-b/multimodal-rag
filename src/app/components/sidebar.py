"""
src/app/components/sidebar.py
------------------------------
Enterprise sidebar — PDF upload, ingestion controls, config panel.
IngestPipeline is imported lazily to avoid loading unstructured at startup.
"""

import os
import tempfile
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

ELEMENT_COLORS = {
    "text":  "#2563EB",
    "table": "#16A34A",
    "image": "#D97706",
}


def render_sidebar(rag_chain) -> None:
    with st.sidebar:
        _render_brand()
        st.divider()
        _render_upload_section()
        st.divider()
        _render_session_controls(rag_chain)
        st.divider()
        _render_config_panel()
        _render_footer()


# ── Brand ──────────────────────────────────────────────────────────────────────

def _render_brand() -> None:
    st.markdown("""
    <div style="padding: 0.25rem 0 0.5rem;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="
          width:34px; height:34px; border-radius:8px;
          background:#1C1B18; color:#F7F6F3;
          display:flex; align-items:center; justify-content:center;
          font-size:0.75rem; font-weight:700; letter-spacing:-0.01em;
        ">PDF</div>
        <div>
          <div style="font-size:0.95rem; font-weight:700; color:#1C1B18;">
            PDF Intelligence
          </div>
          <div style="font-size:0.68rem; color:#8C897F; letter-spacing:0.04em; text-transform:uppercase;">
            Enterprise RAG
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Upload section ─────────────────────────────────────────────────────────────

def _render_upload_section() -> None:
    st.markdown('<div class="section-label">Ingest Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload PDF",
        type=["pdf"],
        help="Upload a banking or financial PDF to parse and store in Pinecone.",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        dry_run = st.toggle("Dry run", value=False, help="Parse only — skip embedding & Pinecone")
    with col2:
        reset_index = st.toggle("Reset index", value=False, help="Wipe Pinecone index before ingesting")

    if uploaded_file and st.button("⚡ Ingest PDF", use_container_width=True, type="primary"):
        _run_ingestion(uploaded_file, dry_run, reset_index)


def _run_ingestion(uploaded_file, dry_run: bool, reset_index: bool) -> None:
    from src.pipeline.ingest_pipeline import IngestPipeline  # lazy import

    progress  = st.progress(0, text="Saving file…")
    status_box = st.empty()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        progress.progress(15, text="Initialising pipeline…")
        pipeline = IngestPipeline(dry_run=dry_run)

        if reset_index and not dry_run:
            status_box.warning("Resetting Pinecone index…")
            from src.vectorstore import PineconeClient
            PineconeClient().delete_index()

        progress.progress(30, text="Parsing PDF (unstructured.io)…")
        summary = pipeline.run(tmp_path)
        progress.progress(100, text="Done!")

        status_box.empty()
        st.success("Ingestion complete!")

        # Stats cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Text",   summary.get("text_chunks", 0))
        c2.metric("Tables", summary.get("table_chunks", 0))
        c3.metric("Images", summary.get("image_chunks", 0))

        st.session_state["ingested_file"] = uploaded_file.name

    except Exception as exc:
        logger.exception("Ingestion failed")
        st.error(f"Ingestion failed: {exc}")
        progress.empty()
    finally:
        if "tmp_path" in locals():
            Path(tmp_path).unlink(missing_ok=True)


# ── Session controls ───────────────────────────────────────────────────────────

def _render_session_controls(rag_chain) -> None:
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)

    ingested = st.session_state.get("ingested_file")
    if ingested:
        st.markdown(
            f'<div style="font-size:0.78rem; color:#16A34A; margin-bottom:0.5rem;">'
            f'✓ <strong>{ingested}</strong> indexed</div>',
            unsafe_allow_html=True,
        )

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_sources = []
        if hasattr(rag_chain, "memory") and rag_chain.memory:
            rag_chain.memory.clear()
        st.rerun()


# ── Config panel ───────────────────────────────────────────────────────────────

def _render_config_panel() -> None:
    with st.expander("⚙️ Retrieval settings", expanded=False):
        st.session_state["top_k"] = st.slider(
            "Top-K chunks", min_value=2, max_value=15,
            value=st.session_state.get("top_k", 5),
            help="Number of chunks retrieved per query",
        )
        st.session_state["show_sources"] = st.toggle(
            "Show citations", value=st.session_state.get("show_sources", True),
        )
        st.session_state["stream_response"] = st.toggle(
            "Stream response", value=st.session_state.get("stream_response", True),
        )


# ── Footer ─────────────────────────────────────────────────────────────────────

def _render_footer() -> None:
    st.markdown("""
    <div style="
      position: absolute; bottom: 1.25rem; left: 0; right: 0;
      text-align: center;
      font-size: 0.68rem;
      color: #8C897F;
      letter-spacing: 0.04em;
    ">
      unstructured.io · Pinecone · LangChain · GPT-4o
    </div>
    """, unsafe_allow_html=True)
