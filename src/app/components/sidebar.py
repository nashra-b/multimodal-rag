"""
src/app/components/sidebar.py
------------------------------
Enterprise sidebar — dark theme.
IngestPipeline is imported lazily to avoid loading unstructured at startup.
"""

import tempfile
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


def render_sidebar(rag_chain) -> None:
    with st.sidebar:
        _render_brand()
        st.divider()
        _render_upload_section()
        st.divider()
        _render_session(rag_chain)
        st.divider()
        _render_config()
        _render_footer()


def _render_brand() -> None:
    st.markdown("""
    <div style="padding:0.4rem 0 0.6rem;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="
          width:32px; height:32px; border-radius:8px;
          background:#E8622A;
          display:flex; align-items:center; justify-content:center;
          font-size:0.68rem; font-weight:700; color:#fff;
        ">PDF</div>
        <div>
          <div style="font-size:0.88rem; font-weight:700; color:#E0DDD8;">
            PDF Intelligence
          </div>
          <div style="font-size:0.62rem; color:#555; letter-spacing:0.08em;
                      text-transform:uppercase; font-family:'DM Mono',monospace;">
            Enterprise RAG
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_upload_section() -> None:
    st.markdown('<div class="section-lbl">Ingest Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a banking or financial PDF",
    )

    c1, c2 = st.columns(2)
    with c1:
        dry_run = st.toggle("Dry run", value=False, help="Parse only — skip Pinecone")
    with c2:
        reset   = st.toggle("Reset index", value=False, help="Wipe Pinecone first")

    if uploaded and st.button("⚡  Ingest PDF", use_container_width=True, type="primary"):
        _run_ingestion(uploaded, dry_run, reset)


def _run_ingestion(uploaded_file, dry_run: bool, reset_index: bool) -> None:
    from src.pipeline.ingest_pipeline import IngestPipeline

    bar    = st.progress(0, text="Saving file…")
    status = st.empty()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        bar.progress(15, text="Initialising pipeline…")
        pipeline = IngestPipeline(dry_run=dry_run)

        if reset_index and not dry_run:
            status.warning("Resetting Pinecone index…")
            from src.vectorstore import PineconeClient
            PineconeClient().delete_index()

        bar.progress(35, text="Parsing with unstructured.io…")
        summary = pipeline.run(tmp_path)
        bar.progress(100, text="Done!")
        status.empty()

        st.markdown("""
        <div style="
          background:#152518; border:1px solid #1E3D22; border-radius:8px;
          padding:0.6rem 0.8rem; margin:0.5rem 0;
          font-size:0.78rem; color:#4ADE80; font-weight:600;
        ">✓ Ingestion complete</div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Text",   summary.get("text_chunks",  0))
        c2.metric("Tables", summary.get("table_chunks", 0))
        c3.metric("Images", summary.get("image_chunks", 0))
        st.session_state["ingested_file"] = uploaded_file.name

    except Exception as exc:
        logger.exception("Ingestion failed")
        st.error(f"Ingestion failed: {exc}")
        bar.empty()
    finally:
        if "tmp_path" in locals():
            Path(tmp_path).unlink(missing_ok=True)


def _render_session(rag_chain) -> None:
    st.markdown('<div class="section-lbl">Session</div>', unsafe_allow_html=True)

    fname = st.session_state.get("ingested_file")
    if fname:
        st.markdown(
            f'<div style="font-size:0.73rem; color:#4ADE80; margin-bottom:0.5rem;">'
            f'✓&nbsp; <span style="font-family:DM Mono,monospace;">{fname}</span></div>',
            unsafe_allow_html=True,
        )

    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_sources = []
        if hasattr(rag_chain, "memory") and rag_chain.memory:
            rag_chain.memory.clear()
        st.rerun()


def _render_config() -> None:
    with st.expander("⚙  Retrieval settings", expanded=False):
        st.session_state["top_k"] = st.slider(
            "Top-K chunks", 2, 15, st.session_state.get("top_k", 5),
            help="Chunks retrieved per query",
        )
        st.session_state["show_sources"]    = st.toggle("Show citations",   value=st.session_state.get("show_sources",    True))
        st.session_state["stream_response"] = st.toggle("Stream responses", value=st.session_state.get("stream_response", True))


def _render_footer() -> None:
    st.markdown("""
    <div style="
      position:absolute; bottom:1rem; left:0; right:0;
      text-align:center;
      font-size:0.62rem; color:#333;
      letter-spacing:0.06em; font-family:'DM Mono',monospace;
    ">
      unstructured.io · Pinecone · LangChain · GPT-4o
    </div>
    """, unsafe_allow_html=True)
