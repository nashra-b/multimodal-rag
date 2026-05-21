"""
src/app/components/sidebar.py — light enterprise theme, no overlap
"""
import tempfile, logging
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)


def render_sidebar(rag_chain) -> None:
    with st.sidebar:
        _brand()
        st.divider()
        _upload()
        st.divider()
        _session(rag_chain)
        st.divider()
        _config()
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        _footer()


def _brand():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding:0.25rem 0 0.5rem;">
      <div style="width:32px; height:32px; border-radius:8px; background:#C95D2A; flex-shrink:0;
                  display:flex; align-items:center; justify-content:center;
                  font-size:0.68rem; font-weight:700; color:#fff;">PDF</div>
      <div>
        <div style="font-size:0.9rem; font-weight:700; color:#1C1B18; line-height:1.2;">
          PDF Intelligence</div>
        <div style="font-size:0.62rem; color:#9B9890; letter-spacing:0.08em;
                    text-transform:uppercase; font-family:monospace;">Enterprise RAG</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _upload():
    st.markdown('<p class="sec-label">Ingest Document</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        dry_run = st.toggle("Dry run", value=False, key="dry_run")
    with col2:
        reset = st.toggle("Reset index", value=False, key="reset_idx")

    if uploaded:
        if st.button("⚡ Ingest PDF", use_container_width=True, type="primary"):
            _ingest(uploaded, dry_run, reset)


def _ingest(uploaded_file, dry_run, reset_index):
    from src.pipeline.ingest_pipeline import IngestPipeline

    bar    = st.progress(0, text="Saving…")
    status = st.empty()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        bar.progress(15, text="Initialising pipeline…")
        pipeline = IngestPipeline(dry_run=dry_run)

        if reset_index and not dry_run:
            status.warning("Resetting index…")
            from src.vectorstore import PineconeClient
            PineconeClient().delete_index()

        bar.progress(35, text="Parsing PDF…")
        summary = pipeline.run(tmp_path)
        bar.progress(100, text="Done!")
        status.empty()

        st.success("Ingestion complete!")
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


def _session(rag_chain):
    st.markdown('<p class="sec-label">Session</p>', unsafe_allow_html=True)

    fname = st.session_state.get("ingested_file")
    if fname:
        st.markdown(
            f'<p style="font-size:0.73rem; color:#16A34A; margin:0 0 0.4rem;">✓ {fname}</p>',
            unsafe_allow_html=True,
        )

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_sources = []
        if hasattr(rag_chain, "memory") and rag_chain.memory:
            rag_chain.memory.clear()
        st.rerun()


def _config():
    with st.expander("⚙ Retrieval settings", expanded=False):
        st.session_state["top_k"] = st.slider(
            "Top-K chunks", 2, 15, st.session_state.get("top_k", 5))
        st.session_state["show_sources"]    = st.toggle(
            "Show citations",    value=st.session_state.get("show_sources", True),    key="tog_src")
        st.session_state["stream_response"] = st.toggle(
            "Stream responses",  value=st.session_state.get("stream_response", True), key="tog_str")


def _footer():
    st.markdown("""
    <p style="font-size:0.62rem; color:#C5C2B8; text-align:center;
              font-family:monospace; letter-spacing:0.04em;">
      unstructured.io · Pinecone · LangChain · GPT-4o
    </p>
    """, unsafe_allow_html=True)
