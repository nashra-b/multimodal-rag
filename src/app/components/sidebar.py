"""
src/app/components/sidebar.py — clean, no overlap
"""
import tempfile, logging
from pathlib import Path
import streamlit as st

logger = logging.getLogger(__name__)


def render_sidebar(rag_chain) -> None:
    with st.sidebar:
        # Brand
        st.markdown("### 📊 PDF Intelligence")
        st.caption("Enterprise RAG · unstructured.io · Pinecone")
        st.divider()

        # Upload
        st.markdown("**📂 Ingest Document**")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        dry_run  = st.checkbox("Dry run (skip Pinecone)", value=False)
        reset    = st.checkbox("Reset Pinecone index first", value=False)

        if uploaded:
            if st.button("⚡ Ingest PDF", type="primary", use_container_width=True):
                _ingest(uploaded, dry_run, reset)

        st.divider()

        # Session
        st.markdown("**💬 Session**")
        fname = st.session_state.get("ingested_file")
        if fname:
            st.success(f"✓ {fname}")

        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.last_sources = []
            if hasattr(rag_chain, "memory") and rag_chain.memory:
                rag_chain.memory.clear()
            st.rerun()

        st.divider()

        # Config
        with st.expander("⚙️ Retrieval settings"):
            st.session_state["top_k"] = st.slider(
                "Top-K chunks", 2, 15, st.session_state.get("top_k", 5))
            st.session_state["show_sources"] = st.checkbox(
                "Show source citations", value=st.session_state.get("show_sources", True))
            st.session_state["stream_response"] = st.checkbox(
                "Stream responses", value=st.session_state.get("stream_response", True))

        st.divider()
        st.caption("unstructured.io · Pinecone · LangChain · GPT-4o")


def _ingest(uploaded_file, dry_run, reset_index):
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

        bar.progress(35, text="Parsing PDF with unstructured.io…")
        summary = pipeline.run(tmp_path)
        bar.progress(100, text="Done!")
        status.empty()

        st.success("✅ Ingestion complete!")
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
