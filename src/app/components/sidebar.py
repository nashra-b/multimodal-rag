"""
src/app/components/sidebar.py
------------------------------
Streamlit sidebar — PDF upload, ingestion controls, and config panel.
"""

import os
import tempfile
import logging
from pathlib import Path

import streamlit as st

from src.pipeline import IngestPipeline

logger = logging.getLogger(__name__)

# ── Element-type colour map (used across the app) ──────────────────────────────
ELEMENT_COLORS = {
    "text":  "#4A90D9",
    "table": "#27AE60",
    "image": "#E67E22",
}


def render_sidebar(rag_chain) -> None:
    """
    Render the full sidebar.

    Args:
        rag_chain : Live RAGChain instance (needed for memory clear button)
    """
    with st.sidebar:
        _render_logo()
        _render_upload_section()
        _render_session_controls(rag_chain)
        _render_config_panel()
        _render_index_stats()
        _render_footer()


# ── Logo / header ──────────────────────────────────────────────────────────────

def _render_logo() -> None:
    st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
            <span style="font-size:2.2rem;">🏦</span>
            <h2 style="margin:0; font-size:1.2rem; color:#1a1a2e; font-weight:700;">
                Multimodal PDF RAG
            </h2>
            <p style="margin:0; font-size:0.75rem; color:#888;">
                Powered by unstructured.io + Pinecone
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()


# ── PDF upload ─────────────────────────────────────────────────────────────────

def _render_upload_section() -> None:
    st.subheader("📂 Ingest a PDF")

    uploaded_file = st.file_uploader(
        label       = "Upload PDF",
        type        = ["pdf"],
        help        = "Upload a banking / financial PDF to ingest into Pinecone.",
        label_visibility = "collapsed",
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"📄 **{uploaded_file.name}** — {file_size_mb:.2f} MB")

        col1, col2 = st.columns(2)

        with col1:
            dry_run = st.checkbox(
                "Dry run",
                value=False,
                help="Parse and chunk without hitting Pinecone or spending API credits."
            )

        with col2:
            reset_index = st.checkbox(
                "Reset index",
                value=False,
                help="⚠️ Deletes all existing vectors before ingesting."
            )

        if st.button("🚀 Ingest PDF", use_container_width=True, type="primary"):
            _run_ingestion(uploaded_file, dry_run=dry_run, reset_index=reset_index)

    st.divider()


def _run_ingestion(uploaded_file, dry_run: bool, reset_index: bool) -> None:
    """Save the uploaded file to a temp path, run the ingestion pipeline."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Ingesting — this may take a few minutes …"):
            progress = st.progress(0, text="Initialising pipeline …")

            pipeline = IngestPipeline(dry_run=dry_run)

            if reset_index and not dry_run:
                progress.progress(5, "Resetting Pinecone index …")
                pipeline.pinecone_client.delete_index()

            progress.progress(15, "Parsing PDF with unstructured.io …")
            parsed = pipeline._step_parse(tmp_path)

            progress.progress(35, "Summarising images with GPT-4o Vision …")
            parsed["image_elements"] = pipeline._step_summarise_images(
                parsed["image_elements"]
            )

            progress.progress(55, "Processing tables …")
            parsed["table_elements"] = pipeline._step_process_tables(
                parsed["table_elements"],
                pdf_path=tmp_path,
                source_file=uploaded_file.name,
            )

            progress.progress(70, "Chunking …")
            chunks = pipeline._step_chunk(parsed, source_file=uploaded_file.name)

            if not dry_run:
                progress.progress(85, "Embedding and upserting to Pinecone …")
                pipeline.upsert_chunks(chunks, source_file=uploaded_file.name)

            progress.progress(100, "Done ✓")

        # ── Success summary ────────────────────────────────────────────────────
        st.success(f"{'[DRY RUN] ' if dry_run else ''}Ingestion complete!")
        _render_ingestion_summary(chunks, uploaded_file.name)

        # Store ingested filename in session state for the chat window
        if "ingested_files" not in st.session_state:
            st.session_state.ingested_files = []
        if uploaded_file.name not in st.session_state.ingested_files:
            st.session_state.ingested_files.append(uploaded_file.name)

    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        logger.error(f"Ingestion error: {e}", exc_info=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _render_ingestion_summary(chunks: dict, filename: str) -> None:
    n_text  = len(chunks.get("text",  []))
    n_table = len(chunks.get("table", []))
    n_image = len(chunks.get("image", []))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Text",   n_text)
    with col2:
        st.metric("📊 Tables", n_table)
    with col3:
        st.metric("🖼️ Images", n_image)


# ── Session controls ───────────────────────────────────────────────────────────

def _render_session_controls(rag_chain) -> None:
    st.subheader("💬 Session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            rag_chain.clear_memory()
            st.rerun()

    with col2:
        if st.button("🔄 Reset all", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            rag_chain.clear_memory()
            st.rerun()

    # Show ingested files
    if st.session_state.get("ingested_files"):
        st.caption("**Ingested files:**")
        for f in st.session_state.ingested_files:
            st.caption(f"  • {f}")

    st.divider()


# ── Config panel ───────────────────────────────────────────────────────────────

def _render_config_panel() -> None:
    with st.expander("⚙️ Retrieval Config", expanded=False):

        st.session_state["top_k"] = st.slider(
            "Top-K chunks",
            min_value=1, max_value=15,
            value=st.session_state.get("top_k", 5),
            help="Number of chunks retrieved per query."
        )

        st.session_state["search_namespaces"] = st.multiselect(
            "Search namespaces",
            options=["text", "table", "image"],
            default=st.session_state.get("search_namespaces", ["text", "table", "image"]),
            help="Restrict search to specific element types."
        )

        st.session_state["show_sources"] = st.toggle(
            "Show source citations",
            value=st.session_state.get("show_sources", True),
        )

        st.session_state["stream_response"] = st.toggle(
            "Stream response",
            value=st.session_state.get("stream_response", True),
        )

    st.divider()


# ── Index stats ────────────────────────────────────────────────────────────────

def _render_index_stats() -> None:
    with st.expander("📊 Pinecone Index Stats", expanded=False):
        if st.button("Refresh stats", use_container_width=True):
            try:
                from src.vectorstore import PineconeClient
                client = PineconeClient()
                stats  = client.index.describe_index_stats()

                total = stats.total_vector_count
                ns    = stats.namespaces or {}

                st.metric("Total vectors", f"{total:,}")
                for name, ns_stats in ns.items():
                    color = ELEMENT_COLORS.get(name, "#888")
                    st.markdown(
                        f'<span style="color:{color}">■</span> '
                        f'**{name}**: {ns_stats.vector_count:,} vectors',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.warning(f"Could not fetch stats: {e}")

    st.divider()


# ── Footer ─────────────────────────────────────────────────────────────────────

def _render_footer() -> None:
    st.markdown("""
        <div style="text-align:center; padding:0.5rem; font-size:0.7rem; color:#aaa;">
            unstructured.io · Pinecone · LangChain · GPT-4o
        </div>
    """, unsafe_allow_html=True)