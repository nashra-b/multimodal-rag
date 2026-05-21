"""
src/app/components/chat_window.py
----------------------------------
Enterprise chat window — conversation history, streaming answers, source badges.
"""

import logging
import streamlit as st

from src.pipeline.rag_chain import RAGChain

logger = logging.getLogger(__name__)

STARTER_QUESTIONS = [
    "What was the net income reported?",
    "Summarise the key risk factors.",
    "What does the revenue breakdown table show?",
    "Describe any charts or diagrams in the document.",
    "What are the capital adequacy ratios?",
    "What did the CEO highlight in their statement?",
]


def render_chat_window(rag_chain: RAGChain) -> None:
    _init_session()
    _render_history()

    if not st.session_state.messages:
        _render_empty_state(rag_chain)

    _render_input(rag_chain)


# ── Session state ──────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "messages":       [],
        "last_sources":   [],
        "top_k":          5,
        "show_sources":   True,
        "stream_response": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── History ────────────────────────────────────────────────────────────────────

def _render_history() -> None:
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "📊"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_source_badges(msg["sources"])


# ── Empty state ────────────────────────────────────────────────────────────────

def _render_empty_state(rag_chain: RAGChain) -> None:
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:2rem; margin-bottom:0.75rem;">📄</div>
      <h3>No conversation yet</h3>
      <p>Upload a PDF in the sidebar, then ask a question below<br>
         or pick one of the suggestions to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Try asking</div>',
                unsafe_allow_html=True)

    cols = st.columns(2)
    for i, q in enumerate(STARTER_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                _handle_question(q, rag_chain)
                st.rerun()


# ── Input ──────────────────────────────────────────────────────────────────────

def _render_input(rag_chain: RAGChain) -> None:
    if prompt := st.chat_input("Ask a question about your PDF…"):
        _handle_question(prompt, rag_chain)


# ── Question handler ───────────────────────────────────────────────────────────

def _handle_question(question: str, rag_chain: RAGChain) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📊"):
        if st.session_state.get("stream_response", True):
            answer, sources = _stream_answer(question, rag_chain)
        else:
            answer, sources = _batch_answer(question, rag_chain)

        if st.session_state.get("show_sources", True) and sources:
            _render_source_badges(sources)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.last_sources = sources


def _stream_answer(question: str, rag_chain: RAGChain) -> tuple[str, list[dict]]:
    placeholder = st.empty()
    full_text   = ""
    try:
        for token in rag_chain.stream(question):
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except Exception as exc:
        logger.exception("Streaming error")
        placeholder.error(f"Error: {exc}")
        return str(exc), []

    # Second call for sources (streaming doesn't return source_documents)
    try:
        result  = rag_chain.invoke(question)
        sources = RAGChain.format_sources(result.get("source_documents", []))
    except Exception:
        sources = []

    return full_text, sources


def _batch_answer(question: str, rag_chain: RAGChain) -> tuple[str, list[dict]]:
    with st.spinner("Retrieving relevant passages…"):
        try:
            result  = rag_chain.invoke(question)
            answer  = result.get("answer", "")
            sources = RAGChain.format_sources(result.get("source_documents", []))
            st.markdown(answer)
            return answer, sources
        except Exception as exc:
            logger.exception("RAG invoke error")
            st.error(f"Error: {exc}")
            return str(exc), []


# ── Inline source badges ───────────────────────────────────────────────────────

def _render_source_badges(sources: list[dict]) -> None:
    if not sources:
        return

    badge_map = {
        "text":  ("badge-text",  "📄 Text"),
        "table": ("badge-table", "📊 Table"),
        "image": ("badge-image", "🖼 Image"),
    }

    badges_html = ""
    for src in sources[:4]:
        etype = src.get("element_type", "text").lower()
        cls, label = badge_map.get(etype, ("badge-text", "📄 Text"))
        page = src.get("page_number")
        page_str = f" · p.{page}" if page else ""
        badges_html += f'<span class="badge {cls}">{label}{page_str}</span>'

    st.markdown(
        f'<div style="margin-top:6px;">{badges_html}</div>',
        unsafe_allow_html=True,
    )
