"""
src/app/components/chat_window.py
----------------------------------
Enterprise chat window — dark theme, streaming, source badges.
"""

import logging
import streamlit as st
from src.pipeline.rag_chain import RAGChain

logger = logging.getLogger(__name__)

STARTERS = [
    "What was the net income reported?",
    "Summarise the key risk factors.",
    "What does the revenue breakdown table show?",
    "Describe any charts or diagrams in the document.",
    "What are the capital adequacy ratios?",
    "What did the CEO highlight in their statement?",
]

BADGE_MAP = {
    "text":  ("pill-text",  "📄 Text"),
    "table": ("pill-table", "📊 Table"),
    "image": ("pill-image", "🖼 Image"),
}


def render_chat_window(rag_chain: RAGChain) -> None:
    _init()
    _history()
    if not st.session_state.messages:
        _empty_state(rag_chain)
    _input(rag_chain)


def _init() -> None:
    for k, v in {
        "messages": [], "last_sources": [],
        "top_k": 5, "show_sources": True, "stream_response": True,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _history() -> None:
    for msg in st.session_state.messages:
        av = "👤" if msg["role"] == "user" else "📊"
        with st.chat_message(msg["role"], avatar=av):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _badges(msg["sources"])


def _empty_state(rag_chain: RAGChain) -> None:
    st.markdown("""
    <div class="empty-card">
      <div class="icon">📄</div>
      <h4>No conversation yet</h4>
      <p>Upload a PDF in the sidebar, then ask a question below<br>
         or pick one of the suggestions to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Try asking</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="starter-grid">', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(STARTERS):
            with cols[i % 2]:
                if st.button(q, key=f"s_{i}", use_container_width=True):
                    _ask(q, rag_chain)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def _input(rag_chain: RAGChain) -> None:
    if prompt := st.chat_input("Ask a question about your PDF…"):
        _ask(prompt, rag_chain)


def _ask(q: str, rag_chain: RAGChain) -> None:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user", avatar="👤"):
        st.markdown(q)

    with st.chat_message("assistant", avatar="📊"):
        if st.session_state.get("stream_response", True):
            answer, sources = _stream(q, rag_chain)
        else:
            answer, sources = _batch(q, rag_chain)
        if st.session_state.get("show_sources", True) and sources:
            _badges(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.last_sources = sources


def _stream(q: str, rag_chain: RAGChain) -> tuple[str, list]:
    ph, full = st.empty(), ""
    try:
        for tok in rag_chain.stream(q):
            full += tok
            ph.markdown(full + "▌")
        ph.markdown(full)
    except Exception as e:
        ph.error(str(e))
        return str(e), []
    try:
        res     = rag_chain.invoke(q)
        sources = RAGChain.format_sources(res.get("source_documents", []))
    except Exception:
        sources = []
    return full, sources


def _batch(q: str, rag_chain: RAGChain) -> tuple[str, list]:
    with st.spinner("Retrieving…"):
        try:
            res     = rag_chain.invoke(q)
            answer  = res.get("answer", "")
            sources = RAGChain.format_sources(res.get("source_documents", []))
            st.markdown(answer)
            return answer, sources
        except Exception as e:
            st.error(str(e))
            return str(e), []


def _badges(sources: list) -> None:
    html = ""
    for s in sources[:5]:
        etype         = s.get("element_type", "text").lower()
        cls, label    = BADGE_MAP.get(etype, ("pill-text", "📄 Text"))
        page          = s.get("page_number")
        page_str      = f" · p.{page}" if page else ""
        html += f'<span class="pill {cls}">{label}{page_str}</span>'
    st.markdown(f'<div style="margin-top:6px; padding-bottom:2px;">{html}</div>', unsafe_allow_html=True)
