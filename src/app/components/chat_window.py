"""
src/app/components/chat_window.py
----------------------------------
Streamlit chat window — renders the conversation history,
handles user input, calls the RAG chain, and displays answers
with optional streaming.
"""

import logging
import streamlit as st

from src.pipeline.rag_chain import RAGChain

logger = logging.getLogger(__name__)

# ── Suggested starter questions (shown on empty chat) ─────────────────────────
STARTER_QUESTIONS = [
    "What was the net income reported?",
    "Summarise the key risk factors.",
    "What does the revenue breakdown table show?",
    "Describe any charts or graphs in the document.",
    "What are the capital adequacy ratios?",
    "What did the CEO highlight in their message?",
]


def render_chat_window(rag_chain: RAGChain) -> None:
    """
    Render the full chat window:
      1. Conversation history
      2. Starter question chips (empty state)
      3. User input box
      4. RAG answer + source passthrough to source_viewer

    Args:
        rag_chain: Live RAGChain instance
    """
    _initialise_session_state()
    _render_history()

    # Show starter chips only when no messages yet
    if not st.session_state.messages:
        _render_starter_questions(rag_chain)

    _render_input_box(rag_chain)


# ── Session state ──────────────────────────────────────────────────────────────

def _initialise_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "stream_response" not in st.session_state:
        st.session_state.stream_response = True


# ── Conversation history ───────────────────────────────────────────────────────

def _render_history() -> None:
    """Replay all messages from session state as chat bubbles."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=_avatar(msg["role"])):
            st.markdown(msg["content"])

            # Re-render inline source badges under assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_inline_source_badges(msg["sources"])


# ── Starter question chips ─────────────────────────────────────────────────────

def _render_starter_questions(rag_chain: RAGChain) -> None:
    st.markdown(
        "<p style='color:#888; font-size:0.9rem; margin-bottom:0.5rem;'>"
        "Try asking:</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, question in enumerate(STARTER_QUESTIONS):
        with cols[i % 2]:
            if st.button(
                question,
                key=f"starter_{i}",
                use_container_width=True,
            ):
                _handle_question(question, rag_chain)
                st.rerun()


# ── User input ─────────────────────────────────────────────────────────────────

def _render_input_box(rag_chain: RAGChain) -> None:
    if prompt := st.chat_input(
        "Ask a question about your PDF…",
        key="chat_input",
    ):
        _handle_question(prompt, rag_chain)


# ── Core question handler ──────────────────────────────────────────────────────

def _handle_question(question: str, rag_chain: RAGChain) -> None:
    """
    1. Display user message
    2. Call RAG chain (streaming or batch)
    3. Display assistant answer
    4. Store sources for the source_viewer panel
    """
    # ── User bubble ────────────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # ── Assistant bubble ───────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        use_streaming = st.session_state.get("stream_response", True)

        if use_streaming:
            answer, sources = _stream_answer(question, rag_chain)
        else:
            answer, sources = _batch_answer(question, rag_chain)

        # Inline source badges beneath the answer
        if st.session_state.get("show_sources", True) and sources:
            _render_inline_source_badges(sources)

    # ── Persist to session state ───────────────────────────────────────────────
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.last_sources = sources


def _stream_answer(question: str, rag_chain: RAGChain) -> tuple[str, list[dict]]:
    """
    Stream tokens into the chat bubble.
    After streaming, does a second invoke() call to get source_documents.

    Returns (full_answer_str, formatted_sources_list).
    """
    placeholder  = st.empty()
    full_answer  = ""

    try:
        with st.spinner(""):
            for token in rag_chain.stream(question):
                full_answer += token
                placeholder.markdown(full_answer + "▌")   # blinking cursor

        placeholder.markdown(full_answer)

        # Fetch sources via a second invoke (streaming doesn't return docs)
        result  = rag_chain.invoke(question)
        sources = RAGChain.format_sources(result.get("source_documents", []))

    except Exception as e:
        full_answer = f"⚠️ Error generating answer: {e}"
        placeholder.markdown(full_answer)
        logger.error(f"Stream error: {e}", exc_info=True)
        sources = []

    return full_answer, sources


def _batch_answer(question: str, rag_chain: RAGChain) -> tuple[str, list[dict]]:
    """
    Non-streaming invoke with a spinner.
    Returns (answer_str, formatted_sources_list).
    """
    try:
        with st.spinner("Thinking …"):
            result  = rag_chain.invoke(question)
        answer  = result.get("answer", "No answer generated.")
        sources = RAGChain.format_sources(result.get("source_documents", []))
        st.markdown(answer)

    except Exception as e:
        answer  = f"⚠️ Error: {e}"
        sources = []
        st.markdown(answer)
        logger.error(f"Batch invoke error: {e}", exc_info=True)

    return answer, sources


# ── Inline source badges ───────────────────────────────────────────────────────

def _render_inline_source_badges(sources: list[dict]) -> None:
    """
    Render compact source badges inline below an answer bubble.
    E.g.: [📊 Table · P.12]  [📄 Text · P.4]
    """
    if not sources:
        return

    badge_html = []
    for s in sources[:5]:     # cap at 5 to avoid clutter
        badge   = RAGChain.element_type_badge(s["element_type"])
        page    = f"P.{s['page_number']}" if s["page_number"] else "P.?"
        color   = _element_color(s["element_type"])
        badge_html.append(
            f'<span style="'
            f'background:{color}22; color:{color}; '
            f'border:1px solid {color}55; border-radius:12px; '
            f'padding:2px 10px; font-size:0.75rem; margin-right:4px;">'
            f'{badge} · {page}</span>'
        )

    st.markdown(
        '<div style="margin-top:6px;">' + "".join(badge_html) + '</div>',
        unsafe_allow_html=True,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avatar(role: str) -> str:
    return "👤" if role == "user" else "🤖"


def _element_color(element_type: str) -> str:
    colors = {
        "text":  "#4A90D9",
        "table": "#27AE60",
        "image": "#E67E22",
    }
    return colors.get(element_type.lower(), "#888888")