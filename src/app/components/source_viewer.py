"""
src/app/components/source_viewer.py
-------------------------------------
Streamlit source citations panel.

Displays the retrieved chunks that backed the last answer:
  - Element type badge (Text / Table / Image)
  - Page number and source file
  - Expandable content snippet
  - Confidence score if available
"""

import streamlit as st

from src.pipeline.rag_chain import RAGChain


# ── Colour and icon config ─────────────────────────────────────────────────────
ELEMENT_CONFIG = {
    "text": {
        "color":  "#4A90D9",
        "bg":     "#EBF4FF",
        "icon":   "📄",
        "label":  "Text",
    },
    "table": {
        "color":  "#27AE60",
        "bg":     "#EAFAF1",
        "icon":   "📊",
        "label":  "Table",
    },
    "image": {
        "color":  "#E67E22",
        "bg":     "#FEF9E7",
        "icon":   "🖼️",
        "label":  "Image",
    },
}

DEFAULT_CONFIG = {
    "color": "#888888",
    "bg":    "#F5F5F5",
    "icon":  "📎",
    "label": "Source",
}


def render_source_viewer() -> None:
    """
    Render the source citations panel using st.session_state.last_sources.
    Call this in the right column of the main layout.
    """
    st.subheader("🔍 Source Citations")

    sources = st.session_state.get("last_sources", [])

    if not sources:
        _render_empty_state()
        return

    _render_summary_bar(sources)
    st.divider()

    for i, source in enumerate(sources):
        _render_source_card(source, index=i)


# ── Empty state ────────────────────────────────────────────────────────────────

def _render_empty_state() -> None:
    st.markdown("""
        <div style="
            text-align:center;
            padding:2rem 1rem;
            color:#aaa;
            border: 1px dashed #ddd;
            border-radius:8px;
            margin-top:1rem;
        ">
            <div style="font-size:2rem;">🔍</div>
            <p style="margin:0.5rem 0 0 0; font-size:0.9rem;">
                Sources will appear here after you ask a question.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ── Summary bar ────────────────────────────────────────────────────────────────

def _render_summary_bar(sources: list[dict]) -> None:
    """Show a compact count of each element type retrieved."""
    counts = {"text": 0, "table": 0, "image": 0}
    for s in sources:
        et = s.get("element_type", "text").lower()
        if et in counts:
            counts[et] += 1

    cols = st.columns(3)
    labels = [("📄 Text", "text"), ("📊 Table", "table"), ("🖼️ Image", "image")]

    for col, (label, key) in zip(cols, labels):
        cfg   = ELEMENT_CONFIG[key]
        count = counts[key]
        with col:
            st.markdown(
                f'<div style="'
                f'background:{cfg["bg"]}; border:1px solid {cfg["color"]}44; '
                f'border-radius:8px; padding:6px; text-align:center;">'
                f'<span style="color:{cfg["color"]}; font-weight:700; font-size:1.1rem;">'
                f'{count}</span>'
                f'<br><span style="font-size:0.72rem; color:#666;">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Individual source card ─────────────────────────────────────────────────────

def _render_source_card(source: dict, index: int) -> None:
    """
    Render one source chunk as an expandable card.

    source dict keys:
        element_type, page_number, source, snippet, chunk_index
    """
    et      = source.get("element_type", "text").lower()
    page    = source.get("page_number")
    file    = source.get("source", "unknown")
    snippet = source.get("snippet", "")
    chunk_i = source.get("chunk_index", 0)
    cfg     = ELEMENT_CONFIG.get(et, DEFAULT_CONFIG)

    page_label  = f"Page {page}" if page else "Page ?"
    chunk_label = f"Chunk {chunk_i}"
    title       = f"{cfg['icon']} {cfg['label']} · {page_label}"

    with st.expander(title, expanded=(index == 0)):   # first card open by default

        # ── Header row ────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="'
            f'display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px;">'

            # Element type badge
            f'<span style="'
            f'background:{cfg["bg"]}; color:{cfg["color"]}; '
            f'border:1px solid {cfg["color"]}66; border-radius:20px; '
            f'padding:2px 10px; font-size:0.75rem; font-weight:600;">'
            f'{cfg["icon"]} {cfg["label"]}</span>'

            # Page badge
            f'<span style="'
            f'background:#f0f0f0; color:#555; '
            f'border:1px solid #ddd; border-radius:20px; '
            f'padding:2px 10px; font-size:0.75rem;">'
            f'📖 {page_label}</span>'

            # Chunk badge
            f'<span style="'
            f'background:#f0f0f0; color:#555; '
            f'border:1px solid #ddd; border-radius:20px; '
            f'padding:2px 10px; font-size:0.75rem;">'
            f'🧩 {chunk_label}</span>'

            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Source file ───────────────────────────────────────────────────────
        st.caption(f"📁 {file}")

        # ── Content snippet ───────────────────────────────────────────────────
        if snippet:
            st.markdown(
                f'<div style="'
                f'background:{cfg["bg"]}; border-left:3px solid {cfg["color"]}; '
                f'padding:8px 12px; border-radius:0 4px 4px 0; '
                f'font-size:0.82rem; color:#333; line-height:1.5; '
                f'white-space:pre-wrap; word-break:break-word;">'
                f'{_escape_html(snippet)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("_No snippet available._")


# ── Utility ────────────────────────────────────────────────────────────────────

def _escape_html(text: str) -> str:
    """Minimal HTML escaping to prevent injection in the snippet box."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )