"""
src/app/components/source_viewer.py
-------------------------------------
Enterprise source citations panel — right column.
Shows element-type badges, page numbers, and content snippets.
"""

import streamlit as st


BADGE_MAP = {
    "text":  ("#DBEAFE", "#2563EB", "📄 Text"),
    "table": ("#DCFCE7", "#16A34A", "📊 Table"),
    "image": ("#FEF3C7", "#D97706", "🖼 Image"),
}


def render_source_viewer() -> None:
    sources: list[dict] = st.session_state.get("last_sources", [])

    # ── Panel header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      display:flex; align-items:center; justify-content:space-between;
      margin-bottom:0.75rem;
    ">
      <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.1em;
                  text-transform:uppercase; color:#8C897F;">
        Source Citations
      </div>
      <div style="font-size:0.7rem; color:#8C897F;">
        {count} chunk{plural}
      </div>
    </div>
    """.format(
        count=len(sources),
        plural="s" if len(sources) != 1 else "",
    ), unsafe_allow_html=True)

    if not sources:
        _render_empty()
        return

    # ── Source cards ──────────────────────────────────────────────────────────
    for i, src in enumerate(sources):
        _render_source_card(src, expanded=(i == 0))

    # ── Element-type legend ───────────────────────────────────────────────────
    _render_legend(sources)


# ── Empty state ───────────────────────────────────────────────────────────────

def _render_empty() -> None:
    st.markdown("""
    <div style="
      text-align:center; padding:2.5rem 1rem;
      border:1px dashed #E4E2DC; border-radius:10px;
    ">
      <div style="font-size:1.6rem; margin-bottom:0.5rem;">🔍</div>
      <div style="font-size:0.82rem; font-weight:600; color:#4A4740;
                  margin-bottom:0.25rem;">No citations yet</div>
      <div style="font-size:0.75rem; color:#8C897F; line-height:1.5;">
        Sources used to answer your question<br>will appear here.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Single source card ────────────────────────────────────────────────────────

def _render_source_card(src: dict, expanded: bool = False) -> None:
    etype   = src.get("element_type", "text").lower()
    page    = src.get("page_number")
    source  = src.get("source", "unknown")
    snippet = src.get("snippet", "")
    chunk   = src.get("chunk_index", 0)

    bg, fg, label = BADGE_MAP.get(etype, BADGE_MAP["text"])

    # Title for expander
    page_str  = f"  ·  p.{page}" if page else ""
    title_str = f"{label}{page_str}"

    with st.expander(title_str, expanded=expanded):
        # Source file pill
        st.markdown(
            f'<div style="font-size:0.7rem; color:#8C897F; margin-bottom:0.5rem;">'
            f'<span style="'
            f'  background:#F7F6F3; border:1px solid #E4E2DC; border-radius:4px;'
            f'  padding:1px 7px; font-family:monospace;">'
            f'{source}</span>'
            f'&nbsp;&nbsp;chunk #{chunk}</div>',
            unsafe_allow_html=True,
        )

        # Snippet
        st.markdown(
            f'<div style="'
            f'  font-size:0.8rem; line-height:1.55; color:#4A4740;'
            f'  border-left:3px solid {fg}; padding-left:0.6rem;'
            f'">{_escape(snippet)}</div>',
            unsafe_allow_html=True,
        )


# ── Legend ────────────────────────────────────────────────────────────────────

def _render_legend(sources: list[dict]) -> None:
    counts = {"text": 0, "table": 0, "image": 0}
    for src in sources:
        etype = src.get("element_type", "text").lower()
        if etype in counts:
            counts[etype] += 1

    active = {k: v for k, v in counts.items() if v > 0}
    if not active:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    pills = ""
    for etype, count in active.items():
        bg, fg, label = BADGE_MAP[etype]
        pills += (
            f'<span style="'
            f'  background:{bg}; color:{fg}; border-radius:99px;'
            f'  padding:2px 9px; font-size:0.7rem; font-weight:600;'
            f'  margin-right:4px;">'
            f'{label} {count}</span>'
        )
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; gap:4px;">{pills}</div>',
        unsafe_allow_html=True,
    )


# ── Utility ───────────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
