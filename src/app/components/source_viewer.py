 """
src/app/components/source_viewer.py
-------------------------------------
Enterprise source citations panel — dark theme.
"""

import streamlit as st

BADGE_MAP = {
    "text":  ("#1A2535", "#60A5FA", "📄 Text"),
    "table": ("#152518", "#4ADE80", "📊 Table"),
    "image": ("#271E10", "#FBB24A", "🖼 Image"),
}


def render_source_viewer() -> None:
    sources: list[dict] = st.session_state.get("last_sources", [])
    count = len(sources)

    st.markdown(f"""
    <div style="
      display:flex; align-items:center; justify-content:space-between;
      margin-bottom:0.75rem;
    ">
      <div class="section-lbl" style="margin:0;">Source Citations</div>
      <div style="
        font-size:0.68rem; font-family:'DM Mono',monospace;
        color:#444; background:#1A1A1A; border:1px solid #252525;
        border-radius:99px; padding:2px 9px;
      ">{count} chunk{"s" if count != 1 else ""}</div>
    </div>
    """, unsafe_allow_html=True)

    if not sources:
        st.markdown("""
        <div style="
          text-align:center; padding:2.5rem 1rem;
          border:1px dashed #252525; border-radius:12px;
        ">
          <div style="font-size:1.5rem; margin-bottom:0.5rem;">🔍</div>
          <div style="font-size:0.82rem; font-weight:600; color:#666;
                      margin-bottom:0.25rem;">No citations yet</div>
          <div style="font-size:0.73rem; color:#444; line-height:1.55;">
            Sources used to answer<br>your question appear here.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    for i, src in enumerate(sources):
        _card(src, expanded=(i == 0))

    _legend(sources)


def _card(src: dict, expanded: bool) -> None:
    etype   = src.get("element_type", "text").lower()
    page    = src.get("page_number")
    source  = src.get("source", "unknown")
    snippet = src.get("snippet", "")
    chunk   = src.get("chunk_index", 0)

    bg, fg, label = BADGE_MAP.get(etype, BADGE_MAP["text"])
    page_str = f"  ·  p.{page}" if page else ""
    title    = f"{label}{page_str}"

    with st.expander(title, expanded=expanded):
        st.markdown(
            f'<div style="margin-bottom:0.4rem;">'
            f'<span style="'
            f'  background:#1A1A1A; border:1px solid #2A2A2A; border-radius:5px;'
            f'  padding:1px 7px; font-size:0.68rem; font-family:DM Mono,monospace;'
            f'  color:#555;">{_esc(source)}</span>'
            f'<span style="font-size:0.68rem; color:#444; margin-left:6px;">chunk #{chunk}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="snippet-block" style="border-left-color:{fg};">'
            f'{_esc(snippet)}</div>',
            unsafe_allow_html=True,
        )


def _legend(sources: list) -> None:
    counts = {"text": 0, "table": 0, "image": 0}
    for s in sources:
        e = s.get("element_type", "text").lower()
        if e in counts:
            counts[e] += 1

    active = {k: v for k, v in counts.items() if v > 0}
    if not active:
        return

    pills = ""
    for etype, n in active.items():
        bg, fg, label = BADGE_MAP[etype]
        pills += (
            f'<span style="background:{bg}; color:{fg}; border-radius:99px;'
            f' padding:2px 9px; font-size:0.68rem; font-weight:600;'
            f' margin-right:4px; font-family:DM Sans,sans-serif;">'
            f'{label} {n}</span>'
        )
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:0.75rem;">{pills}</div>',
        unsafe_allow_html=True,
    )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
