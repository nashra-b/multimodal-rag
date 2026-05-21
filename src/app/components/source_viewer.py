"""
src/app/components/source_viewer.py — light enterprise theme
"""
import streamlit as st

BADGE_MAP = {
    "text":  ("#EFF6FF", "#2563EB", "📄 Text"),
    "table": ("#F0FDF4", "#16A34A", "📊 Table"),
    "image": ("#FFFBEB", "#D97706", "🖼 Image"),
}


def render_source_viewer() -> None:
    sources = st.session_state.get("last_sources", [])
    count   = len(sources)

    st.markdown(
        f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">'
        f'<p class="sec-label" style="margin:0;">Source Citations</p>'
        f'<span style="font-size:0.68rem; font-family:monospace; color:#9B9890;'
        f' background:#F5F4F0; border:1px solid #E5E3DC; border-radius:99px; padding:2px 9px;">'
        f'{count} chunk{"s" if count!=1 else ""}</span></div>',
        unsafe_allow_html=True,
    )

    if not sources:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 1rem;
                    border:1px dashed #D5D2C8; border-radius:12px;">
          <div style="font-size:1.5rem; margin-bottom:0.5rem;">🔍</div>
          <div style="font-size:0.82rem; font-weight:600; color:#6B6860; margin-bottom:0.25rem;">
            No citations yet</div>
          <div style="font-size:0.73rem; color:#9B9890; line-height:1.55;">
            Sources used to answer<br>your question appear here.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for i, src in enumerate(sources):
        _card(src, expanded=(i == 0))
    _legend(sources)


def _card(src, expanded=False):
    etype   = src.get("element_type", "text").lower()
    page    = src.get("page_number")
    source  = src.get("source", "unknown")
    snippet = src.get("snippet", "")
    chunk   = src.get("chunk_index", 0)
    bg, fg, label = BADGE_MAP.get(etype, BADGE_MAP["text"])

    with st.expander(f"{label}{f'  ·  p.{page}' if page else ''}", expanded=expanded):
        st.markdown(
            f'<span style="background:#F5F4F0; border:1px solid #E5E3DC; border-radius:5px;'
            f' padding:1px 7px; font-size:0.68rem; font-family:monospace; color:#6B6860;">'
            f'{_e(source)}</span>'
            f'<span style="font-size:0.68rem; color:#9B9890; margin-left:6px;">chunk #{chunk}</span>'
            f'<div style="font-size:0.78rem; line-height:1.55; color:#6B6860;'
            f' border-left:2.5px solid {fg}; padding-left:0.6rem; margin-top:0.5rem;">'
            f'{_e(snippet)}</div>',
            unsafe_allow_html=True,
        )


def _legend(sources):
    counts = {"text":0,"table":0,"image":0}
    for s in sources:
        e = s.get("element_type","text").lower()
        if e in counts:
            counts[e] += 1
    pills = ""
    for etype, n in counts.items():
        if n == 0:
            continue
        bg, fg, label = BADGE_MAP[etype]
        pills += (f'<span style="background:{bg}; color:{fg}; border-radius:99px;'
                  f' padding:2px 9px; font-size:0.68rem; font-weight:600; margin-right:4px;">'
                  f'{label} {n}</span>')
    if pills:
        st.markdown(f'<div style="margin-top:0.75rem;">{pills}</div>', unsafe_allow_html=True)


def _e(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
