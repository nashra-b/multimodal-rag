"""
components
----------
Streamlit UI component modules.
 
Exports:
    render_sidebar       — Left panel: PDF upload, session controls, config
    render_chat_window   — Centre panel: conversation history and input
    render_source_viewer — Right panel: source citations for the last answer
"""
 
from .sidebar       import render_sidebar        # noqa: F401
from .chat_window   import render_chat_window    # noqa: F401
from .source_viewer import render_source_viewer  # noqa: F401
 
__all__ = [
    "render_sidebar",
    "render_chat_window",
    "render_source_viewer",
]
 