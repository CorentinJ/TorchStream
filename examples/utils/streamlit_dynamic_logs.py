import html
import sys

import streamlit as st
import streamlit.components.v1 as components

from examples.utils.streamlit_run_state import server_run_state, server_state_lock

_LOGBOX_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
}}
#log-box {{
    box-sizing: border-box;
    height: 100%;
    overflow-y: auto;
    white-space: pre;
    font-family: monospace;
    border: 1px solid #CCC;
    padding: 0.5rem;
}}
</style>
</head>
<body>
<div id="log-box"></div>
<script>
(function() {{
    let lastText = "";
    function syncLog() {{
    try {{
        const parentDoc = window.parent && window.parent.document;
        const src = parentDoc && parentDoc.getElementById('{data_id}');
        const box = document.getElementById('log-box');
        if (!src || !box) return;

        const newText = src.textContent || src.innerText || "";
        if (newText !== lastText) {{
        lastText = newText;
        box.textContent = newText;
        box.scrollTop = box.scrollHeight;
        }}
    }} catch (e) {{
        // ignore timing / access issues
    }}
    }}
    setInterval(syncLog, 150);
}})();
</script>
</body>
</html>
"""


def _refresh(key: str) -> None:
    widget, text = server_run_state[key]
    widget.markdown(
        f'<div id="{key}" style="display:none;">{text}</div>',
        unsafe_allow_html=True,
    )


def create_logbox(unique_id: str, height: int = 300) -> None:
    key = f"logbox_{unique_id}"

    with server_state_lock:
        server_run_state[key] = (st.empty(), server_run_state.get(key, (None, ""))[1])

    components.html(_LOGBOX_HTML.format(data_id=key), height=height)
    _refresh(key)


def update_logs(unique_id: str, logs: list[str]) -> None:
    with server_state_lock:
        key = f"logbox_{unique_id}"
        if key not in server_run_state:
            print("Logbox not created, exiting")
            sys.exit(0)

        log_text = "\n".join(logs)
        safe = html.escape(log_text)
        server_run_state[key] = (server_run_state[key][0], safe)
        _refresh(key)
