import html

import streamlit as st
import streamlit.components.v1 as components

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


def create_logbox(unique_id: str, height: int = 300) -> None:
    key = f"logbox_{unique_id}"
    if key in st.session_state:
        return  # Already created

    # Hidden data holder in the main page
    st.session_state[key] = st.empty()

    components.html(_LOGBOX_HTML.format(data_id=key), height=height)


def update_logs(unique_id: str, logs: list[str]) -> None:
    key = f"logbox_{unique_id}"
    assert key in st.session_state, "Logbox not created. Call create_logbox first."

    log_text = "\n".join(logs)
    safe = html.escape(log_text)
    st.session_state[key].markdown(
        f'<div id="{key}" style="display:none;">{safe}</div>',
        unsafe_allow_html=True,
    )
