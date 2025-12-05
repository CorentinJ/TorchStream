import threading

server_state_lock = threading.RLock()
# In-memory store that persists across Streamlit reruns & different users, unlike st.session_state
# To be used with the above lock.
server_run_state = {}
