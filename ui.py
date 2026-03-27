import streamlit as st
import requests
import time

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8001"

st.set_page_config(
    page_title="1P Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a40, #24243e);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* User message bubble */
.user-bubble {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px auto;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 4px 20px rgba(108,99,255,0.3);
    word-wrap: break-word;
}

/* Bot message bubble */
.bot-bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px auto 8px 0;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    word-wrap: break-word;
}

/* Avatar labels */
.avatar-user  { text-align: right; font-size: 0.75rem; color: #a78bfa; margin-bottom: 2px; }
.avatar-bot   { text-align: left;  font-size: 0.75rem; color: #60a5fa; margin-bottom: 2px; }

/* Chat container */
.chat-container {
    max-height: 62vh;
    overflow-y: auto;
    padding: 12px 4px;
    scroll-behavior: smooth;
}

/* Status badge */
.badge-green { background:#10b981; color:white; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; }
.badge-red   { background:#ef4444; color:white; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; }
.badge-gray  { background:#6b7280; color:white; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }

/* Chunk expander */
.chunk-box {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #6c63ff;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.5;
}
.chunk-heading { font-weight: 600; color: #a78bfa; margin-bottom: 4px; font-size: 0.85rem; }
.chunk-score   { font-size: 0.72rem; color: #60a5fa; }

/* Title */
.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.page-sub {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 2px;
    margin-bottom: 16px;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08); }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "show_chunks" not in st.session_state:
    st.session_state.show_chunks = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.json() if r.ok else None
    except Exception:
        return None


def build_index():
    try:
        r = requests.post(f"{API_BASE}/index/build", timeout=300)
        return r.json(), r.ok
    except Exception as e:
        return {"detail": str(e)}, False


def send_chat(query: str, top_k: int):
    try:
        r = requests.post(
            f"{API_BASE}/index/chat",
            json={"query": query, "top_k": top_k},
            timeout=60,
        )
        return r.json(), r.ok
    except Exception as e:
        return {"detail": str(e)}, False


def get_index_info():
    try:
        r = requests.get(f"{API_BASE}/index/info", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="page-title">🤖 1P Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Powered by Groq + Ollama + FAISS</p>', unsafe_allow_html=True)
    st.divider()

    # Health check
    st.markdown("#### 🔌 Server Status")
    health = get_health()
    if health:
        key_ok    = health.get("groq_api_key_set", False)
        idx_ok    = health.get("index_loaded", False)
        n_chunks  = health.get("total_chunks", 0)
        model     = health.get("groq_model", "—")
        embed     = health.get("embed_model", "—")

        col1, col2 = st.columns(2)
        col1.markdown(
            f'**API** <span class="badge-green">Online</span>', unsafe_allow_html=True
        )
        col2.markdown(
            f'**Key** <span class="{"badge-green" if key_ok else "badge-red"}">{"Set ✓" if key_ok else "Missing ✗"}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Index** <span class="{"badge-green" if idx_ok else "badge-gray"}">{"Loaded · " + str(n_chunks) + " chunks" if idx_ok else "Not built"}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"🧠 LLM: `{model}`")
        st.caption(f"📐 Embed: `{embed}`")

        if not key_ok:
            st.error("⚠️ Add `GROQ_API_KEY` to your `.env` file and restart the server.")
    else:
        st.markdown('<span class="badge-red">Offline</span>', unsafe_allow_html=True)
        st.error("Cannot reach the API at `localhost:8001`. Make sure the server is running:\n```\nuvicorn chatbot:app --port 8001\n```")

    st.divider()

    # Build index
    st.markdown("#### 🗂️ Index Management")
    if st.button("⚡ Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building FAISS index from `formatted_Output.txt`…"):
            data, ok = build_index()
        if ok:
            st.success(f"✅ Index built! {data.get('total_chunks', '?')} chunks indexed.")
        else:
            st.error(f"❌ {data.get('detail', 'Unknown error')}")

    st.divider()

    # Settings
    st.markdown("#### ⚙️ Settings")
    st.session_state.top_k = st.slider(
        "Top-K chunks to retrieve", min_value=1, max_value=15, value=st.session_state.top_k
    )
    st.session_state.show_chunks = st.toggle(
        "Show supporting chunks", value=st.session_state.show_chunks
    )

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p style="text-align:center;color:#374151;font-size:0.7rem;margin-top:16px">1P Chatbot · v2.1</p>',
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<p class="page-title">💬 Chat</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Ask anything about your indexed document</p>', unsafe_allow_html=True)

# Chat history
chat_area = st.container()
with chat_area:
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;color:#4b5563;">
                <div style="font-size:3rem;margin-bottom:12px">🔍</div>
                <div style="font-size:1.1rem;font-weight:600;color:#6b7280;">No messages yet</div>
                <div style="font-size:0.85rem;margin-top:6px;color:#4b5563;">
                    Build the index first, then ask a question below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                st.markdown(f'<div class="avatar-user">You</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="avatar-bot">🤖 Assistant</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-bubble">{content}</div>', unsafe_allow_html=True)

                # Show supporting chunks
                if st.session_state.show_chunks and msg.get("chunks"):
                    with st.expander(f"📚 {len(msg['chunks'])} supporting chunks used", expanded=False):
                        for i, chunk in enumerate(msg["chunks"], 1):
                            heading = chunk.get("heading", "")
                            text    = chunk.get("text", "")
                            score   = chunk.get("score", 0)
                            preview = text[:300] + ("..." if len(text) > 300 else "")
                            heading_label = (heading) if heading else ("Chunk " + str(i))

                            st.markdown(
                                '<div class="chunk-box">'
                                '<div class="chunk-heading">' + heading_label + '</div>'
                                '<div class="chunk-score">Score: ' + str(score) + '</div>'
                                '<div style="margin-top:6px">' + preview + '</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )

st.divider()

# Input area
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask a question about your document…",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    # Call API
    with st.spinner("Thinking…"):
        data, ok = send_chat(user_input.strip(), st.session_state.top_k)

    if ok:
        answer = data.get("answer", "No answer returned.")
        chunks = data.get("supporting_chunks", [])
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "chunks":  chunks,
        })
    else:
        error_msg = data.get("detail", "Unknown error from API.")
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"⚠️ Error: {error_msg}",
            "chunks":  [],
        })

    st.rerun()
