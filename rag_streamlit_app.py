# ============================================================
#  streamlit_app.py — RAG Chatbot Web Interface
#  Run with: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import sys, os
sys.path.append(os.path.dirname(__file__))
from rag_chatbot import chat, memory, KNOWLEDGE_BASE

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .user-bubble {
        background: #E3F2FD;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 6px 0 6px 20%;
        color: #1565C0;
        font-weight: 500;
        text-align: right;
    }
    .bot-bubble {
        background: #F3E5F5;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 6px 20% 6px 0;
        border-left: 4px solid #7B1FA2;
        color: #212121;
    }
    .source-badge {
        background: #E8F5E9;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.78em;
        color: #2E7D32;
        display: inline-block;
        margin: 2px;
    }
    .memory-badge {
        background: #FFF3E0;
        border-radius: 8px;
        padding: 3px 8px;
        font-size: 0.75em;
        color: #E65100;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role"   : "bot",
        "content": (
            "👋 Hello! I'm your **RAG-powered AI Assistant**.\n\n"
            "I can answer questions about **AI, Machine Learning, "
            "Deep Learning, NLP, LangChain, RAG pipelines, Vector "
            "Databases, Python**, and more!\n\n"
            "My answers are grounded in a vectorized knowledge base. "
            "What would you like to know?"
        ),
        "sources": [],
    })

# ── LAYOUT ────────────────────────────────────────────────────
col1, col2 = st.columns([2.5, 1])

with col1:
    st.title("🤖 RAG Conversational Chatbot")
    st.caption(
        "LangChain • FAISS Vector Store • "
        "Context Memory • Retrieval-Augmented Generation"
    )

    # ── Chat Display ─────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-bubble">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                if msg.get("sources"):
                    st.markdown("📚 **Sources retrieved:**")
                    for src in msg["sources"]:
                        st.markdown(
                            f'<span class="source-badge">'
                            f'📄 {src}</span>',
                            unsafe_allow_html=True
                        )

    # ── Input ────────────────────────────────────────────────
    st.divider()
    user_input = st.chat_input(
        "Ask me anything about AI, ML, NLP, LangChain..."
    )

    # ── Quick Questions ──────────────────────────────────────
    st.markdown("**💡 Quick questions:**")
    q_cols = st.columns(3)
    quick = None
    questions = [
        "What is RAG?",
        "How does FAISS work?",
        "What is LangChain?",
        "Explain deep learning",
        "What is a transformer?",
        "Python for data science",
    ]
    for i, q in enumerate(questions):
        with q_cols[i % 3]:
            if st.button(q, key=f"q{i}"):
                quick = q

    if quick:
        user_input = quick

    # ── Process ──────────────────────────────────────────────
    if user_input:
        st.session_state.messages.append({
            "role": "user", "content": user_input, "sources": []
        })
        with st.spinner("🔍 Retrieving from knowledge base..."):
            response, retrieved = chat(user_input)
        sources = [d["title"] for d in retrieved] if retrieved else []
        st.session_state.messages.append({
            "role": "bot", "content": response, "sources": sources
        })
        st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────
with col2:
    st.markdown("### 💾 Memory Status")
    turns = len(memory.history)
    st.markdown(
        f'<span class="memory-badge">🧠 {turns} turns stored</span>',
        unsafe_allow_html=True
    )
    if st.button("🗑️ Clear Memory"):
        memory.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    for key, doc in KNOWLEDGE_BASE.items():
        st.markdown(f"📄 {doc['title']}")

    st.divider()
    st.markdown("### ⚙️ RAG Pipeline")
    st.markdown("""
    ```
    Query
      ↓
    Embed Query
      ↓
    FAISS Search
    (top-k chunks)
      ↓
    Build Prompt
    + History
      ↓
    LLM Generate
      ↓
    Response
    ```
    """)

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🔗 **LangChain**
    - 🗄️ **FAISS** vector store
    - 🤗 **HuggingFace** embeddings
    - 🧠 **GPT / Mistral** LLM
    - 💾 **ConversationMemory**
    - 🎈 **Streamlit** UI
    """)
