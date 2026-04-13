import streamlit as st
import os
from rag import hybrid_search, rerank, generate_answer, process_document

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="CoursePack QA", layout="wide")
st.title("📚 CoursePack QA (RAG System)")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Settings")

k = st.sidebar.slider("🔍 Retrieval (Top K)", 1, 10, 3)
top_k = st.sidebar.slider("🎯 Final Answers", 1, 5, 2)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

# -------------------------------
# INIT SESSION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)

    for file in uploaded_files:
        file_path = f"data/{file.name}"

        with open(file_path, "wb") as f:
            f.write(file.read())

        process_document(file_path)

    st.success("✅ Documents processed!")

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------
# INPUT
# -------------------------------
query = st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.spinner("🤖 Thinking..."):
        docs = hybrid_search(query, k=k)
        docs = rerank(query, docs, top_k=top_k)
        answer = generate_answer(query, docs)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    # -------------------------------
    # SOURCES
    # -------------------------------
    with st.expander("📄 Sources"):
        for i, doc in enumerate(docs):
            st.write(f"**Source {i+1}:**")
            st.write(doc.page_content[:300])
            st.divider()