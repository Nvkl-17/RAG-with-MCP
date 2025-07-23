import streamlit as st
from utils import ingest_documents, retrieve_chunks, generate_response

st.set_page_config(page_title="RAG with MCP", page_icon="💬")


st.sidebar.header("📁 Upload Your Documents")
uploaded_files = st.sidebar.file_uploader("Upload documents", type=["pdf", "docx", "csv", "txt", "md", "pptx"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔄 Processing documents..."):
        knowledge_base = ingest_documents(uploaded_files)
    st.sidebar.success("✅ Documents processed!")
else:
    knowledge_base = None

st.markdown("<h1 style='text-align:center;'>💬 RAG with MCP - Let’s Chat!</h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Ask me anything about the documents...")

if user_input:
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("🤖 Thinking..."):
        context_chunks = retrieve_chunks(user_input, knowledge_base)
        response = generate_response(user_input, context_chunks)

    # Add bot response to history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Show full chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
