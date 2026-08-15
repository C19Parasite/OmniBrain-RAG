import streamlit as st
from utils.api import query_api

st.title("🧠 OmniBrain RAG")
st.subheader("Chat with your documents")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        question = st.chat_input("Ask a question about your document...")

question = st.chat_input("Ask a question aboout your document...")

if question:
    result = query_api(question)

    if result["success"]:
        st.write(result["answer"])
    else:
        st.error(result["answer"])