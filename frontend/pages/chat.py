import streamlit as st
from utils.api import ask_question
st.title("Chat with OmniBrain RAG")
st.write("Ask questions about your uploaded PDF document.")
if "messages" not in st.session_state:
    st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask a question about your PDF document")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
            result = ask_question(question)

           if result["success"]:
               response = result["answer"]
            else:
               response = "Unable to get a response."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
            st.write(response)