import streamlit as st
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
        # Here you would typically call your RAG model to get a response
        response = "This is a placeholder response from OmniBrain RAG."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)