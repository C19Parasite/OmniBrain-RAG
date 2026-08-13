import streamlit as st
from utils.api import query_api


st.title("Chat")

question = st.chat_input("Ask a question")

if question:
    result = query_api(question)

    if result["success"]:
        st.write(result["answer"])
    else:
        st.error(result["answer"])