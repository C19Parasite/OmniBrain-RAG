import streamlit as st
from utils.api import query_api

st.title("🧠 OmniBrain RAG")
st.subheader("Chat with your documents")

# 1. Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 3. Chat input
question = st.chat_input("Ask a question about your document...")

# 4. Process question
if question:

    # Save user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Display user's question
    with st.chat_message("user"):
        st.write(question)

    # Send question to backend
    result = query_api(question)

    # Get answer
    if result["success"]:
        answer = result["answer"]
    else:
        answer = result["answer"]

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)