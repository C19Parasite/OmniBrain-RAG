import streamlit as st
from utils.api import query_api

st.title("🧠 OmniBrain RAG")
st.subheader("Chat with your documents")

# Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Ask a question about your document...")

if question:

    # Save user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user's question
    with st.chat_message("user"):
        st.write(question)

    # Send question to API
    result = query_api(question)

    if result["success"]:
        answer = result["answer"]

        # Save assistant answer
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Display assistant answer
        with st.chat_message("assistant"):
            st.write(answer)

    else:
        st.error(result["answer"])