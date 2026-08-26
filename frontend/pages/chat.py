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

if not st.session_state.messages:
    st.info(
        "💬 Start a conversation by asking a question "
        "about your uploaded document."
    )
    if not st.session_state.messages:
    st.write("You can try asking:")
    st.write("• What is this document about?")
    st.write("• Summarize the main points.")
    st.write("• What are the important topics?")
    
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
    with st.spinner("Thinking..."):
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