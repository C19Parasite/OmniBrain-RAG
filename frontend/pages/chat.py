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

        # Display citation if available
        if message.get("source"):
            st.caption(f"📄 Source: {message['source']}")

# Show helpful message when chat is empty
if not st.session_state.messages:
    st.info(
        "💬 Start a conversation by asking a question "
        "about your uploaded document."
    )

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

    # Call API
    with st.spinner("Thinking..."):
        result = query_api(question)

    if result["success"]:

        answer = result["answer"]

        # Get source if API provides one
        source = result.get("source", "Uploaded document")

        # Save assistant answer and citation
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": source
            }
        )

        # Display assistant answer
        with st.chat_message("assistant"):
            st.write(answer)

            # Citation UI
            st.caption(f"📄 Source: {source}")

    else:
        st.error(result["answer"])