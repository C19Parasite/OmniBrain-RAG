import streamlit as st
from utils.api import query_api

# -------------------------------------------------
# Chat page
# -------------------------------------------------
st.title("🧠 OmniBrain RAG")
st.subheader("Chat with your documents")

st.write(
    "Ask questions about your uploaded document "
    "and get AI-powered answers."
)

st.divider()

# -------------------------------------------------
# Create chat history
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Display previous messages
# -------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Display source/citation when available
        if message.get("source"):
            st.caption(
                f"📄 Source: {message['source']}"
            )

# -------------------------------------------------
# Empty chat state
# -------------------------------------------------
if not st.session_state.messages:

    st.info(
        "💬 Start a conversation by asking a question "
        "about your uploaded document."
    )

    st.write("You can try asking:")

    st.write("• What is this document about?")
    st.write("• Summarize the main points.")
    st.write("• What are the important topics?")

# -------------------------------------------------
# Chat input
# -------------------------------------------------
question = st.chat_input(
    "Ask a question about your document..."
)

# -------------------------------------------------
# Process question
# -------------------------------------------------
if question and question.strip():

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

    # -------------------------------------------------
    # Call API with loading state
    # -------------------------------------------------
    with st.spinner("⏳ Thinking..."):
        result = query_api(question)

    # -------------------------------------------------
    # Successful response
    # -------------------------------------------------
    if result["success"]:

        answer = result["answer"]

        source = result.get(
            "source",
            "Uploaded document"
        )

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": source
            }
        )

        # Display assistant response
        with st.chat_message("assistant"):

            st.write(answer)

            st.caption(
                f"📄 Source: {source}"
            )

    # -------------------------------------------------
    # API error
    # -------------------------------------------------
    else:

        st.error(
            f"❌ {result['answer']}"
        )