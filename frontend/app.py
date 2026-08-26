import streamlit as st

# Your existing sidebar/navigation code should remain here

st.title("🧠 OmniBrain RAG")
st.subheader("Chat with your documents")

st.write(
    "Upload your documents and ask questions about their content "
    "using our AI document assistant."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Documents")
    st.write(
        "Upload a PDF document and prepare it for question answering."
    )

with col2:
    st.subheader("💬 Ask Questions")

    st.write(  "Ask questions about your uploaded document and view the answers." )

st.subheader("How it works")

st.write("1. 📄 Upload your PDF")
st.write("2. 🔍 Process your document")
st.write("3. 💬 Ask questions")
st.write("4. 🧠 Get answers")