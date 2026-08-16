import streamlit as st

st.set_page_config(
    page_title="OmniBrain RAG",
    page_icon="🧠",
    layout="wide"
)
with st.sidebar:
    st.title("🧠 OmniBrain")

    st.divider()

    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/upload.py", label="📄 Upload")
    st.page_link("pages/chat.py", label="💬 Chat")

    st.divider()

    st.caption("AI Document Assistant")


st.title("🧠 OmniBrain RAG")
st.subheader("AI Document Assistant")

st.write(
    "Upload your documents and ask questions using AI."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📄 Upload Documents")
    st.write("Upload PDF documents to OmniBrain.")

with col2:
    st.info("💬 Ask Questions")
    st.write("Ask questions about your documents.")

with col3:
    st.info("⚡ Get Answers")
    st.write("Get answers based on your documents.")