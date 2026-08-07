import streamlit as st

st.set_page_config(
    page_title="OmniBrain RAG",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 OmniBrain RAG")

st.write("Welcome to OmniBrain!")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success(f"{uploaded_file.name} uploaded successfully!")