import streamlit as st
from utils.api import upload_file

st.title("📄 Upload Document")
st.subheader("Upload a PDF to start chatting with your document.")

st.info(
    "Upload a PDF document. After uploading, you can ask questions "
    "about its contents in the Chat section."
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file:
    st.success(f"Selected: {uploaded_file.name}")

    if st.button("📤 Upload Document"):

        with st.spinner("Uploading document..."):
            result = upload_file(uploaded_file)

        if result["success"]:
            st.success(result["message"])
        else:
            st.error(result["message"])