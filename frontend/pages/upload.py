import streamlit as st
from utils.api import upload_file


st.title("Upload Document")
st.subheader("Upload a PDF to chat with your document")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file:
    st.write("Selected file:", uploaded_file.name)
if st.button("Upload"):

    progress = st.progress(0)
    status = st.empty()

    status.write("Uploading document...")
    progress.progress(30)

    result = upload_file(uploaded_file)

    progress.progress(100)

    if result["success"]:
        status.empty()
        st.success(result["message"])
    else:
        status.empty()
        st.error(result["message"])