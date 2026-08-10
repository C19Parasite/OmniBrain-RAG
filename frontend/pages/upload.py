import streamlit as st 
from utils.api import upload_file
st.title("Upload Document")
st.write("Upload a PDF document to use with OmniBrain RAG.")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    st.write("Selected file:", uploaded_file.name)

    if st.button("Upload"):
        result = upload_file(uploaded_file)

        if result["success"]:
            st.success(result["message"])
        else:
            st.error("Upload failed")
    
