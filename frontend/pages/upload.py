import streamlit as st 
st.title("Upload Document")
st.write("Upload a PDF document to use with OmniBrain RAG.")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    # You can add additional processing for the uploaded file here
    st.write("**File name:**", uploaded_file.name )
    st.write("**File size:**", uploaded_file.size, "bytes" )
    
