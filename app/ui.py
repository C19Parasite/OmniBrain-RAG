import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OmniBrain RAG Explorer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 OmniBrain RAG Explorer")
st.markdown("Upload documents and query your multi-modal retrieval pipeline effortlessly.")

# Sidebar for document uploads
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF, TXT, MD, or image file", 
    type=["pdf", "txt", "md", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    if st.sidebar.button("Upload & Ingest"):
        with st.spinner("Processing and chunking document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    res_data = response.json()
                    st.sidebar.success(f"Successfully ingested {res_data['chunks_ingested']} chunks!")
                else:
                    st.sidebar.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")

# Main chat/query interface
st.header("💬 Ask OmniBrain")
query = st.text_input("Enter your query about the ingested documents:")

if st.button("Generate Answer"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                payload = {"query": query, "top_k": 4}
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Answer")
                    st.write(data.get("context", "No context returned."))
                    
                    with st.expander("📚 View Retrieved Sources"):
                        sources = data.get("sources", [])
                        if sources:
                            for idx, src in enumerate(sources):
                                st.markdown(f"**Source {idx + 1}:**")
                                st.json(src)
                        else:
                            st.info("No sources cited.")
                else:
                    st.error(f"Query error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {str(e)}")
