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

# No file selected
if uploaded_file is None:
    st.warning("⚠️ Please select a PDF document to continue.")

else:
    # File selected
    st.success(f"📄 Selected: {uploaded_file.name}")

    if st.button("📤 Upload Document", use_container_width=True):

        # Show loading message while uploading
        with st.spinner("⏳ Uploading document..."):
            result = upload_file(uploaded_file)

        # Upload successful
        if result["success"]:
            st.success(f"✅ {result['message']}")

            st.info(
                "💬 Your document is ready. "
                "Go to the Chat section to ask questions."
            )

        # Upload failed
        else:
            st.error(f"❌ {result['message']}")