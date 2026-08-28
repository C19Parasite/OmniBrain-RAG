import streamlit as st

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="OmniBrain RAG",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
with st.sidebar:
    st.title("🧠 OmniBrain")
    st.write("AI Document Assistant")

    st.divider()

    st.page_link(
        "app.py",
        label="🏠 Home"
    )

    st.page_link(
        "pages/upload.py",
        label="📄 Upload"
    )

    st.page_link(
        "pages/chat.py",
        label="💬 Chat"
    )

    st.divider()

    st.caption("Upload a PDF and ask questions about it.")


# -------------------------------------------------
# Home page
# -------------------------------------------------
st.title("🧠 OmniBrain RAG")

st.subheader("Chat with your documents")

st.write(
    "Upload your documents and ask questions about their content "
    "using our AI document assistant."
)

st.divider()


# -------------------------------------------------
# Main actions
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Documents")

    st.write(
        "Upload a PDF document and prepare it for "
        "question answering."
    )

    if st.button("📄 Go to Upload", use_container_width=True):
        st.switch_page("pages/upload.py")


with col2:
    st.subheader("💬 Ask Questions")

    st.write(
        "Ask questions about your uploaded document "
        "and view the answers."
    )

    if st.button("💬 Go to Chat", use_container_width=True):
        st.switch_page("pages/chat.py")


st.divider()


# -------------------------------------------------
# How it works
# -------------------------------------------------
st.subheader("🔍 How it works")

steps = [
    "📄 Upload your PDF",
    "🔍 Process your document",
    "💬 Ask questions",
    "🧠 Get answers"
]

for step in steps:
    st.write(step)


# -------------------------------------------------
# Helpful information
# -------------------------------------------------
st.divider()

st.subheader("💡 Getting started")

st.write(
    "1. Go to the Upload page and select your PDF."
)

st.write(
    "2. Upload the document successfully."
)

st.write(
    "3. Open the Chat page."
)

st.write(
    "4. Ask a question about your document."
)

st.info(
    "💬 Tip: Ask specific questions about the content "
    "of your uploaded document for better answers."
)