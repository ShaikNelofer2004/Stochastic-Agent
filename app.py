import streamlit as st
import os
import tempfile
from src.ingest_data import process_pdfs
from src.agent import DocumentAgent

st.set_page_config(page_title="Enterprise AI Agent", layout="wide")

# Custom CSS for "Enterprise" feel
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stChatInput {
        border-radius: 10px;
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Document Q&A AI Agent")
st.caption("Enterprise-grade RAG with Multi-modal LLM Support")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = DocumentAgent()

# Sidebar - File Upload
with st.sidebar:
    st.header("📂 Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload research papers or agreements."
    )
    
    if uploaded_files and st.button("Ingest Documents"):
        with st.spinner("Processing documents... (Parsing & Embedding)"):
            # Save to temp files
            temp_paths = []
            temp_dir = tempfile.mkdtemp()
            
            for uploaded_file in uploaded_files:
                path = os.path.join(temp_dir, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_paths.append(path)
            
            # Process
            try:
                process_pdfs(temp_paths)
                # Reload agent's store
                st.session_state.agent = DocumentAgent() 
                st.success(f"Successfully ingested {len(uploaded_files)} documents!")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    st.markdown("---")
    st.markdown("### 🛠 Features")
    st.markdown("- **PDF Parsing**: Structure-aware extraction")
    st.markdown("- **Vector Store**: Local NumPy-based Engine")
    st.markdown("- **Model**: Google Gemini 1.5 Flash")
    
    # Check API Key
    if not os.getenv("GOOGLE_API_KEY"):
         st.warning("⚠️ GOOGLE_API_KEY not found in environment!")
         api_key_input = st.text_input("Enter Google API Key", type="password")
         if api_key_input:
             os.environ["GOOGLE_API_KEY"] = api_key_input
             st.success("API Key set for this session!")


# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the documents..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.ask(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
