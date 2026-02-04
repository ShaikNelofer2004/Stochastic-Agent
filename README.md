# Enterprise AI Agent Prototype

A lightweight, enterprise-ready AI agent for Document Q&A, built with Python, Streamlit, and Google Gemini.

## Features

- **📄 Document Ingestion**: Handles multiple PDF documents with structure preservation (Tables, Headers).
- **🧠 Advanced RAG**: Uses custom NumPy-based Vector Store for lightweight, local retrieval.
- **🤖 Multi-modal LLM**: Powered by Google Gemini 1.5 Flash.
- **🔍 Arxiv Integration**: Agent can autonomously search for research papers.
- **🏢 Enterprise-Ready**: Clean UI, source citations, and modular architecture.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd Stochastic
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment**
   - Create a `.env` file in the root directory.
   - Add your Google API Key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload**: Use the sidebar to upload PDF documents. Click "Ingest Documents".
2. **Chat**: Ask questions about the documents.
   - Example directly from docs: *"What is the methodology?"*
   - Example from Arxiv: *"Find papers about Agentic AI"* (The agent will switch tools automatically).

## Architecture

- **Frontend**: Streamlit
- **Vector Store**: Custom `SimpleVectorStore` (NumPy)
- **Embedding**: Gemini Text Embedding 004
- **LLM**: Gemini 1.5 Flash
- **PDF Engine**: PyMuPDF4LLM
