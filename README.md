# Enterprise AI Agent Prototype

A lightweight, enterprise-ready AI agent for Document Q&A, built with Python, Streamlit, and Google Gemini.

## Features

### � Enterprise Performance
- **Custom Vector Engine**: Built on `NumPy` for lightning-fast cosine similarity (no heavyweight bloat).
- **Multi-Turn Memory**: understands context in follow-up questions (e.g., "Summarize this paper" -> "Who wrote **it**?").
- **Real-Time Latency Metrics**: Displays processing time for every interaction to ensure SLA compliance.
- **Precise Citations**: Cites not just the document name, but the **exact page number** where information was found to establish trust.

### 🧠 Core Capabilities
- **📄 Document Ingestion**: Handles multiple PDF documents with structure preservation (Tables, Headers).
- **🤖 Multi-modal LLM**: Powered by **Google Gemini 3 Flash** (or recent Gemini variants).
- **🔍 Arxiv Integration**: Agent autonomously switches tools to search for real-world research papers when asked.

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
   - Example directly from docs: *"What is the methodology?"* -> *"According to Paper X (Page 4)..."*
   - Example from Arxiv: *"Find papers about Agentic AI"* (The agent will switch tools automatically).

## Architecture

- **Frontend**: Streamlit (Enterprise-styled)
- **Vector Store**: Custom `SimpleVectorStore` (NumPy)
- **Embedding**: Gemini Text Embedding 004
- **LLM**: Gemini 1.5/3 Flash
- **PDF Engine**: PyMuPDF4LLM (Structure-aware)

