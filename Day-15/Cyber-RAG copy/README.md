# 🛡️ Cyber Security RAG Assistant

A production-grade **Retrieval Augmented Generation (RAG)** application for cybersecurity Q&A. Upload cybersecurity PDFs and get AI-powered answers using advanced document retrieval and language models.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Ollama Setup](#-ollama-setup)
- [Running the Application](#-running-the-application)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)

---

## ✨ Features

### Core Features
- ✅ **PDF Upload & Processing** - Upload cybersecurity PDFs with validation
- ✅ **Intelligent Retrieval** - ChromaDB with semantic search using Sentence Transformers
- ✅ **LLM Integration** - Ollama with llama3 model for accurate answers
- ✅ **Context-Aware Answers** - RAG chain that answers only from document context
- ✅ **Source Citation** - Displays retrieved document chunks with page numbers
- ✅ **Chat History** - Maintains conversation history in session

### Advanced Features
- 📊 **Vector Persistence** - Persisted ChromaDB for continued sessions
- 🔍 **Similarity Search** - Retrieves top-4 most relevant chunks (k=4)
- 📝 **Logging System** - Complete logging for debugging and monitoring
- 🛡️ **Error Handling** - Comprehensive error handling with user-friendly messages
- 🚀 **Production-Ready** - Type hints, docstrings, modular architecture

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | Streamlit | 1.32.0 |
| **LLM Framework** | LangChain | 0.1.14 |
| **Vector Database** | ChromaDB | 0.4.24 |
| **Embeddings** | Sentence Transformers | 2.2.2 |
| **LLM** | Ollama (llama3) | - |
| **PDF Processing** | PyPDF | 4.0.1 |
| **Language** | Python | 3.12 |

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                         │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ PDF Upload     │  │ Query Input  │  │ Chat History  │   │
│  └────────────────┘  └──────────────┘  └───────────────┘   │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG Pipeline                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐    │
│  │ PDF Ingestion│  │ Chunking     │  │ Embeddings    │    │
│  │ (PyPDF)      │  │ (1000 tokens)│  │ (MiniLM)      │    │
│  └──────────────┘  └──────────────┘  └───────────────┘    │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Vector Database (ChromaDB)                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Persisted Vector Store (chroma_db/)                    │ │
│  │ - Document embeddings                                  │ │
│  │ - Metadata (page, source)                              │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│            Retrieval & LLM Chain                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │
│  │ Retriever  │  │ Prompt     │  │ Ollama LLM         │   │
│  │ (k=4)      │  │ Template   │  │ (llama3)           │   │
│  └────────────┘  └────────────┘  └────────────────────┘   │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Answer with Sources                            │
│  ┌───────────────────┐  ┌──────────────────────────────┐   │
│  │ Generated Answer  │  │ Retrieved Documents (Sources)│   │
│  │ (Context-aware)   │  │ - Page numbers               │   │
│  └───────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

### Prerequisites
- Python 3.12+
- Ollama installed and running
- macOS, Linux, or Windows

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Cyber-RAG.git
cd Cyber-RAG
```

### Step 2: Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import langchain; import chromadb; print('✅ All packages installed successfully')"
```

---

## 🦙 Ollama Setup

### Install Ollama

1. Download from [ollama.ai](https://ollama.ai)
2. Install for your operating system

### Run Ollama

```bash
# Start Ollama (keep running in background)
ollama serve
```

### Pull llama3 Model

In a new terminal:

```bash
ollama pull llama3
```

### Verify Ollama is Running

```bash
curl http://localhost:11434/api/tags
```

You should see a list of available models including `llama3`.

---

## 🚀 Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Usage

1. **Upload PDF**: Click the file uploader in the sidebar
2. **Process Document**: Click "Process Document" button
3. **Ask Questions**: Type your question in the text box
4. **View Answers**: See the AI response with sources

---

## 📁 Project Structure

```
Cyber-RAG/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
│
├── src/                  # Source code module
│   ├── __init__.py       # Package initialization
│   ├── utils.py          # Logging and utilities
│   ├── ingest.py         # PDF loading and chunking
│   ├── retriever.py      # Vector store retrieval
│   └── rag_chain.py      # RAG chain implementation
│
├── chroma_db/            # Persisted vector database
├── data/                 # Storage for uploaded files
└── logs/                 # Application logs
```

---

## 📖 Usage Guide

### Uploading a PDF

1. Click "📤 Upload Document" in the sidebar
2. Select a cybersecurity PDF file
3. Click "🔄 Process Document"
4. Wait for the vectorization to complete

### Asking Questions

```
Good questions:
✅ "What are the best practices for password security?"
✅ "How can I prevent SQL injection attacks?"
✅ "What is XSS and how to prevent it?"

Avoid:
❌ Generic questions not related to the PDF content
❌ Questions requiring real-time external information
```

### Understanding the Response

- **Answer**: AI-generated response based on document context
- **Sources**: Original document chunks with:
  - Page number
  - File name
  - Text preview

---

## 🔧 API Reference

### `src/ingest.py`

#### `load_and_split_pdf(pdf_path: str) -> List`
Load and split PDF into chunks.

```python
from src.ingest import load_and_split_pdf
chunks = load_and_split_pdf("document.pdf")
```

#### `create_vectorstore(chunks: List, persist_directory: str) -> Chroma`
Create persisted vector store from chunks.

```python
from src.ingest import create_vectorstore
vectorstore = create_vectorstore(chunks, "chroma_db")
```

### `src/retriever.py`

#### `get_retriever(persist_directory: str, k: int) -> Retriever`
Load retriever from persisted vector store.

```python
from src.retriever import get_retriever
retriever = get_retriever(k=4)
```

### `src/rag_chain.py`

#### `build_rag_chain(retriever, model_name: str) -> RetrievalQA`
Build complete RAG chain.

```python
from src.rag_chain import build_rag_chain
chain = build_rag_chain(retriever, model_name="llama3")
```

#### `run_query(chain, query: str) -> dict`
Run query through RAG chain.

```python
from src.rag_chain import run_query
result = run_query(chain, "What is cybersecurity?")
# Returns: {"answer": "...", "sources": [...]}
```

---

## 🐛 Troubleshooting

### Issue: "Ollama is not running"

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model if needed
ollama pull llama3
```

### Issue: "Vector store not found"

**Solution:**
1. Upload and process a PDF first
2. Or ensure `chroma_db/` directory exists with data

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Issue: "Streamlit not opening"

**Solution:**
```bash
# Try specifying the host
streamlit run app.py --server.address=localhost
```

### Issue: "Empty PDF error"

**Solution:**
- Ensure PDF has text content (not image-only)
- Try a different PDF
- Check file integrity

---

## 🚀 Future Improvements

### Planned Features
- [ ] Multi-PDF support (process multiple documents simultaneously)
- [ ] Advanced search filters (date, author, category)
- [ ] Export conversation history as PDF/JSON
- [ ] Custom model selection in UI
- [ ] Fine-tuned embeddings for cybersecurity
- [ ] Question suggestions based on content
- [ ] RAG evaluation metrics
- [ ] User authentication
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] GPU acceleration support
- [ ] Response quality scoring

### Optimization Ideas
- Implement caching for frequently asked questions
- Add batch processing for multiple documents
- Optimize chunk size based on document type
- Implement hybrid search (semantic + keyword)
- Add reranking for better relevance

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

## ✅ Verification Checklist

Before running:

- [x] Python 3.12 installed
- [x] Virtual environment created
- [x] Dependencies installed from requirements.txt
- [x] Ollama installed and running
- [x] llama3 model pulled in Ollama
- [x] ChromaDB persistence configured
- [x] All source modules have type hints and docstrings
- [x] Error handling implemented
- [x] Logging configured
- [x] UI components responsive

---

**Built with ❤️ for Cybersecurity Professionals**