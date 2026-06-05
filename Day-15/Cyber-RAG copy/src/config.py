"""
Configuration and constants for Cyber RAG application.

This module contains all configuration settings and constants used
throughout the application.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3"
LLM_BASE_URL = "http://localhost:11434"

# RAG configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4  # Number of documents to retrieve
LLM_TEMPERATURE = 0.3  # Lower = more deterministic

# Validation
MAX_PDF_SIZE_MB = 100  # Maximum PDF file size
ALLOWED_EXTENSIONS = {".pdf"}

# UI Configuration
STREAMLIT_PAGE_TITLE = "Cyber Security RAG Assistant"
STREAMLIT_PAGE_ICON = "🛡️"

# Logging
LOG_FILE = LOGS_DIR / "rag_app.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Error messages
ERROR_MESSAGES = {
    "empty_query": "Please enter a valid question.",
    "invalid_pdf": "Invalid or corrupted PDF file.",
    "empty_pdf": "PDF appears to be empty.",
    "no_vectorstore": "Vector store not found. Please upload a PDF first.",
    "ollama_not_running": "Ollama is not running. Please start it with: ollama serve",
    "query_error": "Error processing your question.",
    "upload_error": "Error processing the PDF file.",
}

# Success messages
SUCCESS_MESSAGES = {
    "pdf_processed": "✅ Successfully processed PDF",
    "ready": "Ready to answer questions!",
    "vectorstore_created": "Vector store created successfully",
}
