"""
Cyber Security RAG Assistant - Streamlit Application

A production-grade web application for cybersecurity Q&A using 
Retrieval Augmented Generation.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Cyber Security RAG Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after Streamlit config
from src.utils import setup_logging, validate_query, get_logger
from src.ingest import load_and_split_pdf, create_vectorstore
from src.retriever import get_retriever, verify_vectorstore_exists
from src.rag_chain import build_rag_chain, run_query


# Setup logging
logger = setup_logging()

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E8F4F8;
        border-left: 4px solid #0088CC;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F8E8;
        border-left: 4px solid #00BB33;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #F8E8E8;
        border-left: 4px solid #BB0000;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


def handle_pdf_upload(uploaded_file):
    """
    Handle PDF file upload and processing.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with st.spinner("📥 Processing PDF..."):
            # Save uploaded file temporarily
            temp_dir = Path(tempfile.gettempdir()) / "cyber_rag"
            temp_dir.mkdir(exist_ok=True)

            temp_pdf_path = temp_dir / uploaded_file.name
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            logger.info(f"PDF uploaded: {uploaded_file.name}")

            # Load and split PDF
            chunks = load_and_split_pdf(str(temp_pdf_path))

            if not chunks:
                st.error("❌ PDF appears to be empty. Please upload a valid PDF.")
                logger.error("PDF is empty")
                return False

            st.info(f"✅ Loaded {len(chunks)} text chunks from PDF")

            # Create vector store
            with st.spinner("� Creating vector embeddings..."):
                vectorstore = create_vectorstore(chunks, persist_directory="chroma_db")
                st.session_state.vectorstore_ready = True
                st.session_state.uploaded_file_name = uploaded_file.name
                logger.info("Vector store created successfully")

            # Load retriever
            st.session_state.retriever = get_retriever()
            logger.info("Retriever loaded successfully")

            # Build RAG chain
            with st.spinner("🔗 Building RAG chain..."):
                st.session_state.rag_chain = build_rag_chain(st.session_state.retriever)
                logger.info("RAG chain built successfully")

            st.success(
                f"✅ Successfully processed **{uploaded_file.name}** \n"
                f"Ready to answer questions!"
            )
            return True

    except ValueError as e:
        st.error(f"❌ Invalid PDF: {str(e)}")
        logger.error(f"Invalid PDF: {str(e)}")
        return False
    except ConnectionError as e:
        st.error(
            f"❌ Cannot connect to Ollama\n\n"
            f"Please ensure Ollama is running:\n"
            f"```bash\nollama serve\n```"
        )
        logger.error(f"Ollama connection error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}")
        return False


def handle_query(query_text):
    """
    Handle user query and generate response.

    Args:
        query_text (str): User's question

    Returns:
        dict or None: Response with answer and sources, or None if error
    """
    if not validate_query(query_text):
        st.warning("⚠️ Please enter a valid question.")
        return None

    if not st.session_state.vectorstore_ready:
        st.error("❌ Please upload and process a PDF first.")
        return None

    try:
        with st.spinner("🔍 Searching documents and generating answer..."):
            result = run_query(st.session_state.rag_chain, query_text)
            logger.info(f"Query processed successfully: {query_text}")
            return result

    except ConnectionError:
        st.error(
            "❌ Cannot connect to Ollama\n\n"
            "Please ensure Ollama is running:\n"
            "```bash\nollama serve\n```"
        )
        logger.error("Ollama connection error during query")
        return None
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        logger.error(f"Query error: {str(e)}")
        return None


def display_chat_history():
    """Display chat history in the sidebar."""
    if st.session_state.chat_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Chat History")

        for i, item in enumerate(st.session_state.chat_history[-5:], 1):  # Show last 5
            with st.sidebar.expander(f"Q {i}: {item['question'][:50]}..."):
                st.write(f"**Q:** {item['question']}")
                st.write(f"**A:** {item['answer'][:200]}...")


# Main Application
def main():
    """Main Streamlit application."""
    initialize_session_state()

    # Header
    st.markdown(
        "<h1 style='text-align: center; color: #FF6B6B;'>🛡️ Cyber Security RAG Assistant</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Upload cybersecurity PDFs and ask questions using AI-powered retrieval</p>",
        unsafe_allow_html=True
    )

    # Sidebar - File Upload
    with st.sidebar:
        st.subheader("📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Select a cybersecurity PDF",
            type=["pdf"],
            help="Upload a PDF document for analysis"
        )

        if uploaded_file is not None:
            if st.button("🔄 Process Document", use_container_width=True):
                handle_pdf_upload(uploaded_file)

        # Display status
        st.markdown("---")
        st.subheader("📊 Status")

        if st.session_state.uploaded_file_name:
            st.success(f"✅ Loaded: {st.session_state.uploaded_file_name}")
        else:
            st.info("⏳ No document loaded yet")

        if st.session_state.vectorstore_ready:
            st.success("✅ Vector store ready")
        else:
            st.warning("⚠️ Vector store not ready")

        display_chat_history()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Ask a Question")

        # Check for persisted vectorstore at startup
        if not st.session_state.vectorstore_ready and verify_vectorstore_exists():
            try:
                with st.spinner("Loading persisted vector store..."):
                    st.session_state.retriever = get_retriever()
                    st.session_state.rag_chain = build_rag_chain(
                        st.session_state.retriever
                    )
                    st.session_state.vectorstore_ready = True
                    st.session_state.uploaded_file_name = "Previous Session"
            except Exception as e:
                logger.error(f"Error loading persisted vectorstore: {str(e)}")

        # Query input
        question = st.text_input(
            "Enter your cybersecurity question:",
            placeholder="e.g., What are the best practices for password security?",
            help="Ask any question about the uploaded documents"
        )

        col_ask, col_clear = st.columns([3, 1])

        with col_ask:
            if st.button("🔍 Ask", use_container_width=True):
                if question:
                    result = handle_query(question)

                    if result:
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"]
                        })

                        # Display answer
                        st.markdown("---")
                        st.subheader("📝 Answer")
                        st.write(result["answer"])

                        # Display sources
                        if result["sources"]:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result["sources"], 1):
                                with st.expander(
                                    f"Source {i} - Page {source['page']}"
                                ):
                                    st.write(f"**File:** {source['source']}")
                                    st.write(f"**Preview:** {source['preview']}")

        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Right column - Information
    with col2:
        st.markdown("### ℹ️ Quick Start")
        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** a cybersecurity PDF
            2. **Process** the document
            3. **Ask** any question
            4. **Review** answers and sources

            ### Tips
            - Be specific in your questions
            - Check source documents
            - Ask follow-up questions
            """)

        st.markdown("### 🛠️ System Info")
        st.markdown("""
        - **LLM:** Ollama (llama3)
        - **Embeddings:** all-MiniLM-L6-v2
        - **Database:** ChromaDB
        - **Chunks:** 1000 tokens
        """)


if __name__ == "__main__":
    main()