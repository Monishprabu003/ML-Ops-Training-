"""
PDF ingestion pipeline for Cyber RAG.

This module handles loading, processing, and splitting PDF documents
into chunks for vectorization.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_logger, validate_pdf_file


logger = get_logger()


def load_and_split_pdf(pdf_path: str) -> List:
    """
    Load PDF file and split it into chunks.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List: List of document chunks

    Raises:
        ValueError: If PDF file is invalid or corrupted
        FileNotFoundError: If PDF file not found
    """
    if not validate_pdf_file(pdf_path):
        logger.error(f"Invalid PDF file: {pdf_path}")
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")

    try:
        logger.info(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found in PDF: {pdf_path}")
            raise ValueError(f"PDF appears to be empty: {pdf_path}")

        logger.info(f"Loaded {len(documents)} pages from PDF")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        return chunks

    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise


def create_vectorstore(chunks: List, persist_directory: str = "chroma_db") -> Chroma:
    """
    Create and persist a Chroma vector store from document chunks.

    Args:
        chunks (List): List of document chunks
        persist_directory (str): Directory to persist the vector store

    Returns:
        Chroma: Vector store object

    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        logger.error("No chunks provided to create vector store")
        raise ValueError("Cannot create vector store with empty chunks list")

    try:
        logger.info(f"Creating embeddings using all-MiniLM-L6-v2 model")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create or update vector store
        logger.info(f"Creating Chroma vector store at: {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        logger.info(f"Vector store created successfully with {len(chunks)} documents")
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise