"""
Retriever module for Cyber RAG.

This module handles loading the persisted vector store and providing
retrieval functionality.
"""

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_logger


logger = get_logger()


def get_retriever(persist_directory: str = "chroma_db", k: int = 4):
    """
    Load persisted Chroma vector store and return retriever.

    Args:
        persist_directory (str): Directory where vector store is persisted
        k (int): Number of documents to retrieve. Defaults to 4

    Returns:
        Retriever: LangChain retriever object

    Raises:
        FileNotFoundError: If persist directory does not exist
        ValueError: If vector store cannot be loaded
    """
    try:
        # Check if persist directory exists
        persist_path = Path(persist_directory)
        if not persist_path.exists():
            logger.error(f"Vector store directory not found: {persist_directory}")
            raise FileNotFoundError(
                f"Vector store not found at {persist_directory}. "
                f"Please upload and process a PDF first."
            )

        logger.info(f"Loading embeddings model: all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info(f"Loading Chroma vector store from: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

        # Create retriever with similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        logger.info(f"Retriever created with k={k}")
        return retriever

    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading retriever: {str(e)}")
        raise ValueError(f"Failed to load vector store: {str(e)}")


def verify_vectorstore_exists(persist_directory: str = "chroma_db") -> bool:
    """
    Check if a persisted vector store exists.

    Args:
        persist_directory (str): Directory to check

    Returns:
        bool: True if vector store exists, False otherwise
    """
    persist_path = Path(persist_directory)
    return persist_path.exists() and list(persist_path.glob("*")) != []
