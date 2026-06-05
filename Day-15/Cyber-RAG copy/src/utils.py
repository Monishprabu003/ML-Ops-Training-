"""
Utility functions for Cyber RAG application.

This module provides logging setup, validation, and helper functions
for the RAG pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(log_file: str = "logs/rag_app.log") -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_file (str): Path to log file. Defaults to "logs/rag_app.log"

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cyber_rag")
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if a file is a valid PDF.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        bool: True if valid PDF, False otherwise
    """
    if not os.path.exists(file_path):
        return False

    if not file_path.lower().endswith('.pdf'):
        return False

    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False


def validate_query(query: str) -> bool:
    """
    Validate if query is not empty.

    Args:
        query (str): Query string

    Returns:
        bool: True if query is valid, False otherwise
    """
    return bool(query and query.strip())


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger("cyber_rag")
