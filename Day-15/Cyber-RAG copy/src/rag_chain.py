"""
RAG Chain module for Cyber RAG.

This module builds the complete RAG chain combining retriever,
prompt template, and LLM.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from utils import get_logger


logger = get_logger()


# System prompt for the RAG chain
SYSTEM_PROMPT = """You are a helpful Cyber Security expert assistant. 
Use the provided document context to answer questions about cybersecurity.

IMPORTANT RULES:
1. Answer ONLY using information from the provided document context.
2. If the answer is not found in the documents, respond with: "I couldn't find that information in the uploaded document."
3. Be specific and cite relevant sections when applicable.
4. Provide practical, actionable advice based on the documents.
5. If a question is outside cybersecurity domain, politely redirect to cybersecurity topics.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_chain(retriever, model_name: str = "llama3", temperature: float = 0.3):
    """
    Build a complete RAG chain combining retriever, prompt, and LLM.

    Args:
        retriever: LangChain retriever object
        model_name (str): Ollama model name. Defaults to "llama3"
        temperature (float): LLM temperature. Defaults to 0.3 (more deterministic)

    Returns:
        RetrievalQA: Complete RAG chain

    Raises:
        ConnectionError: If Ollama is not running
        ValueError: If retriever is invalid
    """
    if retriever is None:
        logger.error("Retriever is None")
        raise ValueError("Retriever cannot be None")

    try:
        logger.info(f"Initializing Ollama with model: {model_name}")
        llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434"
        )

        # Test LLM connectivity
        try:
            logger.info("Testing Ollama connection...")
            test_response = llm.invoke("test")
            logger.info("Ollama connection successful")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            raise ConnectionError(
                "Ollama is not running. Please start Ollama with: ollama serve"
            )

        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=SYSTEM_PROMPT
        )

        logger.info("Building RAG chain...")
        # Create RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            },
            return_source_documents=True
        )

        logger.info("RAG chain built successfully")
        return chain

    except ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error building RAG chain: {str(e)}")
        raise


def run_query(chain, query: str) -> dict:
    """
    Run a query through the RAG chain.

    Args:
        chain: The RAG chain
        query (str): User query

    Returns:
        dict: Contains 'answer' and 'sources' keys
    """
    try:
        logger.info(f"Running query: {query}")
        result = chain.invoke({"query": query})
        logger.info("Query executed successfully")

        # Extract answer and sources
        answer = result.get("result", "No answer generated")
        source_documents = result.get("source_documents", [])

        # Format sources
        sources = []
        for doc in source_documents:
            source_info = {
                "page": doc.metadata.get("page", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "preview": doc.page_content[:100] + "..."
            }
            sources.append(source_info)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error running query: {str(e)}")
        raise
