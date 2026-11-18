"""Configuration and environment management for Medical RAG Assistant."""

import logging
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_provider: str = Field(default="groq", description="LLM provider: groq, ollama, etc.")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    groq_model: str = Field(default="openai/gpt-oss-20b", description="Groq model identifier")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama endpoint")
    ollama_model: str = Field(default="llama3.2", description="Ollama model name")
    temperature: float = Field(default=0.0, description="Sampling temperature for grounded answers")

    # Embeddings & Vector DB Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local HuggingFace embedding model name",
    )
    embedding_device: str = Field(default="cpu", description="Embedding computation device: cpu or cuda")
    chroma_persist_directory: str = Field(
        default=str(BASE_DIR / "data" / "chroma"),
        description="Local directory for Chroma vector store",
    )
    chroma_collection_name: str = Field(default="medical_docs", description="Collection name in Chroma")

    # Reranker Settings
    reranker_model: str = Field(
        default="ms-marco-TinyBERT-L-2-v2",
        description="FlashRank cross-encoder reranker model",
    )
    rerank_top_k: int = Field(default=4, description="Number of reranked candidates fed to generator")

    # Retrieval Settings
    retrieval_top_k: int = Field(default=10, description="Initial candidate count per retriever")
    bm25_weight: float = Field(default=0.4, description="BM25 weight in reciprocal rank fusion")
    dense_weight: float = Field(default=0.6, description="Dense vector weight in reciprocal rank fusion")

    # Chunking Parameters
    chunk_size: int = Field(default=800, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=150, description="Overlap between consecutive chunks")

    # Observability & LangSmith
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="medical-rag", description="LangSmith project name")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )

    def setup_langsmith_tracing(self) -> None:
        """Export LangSmith environment variables if configured."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            logging.info("LangSmith tracing enabled for project '%s'", self.langchain_project)
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"


# Global singleton settings instance
settings = Settings()


def setup_logger(name: str = "medical_rag") -> logging.Logger:
    """Setup structured logger with consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
