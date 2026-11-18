"""Embeddings factory module for local and swappable embeddings."""

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings, setup_logger

logger = setup_logger("indexing.embeddings")


def get_embedding_model(
    model_name: str = settings.embedding_model,
    device: str = settings.embedding_device,
) -> Embeddings:
    """Instantiate HuggingFace local embedding model with standard LangChain interface."""
    logger.info("Initializing embedding model '%s' on device '%s'", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
