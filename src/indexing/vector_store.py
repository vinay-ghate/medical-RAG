"""Persistent Chroma vector store manager."""

from pathlib import Path
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import settings, setup_logger
from src.indexing.embeddings import get_embedding_model

logger = setup_logger("indexing.vector_store")


def get_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
) -> Chroma:
    """Get or initialize persistent Chroma vector database."""
    persist_dir = persist_directory or settings.chroma_persist_directory
    coll_name = collection_name or settings.chroma_collection_name
    embed_model = embeddings or get_embedding_model()

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=coll_name,
        embedding_function=embed_model,
        persist_directory=persist_dir,
    )


def index_documents(
    chunks: List[Document],
    vector_store: Optional[Chroma] = None,
    batch_size: int = 100,
) -> int:
    """Index chunks into Chroma vector store with chunk_id as key to prevent duplicates."""
    if not chunks:
        logger.warning("No documents provided for indexing.")
        return 0

    store = vector_store or get_vector_store()
    ids = [chunk.metadata.get("chunk_id") for chunk in chunks]

    # Process in batches for memory stability
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_docs = chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        store.add_documents(documents=batch_docs, ids=batch_ids)

    logger.info("Successfully indexed %d chunks into Chroma store", total)
    return total
