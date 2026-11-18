"""Structural chunking pipeline with metadata preservation."""

import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings, setup_logger

logger = setup_logger("ingestion.chunking")


def generate_chunk_id(content: str, metadata: dict, index: int) -> str:
    """Generate deterministic chunk identifier based on content hash and index."""
    source = metadata.get("source", "unknown")
    section = metadata.get("section", "general")
    hash_digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"{source}::{section}::{index}::{hash_digest}"


def chunk_documents(
    documents: List[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[Document]:
    """
    Split medical documents into overlapping chunks while preserving rich metadata.
    Prepends section and drug context to each chunk content for better retrieval semantics.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        keep_separator=True,
    )

    chunked_docs: List[Document] = []

    for doc in documents:
        raw_chunks = splitter.split_text(doc.page_content)
        drug = doc.metadata.get("drug", "General")
        section = doc.metadata.get("section", "Overview")

        for idx, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue

            # Context-augmented content for superior dense/keyword match
            context_prefix = f"[{drug} - {section}]\n"
            content_with_context = f"{context_prefix}{text}"

            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_id"] = generate_chunk_id(text, chunk_meta, idx)
            chunk_meta["context_prefix"] = context_prefix.strip()

            chunked_docs.append(
                Document(
                    page_content=content_with_context,
                    metadata=chunk_meta,
                )
            )

    logger.info("Created %d chunks from %d raw documents", len(chunked_docs), len(documents))
    return chunked_docs
