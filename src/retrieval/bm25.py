"""BM25 keyword retriever module."""

import re
from typing import List, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.config import settings, setup_logger

logger = setup_logger("retrieval.bm25")


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer with lowercasing for medical terms."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """In-memory BM25 Okapi index for keyword retrieval."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        logger.info("Initialized BM25 index with %d documents", len(documents))

    def search(self, query: str, top_k: int = settings.retrieval_top_k) -> List[Document]:
        """Perform keyword search and return top-k matching documents."""
        if not self.bm25 or not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Document] = []
        for idx in scored_indices[:top_k]:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            # Copy document to avoid mutating original
            scored_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(scores[idx])},
            )
            results.append(scored_doc)

        return results
