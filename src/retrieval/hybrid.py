"""Hybrid retriever orchestrating Dense Vector Search, BM25, and RRF Fusion."""

from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from src.config import settings, setup_logger
from src.indexing.vector_store import get_vector_store
from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion

logger = setup_logger("retrieval.hybrid")


class HybridRetriever(BaseRetriever):
    """
    Standard LangChain BaseRetriever implementation for hybrid search.
    Combines dense embedding retrieval from Chroma and sparse BM25 retrieval.
    """

    vector_store: Chroma = Field(description="Chroma vector store instance")
    bm25_index: BM25Index = Field(description="BM25 keyword index instance")
    top_k: int = Field(default=settings.retrieval_top_k)
    dense_weight: float = Field(default=settings.dense_weight)
    bm25_weight: float = Field(default=settings.bm25_weight)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Execute hybrid search combining dense and BM25 candidate lists."""
        logger.info("Executing hybrid search for query: '%s'", query)

        # 1. Dense retrieval
        dense_docs = self.vector_store.similarity_search(query, k=self.top_k)

        # 2. Sparse BM25 retrieval
        bm25_docs = self.bm25_index.search(query, top_k=self.top_k)

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            ranked_lists=[(dense_docs, self.dense_weight), (bm25_docs, self.bm25_weight)],
            top_n=self.top_k,
        )

        logger.info(
            "Hybrid retrieval retrieved %d dense, %d BM25, fused to %d candidates",
            len(dense_docs),
            len(bm25_docs),
            len(fused),
        )
        return fused


def build_hybrid_retriever(
    vector_store: Optional[Chroma] = None,
    top_k: int = settings.retrieval_top_k,
) -> HybridRetriever:
    """Factory to create a HybridRetriever synchronized with current Chroma store."""
    store = vector_store or get_vector_store()

    # Extract all documents from Chroma store to initialize BM25 corpus
    all_data = store.get()
    documents: List[Document] = []
    contents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    for content, meta in zip(contents, metadatas):
        documents.append(Document(page_content=content, metadata=meta or {}))

    bm25_index = BM25Index(documents)

    return HybridRetriever(
        vector_store=store,
        bm25_index=bm25_index,
        top_k=top_k,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )
