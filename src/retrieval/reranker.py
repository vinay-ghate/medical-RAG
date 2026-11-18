"""Reranker module using FlashRank cross-encoder for high-precision ranking."""

from typing import List, Optional
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from src.config import settings, setup_logger

logger = setup_logger("retrieval.reranker")


class DocumentReranker:
    """FlashRank-based cross-encoder reranker for re-scoring retrieved candidates."""

    def __init__(self, model_name: str = settings.reranker_model):
        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name, cache_dir="./data/cache")
        logger.info("Initialized FlashRank reranker with model '%s'", model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = settings.rerank_top_k,
    ) -> List[Document]:
        """Rerank candidates using cross-encoder score and return top_k highest scoring."""
        if not documents:
            return []

        # Prepare FlashRank input payload
        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        reranked_docs: List[Document] = []
        for rank_idx, item in enumerate(results[:top_k], start=1):
            original_meta = dict(item.get("meta", {}))
            original_meta["rerank_score"] = round(float(item.get("score", 0.0)), 6)
            original_meta["rerank_rank"] = rank_idx

            doc = Document(
                page_content=item.get("text", ""),
                metadata=original_meta,
            )
            reranked_docs.append(doc)

        logger.info(
            "Reranked %d candidates down to top %d (top score: %s)",
            len(documents),
            len(reranked_docs),
            reranked_docs[0].metadata.get("rerank_score") if reranked_docs else "N/A",
        )
        return reranked_docs


# Global reranker singleton instance
_reranker_instance: Optional[DocumentReranker] = None


def get_reranker() -> DocumentReranker:
    """Get or initialize singleton reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = DocumentReranker()
    return _reranker_instance


def retrieve_and_rerank(
    query: str,
    retriever=None,
    reranker: Optional[DocumentReranker] = None,
    top_k: int = settings.rerank_top_k,
) -> List[Document]:
    """Execute hybrid retrieval followed by cross-encoder reranking."""
    from src.retrieval.hybrid import build_hybrid_retriever

    active_retriever = retriever or build_hybrid_retriever()
    active_reranker = reranker or get_reranker()

    candidates = active_retriever.invoke(query)
    return active_reranker.rerank(query=query, documents=candidates, top_k=top_k)
