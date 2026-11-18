"""Reciprocal Rank Fusion (RRF) for combining dense and sparse retrievals."""

from typing import Dict, List, Tuple
from langchain_core.documents import Document
from src.config import settings, setup_logger

logger = setup_logger("retrieval.fusion")


def reciprocal_rank_fusion(
    ranked_lists: List[Tuple[List[Document], float]],
    rrf_k: int = 60,
    top_n: int = settings.retrieval_top_k,
) -> List[Document]:
    """
    Fuse multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF_Score(d) = SUM( weight_i / (rrf_k + rank_i(d)) )
    
    Args:
        ranked_lists: List of (document_list, weight) tuples.
        rrf_k: Smoothing constant (standard default is 60).
        top_n: Number of fused results to return.
    """
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list, weight in ranked_lists:
        for rank, doc in enumerate(doc_list, start=1):
            doc_id = doc.metadata.get("chunk_id") or doc.page_content
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

            score_increment = weight / (rrf_k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score_increment

    # Sort descending by fused score
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

    fused_docs: List[Document] = []
    for doc_id in sorted_doc_ids[:top_n]:
        base_doc = doc_map[doc_id]
        fused_doc = Document(
            page_content=base_doc.page_content,
            metadata={
                **base_doc.metadata,
                "fusion_score": round(doc_scores[doc_id], 6),
            },
        )
        fused_docs.append(fused_doc)

    logger.debug("Fused %d unique candidates into top %d results", len(doc_map), len(fused_docs))
    return fused_docs
