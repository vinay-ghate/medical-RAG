"""Retrieval evaluation metrics: Recall@k, MRR, and Refusal Accuracy."""

from typing import Dict, List, Optional
from langchain_core.documents import Document


def is_relevant(doc: Document, target_drug: Optional[str], target_section: Optional[str]) -> bool:
    """Check whether a retrieved document chunk matches target drug and section."""
    if not target_drug:
        return False

    meta = doc.metadata
    doc_drug = str(meta.get("drug", "")).strip().lower()
    tgt_drug = target_drug.strip().lower()

    if tgt_drug not in doc_drug and doc_drug not in tgt_drug:
        return False

    if target_section:
        doc_sec = str(meta.get("section", "")).strip().lower()
        tgt_sec = target_section.strip().lower()
        if tgt_sec in doc_sec or doc_sec in tgt_sec:
            return True

    return True


def evaluate_query_retrieval(
    retrieved_docs: List[Document],
    target_drug: Optional[str],
    target_section: Optional[str],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Calculate Recall@k and Reciprocal Rank (RR) for a single query."""
    metrics: Dict[str, float] = {}

    # Calculate Reciprocal Rank
    reciprocal_rank = 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        if is_relevant(doc, target_drug, target_section):
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr"] = reciprocal_rank

    # Calculate Recall@k
    for k in k_values:
        top_k_docs = retrieved_docs[:k]
        hit = any(is_relevant(d, target_drug, target_section) for d in top_k_docs)
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0

    return metrics
