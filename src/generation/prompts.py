"""Grounded prompt templates and citation formatters."""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

GROUNDED_SYSTEM_PROMPT = """You are a grounded clinical information assistant.
Your duty is to answer questions strictly and exclusively using the provided medical context excerpts below.

CRITICAL RULES:
1. ONLY make statements directly supported by the provided context excerpts.
2. DO NOT extrapolate, assume, or add clinical facts not found in the excerpts.
3. Every factual claim must include an inline bracketed citation referencing the excerpt index, e.g., [1] or [1][2].
4. If the context directly addresses the question (e.g., states a contraindication, warning, or guideline recommendation), explain what the referenced excerpts explicitly state.
5. If the provided context does NOT contain any relevant information to answer the question, respond with:
   "Based on the provided medical references, this information was not found."
6. Never diagnose conditions or give personalized dosing advice to the user.

CONTEXT EXCERPTS:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GROUNDED_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_context_with_citations(documents: List[Document]) -> Tuple[str, List[dict]]:
    """
    Format list of retrieved Documents into numbered context blocks and structured citation objects.
    """
    formatted_blocks: List[str] = []
    citations: List[dict] = []

    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        drug = meta.get("drug", "Medical Reference")
        section = meta.get("section", "General")
        source = meta.get("source", "N/A")
        date = meta.get("date", "N/A")

        header = f"[{idx}] {drug} - {section} (Source: {source}, Date: {date})"
        body = doc.page_content.strip()
        formatted_blocks.append(f"{header}\n{body}")

        citations.append(
            {
                "index": idx,
                "drug": drug,
                "section": section,
                "source": source,
                "date": date,
                "excerpt": body[:250] + ("..." if len(body) > 250 else ""),
                "chunk_id": meta.get("chunk_id", ""),
                "rerank_score": meta.get("rerank_score", None),
            }
        )

    full_context_str = "\n\n".join(formatted_blocks)
    return full_context_str, citations
