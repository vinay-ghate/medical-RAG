"""End-to-end grounded generation pipeline with citation tracking."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from src.config import settings, setup_logger
from src.generation.llm import get_llm
from src.generation.prompts import QA_PROMPT, format_context_with_citations
from src.retrieval.reranker import retrieve_and_rerank
from src.safety.guardrails import apply_post_check, run_pre_check

logger = setup_logger("generation.generator")

NOT_FOUND_RESPONSE = "Based on the provided medical references, this information was not found."


class RAGResponse(BaseModel):
    """Structured response container for medical Q&A."""

    query: str
    answer: str
    is_safe: bool
    refusal_reason: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    raw_context: Optional[str] = None


class MedicalRAGPipeline:
    """End-to-end linear RAG pipeline for medical questions."""

    def __init__(self, llm=None, retriever=None, reranker=None):
        settings.setup_langsmith_tracing()
        self.llm = llm or get_llm()
        self.retriever = retriever
        self.reranker = reranker

    def run(self, query: str, top_k: int = settings.rerank_top_k) -> RAGResponse:
        """Execute full RAG pipeline: pre-check -> retrieval -> rerank -> LLM -> post-check."""
        logger.info("Processing medical query: '%s'", query)

        # 1. Safety Pre-check
        pre_check = run_pre_check(query)
        if not pre_check.is_safe:
            return RAGResponse(
                query=query,
                answer=pre_check.redirect_message or "Request refused under safety policy.",
                is_safe=False,
                refusal_reason=pre_check.refusal_reason,
                citations=[],
            )

        # 2. Hybrid Retrieval + Reranking
        retrieved_docs = retrieve_and_rerank(
            query=query,
            retriever=self.retriever,
            reranker=self.reranker,
            top_k=top_k,
        )

        if not retrieved_docs:
            logger.info("No matching context chunks found for query: '%s'", query)
            answer_with_disclaimer = apply_post_check(NOT_FOUND_RESPONSE)
            return RAGResponse(
                query=query,
                answer=answer_with_disclaimer,
                is_safe=True,
                citations=[],
            )

        # 3. Format Context and Citations
        context_str, citations = format_context_with_citations(retrieved_docs)

        # 4. Invoke Grounded LLM Prompt
        chain = QA_PROMPT | self.llm
        raw_output = chain.invoke({"question": query, "context": context_str})
        answer_text = raw_output.content if hasattr(raw_output, "content") else str(raw_output)

        # 5. Safety Post-Check (disclaimer & scrub PHI)
        final_answer = apply_post_check(answer_text)

        return RAGResponse(
            query=query,
            answer=final_answer,
            is_safe=True,
            citations=citations,
            raw_context=context_str,
        )
