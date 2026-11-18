"""Evaluation runner generating comparative metrics for dense vs hybrid vs rerank."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import BASE_DIR, setup_logger
from src.eval.metrics import evaluate_query_retrieval
from src.indexing.vector_store import get_vector_store
from src.retrieval.hybrid import build_hybrid_retriever
from src.retrieval.reranker import get_reranker
from src.safety.guardrails import run_pre_check

logger = setup_logger("eval.runner")


def run_benchmark():
    golden_path = BASE_DIR / "data" / "eval" / "golden_set.json"
    if not golden_path.exists():
        logger.error("Golden set file not found: %s", golden_path)
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    logger.info("Loaded %d golden test cases from %s", len(golden_set), golden_path)

    # Initialize components
    vector_store = get_vector_store()
    hybrid_retriever = build_hybrid_retriever(vector_store=vector_store)
    reranker = get_reranker()

    strategies = ["Dense Only", "Hybrid (BM25+Dense)", "Hybrid + Rerank"]
    results: Dict[str, Dict[str, float]] = {s: {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr": 0.0, "latency_ms": 0.0} for s in strategies}

    safe_cases = [c for c in golden_set if c.get("expected_safe", True)]
    unsafe_cases = [c for c in golden_set if not c.get("expected_safe", True)]

    # 1. Benchmark Safety Refusal Accuracy
    refusal_correct = 0
    for case in unsafe_cases:
        check = run_pre_check(case["query"])
        if not check.is_safe:
            refusal_correct += 1
    refusal_acc = (refusal_correct / len(unsafe_cases)) * 100 if unsafe_cases else 100.0

    # 2. Benchmark Retrieval on Safe Questions
    total_safe = len(safe_cases)

    for case in safe_cases:
        q = case["query"]
        tgt_drug = case.get("target_drug")
        tgt_sec = case.get("target_section")

        # Dense Only
        t0 = time.perf_counter()
        dense_docs = vector_store.similarity_search(q, k=10)
        t_dense = (time.perf_counter() - t0) * 1000
        m_dense = evaluate_query_retrieval(dense_docs, tgt_drug, tgt_sec)
        for k, v in m_dense.items():
            results["Dense Only"][k] += v
        results["Dense Only"]["latency_ms"] += t_dense

        # Hybrid
        t0 = time.perf_counter()
        hybrid_docs = hybrid_retriever.invoke(q)
        t_hybrid = (time.perf_counter() - t0) * 1000
        m_hybrid = evaluate_query_retrieval(hybrid_docs, tgt_drug, tgt_sec)
        for k, v in m_hybrid.items():
            results["Hybrid (BM25+Dense)"][k] += v
        results["Hybrid (BM25+Dense)"]["latency_ms"] += t_hybrid

        # Hybrid + Rerank
        t0 = time.perf_counter()
        reranked_docs = reranker.rerank(query=q, documents=hybrid_docs, top_k=5)
        t_rerank = (time.perf_counter() - t0) * 1000 + t_hybrid
        m_rerank = evaluate_query_retrieval(reranked_docs, tgt_drug, tgt_sec)
        for k, v in m_rerank.items():
            results["Hybrid + Rerank"][k] += v
        results["Hybrid + Rerank"]["latency_ms"] += t_rerank

    # Average metrics
    for s in strategies:
        for metric in ["recall@1", "recall@3", "recall@5", "mrr"]:
            results[s][metric] = round((results[s][metric] / total_safe) * 100, 2)
        results[s]["latency_ms"] = round(results[s]["latency_ms"] / total_safe, 1)

    # Print comparison table
    print("\n" + "=" * 78)
    print("           MEDICAL RAG BENCHMARK EVALUATION RESULTS")
    print("=" * 78)
    print(f"{'Retrieval Pipeline':<24} | {'Recall@1':<9} | {'Recall@3':<9} | {'Recall@5':<9} | {'MRR (%)':<9} | {'Avg Latency':<10}")
    print("-" * 78)
    for s in strategies:
        r = results[s]
        print(f"{s:<24} | {r['recall@1']:>7.1f}% | {r['recall@3']:>7.1f}% | {r['recall@5']:>7.1f}% | {r['mrr']:>7.1f}% | {r['latency_ms']:>6.1f} ms")
    print("-" * 78)
    print(f"Safety Refusal Accuracy on Unsafe / Emergency Queries: {refusal_acc:.1f}% ({refusal_correct}/{len(unsafe_cases)})")
    print("=" * 78 + "\n")

    # Save to disk
    output_path = BASE_DIR / "data" / "eval" / "eval_results.json"
    eval_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_test_cases": len(golden_set),
        "safe_cases": total_safe,
        "unsafe_cases": len(unsafe_cases),
        "refusal_accuracy_pct": refusal_acc,
        "pipelines": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    logger.info("Evaluation metrics successfully written to %s", output_path)


if __name__ == "__main__":
    run_benchmark()
