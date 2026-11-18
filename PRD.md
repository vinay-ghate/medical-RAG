# PRD: Medical Q&A Assistant (LangChain RAG)

## 1. Overview
| Item | Detail |
|---|---|
| **Product** | RAG assistant that answers drug and guideline questions with source citations |
| **Purpose** | Learning project and interview portfolio piece |
| **Users** | Primarily the developer; secondarily anyone reviewing the demo |
| **Scope** | Information only. No diagnosis, no personalized dosing, no real patient data |

## 2. Goals and Non-Goals
| Goals | Non-Goals |
|---|---|
| Grounded answers with citations | Diagnosis or treatment advice |
| Demonstrate hybrid search + reranking | Real patient data (PHI) |
| Free-tier or local only | Multi-user auth, billing |
| Easy to swap any component | Production-scale deployment (v1) |
| Measurable quality (eval metrics) | Fine-tuning models |

## 3. Design Principles (Simple, Reusable, Upgradable)
| Principle | How it's applied |
|---|---|
| **Simple** | One linear pipeline first; add an agent only after the baseline works. Few dependencies, one config file. |
| **Reusable** | Domain-agnostic core: swap the data folder and prompt to get a legal, HR, or any other Q&A bot. Medical logic lives only in config, prompts, and data. |
| **Upgradable** | Every component sits behind a standard LangChain interface (model, embeddings, vector store, retriever, reranker). Change a component by editing config, not code structure. |
| **Provider-agnostic** | Model, embeddings, and reranker chosen via config (NVIDIA, Ollama, HuggingFace, etc.) |
| **Measurable** | Every upgrade (hybrid, reranker, agent) must show a metric gain on the same golden set |
| **Replaceable storage** | Chroma now, Pinecone later via the same vector store interface |

## 4. Functional Requirements
| ID | Requirement | Priority |
|---|---|---|
| F1 | Ingest PDFs, web pages, and JSON into a local index | Must |
| F2 | Chunk by document structure and attach metadata (drug, section, source, date) | Must |
| F3 | Hybrid retrieval (BM25 + dense) with fusion | Must |
| F4 | Rerank the top candidates | Must |
| F5 | Generate answers only from retrieved context, with citations | Must |
| F6 | Return "not found" when context is insufficient | Must |
| F7 | Refuse diagnosis, personal dosing, and emergencies, with a safe redirect | Must |
| F8 | Chat UI with source display | Should |
| F9 | Conversation memory for follow-ups | Should |
| F10 | Agent with extra tools (interaction check, PubMed search) | Could (v2) |
| F11 | Re-index only changed documents | Could |

## 5. Non-Functional Requirements
| Area | Target |
|---|---|
| **Cost** | Zero (free tiers or local models) |
| **Latency** | Under ~10 s per answer on the free tier |
| **Quality** | Faithfulness and citation accuracy tracked, with target thresholds set after the baseline |
| **Maintainability** | Single config file; modules under ~150 lines; README with architecture |
| **Safety** | Disclaimer shown; grounded-only answers; no PHI stored |
| **Observability** | Tracing enabled (LangSmith free tier or local logs) |

## 6. Architecture (Modular)
| Module | Responsibility | Swappable with |
|---|---|---|
| **Config** | Model names, chunk size, k values, paths | Env/YAML file |
| **Ingestion** | Load, clean, chunk, add metadata | New loaders |
| **Index** | Embeddings + vector store | Chroma → Pinecone |
| **Retrieval** | BM25 + dense + fusion | Different retrievers or fusion weights |
| **Rerank** | Cross-encoder on top-N | NVIDIA, bge, Cohere |
| **Generation** | Prompt + LLM + citation formatting | Any chat model |
| **Safety** | Pre-check and post-check rules | Stricter policy |
| **UI** | Chat interface | Streamlit → Gradio → API |
| **Eval** | Golden set + metrics runner | More datasets |

**Flow:** Query → safety pre-check → hybrid retrieval → rerank → grounded prompt → LLM → cited answer

**Rule:** modules talk only through standard interfaces (documents in, documents out), so none depends on a specific provider.

## 7. Tech Stack (Free)
| Layer | v1 choice |
|---|---|
| Framework | LangChain v1 (agent/LangGraph later) |
| LLM | NVIDIA hosted free model or Ollama |
| Embeddings | HuggingFace bge or NVIDIA embeddings |
| Reranker | bge-reranker or NVIDIA rerank |
| Vector DB | Chroma |
| Keyword | BM25 |
| UI | Streamlit |
| Eval/tracing | LangSmith free tier |

Check current docs for exact package locations and free-tier limits, since both change.

## 8. Data
| Source | Use |
|---|---|
| openFDA / DailyMed labels | Core drug facts |
| MedlinePlus | Plain-language info |
| WHO/CDC guideline PDFs | Guidelines |
| PubMed abstracts | Optional, v2 |

Start with 20-50 documents. Record the source URL and date for freshness.

## 9. Success Metrics
| Metric | Purpose |
|---|---|
| Recall@k, MRR, NDCG@k | Retrieval quality |
| Faithfulness | Answer supported by context |
| Citation accuracy | Cited source actually supports the claim |
| Refusal accuracy | Correct refusals on unsafe queries |
| Latency | Usability |

**Eval method:** a golden set of 30-50 Q&A pairs, plus a comparison table: dense only → hybrid → hybrid + rerank.

## 10. Milestones
| Phase | Deliverable | Est. time |
|---|---|---|
| M1 | Ingestion + index + retrieval-only test | 2 days |
| M2 | Hybrid + rerank, with metrics | 2 days |
| M3 | Grounded generation + safety + UI | 2 days |
| M4 | Evaluation report + README | 1-2 days |
| M5 (v2) | Agent, memory, extra tools | 2-3 days |

## 11. Risks and Mitigations
| Risk | Mitigation |
|---|---|
| Hallucinated medical info | Grounded-only prompt, citations, "not found" fallback |
| Free-tier rate limits | Local model fallback; cache responses |
| Outdated documents | Store dates; re-index flow |
| Similar drug names confuse dense search | BM25 + metadata filters |
| Over-engineering | Ship the linear pipeline first; add features only if metrics justify them |
| Dependency churn (LangChain changes) | Pin versions; isolate integrations in the modules above |

## 12. Upgrade Roadmap
1. **v1:** linear RAG, Chroma, hybrid + rerank, Streamlit
2. **v2:** agent with tools, memory, PubMed
3. **v3:** LangGraph workflow (classify → retrieve → verify → answer), Pinecone, scheduled re-indexing
4. **v4:** reuse the core for another domain by swapping data and prompts only