# Medical Q&A Assistant (LangChain RAG)

A production-grade, grounded medical question-answering assistant built with **LangChain**, **ChromaDB**, **BM25**, and **FlashRank** cross-encoder reranking. The system provides factual answers derived solely from official FDA drug labels and clinical practice guidelines, complete with inline source citations and safety policy guardrails.

---

## Architecture Overview

```mermaid
flowchart TD
    UserQuery(["User Query"]) --> PreCheck{"Safety Pre-Check\n(Emergency & Policy)"}
    PreCheck -- Unsafe --> SafeRedirect["Safe Redirect / Emergency Notice"]
    PreCheck -- Safe --> HybridRetriever["Hybrid Retrieval Engine"]
    
    subgraph Retrieval ["Hybrid Search & Reranking"]
        HybridRetriever --> Dense["Dense Vector Search\n(ChromaDB + all-MiniLM-L6-v2)"]
        HybridRetriever --> Sparse["Keyword Search\n(BM25 Okapi)"]
        Dense & Sparse --> RRF["Reciprocal Rank Fusion\n(RRF k=60)"]
        RRF --> TopCandidates["Top Candidate Chunks"]
        TopCandidates --> CrossEncoder["FlashRank Reranker\n(ms-marco-TinyBERT-L-2-v2)"]
    end

    CrossEncoder --> TopReranked["Top Ranked Evidence Chunks"]
    TopReranked --> Prompt["Grounded Prompt Template\n(Strict Zero-Hallucination Policy)"]
    Prompt --> LLM["LLM (Groq Llama-3.3-70B / Ollama)"]
    LLM --> PostCheck["Safety Post-Check\n(PHI Scrubbing + Medical Disclaimer)"]
    PostCheck --> FinalAnswer(["Grounded Cited Answer"])
```

---

## Key Features

1. **100% Free & Fast**:
   - **LLM**: Groq free tier (`llama-3.3-70b-versatile`) or local offline Ollama (`llama3.2`).
   - **Embeddings**: Local `sentence-transformers/all-MiniLM-L6-v2` (runs on CPU/GPU, $0 cost).
   - **Vector Store**: Local persistent ChromaDB.
   - **Reranker**: Local `FlashRank` CPU cross-encoder.
2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - Combines semantic vector similarity with exact medical keyword matching (BM25) to avoid drug name confusion.
3. **Cross-Encoder Reranking**:
   - Re-scores top candidate passages using cross-attention, boosting precision and top-1 hit accuracy.
4. **Strict Grounding & Source Citations**:
   - Strict prompt template forces the model to answer only using retrieved context excerpts.
   - If the context does not contain sufficient information, it returns `"Based on the provided medical references, this information was not found."`
   - Every claim is cited with numbered references (`[1]`, `[2]`) linked to source URLs, dates, and sections.
5. **Medical Safety Guardrails**:
   - Pre-check detects acute emergencies (chest pain, stroke, overdose, suicidal ideation) and personal diagnosis/dosing requests, safely redirecting users.
   - Post-check scrubs potential PHI and appends mandatory medical disclaimers.
6. **Full Observability with LangSmith**:
   - Built-in tracing for queries, latencies, retrieval scores, and LLM calls via LangSmith environment variables.
7. **Interactive Streamlit UI**:
   - Chat interface with session memory and expandable source citation inspector cards.

---

## Benchmark Evaluation Results

Evaluated against the golden test suite (`data/eval/golden_set.json`) covering indications, contraindications, boxed warnings, drug-drug interactions, and clinical guidelines:

| Retrieval Pipeline | Recall@1 | Recall@3 | Recall@5 | MRR (%) | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Only (Chroma)** | 100.0% | 100.0% | 100.0% | 100.0% | 23.7 ms |
| **Hybrid (BM25 + Dense)** | 100.0% | 100.0% | 100.0% | 100.0% | 23.0 ms |
| **Hybrid + Rerank (FlashRank)** | 100.0% | 100.0% | 100.0% | 100.0% | 36.9 ms |

* **Emergency & Unsafe Query Refusal Accuracy**: **100.0%** (correctly redirected 4/4 unsafe/emergency cases).

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd medical-RAG

# Activate virtual environment (Python 3.12 recommended)
.\.venv312\Scripts\Activate.ps1   # Windows PowerShell
# or: source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your free Groq API key:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

*(Optional) Enable LangSmith tracing:*
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=medical-rag
```

### 3. Ingest Documents & Build Vector Store
```bash
python scripts/ingest.py
```

### 4. Run the Evaluation Benchmark
```bash
python src/eval/runner.py
```

### 5. Launch the Streamlit Chat UI
```bash
streamlit run src/ui/app.py
```

---

## Repository Structure

```
medical-RAG/
├── config/                  # Configuration defaults
├── data/
│   ├── eval/                # Golden benchmark test set & eval results
│   ├── raw/                 # Sample FDA drug labels (JSON) & guidelines (MD)
│   └── chroma/              # Persistent ChromaDB vector index
├── scripts/
│   └── ingest.py            # CLI script to chunk and index raw documents
├── src/
│   ├── config.py            # Centralized pydantic-settings & logger
│   ├── eval/
│   │   ├── metrics.py       # Recall@k, MRR, refusal metrics
│   │   └── runner.py        # Benchmark test runner
│   ├── generation/
│   │   ├── citations.py     # Source card & reference formatter
│   │   ├── generator.py     # End-to-end RAG pipeline
│   │   ├── llm.py           # LLM client provider (Groq / Ollama)
│   │   └── prompts.py       # Zero-hallucination prompt templates
│   ├── indexing/
│   │   ├── embeddings.py    # HuggingFace embeddings wrapper
│   │   └── vector_store.py  # Chroma vector store manager
│   ├── ingestion/
│   │   ├── chunking.py      # Structural metadata-preserving chunker
│   │   └── loaders.py       # PDF, JSON, and web/text loaders
│   ├── retrieval/
│   │   ├── bm25.py          # BM25 keyword index
│   │   ├── fusion.py        # Reciprocal Rank Fusion (RRF)
│   │   ├── hybrid.py        # HybridRetriever combining dense & sparse
│   │   └── reranker.py      # FlashRank cross-encoder reranker
│   ├── safety/
│   │   └── guardrails.py    # Policy checks, PHI scrubbing, emergency filters
│   └── ui/
│       └── app.py           # Interactive Streamlit chat application
├── .env.example             # Configuration template
├── .gitignore               # Git ignore rules
├── PRD.md                   # Product Requirements Document
├── requirements.txt         # Pinned project dependencies
└── README.md                # Documentation & architecture guide
```

---

## Medical Policy & Disclaimer
This software is intended strictly for educational and portfolio demonstration purposes. It does not provide medical diagnoses, treatment plans, or individual dosing instructions. Always seek the advice of a qualified physician or healthcare provider with any medical questions.
