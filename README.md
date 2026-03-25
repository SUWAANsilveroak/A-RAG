# A-RAG (Agentic Retrieval-Augmented Generation)

## Overview

A-RAG is a **research-driven, local-first Retrieval-Augmented Generation system** designed to:

- Maximize accuracy using **hierarchical retrieval**
- Minimize hallucination through **strict grounding**
- Enable **agentic reasoning (ReAct loop)**
- Operate efficiently with **Small Language Models (SLMs)**

This project is not just a tool — it is a **framework to study and compare RAG architectures**.

---

## Goals

- Build a **high-accuracy RAG system using SLMs**
- Reduce dependency on large models
- Ensure:
  - explainability
  - traceability
  - reproducibility
- Compare:
  - basic RAG
  - hybrid RAG
  - agentic RAG (A-RAG)

---

## Core Principles

- No hallucination → answers must be grounded
- Token efficiency → minimal context, maximum relevance
- Full traceability → every step is logged
- Modular design → components can be swapped and compared
- No hidden logic → everything is explicit

---

## Architecture

```text
User Query
   ↓
Query Planner
   ↓
Agent Loop (ReAct)
   ├── keyword_search (lexical)
   ├── semantic_search (embedding)
   ├── hybrid_search (BM25 + vector)
   ├── chunk_read (full context)
   ↓
Re-ranking Layer
   ↓
Context Builder (compressed)
   ↓
LLM (Local / API via LiteLLM)
   ↓
Answer Validator
   ↓
Final Output

---


## Key Components

### 1. Indexing Pipeline
- Document loading
- Chunking (~1000 tokens)
- Sentence segmentation
- Embedding generation

---

### 2. Retrieval Layer
- Keyword search (lexical)
- Semantic search (vector-based)
- Hybrid search (BM25 + embeddings)
- Re-ranking (cross-encoder)

---

### 3. Agent System
- Query planning
- Tool selection
- Iterative reasoning (ReAct loop)
- Context tracking (avoid redundancy)

---

### 4. Validation Layer
- Grounding check
- Conflict detection
- Completeness check

---

## Tech Stack

- **LLM**: Ollama (Llama3 / Phi-3 / Mistral)
- **Embeddings**: BGE-Small-en-v1.5
- **Vector DB**: FAISS
- **Orchestration**:
  - LangGraph (agent loop)
  - LiteLLM (model gateway)
- **Utilities**:
  - spaCy / NLTK (text processing)
  - rank_bm25 (keyword search)

---

## Project Structure
