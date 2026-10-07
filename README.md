# AI Research Paper Assistant with Explainable Similarity Ranking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdullahf-77/ai-research-paper-assistant/blob/main/research_paper_assistant.ipynb)

Upload one research paper (PDF). The system:

1. **Answers questions about the paper** with RAG (grounded answers with page citations).
2. **Discovers, ranks, and explains similar papers** from Semantic Scholar (references, citations,
   recommendations) and arXiv (LLM-written queries). Scores are computed in Python (embeddings + cosine
   similarity), and the LLM only explains them.

Built with **LangChain `create_agent`** (exactly two agents) orchestrated by a **LangGraph `StateGraph`**
with conditional routing and a parallel fan-out.

## Run in Google Colab

1. Click the **Open in Colab** badge above.
2. Add a secret (🔑 icon in the left sidebar) and enable *Notebook access*:
   - `GROQ_API_KEY` (free at https://console.groq.com), **or** `GOOGLE_API_KEY` with `LLM_PROVIDER = 'gemini'`
   - optional: `SEMANTIC_SCHOLAR_API_KEY` (higher rate limits)
3. **Runtime → Run all**. Upload your PDF when asked (or set `PDF_SOURCE = 'sample'`).

## Run locally

```bash
uv sync
export GROQ_API_KEY=...        # never commit keys
uv run jupyter lab research_paper_assistant.ipynb
```
Locally, set `PDF_SOURCE = 'sample'` (the upload widget only works in Colab), or point `PDF_PATH` to a file.

## Architecture

```
START ─route─► load_pdf ─► extract_profile ─┬─► build_rag_index ─route─► paper_rag_agent ─► END
  │                                         │                    └─► END
  │                                         └─► discovery_agent ─► merge_dedup ─route─► score_rank ─► explain_ranking ─► END
  └─(paper already processed)─► paper_rag_agent ─► END                         └─► END (no candidates)
```

| Agent | Tools |
|---|---|
| Paper RAG Agent | `search_paper` (FAISS retriever over the PDF) |
| Discovery & Ranking Agent | `get_semantic_scholar_candidates`, `search_arxiv` |

**Overall Relatedness** = 100 × (0.70·semantic_similarity + 0.15·citation_relationship
+ 0.10·multi_source_agreement + 0.05·topic_overlap)

Reliability and observability: a custom trace tree (nodes, agents, tools, LLM calls, tokens),
`ToolCallLimitMiddleware`, LoopDetector (repetition + stagnation), checkpointer memory for follow-up
questions, and Semantic Scholar 429 retries.

## Files

```
research_paper_assistant.ipynb   # the whole project
pyproject.toml                   # dependencies (uv)
README.md
```
