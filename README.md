# OSCAR

**[EMNLP 2026 Main Conference]**

Offical Code of Paper [https://arxiv.org/abs/2602.08603](***OSCAR: Optimization-Steered Agentic Planning for Composed Image Retrieval***)

Framework overview: Planner -> Retriever -> Ranker (LangGraph).

- Planner generates tool-call trajectories.
- Retriever executes tool calls and composes retrieved results into a candidate set.
- Ranker scores each candidate image.

Repository structure:
- `src/mip_optimizer/`: MIP construction and solvers.
- `src/ranker/`: VLM ranker server for reranking image candidates.

