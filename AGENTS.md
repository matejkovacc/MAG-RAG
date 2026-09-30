# Working on MAG – RAG

## Purpose

This repository contains the standalone thesis prototype for answering student
affairs questions from public UL FRI documents. It prepares traceable evidence,
retrieves passages, returns cited answers and evaluates retrieval/answer quality.
It contains no historical student correspondence. Document-derived questions are
synthetic and must never be described as real user questions.

## Structure

- `src/knowledge_base/`: PDF/HTML/TXT/Markdown preparation, source provenance,
  MongoDB/Qdrant storage and retrieval.
- `src/rag/`: typed answer flow, offline/live providers and the local web UI.
- `src/evaluation/`: dataset validation, reference review, metrics and reports.
- `evaluation/`: development questions and review templates; never index them.
- `config/`: explicit source allowlists and dataset-license records.
- `tests/knowledge_base/`: unit/integration tests for this standalone project.
- `docs/`: setup, architecture, evaluation and limitations.

## Commands

Run from the repository root with Python 3.12:

- Install offline runtime: `python -m pip install -r requirements.txt`
- Install test/live extras: `python -m pip install -r requirements-dev.txt`
- Run tests: `python -m pytest tests/knowledge_base -q`
- Run UI tests: `node --test tests/knowledge_base/chat_ui.test.cjs`
- Format check: `python -m black --check src tests/knowledge_base`
- Start offline demo: `python -m src.rag --corpus <corpus.json> --port 10130`
- Start databases: `docker compose up -d`

There is no package build, type-check or standalone lint command. Live Azure
operations require a configured `.env`, explicit user approval and the command's
`--allow-external-api` flag. Never make model calls merely to verify local work.

## Implementation rules

Follow neighboring typed Pydantic/Python patterns, reuse the existing provider
boundaries and keep changes focused. Preserve stable source/chunk IDs, exact source
offsets, corpus/snapshot filters and trusted citation metadata. Keep local lexical
vectors separate from Azure vectors. Never add API keys, `.env`, databases, raw
source documents or student personal data to Git.

Preserve current user changes. Inspect status/diffs before editing. For complex
work, briefly plan, implement the smallest coherent change, run relevant tests and
report exact results. Do not claim tests or live behavior were verified unless the
commands actually ran.

For reviews, compare the change with the request and report functional bugs,
regressions, edge cases and missing tests with file/line, consequence and trigger.
Do not modify code during a review.

Completion requires implemented behavior, appropriate checks and explicit mention
of unverified behavior or remaining issues. Human review is required before an
evaluation item becomes gold; automatic citation/overlap metrics do not establish
factual correctness or legal applicability.

## Repository boundaries

This is the sole application checkout. Before committing or pushing, verify that
`git remote get-url origin` points to `matejkovacc/MAG-RAG`. Thesis LaTeX belongs
in the sibling `mag-rag-thesis` repository, primarily `thesis_template.tex`.
Do not create another application or manuscript copy inside this repository.
Keep `data/private/`, archives and frozen local runs out of Git. Select the corpus
explicitly: the 26 development questions use the two-document snapshot under
`data/thesis/development/`, not the expanded regulations corpus.
Storage namespaces are configurable in the private environment; changing a
namespace does not migrate an index. Preserve snapshot checks and per-run API gates.
