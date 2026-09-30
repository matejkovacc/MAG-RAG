# Student-affairs document preparation

This module prepares public PDF and selected official FRI website evidence.

`catalog.py` additionally accepts an explicit catalog of public local PDF, HTML,
UTF-8 TXT and Markdown files, or opt-in official FRI HTTPS downloads. Configure
`config/evaluation-sources.json`; see the [data guide](../../docs/data.md).
Catalog preparation produces the same `PreparedCorpus` consumed by existing offline
and semantic indexes; it never publishes or embeds automatically. Local-only
sources keep an empty citation URL instead of an invented external link.
It needs only the dependencies in `requirements-base.txt`; imports do not create
LLM, MongoDB, Qdrant, or CUDA clients.

From the repository root:

```powershell
.venv/Scripts/python.exe -m src.knowledge_base prepare --manifest config/thesis-sources.json --output data/thesis/new-pdf/corpus.json
.venv/Scripts/python.exe -m src.knowledge_base inspect --corpus data/thesis/new-pdf/corpus.json --query "menjave predmetov" --limit 3
```

Local PDFs must exist at the manifest paths. See [setup, source acquisition and
limitations](../../docs/data.md). `inspect` performs literal
keyword matching for evidence inspection; it does not generate answers or perform
semantic retrieval. The export is a preparation artifact, not a live database.

- `models.py`: manifest, source, page, chunk and export contracts.
- `prepare.py`: native PDF extraction and page/article chunking.
- `website.py`: static HTML extraction by configured content ID, FAQ/heading sections,
  table row/cell boundaries, exact normalized-text offsets and original link inventory.
- `refresh.py`: bounded HTTPS downloads, immutable captures, change/failure reports
  and a separate expanded corpus. No model calls or automatic database publication.
- `__main__.py`: prepare/inspect commands and atomic JSON export.

Optional semantic retrieval is implemented in `vector_store.py` and `runtime.py`,
with `index` and `search` CLI commands. See [configuration and live-service
limitations](../../docs/live-mode.md). Install
`requirements-dev.txt` to run the complete suite, including local-vector
tests; use `tests/knowledge_base/test_preparation.py` with preparation-only dependencies.

Run offline tests with `python -m pytest tests/knowledge_base -q` from the root.

`MongoEvidenceStore.article_neighbors` reads immediate previous/next chunks of the
same article, source, version and captured snapshot for live RAG context expansion.
It does not embed or publish data. Expanded hits have `retrieval_origin` set to
`article_neighbor`, `seed_chunk_id` set, and `score=None` rather than an invented
dense score. Direct semantic `search` continues returning scored ranked hits.

For website refresh commands and citation semantics, see the [data guide](../../docs/data.md).

Regulation discovery is in `discovery.py`; `verification.py` audits a published
snapshot. See [regulations](../../docs/regulations.md) for `refresh`, `verify` and
`smoke`. Verification does not embed; smoke queries require separate API approval.
