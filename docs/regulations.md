# FRI regulations ingestion

The `fri-regulations` corpus is discovered from the official UL FRI landing page:

<https://fri.uni-lj.si/sl/pravilniki-vloge-ceniki>

The landing page is the source of truth. `config/fri-regulations.json` contains the
landing URL, not a maintained list of PDF URLs. Discovery follows the page's content
structure, headings, nested lists, link labels and document URLs. It follows only
explicit official links on FRI, UL and PISRS hosts. UL collection pages are narrowed
to the linked regulation family; PISRS metadata resolves the linked record to its
latest listed consolidated PDF version without fixing a PDF ID in configuration.

## Scope

Included sources cover general regulations, study rules and amendments, exceptional
enrolment/status rules, disciplinary responsibility, tutoring, international
exchange and practical training, Prešeren awards, first- and second-cycle thesis
rules/instructions, and doctoral rules/instructions.

Forms, requests, empty templates, thesis templates, doctoral application templates,
price lists, unrelated navigation and non-official destinations are excluded by
section and link meaning. A DOC/DOCX suffix alone is not an exclusion: a normative
document with an unsupported format is reported as a failed source instead of being
silently classified as a form.

Each source records its landing and linked URLs, resolved source URL, category,
subcategory, document type, known publication/effective dates, version label,
amendment/parent relationship, content hash, retrieval time and discovery path.
Unknown dates remain null. The metadata records discovery and version evidence; it
does not assert legal applicability or supervisor review.

## Refresh and preparation

`refresh` performs discovery, downloads immutable raw captures, extracts PDFs/HTML,
and produces the existing `PreparedCorpus` consumed by the indexer. PDFs retain raw
physical pages and chunks never cross page boundaries. Article labels such as
`1. člen` are inherited by continuation chunks; Roman-numbered chapter headings are
retained where detectable. A textless PDF page is excluded only when its content
stream is provably blank. Images, drawing operators, annotations or unknown content
still fail extraction for review.

A run writes `discovered-sources.json`, `refresh-report.json`, `raw/`,
`prepare-manifest.json` and, only when every selected source succeeds, `corpus.json`.
Failures never replace a published database snapshot or create a reduced candidate.
An unchanged refresh retains prior evidence identities. Changed sources receive new
content/chunk IDs while unchanged chunks can reuse vectors from the active compatible
snapshot.

The reviewed local candidate from 2026-09-23 is:

```text
data/thesis/regulations/2026-09-23-ready/corpus.json
```

It contains 22 source documents, 230 stored physical pages and 903 chunks. This is a
development corpus with source review status `pending`.

## Rebuild, publish and verify

Run this sequence from the repository root. The output directory must not already
exist; use a new name for every refresh. Website access and Azure embedding access
are separate explicit opt-ins.

```powershell
$run = "data/thesis/regulations/next"
.venv/Scripts/python.exe -m src.knowledge_base refresh --manifest config/fri-regulations.json --output $run --allow-website-fetch --transport curl
.venv/Scripts/python.exe -m src.knowledge_base index --corpus "$run/corpus.json" --allow-external-api
.venv/Scripts/python.exe -m src.knowledge_base verify --corpus "$run/corpus.json" --output "$run/database-verification.json" --allow-external-api
.venv/Scripts/python.exe -m src.knowledge_base smoke --corpus-id fri-regulations --output "$run/retrieval-smoke.json" --limit 5 --allow-external-api
```

For a future comparison, add `--previous <previous-run>/corpus.json` to `refresh`.
Do not run `prepare` again after a successful regulation refresh: `refresh` has
already created the prepared corpus.

Indexing stages a complete snapshot in the existing dedicated thesis MongoDB and
Qdrant collection, then atomically changes the MongoDB active pointer. Search always
filters by corpus and active snapshot, so retained history is not searchable.
Compatible unchanged chunks reuse existing vectors; changed or missing chunks are
embedded. There is no automatic garbage collection of inactive snapshots.

`verify` makes no embedding request. It checks the candidate fingerprint, MongoDB
source/page/chunk records, exact page offsets, duplicate URLs/IDs, Qdrant payloads,
active point parity, inactive history and orphan vectors. `smoke` performs seven
query embeddings and writes ranked excerpts with scores, title, page, article,
section and source URL. It does not generate answers or measure correctness.

Run the offline regression suite with:

```powershell
.venv/Scripts/python.exe -m pytest tests/knowledge_base -q
```
