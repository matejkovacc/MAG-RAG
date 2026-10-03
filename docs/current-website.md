# PDF coverage and website retrieval — 3 October 2026

## PDF inventory

`python -m src.knowledge_base sync-pdfs --output <new-directory> --previous <previous-corpus.json> --allow-website-fetch`

This command downloads public documents, writes their provenance and prepares
text locally. It never calls a model or publishes a database index. The output
directory must be new. Scope: direct PDFs in the FRI regulations/forms/pricing
directory, the existing resolver's linked UL/PISRS regulations, and PDFs on the
explicitly linked UL pricing page. It does not recursively crawl the whole site.
DOCX alternatives, XLSX price tables and GitHub templates are outside PDF scope.

The actual run `data/thesis/all-pdfs/2026-10-03/` acquired **42/42 PDFs**:
29 direct links, eight additional resolved regulations and five UL pricing PDFs.
Twenty URLs were absent from the previous 22-document corpus. All 42 prepared:
282 physical pages and 1,077 chunks, with one provably blank page excluded from
chunks. `inventory.json` records every outcome, URL, discovery path and SHA-256;
`raw/` retains the PDFs. Native text extraction is not a layout or legal-validity
review. Forms/templates are labelled in source notes and must not become rules.
Both 2025/26 and 2026/27 pricing documents remain explicit sources.

The corpus `fri-all-pdfs` is now **published and verified** in the local semantic
index: 42 Mongo sources, 282 pages, 1,077 chunks and 1,077 active Qdrant points,
with zero orphan points or verification errors. The original 22-document snapshot
and development evaluation IDs remain intact. Raw files, corpus exports and local
audit outputs stay out of Git; this publication exists in this local installation.

The initial run received HTTP 429 on its third embedding request. Recovery reused
the 64 already written vectors, paced subsequent requests under the observed
20,000-token/minute limit and stayed inside the original 34-request cap: **32 total
index requests including the failed one**, 252,082 reported embedding tokens.
Identical text reused a vector within recovery. Only the failed run's 64 explicitly
identified, backed-up unpublished points were removed after the replacement
publication. The final verification is saved at
`data/thesis/all-pdfs/2026-10-03/publication-and-pilot/recovery/database-verification.json`.
Saved operation scripts, counters and checkpoints document recovery; SDK retries
were disabled. This was a storage/provenance audit, not semantic answer evaluation.

## Fetch pages while answering

`config/fri-current-website.json` contains ten fixed official FRI HTML pages.
Every question downloads these pages with at most four concurrent requests.
Question text is never sent to FRI. HTTPS, validated redirects, time/size limits
and an explicit content selector apply. Page bytes are processed in memory;
the normal answer server does not persist questions or public-page captures.

Start a preview with public website requests but **zero model calls**:

```powershell
.venv/Scripts/python.exe -m src.rag --corpus data/thesis/all-pdfs/2026-10-03/corpus.json --current-website --allow-website-fetch --port 10130
```

This mode quotes retrieved passages. For **generated answers**, after explicit
approval for the question allowance, use the verified 42-document PDF snapshot:

```powershell
.venv/Scripts/python.exe -m src.rag --live --allow-external-api --max-questions 5 --corpus data/thesis/all-pdfs/2026-10-03/corpus.json --current-website --allow-website-fetch --port 10131
```

Startup verifies that the selected PDF file matches the published snapshot. No startup path indexes data. The existing live
allowance is checked before public fetching and paid retrieval/generation.

Fresh page chunks are ranked lexically, without embedding or indexing the pages.
At a context limit of five, up to three slots are initially reserved for fresh
HTML, with the remaining slots for PDFs; unused slots can receive more HTML.
This is a bounded retrieval policy, not a verified optimum or whole-site search.
Only these ten pages are fetched; links to further HTML/PDF pages are not followed.
Failures are disclosed, and saved HTML is never silently substituted as current.
The UI shows fetch times and per-page outcomes. A fetch time is not an effective
date; pages may contain old deadlines or different academic years.

## Technical pilot and its limits

Two saved technical runs are under `data/thesis/site-audit/2026-10-03/`.
The first exposed standalone headings outranking useful text and the enrolment
page failing on a full-width table title. The fixes exclude heading-only chunks,
bound length normalization and support the observed title-plus-two-column table.
Arbitrary merged data cells still fail explicitly. New HTML captures identify
the extractor as `stdlib-html-v2`; frozen snapshots are not rewritten.

The second run fetched all ten pages successfully for each of three manually
written development questions. It retrieved the office-hours passage, the
2026/27 calendar and the master's-topic application procedure. Raw page bytes
are preserved under `pilot-v2-raw/`, with hashes/timestamps in the answer records.
The calendar question asked about autumn holidays: retrieving a calendar does
**not** establish that it answers that question (it lists examination periods).
That item should test abstention, not be counted as answer success.

These runs use literal-excerpt simulation with an empty PDF retriever. They
verify fresh acquisition and candidate retrieval, not semantic retrieval,
generated-answer correctness, or a representative student-office evaluation.
The new unit checks also cover changing page content between questions,
failed-page handling, unsafe redirects, explicit years, concurrent requests,
abstention metadata, UI disclosure and the live allowance. Any paid answer pilot
requires a separate approved batch and human review of the actual claims.

## Approved generated-answer pilot

The subsequent approved batch on 3 October 2026 used the existing published
22-document PDF snapshot plus current pages: **5 embedding and 5 chat requests**,
no retries, all five returned a valid response. Usage was 127 embedding tokens,
13,593 chat input tokens and 343 chat output tokens. The five manually written
questions and actual evidence are saved in
`data/thesis/site-audit/2026-10-03/approved-answer-pilot/`.

Evidence inspection by Codex (not independent human annotation) found:

| Question | Observed outcome |
| --- | --- |
| Tuesday office hours | Answer matches the cited office hours; telephone hours remain separate. |
| Winter exams, 2026/27 | Correctly selects the labelled year and dates in its cited calendar passage. |
| Master's-topic application | Partially supported: omitted the explicit year and added contact advice unsupported by its sole citation. |
| Autumn holidays | Abstains; autumn examination dates are not treated as holiday dates. |
| Personal application approval | Abstains; no claim of accessing an individual student record. |

This is not a 5/5 correctness result or thesis benchmark. The prompt now explicitly
requires retaining deadline years and limits added contact advice to what the cited
passage supports. That revision has **not** been model-tested; the five-question
allowance was exhausted. Preserve this run when planning a new approved batch.
The prompt actually used, source bytes, single retrieval capture, usage, and an
explicitly non-human review are retained separately from the revised prompt.
