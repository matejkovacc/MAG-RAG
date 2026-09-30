# Real student-office question intake

Choose the corpus explicitly with `--corpus`. The copied regulation candidate is `data/thesis/regulations/2026-09-23-ready/corpus.json` (`fri-regulations`). Its saved audit records snapshot `f12b4035-d561-4b93-a971-5e83971cd199`, 22 sources and 903 chunks. That is a historical artifact, not a verification of the current database. Live mode checks the candidate against the active snapshot before answering. Review evidence again after any publication.

The controlled evaluation runner always requires `--corpus`. It validates the dataset's corpus ID, digest and exact copied passages before creating a provider. For newly imported office cases it also checks the recorded snapshot against the active snapshot in live mode. Existing `evaluation/pilot.json`, `controlled-silver-v1.json`, `controlled-review-v1.json` and `question-variants.json` belong to the earlier `fri-student-affairs-development` corpus or derive from it. Their chunk IDs and review decisions have not been transferred to `fri-regulations`.

## Input

Prepare a UTF-8 JSON file outside retrieval storage. The shape below uses placeholders, not example student data:

Required top-level fields are `name` (nonempty dataset label), `data_origin` (exactly
`"real"`), `anonymized` (exactly `true`), and a nonempty `cases` array. For an
initial 10–20-pair batch, the count is a collection target, not a software limit.
Each case requires `id` (unique anonymous ID), `question`, `original_response`,
`reference_answer`, `answerable` (Boolean), `expected_status`, `family_id`, and
`split`. The question must have 2–2,000 characters; both answers must be nonempty;
the reference answer has a 12,000-character maximum. `split` is `"development"` or
`"test"`. Optional fields are `source_ids` and `required_source_groups` (both
default to empty lists), `category` (defaults to `"unspecified"`), and `difficulty`
(`"easy"`, `"medium"`, or `"hard"`; defaults to `"easy"`). Unknown fields are rejected.

```json
{
  "name": "<dataset version>",
  "data_origin": "real",
  "anonymized": true,
  "cases": [
    {
      "id": "<stable anonymous case ID>",
      "question": "<anonymized original student question>",
      "original_response": "<anonymized original office response>",
      "reference_answer": "<draft answer for human source review>",
      "source_ids": ["<chunk ID from the selected corpus>"],
      "required_source_groups": [["<same chunk ID>"]],
      "category": "<topic>",
      "difficulty": "easy",
      "answerable": true,
      "expected_status": "evidence_found",
      "family_id": "<shared ID for related enquiries and paraphrases>",
      "split": "development"
    }
  ]
}
```

`original_response` is historical office text. `reference_answer` is a separate draft annotation to check against the selected regulation version; the importer never treats the office response as an approved reference. For a question the regulations can answer, set `answerable: true`, `expected_status: "evidence_found"`, and provide at least one valid `source_ids` chunk ID. For a question the regulations cannot answer, use `answerable: false`, empty or omitted `source_ids` and `required_source_groups`, and `expected_status: "no_evidence"`. If the question needs more information before it can be answered, use the same empty evidence fields with `expected_status: "needs_clarification"`. In both cases, the reference answer should describe the expected safe response; the office's historical response is still kept separately. Source groups may list alternative IDs within one group. Every source ID must occur in exactly one group; omitted groups become one group per source. Locate IDs in the prepared corpus or saved retrieval results and verify their actual text, page, article, version and applicability. Do not copy IDs from the older corpus.

Remove names, addresses, student numbers and other identifying details from the question, original response, and draft reference answer before writing the file; use an ID unrelated to student identity. `anonymized: true` is an intake declaration; software cannot certify de-identification. Give every case and paraphrase from one underlying enquiry the same `family_id` and split. The validator rejects duplicate questions, family leakage across development/test, and highly similar questions assigned to different families or splits **within this dataset**; similarity checks cannot find every paraphrase. Keep a family register across later batches, because the importer cannot detect cross-file leakage. Reserve test families before tuning; exposed development cases stay in development.

## Offline import and review

Run from the repository root. Substitute an actual input path and new output names; output files are never overwritten.

```powershell
.venv/Scripts/python.exe -m src.evaluation import-office --input data/private/incoming-anonymized.json --corpus data/thesis/regulations/2026-09-23-ready/corpus.json --verification data/thesis/regulations/2026-09-23-ready/database-verification.json --output data/private/office-draft-v1.json
.venv/Scripts/python.exe -m src.evaluation validate-set --dataset data/private/office-draft-v1.json --corpus data/thesis/regulations/2026-09-23-ready/corpus.json
.venv/Scripts/python.exe -m src.evaluation review-template --dataset data/private/office-draft-v1.json --corpus data/thesis/regulations/2026-09-23-ready/corpus.json --output data/private/office-review-v1.json
```

A human reviewer checks de-identification, each question and original response, the reference answer, answerability, current source applicability, and every cited passage. The review template includes five required approval checks: `question_clear`, `answer_correct`, `source_correct`, `sufficient_evidence`, and `answerability_correct`. Correct the draft dataset first if any reference or evidence is wrong, then make a fresh template because approval is bound to its content hash. Record the reviewer code and decision (`approved`, `needs_revision`, or `rejected`); only `approved` cases with all five checks true become reviewed references. Apply the completed template to a **new** file:

```powershell
.venv/Scripts/python.exe -m src.evaluation apply-review --dataset data/private/office-draft-v1.json --corpus data/thesis/regulations/2026-09-23-ready/corpus.json --reviews data/private/office-review-v1.json --output data/private/office-reviewed-v1.json
.venv/Scripts/python.exe -m src.evaluation validate-set --dataset data/private/office-reviewed-v1.json --corpus data/thesis/regulations/2026-09-23-ready/corpus.json
```

Only after real pairs and human review arrive, an offline diagnostic can run without external calls:

```powershell
.venv/Scripts/python.exe -m src.evaluation evaluate --dataset data/private/office-reviewed-v1.json --corpus data/thesis/regulations/2026-09-23-ready/corpus.json --split test --reviewed-only --origin real_student_office --mode offline --limit 100 --output data/thesis/data/private/office-offline-v1
```

Live evaluation requires a separate, explicitly approved bounded batch and `--mode live --allow-external-api`. It is not authorized by importing or reviewing data. The runner captures one retrieval per case and records the selected snapshot. Its deterministic source and text-overlap metrics do not establish factual correctness or RAGAs scores. Human answer ratings remain a separate step.

Still needed: actual anonymized historical question–response pairs, case grouping and split decisions, human reference and source review, source applicability review for the regulations, and agreement on the thesis evaluation methodology. No real cases or evaluation results are present yet.
