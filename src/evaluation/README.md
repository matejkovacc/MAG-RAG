# Evaluation

Compare answers with references and supporting sources. The normal workflow uses
only question origin, reference-review status and answerability. Read the
[usage guide](../../docs/evaluation.md) for the existing 26 document-derived
questions and 10 official FAQs, reference review and interpretation.

Run from the project root using the existing thesis environment:

```powershell
.venv/Scripts/python.exe -m src.evaluation --help
.venv/Scripts/python.exe -m src.evaluation evaluate --dataset evaluation/controlled-silver-v1.json --corpus data/thesis/development/corpus.json --allow-unreviewed --limit 26 --output data/thesis/evaluation/my-offline-run
```

Default execution uses temporary local stores, word-based retrieval and quoted
source passages. No model calls occur. Live runs require separate batch approval
and explicit flags; all existing source, split and review checks remain active.

Use `report --run ... --output ...` to reformat saved results without repeating
retrieval or generation. Use `export-questions --dataset ... --corpus ... --output
...` for a readable question list. Review templates now include the question,
reference answer and source text alongside the decision fields.

Modules:

- `controlled.py`: frozen annotations, provenance and human-review validation.
- `metrics.py`: deterministic retrieval and answer metrics.
- `controlled_runner.py`: single-retrieval execution, persisted results and human scores.
- `reporting.py`: plain-language report, answer list and CSV.
- `controlled_cli.py`: current commands and compatibility aliases.
- `faq.py`: official FAQ import and explicit lookup protocol.
- `public_qa.py`: separately reported public-language benchmarks.

The legacy `validate`/`run` pilot commands and old generation/selection aliases
remain available. Existing datasets and approvals are not rewritten. Their
historical field names do not determine the current report layout. RAGAs and
embedding-based answer similarity are not run by the local scorer.
