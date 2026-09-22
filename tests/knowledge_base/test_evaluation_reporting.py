"""Readable evaluation reports preserve saved results and honest review provenance."""

import copy
import csv
import json
import socket

import pytest

from src.evaluation.controlled import (
    ControlledDataset,
    ControlledItem,
    apply_reviews,
    review_template,
)
from src.evaluation.controlled_cli import main, parser
from src.evaluation.controlled_runner import export_saved_run, result_groups
from src.evaluation.reporting import render_questions, render_report


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Reformatting and review views cannot call providers or fetch sources."""

    def fail(*args, **kwargs):
        """Disallow outgoing connections."""
        raise AssertionError("Unexpected network request")

    monkeypatch.setattr(socket.socket, "connect", fail)


@pytest.fixture
def saved_run(tmp_path):
    """Small saved-run fixture with historical grouping names and explicit outputs."""
    path = tmp_path / "original"
    path.mkdir()
    row = {
        "id": "fixture-1",
        "question": "Which test colour?",
        "expected_answer": "Blue",
        "response": "Red",
        "answer": {"citations": []},
        "generation_method": "official_faq",
        "review_status": "unreviewed",
        "answerable": True,
        "supporting_passages": [],
        "category": "direct",
        "difficulty": "easy",
        "tier": "official_faq",
        "error": None,
        "latency_seconds": 1.25,
        "metrics": {
            "token_f1": 0.0,
            "status_match": 1.0,
            "exact_match": 0.0,
            "answer_with_correct_source": 0.0,
        },
    }
    meta = {
        "mode": "simulated",
        "protocol": "official_faq_lookup",
        "selected_ids": [row["id"]],
    }
    summary = {
        "metadata": meta,
        "gold": {},
        "silver": {},
        "all": {},
        "unattempted_ids": [],
        "elapsed_seconds": 1.25,
        "usage": {"embedding_requests": 0, "chat_requests": 0},
        "ragas": "not run",
        "semantic_similarity": None,
        "human_ratings": "not collected",
    }
    files = {
        "summary.json": summary,
        "metadata.json": meta,
        "dataset.json": {"schema_version": 2, "items": [row]},
        "answer-review.json": [],
    }
    for name, data in files.items():
        (path / name).write_text(json.dumps(data), encoding="utf-8")
    (path / "attempts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path, row, summary


def test_reformat_preserves_outputs_and_separates_review_from_origin(
    saved_run, tmp_path
):
    """Saved metrics/attempts stay intact; refreshed report and CSV omit old taxonomies."""
    path, row, _ = saved_run
    before = {file.name: file.read_bytes() for file in path.iterdir()}
    output = tmp_path / "simple"
    summary = export_saved_run(path, output)
    assert {file.name: file.read_bytes() for file in path.iterdir()} == before
    assert (output / "attempts.jsonl").read_bytes() == before["attempts.jsonl"]
    assert summary["overall"]["metrics"]["token_f1"] == {"mean": 0.0, "n": 1}
    assert summary["by_origin"]["official_faq"]["n"] == 1
    assert summary["by_review_status"]["needs_review"]["n"] == 1
    assert not {"gold", "silver", "by_category", "by_difficulty"}.intersection(summary)
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "not independent regulation-based" in report
    assert "No model or retrieval was rerun" in report
    assert all(
        word not in report.lower()
        for word in ("gold", "silver", "difficulty", "category")
    )
    assert "**Actual answer**\n\nRed" in (output / "answers.md").read_text(
        encoding="utf-8"
    )
    with (output / "results.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        records = list(reader)
        assert not {"tier", "category", "difficulty"}.intersection(reader.fieldnames)
    assert (
        records[0]["origin"] == "Official FAQ"
        and records[0]["review_status"] == "Needs review"
    )
    with pytest.raises(FileExistsError):
        export_saved_run(path, output)


@pytest.mark.parametrize("change", ["question", "duplicate", "selection"])
def test_inconsistent_saved_run_fails_before_writing(saved_run, tmp_path, change):
    """Do not give corrupted saved attempts a fresh, apparently valid presentation."""
    path, row, summary = saved_run
    if change == "question":
        row["question"] = "Changed question"
    elif change == "selection":
        summary["metadata"]["selected_ids"] = []
        (path / "summary.json").write_text(json.dumps(summary))
    (path / "attempts.jsonl").write_text(
        json.dumps(row)
        + "\n"
        + (json.dumps(row) + "\n" if change == "duplicate" else "")
    )
    with pytest.raises(ValueError):
        export_saved_run(path, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_reviewed_faq_keeps_origin_and_empty_metrics_are_na(saved_run):
    """Review status does not turn official material into document-derived examples."""
    _, row, saved = saved_run
    other = copy.deepcopy(row)
    other.update(id="reviewed", review_status="approved")
    summary = {"metadata": saved["metadata"], **result_groups([row, other])}
    assert summary["by_origin"]["official_faq"]["n"] == 2
    assert summary["by_review_status"]["reviewed"]["n"] == 1
    report = render_report(summary, [row, other])
    assert "References reviewed: **1/2**" in report
    assert "N/A" in report


def test_readable_review_context_cannot_be_changed_silently():
    """Approvals must refer to the same question and answer the person was shown."""
    item = ControlledItem(
        id="fixture",
        question="Which invented answer?",
        expected_answer="No information in the fixture.",
        answerable=False,
        expected_status="no_evidence",
        generation_method="manual",
        family_id="fixture",
    )
    dataset = ControlledDataset(
        name="fixture",
        corpus_id="fixture",
        corpus_sha256="fixture",
        generation_description="Unit fixture",
        items=[item],
    )
    assert item.category == "unspecified"
    decisions = review_template(dataset)
    assert decisions[0]["question"] == item.question
    assert decisions[0]["origin"] == "Created from documents"
    view = render_questions([item.model_dump(mode="json")])
    assert "Needs review" in view and "No supporting passage" in view
    item.review_notes = "Automatic preparation note about difficulty and category"
    assert "Automatic preparation note" not in render_questions(
        [item.model_dump(mode="json")]
    )
    item.reviewer_code = "human-unit-fixture"
    assert "Automatic preparation note" in render_questions(
        [item.model_dump(mode="json")]
    )
    decisions[0]["expected_answer"] = "Different answer"
    with pytest.raises(ValueError, match="context was edited"):
        apply_reviews(dataset, decisions)


def test_cli_exposes_plain_filters_and_can_reformat(saved_run, tmp_path):
    """Simple commands are discoverable, with old tier flags retained only for compatibility."""
    help_text = parser().format_help()
    assert "make-questions" in help_text and "make-silver" not in help_text
    path, _, _ = saved_run
    assert (
        main(["report", "--run", str(path), "--output", str(tmp_path / "from-cli")])
        == 0
    )
