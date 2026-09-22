"""Official FAQ provenance, intentional overlap and separation from synthetic results."""

import hashlib
import json
import socket
from datetime import datetime, timezone

import pytest

from src.evaluation.__main__ import main
from src.evaluation.controlled import (
    CHECKS,
    ControlledDataset,
    apply_reviews,
    audit_dataset,
    evaluation_tier,
    load_controlled,
    review_template,
)
from src.evaluation.controlled_runner import run_controlled, score_human_reviews
from src.evaluation.faq import FAQ_URL, RETURN_LINK, import_faq
from src.knowledge_base.models import PreparedCorpus
from src.rag.offline import offline_service


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No source downloads, databases or model endpoints are reached by these tests."""

    def fail(*args, **kwargs):
        """Reject outgoing connections."""
        raise AssertionError("Network forbidden in FAQ tests")

    monkeypatch.setattr(socket.socket, "connect", fail)


@pytest.fixture
def faq(tmp_path):
    """Invented test prose in the real page's structural pattern, not institutional rules."""
    content = f"""<html><nav>unrelated navigation</nav><div id="katedre-container">
    <h1>Pogosta vprašanja</h1><ol><li><a href="#colour">What colour is the test form?</a></li>
    <li><a href="#location">Where is the test file?</a></li></ol>
    <h1><a id="colour"></a>What colour is the test form?</h1>
    <p>The invented test form is blue. This is a test fixture, not an actual faculty rule.</p>
    <h1><a id="location"></a>Where is the test file?</h1>
    <p>The invented test file is in the test directory. These words describe only the unit fixture.</p>
    <p>{RETURN_LINK}</p></div></html>"""
    path = tmp_path / "source.html"
    path.write_text(content, encoding="utf-8")
    stamp = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    dataset = import_faq(path, tmp_path / "bundle", stamp)
    corpus = PreparedCorpus.model_validate_json(
        (tmp_path / "bundle" / "corpus.json").read_text(encoding="utf-8")
    )
    return path, stamp, dataset, corpus


def test_import_exact_answers_skips_toc_and_does_not_self_approve(faq, tmp_path):
    """Capture each question once, preserve links, and never manufacture human review."""
    path, stamp, dataset, corpus = faq
    assert len(dataset.items) == len(corpus.documents[0].chunks) == 2
    assert all(
        item.generation_method == "official_faq" and item.review_status == "unreviewed"
        for item in dataset.items
    )
    assert (
        dataset.items[0].expected_answer
        == "The invented test form is blue. This is a test fixture, not an actual faculty rule."
    )
    assert RETURN_LINK not in dataset.items[-1].expected_answer
    assert dataset.items[0].supporting_passages[0].url == FAQ_URL + "#colour"
    assert (
        dataset.items[0].official_faq.snapshot_sha256
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert dataset.items[0].official_faq.captured_at == stamp
    loaded, audit = load_controlled(tmp_path / "bundle" / "dataset.json", corpus)
    assert loaded == dataset and audit["valid"]
    assert len(audit["warnings"]) == 2 and all(
        "not an independent" in message for message in audit["warnings"]
    )
    assert all(evaluation_tier(item) == "official_faq" for item in dataset.items)
    with pytest.raises(ValueError, match="new directory"):
        import_faq(path, tmp_path / "bundle", stamp)


@pytest.mark.parametrize(
    "change", ["answer", "question", "hash", "section", "url", "capture"]
)
def test_overlap_exception_requires_verified_published_reference(faq, change):
    """The FAQ label cannot suppress leak checks for edited or incorrectly attributed data."""
    _, _, dataset, corpus = faq
    altered = dataset.model_copy(deep=True)
    item = altered.items[0]
    if change == "answer":
        item.expected_answer = "An unsupported claim"
    elif change == "question":
        item.question = "What is the invented test form like?"
    elif change == "hash":
        item.official_faq.snapshot_sha256 = "0" * 64
    elif change == "section":
        item.official_faq.section_index = 999
    elif change == "url":
        item.official_faq.source_url = "https://example.org/faq"
    else:
        item.official_faq.captured_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert not audit_dataset(altered, corpus)["valid"]


def test_protocol_is_separate_and_existing_leak_detection_stays_strict(faq):
    """Conventional cases cannot opt into FAQ overlap without official origin/protocol."""
    _, _, dataset, corpus = faq
    data = dataset.model_dump(mode="json")
    data["protocol"] = "controlled"
    with pytest.raises(ValueError, match="separate"):
        ControlledDataset.model_validate(data)
    for item in data["items"]:
        item["generation_method"] = "manual"
        item.pop("official_faq")
    ordinary = ControlledDataset.model_validate(data)
    audit = audit_dataset(ordinary, corpus)
    assert not audit["valid"]
    assert any(
        "question appears in knowledge base" in error for error in audit["errors"]
    )


def test_faq_review_and_reports_remain_distinct(faq, tmp_path):
    """Publisher references are a third tier until a human explicitly approves them."""
    _, _, dataset, corpus = faq
    decisions = review_template(dataset)
    decisions[0].update(
        review_status="approved",
        reviewer_code="unit-fixture-reviewer",
        checks={key: True for key in CHECKS},
    )
    reviewed = apply_reviews(dataset, decisions)
    assert evaluation_tier(reviewed.items[0]) == "gold"
    assert reviewed.protocol == "official_faq_lookup"
    changed = reviewed.model_dump(mode="json")
    changed["items"][0]["official_faq"]["anchor"] = "wrong"
    with pytest.raises(ValueError, match="stale"):
        ControlledDataset.model_validate(changed)
    with offline_service(corpus) as service:
        summary = run_controlled(
            service, reviewed, reviewed.items, tmp_path / "run", {}
        )
    assert summary["by_origin"]["official_faq"]["n"] == 2
    assert summary["by_review_status"]["needs_review"]["n"] == 1
    assert summary["by_review_status"]["reviewed"]["n"] == 1
    assert "document_derived" not in summary["by_origin"]
    assert summary["usage"] == {"embedding_requests": 0, "chat_requests": 0}
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "not independent regulation-based" in report
    assert "Official FAQ" in report
    scores = score_human_reviews(
        tmp_path / "run", tmp_path / "run" / "answer-review.json"
    )
    assert scores["by_origin"] == {}


def test_cli_selects_faq_not_silver_and_keeps_api_guard(faq, tmp_path):
    """Existing evaluate command can run this set without silently calling Azure."""
    args = [
        "evaluate",
        "--dataset",
        str(tmp_path / "bundle" / "dataset.json"),
        "--corpus",
        str(tmp_path / "bundle" / "corpus.json"),
        "--allow-unreviewed",
        "--output",
        str(tmp_path / "cli-run"),
    ]
    with pytest.raises(SystemExit):
        main(args + ["--tier", "silver"])
    with pytest.raises(SystemExit):
        main(args + ["--mode", "live"])
    with pytest.raises(SystemExit):
        main(args + ["--reviewed-only"])
    with pytest.raises(SystemExit):
        main(args + ["--origin", "document_derived"])
    assert main(args + ["--origin", "official_faq"]) == 0
    summary = json.loads(
        (tmp_path / "cli-run" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["by_origin"]["official_faq"]["n"] == 2


def test_import_fails_closed_on_missing_answers(faq, tmp_path):
    """A redesigned or empty anchored answer is never assigned a reference guess."""
    path, stamp, _, _ = faq
    path.write_text(
        '<div id="katedre-container"><p>'
        + "Introductory fixture text. " * 8
        + '</p><h1><a id="empty"></a>What is missing?</h1></div>'
    )
    with pytest.raises(ValueError, match="question heading"):
        import_faq(path, tmp_path / "invalid", stamp)
    assert not (tmp_path / "invalid").exists()


def test_missing_anchored_section_cannot_silently_shrink_benchmark(faq, tmp_path):
    """The source table of contents independently checks answer extraction coverage."""
    path, stamp, _, _ = faq
    content = path.read_text(encoding="utf-8").replace('<a id="colour"></a>', "")
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="table of contents"):
        import_faq(path, tmp_path / "missing-section", stamp)
