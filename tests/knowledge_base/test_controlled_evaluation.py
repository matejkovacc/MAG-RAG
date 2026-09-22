"""Offline regressions for controlled QA, human approval and experiment integrity."""

import json
import socket
from pathlib import Path

import pytest

from src.evaluation.__main__ import main
from src.evaluation.controlled import (
    CHECKS,
    ControlledDataset,
    ControlledItem,
    apply_reviews,
    audit_dataset,
    make_silver,
    passage,
    review_template,
)
from src.evaluation.controlled_runner import (
    RATING_FIELDS,
    run_controlled,
    score_human_reviews,
)
from src.evaluation.metrics import answer_scores, retrieval_scores, text_metrics
from src.evaluation.models import corpus_digest, load_benchmark
from src.knowledge_base.catalog import prepare_text
from src.knowledge_base.models import PreparedCorpus, SourceSpec
from src.rag.answers import Answer, AnswerPart
from src.rag.offline import offline_service


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Evaluation must not reach external services, even through fake-live paths."""

    def fail(*args, **kwargs):
        """Fail immediately on attempted network access."""
        raise AssertionError("No network in controlled evaluation tests")

    monkeypatch.setattr(socket.socket, "connect", fail)


@pytest.fixture
def fixture_set(tmp_path):
    """Use invented unit-test prose, never fabricated university regulations."""
    text = "Testni dokument opisuje poskusni obrazec. Poskusni obrazec je moder in se hrani v testni mapi. To ni fakultetni pravilnik."
    doc = prepare_text(
        SourceSpec(
            key="fixture", title="Invented test fixture", local_path="fixture.txt"
        ),
        "fixture",
        text.encode(),
        "txt",
        [],
    )
    corpus = PreparedCorpus(corpus_id="fixture", documents=[doc])
    ref = passage(doc, doc.chunks[0])
    item = ControlledItem(
        id="one",
        question="Kakšne barve je poskusni obrazec?",
        expected_answer="Poskusni obrazec je moder",
        source_ids=[ref.source_id],
        supporting_passages=[ref],
        category="direct",
        answerable=True,
        expected_status="evidence_found",
        generation_method="synthetic",
        family_id="colour",
    )
    refusal = ControlledItem(
        id="two",
        question="Kdo je zmagal na olimpijskih igrah?",
        expected_answer="Podatka v testnem dokumentu ni.",
        category="out_of_scope",
        answerable=False,
        expected_status="no_evidence",
        generation_method="manual",
        family_id="sport",
    )
    dataset = ControlledDataset(
        name="unit fixture",
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus_digest(corpus),
        generation_description="Invented test fixture",
        items=[item, refusal],
    )
    (tmp_path / "corpus.json").write_text(corpus.model_dump_json(), encoding="utf-8")
    (tmp_path / "dataset.json").write_text(dataset.model_dump_json(), encoding="utf-8")
    return corpus, dataset


def test_lexical_and_retrieval_arithmetic():
    """Repeated tokens, reordered answers, alternatives and missing hits have known scores."""
    assert text_metrics("ČLEN, MODER!", "člen moder") == {
        "exact_match": 1,
        "token_f1": 1,
        "rouge_l": 1,
    }
    assert text_metrics("a a b", "a b b")["token_f1"] == pytest.approx(2 / 3)
    assert text_metrics("a b", "b a") == {
        "exact_match": 0,
        "token_f1": 1,
        "rouge_l": 0.5,
    }
    assert all(value == 0 for value in text_metrics("", "moder").values())
    with pytest.raises(ValueError):
        text_metrics("answer", "!!!")
    scores = retrieval_scores(
        ["a", "alternative", "b"],
        ["noise", "alternative", "other", "b"],
        [["a", "alternative"], ["b"]],
    )
    assert scores == {
        "hit_rate_at_1": 0,
        "recall_at_1": 0,
        "hit_rate_at_3": 1,
        "recall_at_3": 0.5,
        "hit_rate_at_5": 1,
        "recall_at_5": 1,
        "reciprocal_rank": 0.5,
        "any_correct_source": 1,
    }
    assert all(value is None for value in retrieval_scores([], []).values())
    assert retrieval_scores(["a"], ["z"])["reciprocal_rank"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval_scores(["a"], ["a", "a"])


def test_answer_metrics_and_refusals(fixture_set):
    """Citation correctness is an ID proxy; a clarification is not a correct refusal."""
    _, dataset = fixture_set
    item, refusal = dataset.items
    answer = Answer(
        status="evidence_found",
        message="",
        parts=[
            AnswerPart(
                text=item.expected_answer, citation_ids=[item.source_ids[0], "wrong"]
            )
        ],
    )
    scores = answer_scores(item, answer)
    assert scores["exact_match"] == scores["token_f1"] == scores["rouge_l"] == 1
    assert scores["cited_source_precision"] == 0.5
    assert (
        scores["required_source_citation_recall"]
        == scores["answer_with_correct_source"]
        == 1
    )
    assert scores["unanswerable_detection"] is None
    refused = Answer(status="no_evidence", message="Not enough evidence")
    assert answer_scores(refusal, refused)["unanswerable_detection"] == 1
    assert answer_scores(refusal, refused)["token_f1"] is None
    assert answer_scores(item, refused)["false_refusal"] == 1
    assert answer_scores(refusal, None)["unanswerable_detection"] == 0
    clarification = Answer(status="needs_clarification", message="Which document?")
    assert answer_scores(refusal, clarification)["unanswerable_detection"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["empty", "missing", "duplicate_id", "duplicate_question", "split", "groups"],
)
def test_invalid_schema(fixture_set, mutation):
    """Reject incomplete cases, duplicate identity and family leakage before providers."""
    _, dataset = fixture_set
    data = dataset.model_dump()
    if mutation == "empty":
        data["items"][0]["expected_answer"] = "  "
    elif mutation == "missing":
        data["items"][0]["supporting_passages"] = []
    elif mutation == "duplicate_id":
        data["items"][1]["id"] = "one"
    elif mutation == "duplicate_question":
        data["items"][1]["question"] = data["items"][0]["question"].upper()
    elif mutation == "groups":
        data["items"][0]["required_source_groups"] = [["unknown"]]
    else:
        data["items"][1].update(family_id="colour", split="test")
    with pytest.raises(ValueError):
        ControlledDataset.model_validate(data)


def test_audit_missing_altered_unsupported_and_leaked(fixture_set):
    """Exact provenance and obvious numeric inconsistencies fail; semantics stay explicit."""
    corpus, dataset = fixture_set
    assert audit_dataset(dataset, corpus)["valid"]
    changed = dataset.model_copy(deep=True)
    changed.items[0].supporting_passages[0].text = "tampered"
    assert not audit_dataset(changed, corpus)["valid"]
    changed = dataset.model_copy(deep=True)
    changed.items[0].expected_answer = "Deadline is 999 days"
    assert not audit_dataset(changed, corpus)["valid"]
    changed.items[0].expected_answer = "Something not entailed"
    audit = audit_dataset(changed, corpus)
    assert audit["valid"] and any(
        "entailment" in warning for warning in audit["warnings"]
    )
    changed.items[0].question = "Poskusni obrazec je moder"
    assert any(
        "knowledge base" in error for error in audit_dataset(changed, corpus)["errors"]
    )
    changed = dataset.model_copy(deep=True)
    changed.corpus_sha256 = "wrong"
    assert not audit_dataset(changed, corpus)["valid"]
    changed = dataset.model_copy(deep=True)
    changed.items[1].question = "Kakšne barve je poskusni obrazec danes?"
    assert any(
        "Near-duplicate" in error for error in audit_dataset(changed, corpus)["errors"]
    )


def test_review_binding_and_gold(fixture_set):
    """Gold requires five explicit human decisions and stale approvals fail closed."""
    _, dataset = fixture_set
    decisions = review_template(dataset)
    decisions[0].update(review_status="approved", reviewer_code="human-unit-fixture")
    with pytest.raises(ValueError, match="five"):
        apply_reviews(dataset, decisions)
    decisions[0]["checks"] = {key: True for key in CHECKS}
    gold = apply_reviews(dataset, decisions)
    assert gold.items[0].review_status == "approved"
    assert apply_reviews(gold, review_template(gold)).items[0] == gold.items[0]
    assert dataset.items[0].review_status == "unreviewed"
    data = gold.model_dump()
    data["items"][0]["expected_answer"] = "Changed reference"
    with pytest.raises(ValueError, match="stale"):
        ControlledDataset.model_validate(data)
    decisions[0]["content_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different"):
        apply_reviews(dataset, decisions)


def test_runner_separates_tiers_single_retrieval_and_human_ratings(
    fixture_set, tmp_path
):
    """One real retrieval per question, safe saved artifacts and reference-free calls."""
    corpus, dataset = fixture_set
    reviews = review_template(dataset)
    reviews[0].update(
        review_status="approved",
        reviewer_code="human-unit-fixture",
        checks={key: True for key in CHECKS},
    )
    dataset = apply_reviews(dataset, reviews)
    calls = []
    with offline_service(corpus) as service:
        original = service.retriever
        search = original.search

        def capture(question, limit):
            """Record only the user question supplied to retrieval."""
            calls.append(question)
            return search(question, limit)

        original.search = capture
        summary = run_controlled(service, dataset, dataset.items, tmp_path / "run", {})
        assert service.retriever is original
        assert calls == [item.question for item in dataset.items]
        assert (
            summary["by_review_status"]["reviewed"]["n"]
            == summary["by_review_status"]["needs_review"]["n"]
            == 1
        )
        assert summary["answerable"]["n"] == summary["unanswerable"]["n"] == 1
        assert summary["usage"] == {"embedding_requests": 0, "chat_requests": 0}
        with pytest.raises(FileExistsError):
            run_controlled(service, dataset, dataset.items, tmp_path / "run", {})
    saved = tmp_path / "run"
    assert all(
        (saved / name).exists()
        for name in [
            "dataset.json",
            "metadata.json",
            "attempts.jsonl",
            "summary.json",
            "results.csv",
            "report.md",
            "answer-review.json",
        ]
    )
    assert score_human_reviews(saved, saved / "answer-review.json")["reviewed_n"] == 0
    ratings = json.loads((saved / "answer-review.json").read_text())
    ratings[0].update(
        reviewer_code="human-unit-fixture",
        **{key: 4 for key in RATING_FIELDS},
        tags=["incomplete"],
    )
    review_path = tmp_path / "ratings.json"
    review_path.write_text(json.dumps(ratings))
    scored = score_human_reviews(saved, review_path)
    assert scored["reviewed_n"] == 1
    assert scored["by_review_status"]["reviewed"]["scores"]["correctness"] == 4
    ratings[0]["correctness"] = 6
    review_path.write_text(json.dumps(ratings))
    with pytest.raises(ValueError, match="integers"):
        score_human_reviews(saved, review_path)
    ratings[0]["response_sha256"] = "stale"
    review_path.write_text(json.dumps(ratings))
    with pytest.raises(ValueError, match="stale"):
        score_human_reviews(saved, review_path)


def test_provider_failure_stops_and_sanitizes(fixture_set, tmp_path):
    """Save failed attempts without provider messages, retries or fabricated scores."""
    corpus, dataset = fixture_set
    with offline_service(corpus) as service:

        def fail(question, evidence):
            """Mimic a provider exception containing sensitive context."""
            raise RuntimeError("SECRET-PROVIDER-TEXT")

        service.generator.generate = fail
        summary = run_controlled(
            service, dataset, dataset.items, tmp_path / "failure", {}
        )
    assert summary["overall"]["errors"] == 1
    assert summary["unattempted_ids"] == ["two"]
    assert "reviewed" not in summary["by_review_status"]
    assert (
        "SECRET-PROVIDER-TEXT"
        not in (tmp_path / "failure" / "attempts.jsonl").read_text()
    )


def test_cli_guards_and_old_corpus_compatibility(fixture_set, tmp_path):
    """Approval and held-out review gates happen before any model provider is loaded."""
    args = [
        "evaluate",
        "--dataset",
        str(tmp_path / "dataset.json"),
        "--corpus",
        str(tmp_path / "corpus.json"),
        "--output",
        str(tmp_path / "never"),
    ]
    with pytest.raises(SystemExit):
        main(args)
    with pytest.raises(SystemExit):
        main(args + ["--mode", "live", "--allow-unreviewed"])
    assert not (tmp_path / "never").exists()
    corpus_path = Path("data/thesis/corpus.json")
    if not corpus_path.exists():
        pytest.skip("Public-source corpus is intentionally not distributed")
    corpus = PreparedCorpus.model_validate_json(corpus_path.read_text(encoding="utf-8"))
    old = load_benchmark(Path("evaluation/pilot.json"), corpus)
    imported = make_silver(corpus, legacy=Path("evaluation/pilot.json"))
    assert len(old.items) == len(imported.items) == 24
    assert imported.items[17].required_source_groups == old.items[17].evidence_groups
    assert audit_dataset(imported, corpus)["valid"]
