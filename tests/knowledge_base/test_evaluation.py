"""Evaluation arithmetic, leakage prevention, provenance and no-network regressions."""

import hashlib
import json
import socket
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from src.evaluation.__main__ import main
from src.evaluation.models import (
    Benchmark,
    BenchmarkItem,
    EvidenceReference,
    corpus_digest,
    load_benchmark,
)
from src.evaluation.runner import CapturingRetriever, retrieval_metrics, run_benchmark
from src.knowledge_base.models import (
    PageText,
    PreparedCorpus,
    PreparedDocument,
    SourceSpec,
)
from src.knowledge_base.prepare import split_pages, stable_id
from src.knowledge_base.vector_store import fingerprint
from src.rag.answers import AnswerService, SimulatedGenerator
from src.rag.offline import offline_service


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """All evaluation tests, including fake-live paths, must work without sockets."""

    def fail(*args, **kwargs):
        """Reject any attempt to open an outgoing socket."""
        raise AssertionError("Network forbidden during evaluation tests")

    monkeypatch.setattr(socket.socket, "connect", fail)


@pytest.fixture
def annotated(tmp_path):
    """Invented unit-test source, unrelated to actual university requirements."""
    text = "1. člen\nSintetični primer: prijava na izpit zahteva testni obrazec."
    version = hashlib.sha256(text.encode()).hexdigest()
    source_id = stable_id("evaluation-test", "rules")
    pages = [PageText(page=1, text=text)]
    chunks = split_pages(pages, source_id, version, [])
    source = SourceSpec(
        key="fixture",
        title="Synthetic fixture",
        url="https://example.org/test.pdf",
        local_path="unused",
    )
    corpus = PreparedCorpus(
        corpus_id="evaluation-test",
        documents=[
            PreparedDocument(
                source_id=source_id,
                source=source,
                content_sha256=version,
                extraction_version="fixture",
                chunking_version="fixture",
                prepared_at=datetime.now(timezone.utc),
                pages=pages,
                chunks=chunks,
            )
        ],
    )
    ref = EvidenceReference(
        chunk_id=chunks[0].id,
        source_key=source.key,
        version=version,
        page=1,
        article="1",
        url="https://example.org/test.pdf#page=1",
        excerpt=text,
    )
    item = BenchmarkItem(
        id="test-1",
        family_id="registration",
        question="Kako poteka prijava na izpit?",
        category="direct",
        reference_answer="GOLD ANSWER MUST NEVER REACH RAG",
        expected_status="evidence_found",
        evidence_groups=[[ref.chunk_id]],
        evidence=[ref],
        review_notes="Synthetic unit fixture",
    )
    benchmark = Benchmark(
        name="unit-test",
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus_digest(corpus),
        scope="unit test",
        authorship="unit fixture",
        items=[item],
    )
    (tmp_path / "corpus.json").write_text(corpus.model_dump_json(), encoding="utf-8")
    (tmp_path / "benchmark.json").write_text(
        benchmark.model_dump_json(), encoding="utf-8"
    )
    return corpus, benchmark


def test_required_evidence_groups_and_alternatives():
    """Any alternative covers a group; multiple required passages need all groups."""
    scores = retrieval_metrics(
        [["a", "a-alternative"], ["b"]], ["noise", "a-alternative", "other", "b"]
    )
    assert scores["evidence_recall_at_1"] == 0
    assert scores["evidence_recall_at_3"] == 0.5
    assert scores["all_evidence_at_3"] == 0
    assert scores["evidence_recall_at_5"] == 1
    assert scores["reciprocal_rank_at_5"] == 0.5
    assert all(value is None for value in retrieval_metrics([], []).values())
    assert retrieval_metrics([["a"]], ["noise"])["reciprocal_rank_at_5"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval_metrics([["a"]], ["a", "a"])


def test_frozen_corpus_and_exact_citations(annotated, tmp_path):
    """Reject altered source identity, quote or corpus even when IDs still match."""
    corpus, benchmark = annotated
    path = tmp_path / "benchmark.json"
    assert load_benchmark(path, corpus) == benchmark
    data = benchmark.model_dump()
    data["items"][0]["evidence"][0]["excerpt"] = "tampered"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_benchmark(path, corpus)
    path.write_text(benchmark.model_dump_json(), encoding="utf-8")
    changed = corpus.model_copy(deep=True)
    changed.documents[0].source.notes = "Changed applicability metadata"
    with pytest.raises(ValueError, match="corpus mismatch"):
        load_benchmark(path, changed)


def test_human_review_and_family_split_validation(annotated):
    """No human-review claims without a reviewer; no paraphrases across splits."""
    _, benchmark = annotated
    data = benchmark.model_dump()
    data["items"][0]["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="reviewer code"):
        Benchmark.model_validate(data)
    data = benchmark.model_dump()
    other = {
        **data["items"][0],
        "id": "test-2",
        "question": "A different question in the same family?",
        "split": "test",
    }
    data["items"].append(other)
    with pytest.raises(ValueError, match="family"):
        Benchmark.model_validate(data)


def test_runner_retrieves_once_and_never_sends_gold(annotated, tmp_path):
    """Capture exactly the evidence supplied to the generator and preserve UTF-8."""
    corpus, benchmark = annotated
    calls = []
    with offline_service(corpus) as service:
        original = service.retriever
        delegate = original.search

        def search(question, limit):
            """Record the sole question sent to retrieval."""
            calls.append(question)
            return delegate(question, limit)

        original.search = search
        summary = run_benchmark(
            service, benchmark, benchmark.items, tmp_path / "run", {}
        )
        assert service.retriever is original
        assert calls == [benchmark.items[0].question]
        assert summary["overall"]["metrics"]["evidence_recall_at_5"] == {
            "mean": 1.0,
            "n": 1,
        }
        assert summary["overall"]["errors"] == 0
        assert summary["usage"] == {"embedding_requests": 0, "chat_requests": 0}
        assert summary["human_answer_correctness"] is None
        assert summary["ragas"] == "not run"
        with pytest.raises(FileExistsError):
            run_benchmark(service, benchmark, benchmark.items, tmp_path / "run", {})
    row = json.loads((tmp_path / "run" / "attempts.jsonl").read_text(encoding="utf-8"))
    assert row["human_review"]["correctness"] is None
    assert row["retrieved"][0]["chunk"]["text"] in row["answer"]["parts"][0]["text"]
    assert "člen" in (tmp_path / "run" / "review.md").read_text(encoding="utf-8")


def test_failure_stops_batch_and_sanitizes_error(annotated, tmp_path):
    """A failed attempt counts against status metrics and never triggers a retry."""
    corpus, benchmark = annotated
    second = benchmark.items[0].model_copy(
        update={"id": "test-2", "question": "Another question?"}
    )
    with offline_service(corpus) as service:

        class Broken:
            """Fail before obtaining evidence with a secret-like provider message."""

            def search(self, question, limit):
                """Raise a message that must never reach saved evaluation outputs."""
                raise RuntimeError("SECRET-FAKE-KEY")

        service.retriever = Broken()
        summary = run_benchmark(
            service, benchmark, [benchmark.items[0], second], tmp_path / "failed", {}
        )
    assert summary["overall"]["attempted"] == 1
    assert summary["overall"]["errors"] == 1
    assert summary["unattempted_ids"] == ["test-2"]
    assert summary["overall"]["metrics"]["status_match"]["mean"] == 0
    assert "SECRET-FAKE-KEY" not in (tmp_path / "failed" / "attempts.jsonl").read_text()


def test_snapshot_change_prevents_generation(annotated):
    """A changed snapshot must fail before the answer generator is invoked."""
    corpus, _ = annotated
    with offline_service(corpus) as service:
        recorder = CapturingRetriever(service.retriever, "different-snapshot")
        with pytest.raises(ValueError, match="snapshot changed"):
            recorder.search("prijava izpit", 5)


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--mode", "live"],
        ["run", "--allow-external-api"],
        ["run", "--limit", "101"],
    ],
)
def test_cli_gates_before_reading_configuration(args):
    """Invalid paid-call flags are rejected before corpus or credentials are read."""
    with pytest.raises(SystemExit) as exc:
        main([*args, "--corpus", "does-not-exist.json"])
    assert exc.value.code == 2


def test_cli_draft_gate_and_fake_live_snapshot_check(annotated, tmp_path, monkeypatch):
    """Live CLI validates published content and gives its batch an exact allowance."""
    from src.rag import live

    corpus, benchmark = annotated
    args = [
        "run",
        "--corpus",
        str(tmp_path / "corpus.json"),
        "--benchmark",
        str(tmp_path / "benchmark.json"),
        "--output",
        str(tmp_path / "live-run"),
    ]
    with pytest.raises(SystemExit):
        main(args)
    with offline_service(corpus) as offline:
        index = offline.retriever.index
        active = index.store.get_active(corpus.corpus_id)
        assert active["fingerprint"] == fingerprint(corpus, index.profile)
        generator = SimulatedGenerator()
        generator.deployment = "fake-chat"
        generator.prompt = "unit-test-prompt"
        service = AnswerService(
            offline.retriever, generator, mode="generated", retrieval_limit=5
        )

        @contextmanager
        def fake_service(corpus_id, *, allow_external_api, max_questions):
            """Assert CLI bounds, then inject local providers with no Azure client."""
            assert allow_external_api is True
            assert max_questions == 1
            yield service

        monkeypatch.setattr(live, "live_service", fake_service)
        assert (
            main([*args, "--mode", "live", "--allow-external-api", "--allow-draft"])
            == 0
        )
        active["fingerprint"] = "stale"
        monkeypatch.setattr(index.store, "get_active", lambda _: active)
        args[args.index("--output") + 1] = str(tmp_path / "stale-run")
        with pytest.raises(ValueError, match="Published corpus differs"):
            main([*args, "--mode", "live", "--allow-external-api", "--allow-draft"])
        assert not (tmp_path / "stale-run").exists()


def test_live_http_diagnostic_survives_evaluation_boundary(annotated, tmp_path):
    """A fake HTTP error keeps safe metadata in the artifact and stops the batch."""
    import httpx
    from src.rag.live import LiveAnswerService, Usage

    _, benchmark = annotated
    usage = Usage()

    class FailingRetriever:
        """Simulate a rejected embedding request with no socket or model call."""

        def search(self, question, limit):
            """Drive production hooks before raising a private provider exception."""
            request = httpx.Request("POST", "https://example.org/embeddings")
            usage.request(request)
            usage.response(
                httpx.Response(
                    429,
                    request=request,
                    headers={"apim-request-id": "f36f9ec3-8909-4374-8522-e0a8fbfa96f5"},
                    json={"error": {"message": "SECRET"}},
                )
            )
            raise RuntimeError("SECRET")

    service = LiveAnswerService(FailingRetriever(), SimulatedGenerator(), usage, 1)
    result = run_benchmark(
        service, benchmark, benchmark.items, tmp_path / "diagnostic", {}
    )
    text = (tmp_path / "diagnostic" / "attempts.jsonl").read_text(encoding="utf-8")
    row = json.loads(text)
    assert row["error"] == "LiveAnswerFailure"
    assert row["error_diagnostic"] == {
        "category": "rate_limit",
        "operation": "embedding",
        "http_status": 429,
        "request_id": "f36f9ec3-8909-4374-8522-e0a8fbfa96f5",
    }
    assert result["overall"]["errors"] == 1
    assert result["usage"]["embedding_requests"] == 1
    assert result["usage"]["chat_requests"] == 0
    assert "SECRET" not in text
