"""Offline regression coverage for scoped article expansion and safe diagnostics."""

import socket
from types import SimpleNamespace
from uuid import uuid4

import httpx
import mongomock
import pytest

from src.knowledge_base.models import EvidenceChunk, SourceSpec
from src.knowledge_base.prepare import stable_id
from src.knowledge_base.vector_store import MongoEvidenceStore, SearchHit
from src.rag.answers import Question, SimulatedGenerator
from src.rag.diagnostics import LiveAnswerFailure
from src.rag.live import LiveAnswerService, SemanticRetriever, Usage


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests must not call databases or cloud endpoints."""

    def fail(*args, **kwargs):
        """Block every outgoing socket, including loopback."""
        raise AssertionError("No network allowed")

    monkeypatch.setattr(socket.socket, "connect", fail)


def hit(
    article="7",
    page=2,
    start=0,
    *,
    snapshot="snapshot",
    source_id="source",
    version="version",
):
    """Create a standalone synthetic passage, never an actual university rule."""
    text = (
        "Synthetic fixture sentence with enough words to be a substantive passage for the retrieval filter. "
        * 2
    )
    chunk_id = stable_id(
        source_id, version, article or "heading", str(page), str(start)
    )
    chunk = EvidenceChunk(
        id=chunk_id,
        source_id=source_id,
        version=version,
        page=page,
        start=start,
        end=start + len(text),
        article=article,
        text=text,
    )
    source = SourceSpec(
        key=source_id,
        title="Unit fixture",
        url="https://example.org/rules.pdf",
        local_path="unused",
    )
    return SearchHit(
        rank=1,
        score=0.7,
        corpus_id="corpus",
        snapshot_id=snapshot,
        point_id=stable_id(snapshot, chunk_id),
        chunk=chunk,
        source=source,
        citation_url=f"https://example.org/rules.pdf#page={page}",
    )


def store_hits(store, hits):
    """Insert isolated test records using the existing Mongo evidence schema."""
    for item in hits:
        store.chunks.insert_one(
            {
                "_id": item.point_id,
                "corpus_id": item.corpus_id,
                "snapshot_id": item.snapshot_id,
                "chunk": item.chunk.model_dump(),
                "source": item.source.model_dump(mode="json"),
            }
        )


def test_neighbors_stay_in_exact_article_source_version_and_snapshot():
    """Same article numbers in other documents or snapshots cannot contaminate context."""
    store = MongoEvidenceStore(mongomock.MongoClient()["mag_rag_test"])
    seed, previous, following = hit(), hit(page=1), hit(page=2, start=300)
    distractors = [
        hit(article="8", start=200),
        hit(start=200, snapshot="other"),
        hit(start=200, source_id="other"),
        hit(start=200, version="old"),
    ]
    store_hits(store, [seed, previous, following, *distractors])
    neighbors = store.article_neighbors(seed)
    assert [item.chunk.id for item in neighbors] == [
        previous.chunk.id,
        following.chunk.id,
    ]
    assert all(
        item.score is None and item.retrieval_origin == "article_neighbor"
        for item in neighbors
    )
    assert all(item.seed_chunk_id == seed.chunk.id for item in neighbors)
    assert neighbors[0].citation_url.endswith("#page=1")
    assert store.article_neighbors(hit(article=None)) == []
    # Publishing another pointer must not move this read off the captured snapshot.
    store.active.insert_one({"_id": "corpus", "snapshot_id": "other"})
    assert store.article_neighbors(seed) == neighbors


def test_one_neighbor_within_budget_and_one_query():
    """A split rule is available to generation without an extra query or more context slots."""
    store = MongoEvidenceStore(mongomock.MongoClient()["mag_rag_test"])
    primary = [hit(article=str(i)) for i in range(1, 6)]
    neighbor = hit(article="2", page=1)
    store_hits(store, [*primary, neighbor])
    calls = []

    def search(corpus_id, question, limit):
        """Return a fixed ranked result set instead of embedding the question."""
        calls.append((corpus_id, question, limit))
        return primary

    retriever = SemanticRetriever(SimpleNamespace(store=store, search=search), "corpus")
    results = retriever.search("question", 5)
    assert len(calls) == 1 and calls[0][2] == 15
    assert [item.chunk.id for item in results] == [
        primary[0].chunk.id,
        primary[1].chunk.id,
        neighbor.chunk.id,
        primary[2].chunk.id,
        primary[3].chunk.id,
    ]
    assert [item.rank for item in results] == [1, 2, 3, 4, 5]
    assert sum(item.retrieval_origin == "article_neighbor" for item in results) == 1
    assert len({item.chunk.id for item in results}) == 5


def test_already_ranked_neighbor_is_not_duplicated_or_promoted():
    """Preserve existing dense results when the companion is already selected."""
    store = MongoEvidenceStore(mongomock.MongoClient()["mag_rag_test"])
    primary = [hit(), hit(page=1)]
    store_hits(store, primary)
    retriever = SemanticRetriever(
        SimpleNamespace(store=store, search=lambda *args, **kwargs: primary), "corpus"
    )
    results = retriever.search("question", 5)
    assert [item.chunk.id for item in results] == [item.chunk.id for item in primary]
    assert all(item.retrieval_origin == "ranked" for item in results)
    assert len(retriever.search("question", 1)) == 1


@pytest.mark.parametrize(
    "status,category",
    [
        (401, "authentication"),
        (403, "permission"),
        (429, "rate_limit"),
        (500, "provider_server"),
        (400, "provider_request"),
    ],
)
def test_http_diagnostics_retain_only_safe_metadata(status, category):
    """Raw error text, credentials, arbitrary headers and provider codes are discarded."""
    usage = Usage()
    request_id = str(uuid4())
    request = httpx.Request(
        "POST", "https://example.org/chat/completions", headers={"api-key": "SECRET"}
    )
    response = httpx.Response(
        status,
        request=request,
        headers={"apim-request-id": request_id, "authorization": "SECRET"},
        json={"error": {"message": "SECRET prompt text", "code": "SECRET"}},
    )
    usage.request(request)
    usage.response(response)
    assert usage.last_failure.model_dump() == {
        "category": category,
        "http_status": status,
        "request_id": request_id,
        "operation": "chat",
    }
    assert "SECRET" not in str(usage.__dict__)
    usage.begin_question()
    assert usage.last_failure is None and usage.operation is None
    assert usage.counters["failed_requests"] == 1
    assert usage.counters["chat_requests"] == 1


def test_unknown_request_id_is_not_logged_and_non_json_body_is_safe():
    """Reject arbitrary header content rather than trying to redact it heuristically."""
    usage = Usage()
    request = httpx.Request("POST", "https://example.org/embeddings")
    usage.request(request)
    usage.response(
        httpx.Response(
            502,
            request=request,
            headers={"x-request-id": "SECRET key"},
            text="SECRET html error page",
        )
    )
    assert usage.last_failure.request_id is None
    assert usage.last_failure.operation == "embedding"
    assert "SECRET" not in str(usage.__dict__)


def test_transport_failure_and_stale_failure_do_not_leak(monkeypatch):
    """Keep timeout category, discard private error text and clear previous request metadata."""
    usage = Usage()

    class Retriever:
        """Raise a local fake timeout at the model boundary."""

        def search(self, question, limit):
            """Record an attempted embedding without creating an HTTP connection."""
            usage.request(httpx.Request("POST", "https://example.org/embeddings"))
            raise TimeoutError("SECRET credential and prompt")

    service = LiveAnswerService(Retriever(), SimulatedGenerator(), usage, 1)
    with pytest.raises(LiveAnswerFailure) as error:
        service.answer(Question(question="question"))
    assert error.value.diagnostic.category == "timeout"
    assert error.value.diagnostic.http_status is None
    assert error.value.diagnostic.operation == "embedding"
    assert "SECRET" not in str(error.value)
    assert service.session_status()["questions_remaining"] == 0
