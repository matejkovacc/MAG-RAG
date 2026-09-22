"""Real local Qdrant tests with synthetic vectors and a MongoDB test double."""

import hashlib
from datetime import datetime, timedelta, timezone

import mongomock
import pytest
from qdrant_client import QdrantClient, models

from src.knowledge_base.models import (
    PageText,
    PreparedCorpus,
    PreparedDocument,
    SourceSpec,
)
from src.knowledge_base.prepare import split_pages, stable_id
from src.knowledge_base.vector_store import (
    EmbeddingProfile,
    MongoEvidenceStore,
    RetrievalError,
    ThesisVectorIndex,
)


class SyntheticEmbedder:
    """Deterministic three-dimensional vectors; never evidence of semantic quality."""

    def __init__(self) -> None:
        """Track calls to verify idempotency and validation before embedding."""
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Assign controlled directions to synthetic fixture words."""
        self.calls += 1
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a known direction for deterministic ranking tests."""
        if "alpha" in text:
            return [1.0, 0.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def corpus(
    corpus_id: str = "test", texts: tuple[str, ...] = ("beta rule", "alpha rule")
) -> PreparedCorpus:
    """Create synthetic sources with exact page locations and distinct source IDs."""
    documents = []
    for index, text in enumerate(texts):
        key = f"source-{index}"
        source_id = stable_id(corpus_id, key)
        content = f"1. člen\n{text}"
        version = hashlib.sha256(content.encode()).hexdigest()
        pages = [PageText(page=1, text=content)]
        documents.append(
            PreparedDocument(
                source_id=source_id,
                source=SourceSpec(
                    key=key,
                    title=key,
                    url=f"https://example.org/{key}.pdf",
                    local_path=f"{key}.pdf",
                    notes="Synthetic test fixture",
                ),
                content_sha256=version,
                extraction_version="synthetic",
                chunking_version="test",
                prepared_at=datetime.now(timezone.utc),
                pages=pages,
                chunks=split_pages(pages, source_id, version, []),
            )
        )
    return PreparedCorpus(corpus_id=corpus_id, documents=documents)


@pytest.fixture
def index():
    """Isolate each test from all real databases and cloud services."""
    vectors = QdrantClient(":memory:")
    database = mongomock.MongoClient()["mag_rag_test"]
    profile = EmbeddingProfile(
        provider="synthetic-test",
        model="fixture",
        deployment="fixture",
        endpoint="local",
        api_version="test",
        dimensions=3,
    )
    service = ThesisVectorIndex(
        MongoEvidenceStore(database), vectors, SyntheticEmbedder(), profile
    )
    yield service
    vectors.close()


def test_legacy_thesis_store_reuses_snapshot_without_reembedding(index) -> None:
    """A renamed application can read the original thesis snapshot unchanged."""
    index.store = MongoEvidenceStore(mongomock.MongoClient()["finrep_thesis"])
    legacy = ThesisVectorIndex(
        index.store,
        index.vectors,
        index.embedder,
        index.profile,
        "finrep_thesis_chunks",
    )
    prepared = corpus()
    result = legacy.index(prepared)
    calls = index.embedder.calls
    reopened = ThesisVectorIndex(
        legacy.store,
        legacy.vectors,
        index.embedder,
        index.profile,
        "finrep_thesis_chunks",
    )
    hits = reopened.search(prepared.corpus_id, "alpha", limit=1)
    assert hits[0].snapshot_id == result["snapshot_id"]
    assert "alpha rule" in hits[0].chunk.text
    assert index.embedder.calls == calls


@pytest.mark.parametrize("name", ["finrep", "finrep_thesis_unrelated"])
def test_legacy_compatibility_still_rejects_unrelated_storage(index, name) -> None:
    """Legacy compatibility is an exact exception, not a relaxed prefix guard."""
    with pytest.raises(RetrievalError):
        MongoEvidenceStore(mongomock.MongoClient()[name])
    with pytest.raises(RetrievalError):
        ThesisVectorIndex(
            index.store, index.vectors, index.embedder, index.profile, name
        )


def test_index_preserves_shared_ids_sources_and_vector_rank(
    index: ThesisVectorIndex,
) -> None:
    """Mongo insertion order differs from vector rank but citations remain correct."""
    prepared = corpus()
    result = index.index(prepared)
    hits = index.search("test", "alpha", limit=2)
    assert result["status"] == "indexed"
    assert [hit.chunk.text for hit in hits] == [
        "1. člen\nalpha rule",
        "1. člen\nbeta rule",
    ]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score > hits[1].score
    assert hits[0].citation_url == "https://example.org/source-1.pdf#page=1"
    assert hits[0].chunk.article == "1"
    assert hits[0].source.review_status == "pending"
    assert (
        index.store.chunks.find_one({"_id": hits[0].point_id})["chunk"]["id"]
        == hits[0].chunk.id
    )
    assert index.store.pages.count_documents({}) == 2
    assert index.store.sources.count_documents({}) == 2


def test_repeat_index_is_idempotent_ignoring_timestamps(
    index: ThesisVectorIndex,
) -> None:
    """Unchanged evidence skips embeddings and does not duplicate stored records."""
    prepared = corpus()
    first = index.index(prepared)
    for document in prepared.documents:
        document.prepared_at += timedelta(days=1)
    prepared.documents.reverse()
    repeated = index.index(prepared)
    assert repeated["status"] == "unchanged"
    assert repeated["snapshot_id"] == first["snapshot_id"]
    assert index.embedder.calls == 1
    assert index.store.chunks.count_documents({}) == 2
    assert index.vectors.count(index.collection).count == 2


def test_updated_and_other_corpora_are_isolated(index: ThesisVectorIndex) -> None:
    """Both previous versions and other corpora are excluded from active queries."""
    old = index.index(corpus())
    index.index(corpus("other", ("alpha outside corpus",)))
    new = index.index(corpus(texts=("alpha updated",)))
    hits = index.search("test", "alpha", limit=10)
    assert len(hits) == 1
    assert hits[0].chunk.text.endswith("alpha updated")
    assert hits[0].snapshot_id == new["snapshot_id"] != old["snapshot_id"]
    assert index.store.chunks.count_documents({}) == 4


def test_embedding_configuration_mismatch_requires_reindex(
    index: ThesisVectorIndex,
) -> None:
    """Same dimensionality does not imply two deployments share a vector space."""
    index.index(corpus())
    index.profile = index.profile.model_copy(update={"deployment": "different"})
    with pytest.raises(RetrievalError, match="configuration differs"):
        index.search("test", "alpha")
    assert index.index(corpus())["status"] == "indexed"
    assert index.search("test", "alpha")[0].rank == 1


def test_incompatible_collection_is_not_recreated(index: ThesisVectorIndex) -> None:
    """Wrong-sized existing collections fail before embedding or deleting data."""
    index.vectors.create_collection(
        index.collection,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    with pytest.raises(RetrievalError, match="incompatible"):
        index.index(corpus())
    assert index.embedder.calls == 0
    assert (
        index.vectors.get_collection(index.collection).config.params.vectors.size == 2
    )


@pytest.mark.parametrize(
    "bad_vectors",
    [
        [],
        [[1.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[float("nan"), 1.0, 0.0]],
        [[float("inf"), 0.0, 1.0]],
    ],
)
def test_bad_embeddings_cannot_publish(
    index: ThesisVectorIndex, monkeypatch, bad_vectors: list[list[float]]
) -> None:
    """Count/dimension/numeric errors never publish an incomplete corpus."""
    monkeypatch.setattr(index.embedder, "embed_documents", lambda texts: bad_vectors)
    with pytest.raises(RetrievalError):
        index.index(corpus(texts=("alpha",)))
    assert index.store.get_active("test") is None


def test_mongo_failure_leaves_old_snapshot_searchable(
    index: ThesisVectorIndex, monkeypatch
) -> None:
    """New vectors alone cannot become visible if MongoDB staging fails."""
    old = index.index(corpus())

    def fail(*args):
        """Simulate a database failure before publication."""
        raise RuntimeError("simulated MongoDB outage")

    monkeypatch.setattr(index.store, "stage", fail)
    with pytest.raises(RuntimeError, match="outage"):
        index.index(corpus(texts=("alpha updated",)))
    assert index.store.get_active("test")["snapshot_id"] == old["snapshot_id"]
    assert index.search("test", "alpha")[0].chunk.text.endswith("alpha rule")


def test_partial_vector_failure_never_publishes(
    index: ThesisVectorIndex, monkeypatch
) -> None:
    """A failure after the first batch leaves only invisible staged vectors."""
    original = index.vectors.upsert
    calls = 0

    def fail_second(*args, **kwargs):
        """Commit the first batch and fail the second."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated Qdrant outage")
        return original(*args, **kwargs)

    monkeypatch.setattr(index.vectors, "upsert", fail_second)
    with pytest.raises(RuntimeError, match="outage"):
        index.index(corpus(), batch_size=1)
    assert index.vectors.count(index.collection).count == 1
    assert index.store.get_active("test") is None


def test_missing_hydrated_evidence_fails_explicitly(index: ThesisVectorIndex) -> None:
    """Missing MongoDB records cannot silently remove or fabricate citations."""
    index.index(corpus())
    index.store.chunks.delete_many({})
    with pytest.raises(RetrievalError, match="missing MongoDB evidence"):
        index.search("test", "alpha")


def test_reindex_repairs_incomplete_snapshot(index: ThesisVectorIndex) -> None:
    """An identical input must be rebuilt if published records have been lost."""
    old = index.index(corpus())
    index.store.pages.delete_many({})
    repaired = index.index(corpus())
    assert repaired["status"] == "indexed"
    assert repaired["snapshot_id"] != old["snapshot_id"]
    assert index.search("test", "alpha")[0].chunk.text.endswith("alpha rule")


def test_query_uses_one_snapshot_during_concurrent_publication(
    index: ThesisVectorIndex, monkeypatch
) -> None:
    """A pointer change during hydration does not mix versions within one response."""
    old = index.index(corpus())
    hydrate = index.store.hydrate

    def publish_then_hydrate(corpus_id, snapshot_id, ids):
        """Publish an update between vector search and Mongo hydration."""
        index.index(corpus(texts=("alpha updated",)))
        return hydrate(corpus_id, snapshot_id, ids)

    monkeypatch.setattr(index.store, "hydrate", publish_then_hydrate)
    hits = index.search("test", "alpha")
    assert {hit.snapshot_id for hit in hits} == {old["snapshot_id"]}
    assert hits[0].chunk.text.endswith("alpha rule")


@pytest.mark.parametrize(
    "corpus_id,query,limit",
    [("", "alpha", 1), ("test", " ", 1), ("test", "alpha", 0), ("test", "alpha", 101)],
)
def test_invalid_search_parameters(
    index: ThesisVectorIndex, corpus_id: str, query: str, limit: int
) -> None:
    """Invalid user input fails before model or vector requests."""
    with pytest.raises(RetrievalError):
        index.search(corpus_id, query, limit)


def test_unknown_corpus_and_non_project_names_are_rejected(
    index: ThesisVectorIndex,
) -> None:
    """Missing scope cannot fall back to another database or vector collection."""
    with pytest.raises(RetrievalError, match="No published"):
        index.search("missing", "alpha")
    with pytest.raises(RetrievalError):
        MongoEvidenceStore(mongomock.MongoClient()["other"])
    with pytest.raises(RetrievalError):
        ThesisVectorIndex(
            index.store, index.vectors, index.embedder, index.profile, "other"
        )


def test_qdrant_disk_store_survives_reopen(index: ThesisVectorIndex, tmp_path) -> None:
    """Exercise actual local Qdrant persistence, not a mock vector API."""
    vectors = QdrantClient(path=str(tmp_path / "vectors"))
    index.vectors = vectors
    indexed = index.index(corpus())
    vectors.close()
    reopened = QdrantClient(path=str(tmp_path / "vectors"))
    index.vectors = reopened
    try:
        assert index.search("test", "alpha")[0].snapshot_id == indexed["snapshot_id"]
    finally:
        reopened.close()
