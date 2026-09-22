"""Snapshot-based MongoDB/Qdrant indexing with ranked citation hydration."""

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field
from pymongo.database import Database
from qdrant_client import QdrantClient, models

from src.knowledge_base.models import (
    EvidenceChunk,
    PreparedCorpus,
    SourceSpec,
    citation_url,
)
from src.knowledge_base.prepare import stable_id


class RetrievalError(ValueError):
    """A corpus cannot be indexed or searched consistently."""


class EmbeddingProfile(BaseModel):
    """Non-secret identity of the vector space used for a corpus snapshot."""

    provider: str = "azure-openai"
    model: str = Field(min_length=1)
    deployment: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    api_version: str = Field(min_length=1)
    dimensions: int = Field(gt=0)


class Embedder(Protocol):
    """Small interface already implemented by LangChain embedding clients."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts in input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a query in the same vector space as the indexed documents."""
        ...


class SearchHit(BaseModel):
    """Ranked evidence with original source metadata and a physical-page link."""

    rank: int
    score: float | None
    corpus_id: str
    snapshot_id: str
    point_id: str
    chunk: EvidenceChunk
    source: SourceSpec
    citation_url: str
    retrieval_origin: Literal["ranked", "article_neighbor"] = "ranked"
    seed_chunk_id: str | None = None


def fingerprint(corpus: PreparedCorpus, profile: EmbeddingProfile) -> str:
    """Hash evidence/configuration while ignoring preparation time and source order."""
    documents = [
        doc.model_dump(mode="json", exclude={"prepared_at"}) for doc in corpus.documents
    ]
    documents.sort(key=lambda doc: doc["source_id"])
    payload = {
        "corpus_id": corpus.corpus_id,
        "documents": documents,
        "embedding": profile.model_dump(),
        "schema": 1,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_vectors(vectors: list[list[float]], count: int, dimensions: int) -> None:
    """Reject missing, wrong-sized, nonfinite and zero cosine vectors."""
    if len(vectors) != count:
        raise RetrievalError("Embedding count does not match input count")
    for vector in vectors:
        if (
            len(vector) != dimensions
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
        ):
            raise RetrievalError("Embedding dimension or numeric values are invalid")


class MongoEvidenceStore:
    """Stage immutable snapshots; publish through a single atomic pointer update."""

    def __init__(self, database: Database) -> None:
        """Accept an existing client database without opening network connections."""
        if not database.name.startswith("mag_rag") and database.name != "finrep_thesis":
            raise RetrievalError(
                "The evidence store must use mag_rag or the legacy finrep_thesis database"
            )
        if not database.write_concern.acknowledged:
            raise RetrievalError("Thesis storage requires acknowledged MongoDB writes")
        self.database = database
        self.active = database["thesis_active_corpora"]
        self.chunks = database["thesis_chunks"]
        self.sources = database["thesis_sources"]
        self.pages = database["thesis_pages"]

    def get_active(self, corpus_id: str) -> dict[str, Any] | None:
        """Read one complete published snapshot pointer."""
        return self.active.find_one({"_id": corpus_id})

    def stage(self, corpus: PreparedCorpus, snapshot_id: str) -> None:
        """Write original pages and chunk records without changing visible evidence."""
        for collection in [self.chunks, self.sources, self.pages]:
            collection.create_index([("corpus_id", 1), ("snapshot_id", 1)])
        context = {"corpus_id": corpus.corpus_id, "snapshot_id": snapshot_id}
        for document in corpus.documents:
            source_id = document.source_id
            source_record = document.model_dump(
                mode="json", exclude={"chunks", "pages"}
            )
            self.sources.replace_one(
                {"_id": stable_id(snapshot_id, source_id)},
                {"_id": stable_id(snapshot_id, source_id), **context, **source_record},
                upsert=True,
            )
            for page in document.pages:
                key = stable_id(snapshot_id, source_id, str(page.page))
                self.pages.replace_one(
                    {"_id": key},
                    {
                        "_id": key,
                        **context,
                        "source_id": source_id,
                        **page.model_dump(),
                    },
                    upsert=True,
                )
            for chunk in document.chunks:
                key = stable_id(snapshot_id, chunk.id)
                self.chunks.replace_one(
                    {"_id": key},
                    {
                        "_id": key,
                        **context,
                        "chunk": chunk.model_dump(),
                        "source": document.source.model_dump(mode="json"),
                    },
                    upsert=True,
                )

    def publish(
        self,
        corpus_id: str,
        snapshot_id: str,
        digest: str,
        profile: EmbeddingProfile,
        count: int,
        collection: str,
    ) -> None:
        """Publish only after all synchronous MongoDB/Qdrant writes succeeded."""
        self.active.replace_one(
            {"_id": corpus_id},
            {
                "_id": corpus_id,
                "snapshot_id": snapshot_id,
                "fingerprint": digest,
                "embedding": profile.model_dump(),
                "chunks": count,
                "vector_collection": collection,
                "published_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )

    def hydrate(
        self, corpus_id: str, snapshot_id: str, ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Load scoped evidence by ID; callers explicitly restore vector ranking."""
        records = self.chunks.find(
            {"_id": {"$in": ids}, "corpus_id": corpus_id, "snapshot_id": snapshot_id}
        )
        return {record["_id"]: record for record in records}

    def article_neighbors(self, seed: SearchHit) -> list[SearchHit]:
        """Fetch immediate same-article neighbors from the seed's immutable snapshot."""
        chunk = seed.chunk
        if not chunk.article:
            return []
        scope = {
            "corpus_id": seed.corpus_id,
            "snapshot_id": seed.snapshot_id,
            "chunk.source_id": chunk.source_id,
            "chunk.version": chunk.version,
            "chunk.article": chunk.article,
        }
        result = []
        for operator, order in (("$lt", -1), ("$gt", 1)):
            query = {
                **scope,
                "$or": [
                    {"chunk.page": {operator: chunk.page}},
                    {"chunk.page": chunk.page, "chunk.start": {operator: chunk.start}},
                ],
            }
            record = self.chunks.find_one(
                query, sort=[("chunk.page", order), ("chunk.start", order)]
            )
            if record is None:
                continue
            neighbor = EvidenceChunk.model_validate(record["chunk"])
            source = SourceSpec.model_validate(record["source"])
            if source != seed.source or record["_id"] != stable_id(
                seed.snapshot_id, neighbor.id
            ):
                raise RetrievalError(
                    "Article neighbor identity or source metadata disagrees"
                )
            result.append(
                SearchHit(
                    rank=0,
                    score=None,
                    corpus_id=seed.corpus_id,
                    snapshot_id=seed.snapshot_id,
                    point_id=record["_id"],
                    chunk=neighbor,
                    source=source,
                    citation_url=citation_url(source, neighbor),
                    retrieval_origin="article_neighbor",
                    seed_chunk_id=chunk.id,
                )
            )
        return result


class ThesisVectorIndex:
    """Index complete corpus snapshots and retrieve their cited passages."""

    def __init__(
        self,
        store: MongoEvidenceStore,
        vectors: QdrantClient,
        embedder: Embedder,
        profile: EmbeddingProfile,
        collection: str = "mag_rag_chunks",
    ) -> None:
        """Inject storage and embedding clients for production or isolated tests."""
        if (
            not collection.startswith("mag_rag_")
            and collection != "finrep_thesis_chunks"
        ):
            raise RetrievalError(
                "Use a mag_rag_ collection or legacy finrep_thesis_chunks"
            )
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.profile = profile
        self.collection = collection

    def check_collection(self, create: bool = False) -> None:
        """Check cosine/dimension compatibility without recreating existing data."""
        if not self.vectors.collection_exists(self.collection):
            if not create:
                raise RetrievalError(
                    "The thesis vector collection does not exist; index first"
                )
            self.vectors.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.profile.dimensions, distance=models.Distance.COSINE
                ),
            )
        config = self.vectors.get_collection(self.collection).config.params.vectors
        if (
            not isinstance(config, models.VectorParams)
            or config.size != self.profile.dimensions
            or config.distance != models.Distance.COSINE
        ):
            raise RetrievalError(
                "Existing Qdrant collection has incompatible dimensions or distance"
            )

    def index(self, corpus: PreparedCorpus, batch_size: int = 32) -> dict[str, Any]:
        """Build a staged snapshot; failures before publication leave the prior pointer untouched."""
        corpus = PreparedCorpus.model_validate(corpus.model_dump())
        chunks = [chunk for doc in corpus.documents for chunk in doc.chunks]
        if not corpus.corpus_id.strip() or not chunks or batch_size < 1:
            raise RetrievalError(
                "Indexing requires a nonempty corpus and positive batch size"
            )
        digest = fingerprint(corpus, self.profile)
        self.check_collection(create=True)
        active = self.store.get_active(corpus.corpus_id)
        if (
            active
            and active["fingerprint"] == digest
            and active.get("vector_collection") == self.collection
        ):
            scope = {
                "corpus_id": corpus.corpus_id,
                "snapshot_id": active["snapshot_id"],
            }
            filter_ = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in scope.items()
                ]
            )
            complete = (
                self.vectors.count(
                    self.collection, count_filter=filter_, exact=True
                ).count
                == len(chunks)
                and self.store.chunks.count_documents(scope) == len(chunks)
                and self.store.sources.count_documents(scope) == len(corpus.documents)
                and self.store.pages.count_documents(scope)
                == sum(len(doc.pages) for doc in corpus.documents)
            )
            if complete:
                return {
                    "snapshot_id": active["snapshot_id"],
                    "chunks": active["chunks"],
                    "status": "unchanged",
                }
        snapshot_id = str(uuid4())
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            embeddings = self.embedder.embed_documents([chunk.text for chunk in batch])
            validate_vectors(embeddings, len(batch), self.profile.dimensions)
            points = [
                models.PointStruct(
                    id=stable_id(snapshot_id, chunk.id),
                    vector=embedding,
                    payload={
                        "corpus_id": corpus.corpus_id,
                        "snapshot_id": snapshot_id,
                        "chunk_id": chunk.id,
                    },
                )
                for chunk, embedding in zip(batch, embeddings)
            ]
            result = self.vectors.upsert(self.collection, points=points, wait=True)
            if result.status != models.UpdateStatus.COMPLETED:
                raise RetrievalError(
                    "Vector write has not completed; snapshot was not published"
                )
        self.store.stage(corpus, snapshot_id)
        self.store.publish(
            corpus.corpus_id,
            snapshot_id,
            digest,
            self.profile,
            len(chunks),
            self.collection,
        )
        return {"snapshot_id": snapshot_id, "chunks": len(chunks), "status": "indexed"}

    def search(self, corpus_id: str, query: str, limit: int = 5) -> list[SearchHit]:
        """Retrieve one active snapshot, preserve rank, and fail on missing evidence."""
        if not corpus_id.strip() or not query.strip() or not 1 <= limit <= 100:
            raise RetrievalError(
                "Supply a corpus, a nonempty query and limit between 1 and 100"
            )
        active = self.store.get_active(corpus_id)
        if active is None:
            raise RetrievalError("No published snapshot for this corpus; index first")
        if active["embedding"] != self.profile.model_dump():
            raise RetrievalError(
                "Embedding configuration differs from the indexed snapshot; reindex with the intended configuration"
            )
        if active.get("vector_collection") != self.collection:
            raise RetrievalError(
                "Configured vector collection differs from the published snapshot"
            )
        self.check_collection()
        embedding = self.embedder.embed_query(query)
        validate_vectors([embedding], 1, self.profile.dimensions)
        snapshot_id = active["snapshot_id"]
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="corpus_id", match=models.MatchValue(value=corpus_id)
                ),
                models.FieldCondition(
                    key="snapshot_id", match=models.MatchValue(value=snapshot_id)
                ),
            ]
        )
        points = self.vectors.query_points(
            self.collection,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        records = self.store.hydrate(
            corpus_id, snapshot_id, [str(point.id) for point in points]
        )
        hits = []
        for rank, point in enumerate(points, start=1):
            record = records.get(str(point.id))
            if record is None:
                raise RetrievalError(
                    "Vector index references missing MongoDB evidence; restore the snapshot"
                )
            chunk = EvidenceChunk.model_validate(record["chunk"])
            source = SourceSpec.model_validate(record["source"])
            if point.payload is None or point.payload.get("chunk_id") != chunk.id:
                raise RetrievalError("Vector and document chunk identities disagree")
            hits.append(
                SearchHit(
                    rank=rank,
                    score=point.score,
                    corpus_id=corpus_id,
                    snapshot_id=snapshot_id,
                    point_id=str(point.id),
                    chunk=chunk,
                    source=source,
                    citation_url=citation_url(source, chunk),
                )
            )
        return hits
