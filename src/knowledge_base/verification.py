"""Read-only published-corpus audits and retrieval-only regulation smoke checks."""

from collections import Counter

from qdrant_client import models

from src.knowledge_base.models import PreparedCorpus
from src.knowledge_base.prepare import stable_id
from src.knowledge_base.vector_store import (
    RetrievalError,
    ThesisVectorIndex,
    fingerprint,
)

SMOKE_QUERIES = (
    ("Kolikokrat lahko študent opravlja izpit?", "study_rules"),
    ("Kakšni so pogoji za podaljšanje statusa študenta?", "student_status"),
    ("Kako poteka prijava magistrskega dela?", "master_thesis"),
    ("Kakšna so pravila glede diplomskega dela?", "bachelor_thesis"),
    (
        "Kaj določa pravilnik o mednarodnih študijskih izmenjavah?",
        "international_exchange",
    ),
    ("Kako je urejen doktorski študij?", "doctoral"),
    ("Kakšne so naloge tutorjev?", "tutoring"),
)


def duplicate_values(values: list[str]) -> list[str]:
    """Return duplicate logical identities within the selected active snapshot."""
    return sorted(value for value, count in Counter(values).items() if count > 1)


def verify_index(index: ThesisVectorIndex, corpus: PreparedCorpus) -> dict:
    """Audit real Mongo evidence and every vector for this corpus, including history.

    Historical snapshots may repeat logical IDs by design; duplicates are checked
    within the active snapshot. No query embeddings or model calls are made.
    """
    active = index.store.get_active(corpus.corpus_id)
    if not active or active["fingerprint"] != fingerprint(corpus, index.profile):
        raise RetrievalError("The candidate is not the active published corpus")
    if active.get("vector_collection") != index.collection:
        raise RetrievalError("Published vector collection differs from configuration")
    index.check_collection()
    scope = {"corpus_id": corpus.corpus_id, "snapshot_id": active["snapshot_id"]}
    sources = list(index.store.sources.find(scope))
    chunks = list(index.store.chunks.find(scope))
    pages = list(index.store.pages.find(scope))
    source_ids = [r["source_id"] for r in sources]
    urls = [r["source"].get("url") or "" for r in sources]
    chunk_ids = [r["chunk"]["id"] for r in chunks]
    errors = []
    expected_sources = {d.source_id: d for d in corpus.documents}
    expected_chunks = {c.id: c for d in corpus.documents for c in d.chunks}
    if set(source_ids) != set(expected_sources):
        errors.append("Mongo source identities differ from candidate")
    if set(chunk_ids) != set(expected_chunks):
        errors.append("Mongo chunk identities differ from candidate")
    for record in sources:
        expected = expected_sources.get(record["source_id"])
        if expected and (
            record["content_sha256"] != expected.content_sha256
            or record["source"] != expected.source.model_dump(mode="json")
        ):
            errors.append("Mongo source metadata differs from candidate")
    page_map = {(r["source_id"], r["page"]): r["text"] for r in pages}
    for record in chunks:
        chunk = record["chunk"]
        expected = expected_chunks.get(chunk["id"])
        text = page_map.get((chunk["source_id"], chunk["page"]))
        if (
            expected is None
            or chunk != expected.model_dump()
            or text is None
            or text[chunk["start"] : chunk["end"]] != chunk["text"]
        ):
            errors.append(f"Invalid Mongo evidence: {chunk['id']}")
    duplicates = {
        "source_urls": duplicate_values(urls),
        "document_ids": duplicate_values(source_ids),
        "chunk_ids": duplicate_values(chunk_ids),
    }
    if any(duplicates.values()):
        errors.append("Duplicate active identities")
    if len(pages) != sum(len(d.pages) for d in corpus.documents):
        errors.append("Mongo page count differs from candidate")
    corpus_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="corpus_id", match=models.MatchValue(value=corpus.corpus_id)
            )
        ]
    )
    offset = None
    total = active_count = 0
    active_point_ids = []
    orphan_point_ids = []
    inactive_snapshot_ids = set()
    while True:
        points, offset = index.vectors.scroll(
            index.collection,
            scroll_filter=corpus_filter,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            total += 1
            payload = point.payload or {}
            snapshot = payload.get("snapshot_id")
            record = index.store.chunks.find_one(
                {
                    "_id": str(point.id),
                    "corpus_id": corpus.corpus_id,
                    "snapshot_id": snapshot,
                }
            )
            if not record:
                errors.append(f"Vector lacks Mongo chunk: {point.id}")
                orphan_point_ids.append(str(point.id))
                continue
            chunk = record["chunk"]
            source = index.store.sources.find_one(
                {
                    "_id": stable_id(snapshot, chunk["source_id"]),
                    "corpus_id": corpus.corpus_id,
                    "snapshot_id": snapshot,
                }
            )
            expected_payload = {
                "corpus_id": corpus.corpus_id,
                "snapshot_id": snapshot,
                "chunk_id": chunk["id"],
                "source_id": chunk["source_id"],
                "version": chunk["version"],
                "page": chunk["page"],
                "article": chunk.get("article"),
                "section": chunk.get("section"),
            }
            if (
                not source
                or any(
                    payload.get(key) != value for key, value in expected_payload.items()
                )
                or str(point.id) != stable_id(snapshot, chunk["id"])
            ):
                errors.append(f"Vector/source identity mismatch: {point.id}")
            if snapshot == active["snapshot_id"]:
                active_count += 1
                active_point_ids.append(str(point.id))
            else:
                inactive_snapshot_ids.add(snapshot)
        if offset is None:
            break
    if set(active_point_ids) != {r["_id"] for r in chunks} or active_count != len(
        expected_chunks
    ):
        errors.append("Active Mongo/Qdrant point sets differ")
    per_source = []
    for record in sources:
        source_chunks = [
            r["chunk"] for r in chunks if r["chunk"]["source_id"] == record["source_id"]
        ]
        per_source.append(
            {
                "document_id": record["source_id"],
                "title": record["source"]["title"],
                "source_url": record["source"].get("url"),
                "pages": sum(p["source_id"] == record["source_id"] for p in pages),
                "chunks": len(source_chunks),
                "articles": sorted(
                    {c["article"] for c in source_chunks if c["article"]}
                ),
            }
        )
    return {
        "corpus_id": corpus.corpus_id,
        "snapshot_id": active["snapshot_id"],
        "ok": not errors,
        "errors": errors,
        "duplicates": duplicates,
        "mongo_sources": len(sources),
        "mongo_pages": len(pages),
        "mongo_chunks": len(chunks),
        "qdrant_active_points": active_count,
        "qdrant_inactive_points": total - active_count,
        "qdrant_inactive_snapshots": len(inactive_snapshot_ids),
        "qdrant_orphan_points": len(orphan_point_ids),
        "qdrant_orphan_point_ids": orphan_point_ids,
        "qdrant_corpus_points_including_history": total,
        "qdrant_collection_points_including_other_corpora": index.vectors.count(
            index.collection, exact=True
        ).count,
        "documents_with_articles": [r["title"] for r in per_source if r["articles"]],
        "documents_without_articles": [
            r["title"] for r in per_source if not r["articles"]
        ],
        "sources": per_source,
    }


def smoke_search(
    index: ThesisVectorIndex, corpus_id: str, limit: int = 5
) -> list[dict]:
    """Run seven query embeddings, no generated answers or automatic correctness score."""
    result = []
    for query, expected in SMOKE_QUERIES:
        hits = index.search(corpus_id, query, limit)
        result.append(
            {
                "query": query,
                "expected_family": expected,
                "expected_family_in_top_k": any(
                    h.source.regulation and h.source.regulation.subcategory == expected
                    for h in hits
                ),
                "hits": [
                    {
                        "score": h.score,
                        "title": h.source.title,
                        "page": None if h.source.web else h.chunk.page,
                        "article": h.chunk.article,
                        "section": h.chunk.section,
                        "source_url": str(h.source.url),
                        "citation_url": h.citation_url,
                        "chunk_id": h.chunk.id,
                        "preview": h.chunk.text[:500],
                    }
                    for h in hits
                ],
            }
        )
    return result
