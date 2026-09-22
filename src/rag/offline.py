"""Local lexical vectors and MongoDB simulation using the existing storage pipeline."""

import hashlib
import json
import os
import re
from contextlib import contextmanager
from typing import Iterator, Mapping

import mongomock
from qdrant_client import QdrantClient
from pymongo import MongoClient

from src.knowledge_base.models import PreparedCorpus
from src.knowledge_base.vector_store import (
    EmbeddingProfile,
    MongoEvidenceStore,
    SearchHit,
    ThesisVectorIndex,
    RetrievalError,
)
from src.rag.answers import AnswerService, SimulatedGenerator


STOP_WORDS = frozenset(
    "a ali ampak bi bil bila bilo biti bo bodo bom da do in is iz je jih jo kaj kako kam kdaj ki ko kot lahko me mi na naj ne ni o ob od pa po pri s se sem si so ta te ter the to v ve za z že what when how can i my".split()
)


def terms(text: str) -> set[str]:
    """Crude five-character prefixes for a lexical demo, not Slovenian NLP."""
    return {
        word[:5]
        for word in re.findall(r"[^\W\d_]+", text.casefold())
        if len(word) > 2 and word not in STOP_WORDS
    }


class LexicalEmbedder:
    """Deterministic binary term vectors; no model, SDK or network connection."""

    def __init__(self, texts: list[str]) -> None:
        """Build a reproducible vocabulary from document content only."""
        self.vocabulary = sorted(set().union(*(terms(text) for text in texts)))
        self.positions = {term: index for index, term in enumerate(self.vocabulary)}
        self.dimensions = len(self.vocabulary) + 1
        self.identity = hashlib.sha256(json.dumps(self.vocabulary).encode()).hexdigest()

    def embed_query(self, text: str) -> list[float]:
        """Reserve an unmatched-query dimension to keep cosine vectors nonzero."""
        vector = [0.0] * self.dimensions
        for term in terms(text):
            if term in self.positions:
                vector[self.positions[term]] = 1.0
        if not any(vector):
            vector[-1] = 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Use the same term-vector mapping for passages and enquiries."""
        return [self.embed_query(text) for text in texts]


class OfflineRetriever:
    """Reuse Qdrant ranking and Mongo hydration; discard non-overlapping results."""

    def __init__(self, index: ThesisVectorIndex, corpus_id: str) -> None:
        """Scope every request to the preview corpus."""
        self.index = index
        self.corpus_id = corpus_id

    def search(self, question: str, limit: int) -> list[SearchHit]:
        """Lexical overlap is a demo filter, not a calibrated answer confidence."""
        query_terms = terms(question)
        if not query_terms:
            return []
        return [
            hit
            for hit in self.index.search(self.corpus_id, question, limit)
            if hit.score > 0 and query_terms & terms(hit.chunk.text)
        ]


@contextmanager
def demo_clients(local_databases: bool = False) -> Iterator[tuple]:
    """Select temporary stores or fixed-loopback database servers; never cloud clients."""
    mongo = None
    vectors = None
    try:
        if local_databases:
            from dotenv import dotenv_values

            settings = local_database_settings({**dotenv_values(".env"), **os.environ})
            mongo = MongoClient(
                "127.0.0.1",
                settings["mongo_port"],
                username=settings["username"],
                password=settings["password"],
                authSource="admin",
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
            )
            mongo.admin.command("ping")
            vectors = QdrantClient(
                url=f"http://127.0.0.1:{settings['qdrant_port']}",
                api_key=settings["qdrant_key"],
                timeout=10,
            )
            vectors.get_collections()
        else:
            mongo = mongomock.MongoClient()
            vectors = QdrantClient(":memory:")
        yield mongo, vectors
    except Exception as exc:
        raise RetrievalError(
            f"Local demo storage failed ({type(exc).__name__}); check local database services and credentials"
        ) from None
    finally:
        if vectors is not None:
            vectors.close()
        if mongo is not None:
            mongo.close()


def local_database_settings(values: Mapping[str, str | None]) -> dict:
    """Read only Compose credentials/ports; ignore all configured remote hosts/URIs."""
    required = [
        "MONGO_INITDB_ROOT_USERNAME",
        "MONGO_INITDB_ROOT_PASSWORD",
        "QDRANT_API_KEY",
    ]
    if any(not values.get(key) for key in required):
        raise ValueError("Local Compose database credentials are required")
    mongo_port = int(values.get("MONGO_PORT") or "10100")
    qdrant_port = int(values.get("QDRANT_HTTP_PORT") or "10060")
    if not all(1 <= port <= 65535 for port in (mongo_port, qdrant_port)):
        raise ValueError("Invalid local database port")
    return {
        "mongo_port": mongo_port,
        "qdrant_port": qdrant_port,
        "username": values[required[0]],
        "password": values[required[1]],
        "qdrant_key": values[required[2]],
    }


@contextmanager
def offline_service(
    corpus: PreparedCorpus, *, local_databases: bool = False
) -> Iterator[AnswerService]:
    """Use local lexical vectors with temporary stores or opt-in loopback servers."""
    embedder = LexicalEmbedder(
        [chunk.text for doc in corpus.documents for chunk in doc.chunks]
    )
    with demo_clients(local_databases) as (mongo, vectors):
        profile = EmbeddingProfile(
            provider="offline-lexical-demo",
            model="prefix5-binary-v1",
            deployment=embedder.identity,
            endpoint="local",
            api_version="1",
            dimensions=embedder.dimensions,
        )
        index = ThesisVectorIndex(
            MongoEvidenceStore(mongo["mag_rag_demo"]),
            vectors,
            embedder,
            profile,
            "mag_rag_demo_lexical_" + embedder.identity,
        )
        index.index(corpus)
        yield AnswerService(
            OfflineRetriever(index, corpus.corpus_id), SimulatedGenerator()
        )
