"""Explicit configuration and lazy clients for the thesis retrieval CLI."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from dotenv import dotenv_values

from src.knowledge_base.vector_store import (
    EmbeddingProfile,
    MongoEvidenceStore,
    RetrievalError,
    ThesisVectorIndex,
)


@dataclass(frozen=True)
class RetrievalSettings:
    """Validated connection settings; credential-bearing fields are not printable."""

    mongo_uri: str = field(repr=False)
    mongo_database: str
    qdrant_url: str = field(repr=False)
    qdrant_key: str | None = field(repr=False)
    collection: str
    azure_key: str = field(repr=False)
    profile: EmbeddingProfile

    @classmethod
    def from_values(cls, values: Mapping[str, str | None]) -> "RetrievalSettings":
        """Validate required names without displaying their values in errors."""
        required = [
            "THESIS_MONGO_URI",
            "THESIS_QDRANT_URL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "OPENAI_API_VERSION",
        ]
        missing = [key for key in required if not (values.get(key) or "").strip()]
        if missing:
            raise RetrievalError("Missing configuration: " + ", ".join(missing))
        database = values.get("THESIS_MONGO_DB") or "mag_rag"
        collection = values.get("THESIS_QDRANT_COLLECTION") or "mag_rag_chunks"
        # Reuse the explicitly selected thesis index from the original project.
        if not database.startswith("mag_rag") and database != "finrep_thesis":
            raise RetrievalError(
                "THESIS_MONGO_DB must use mag_rag or the legacy finrep_thesis database"
            )
        if (
            not collection.startswith("mag_rag_")
            and collection != "finrep_thesis_chunks"
        ):
            raise RetrievalError(
                "THESIS_QDRANT_COLLECTION must use mag_rag_ or legacy finrep_thesis_chunks"
            )
        try:
            dimensions = int(values.get("AZURE_OPENAI_EMBEDDER_DIM") or "3072")
        except ValueError:
            raise RetrievalError(
                "AZURE_OPENAI_EMBEDDER_DIM must be a positive integer"
            ) from None
        if dimensions < 1:
            raise RetrievalError("AZURE_OPENAI_EMBEDDER_DIM must be a positive integer")
        return cls(
            mongo_uri=str(values["THESIS_MONGO_URI"]),
            mongo_database=database,
            qdrant_url=str(values["THESIS_QDRANT_URL"]),
            qdrant_key=values.get("QDRANT_API_KEY") or None,
            collection=collection,
            azure_key=str(values["AZURE_OPENAI_API_KEY"]),
            profile=EmbeddingProfile(
                model=values.get("AZURE_OPENAI_EMBEDDER") or "text-embedding-3-large",
                deployment=str(values["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]),
                endpoint=str(values["AZURE_OPENAI_ENDPOINT"]).rstrip("/"),
                api_version=str(values["OPENAI_API_VERSION"]),
                dimensions=dimensions,
            ),
        )

    @classmethod
    def from_environment(cls) -> "RetrievalSettings":
        """Read local .env with process environment taking precedence; no global mutation."""
        return cls.from_values({**dotenv_values(Path(".env")), **os.environ})


@contextmanager
def configured_index(
    *,
    allow_external_api: bool = False,
    http_client: "httpx.Client | None" = None,
    max_retries: int = 2,
) -> Iterator[ThesisVectorIndex]:
    """Create clients only when indexing/searching; close them and sanitize remote errors."""
    if not allow_external_api:
        raise RetrievalError(
            "External API calls are disabled. Obtain explicit approval before enabling this operation."
        )
    settings = RetrievalSettings.from_environment()
    from langchain_openai import AzureOpenAIEmbeddings
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    mongo = None
    vectors = None
    try:
        mongo = MongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=30000,
        )
        mongo.admin.command("ping")
        vectors = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_key, timeout=30
        )
        embedder = AzureOpenAIEmbeddings(
            azure_endpoint=settings.profile.endpoint,
            azure_deployment=settings.profile.deployment,
            api_key=settings.azure_key,
            api_version=settings.profile.api_version,
            model=settings.profile.model,
            dimensions=settings.profile.dimensions,
            request_timeout=30,
            max_retries=max_retries,
            http_client=http_client,
        )
        yield ThesisVectorIndex(
            MongoEvidenceStore(mongo[settings.mongo_database]),
            vectors,
            embedder,
            settings.profile,
            settings.collection,
        )
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            f"Remote service operation failed ({type(exc).__name__}); check configured service access and availability"
        ) from None
    finally:
        if vectors is not None:
            vectors.close()
        if mongo is not None:
            mongo.close()
