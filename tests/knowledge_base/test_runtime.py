"""Validate configuration without accessing real credentials or services."""

import pytest

from src.knowledge_base.runtime import RetrievalSettings, configured_index
from src.knowledge_base.vector_store import RetrievalError


def values() -> dict[str, str]:
    """Return explicitly fake settings; never read the user's local .env."""
    return {
        "THESIS_MONGO_URI": "mongodb://user:FAKE_SECRET@localhost:27017/",
        "THESIS_QDRANT_URL": "http://localhost:6333",
        "AZURE_OPENAI_API_KEY": "FAKE_SECRET",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embeddings",
        "OPENAI_API_VERSION": "test",
    }


def test_missing_settings_list_names_only() -> None:
    """Missing deployment information is actionable without revealing credentials."""
    settings = values()
    del settings["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    with pytest.raises(RetrievalError) as error:
        RetrievalSettings.from_values(settings)
    assert "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in str(error.value)
    assert "FAKE_SECRET" not in str(error.value)


def test_defaults_are_isolated_and_repr_excludes_secrets() -> None:
    """Default names are isolated within the standalone application."""
    settings = RetrievalSettings.from_values(values())
    assert settings.mongo_database == "mag_rag"
    assert settings.collection == "mag_rag_chunks"
    assert settings.profile.dimensions == 3072
    assert settings.profile.endpoint == "https://example.openai.azure.com"
    assert "FAKE_SECRET" not in repr(settings)


@pytest.mark.parametrize(
    "key,value",
    [
        ("THESIS_MONGO_DB", "other"),
        ("THESIS_QDRANT_COLLECTION", "other"),
        ("THESIS_MONGO_DB", "finrep"),
        ("THESIS_MONGO_DB", "finrep_thesis_unrelated"),
        ("THESIS_QDRANT_COLLECTION", "finrep"),
        ("THESIS_QDRANT_COLLECTION", "finrep_thesis_unrelated"),
        ("AZURE_OPENAI_EMBEDDER_DIM", "0"),
        ("AZURE_OPENAI_EMBEDDER_DIM", "invalid"),
    ],
)
def test_bad_settings_rejected(key: str, value: str) -> None:
    """Isolation/dimension errors are caught before constructing service clients."""
    with pytest.raises(RetrievalError):
        RetrievalSettings.from_values({**values(), key: value})


def test_remote_errors_do_not_expose_connection_strings(monkeypatch) -> None:
    """Driver errors may contain a URI; the CLI-facing error must not repeat it."""
    import pymongo

    monkeypatch.setattr(
        RetrievalSettings,
        "from_environment",
        lambda: RetrievalSettings.from_values(values()),
    )

    def fail(*args, **kwargs):
        """Simulate a driver exception containing sensitive connection details."""
        raise RuntimeError("connection failed for FAKE_SECRET")

    monkeypatch.setattr(pymongo, "MongoClient", fail)
    with pytest.raises(RetrievalError) as error:
        with configured_index(allow_external_api=True):
            pytest.fail("Should not create a configured index")
    assert "FAKE_SECRET" not in str(error.value)
    assert "RuntimeError" in str(error.value)


def test_cloud_disabled_before_reading_configuration(monkeypatch) -> None:
    """The default path cannot read credentials or construct remote clients."""

    def fail():
        raise AssertionError("Configuration must not be read before approval")

    monkeypatch.setattr(RetrievalSettings, "from_environment", fail)
    with pytest.raises(RetrievalError, match="External API calls are disabled"):
        with configured_index():
            pytest.fail("Cloud access must be opt-in per invocation")


def test_explicit_legacy_thesis_storage_is_supported() -> None:
    """Reuse the old thesis storage only when selected explicitly in settings."""
    settings = RetrievalSettings.from_values(
        {
            **values(),
            "THESIS_MONGO_DB": "finrep_thesis",
            "THESIS_QDRANT_COLLECTION": "finrep_thesis_chunks",
        }
    )
    assert settings.mongo_database == "finrep_thesis"
    assert settings.collection == "finrep_thesis_chunks"
