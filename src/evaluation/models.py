"""Validate benchmark provenance and keep draft annotations explicit."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.knowledge_base.models import PreparedCorpus, citation_url


def corpus_digest(corpus: PreparedCorpus) -> str:
    """Identify the complete evidence snapshot, independent of preparation time."""
    data = corpus.model_dump(mode="json")
    for document in data["documents"]:
        document.pop("prepared_at")
    data["documents"].sort(key=lambda item: item["source_id"])
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class EvidenceReference(BaseModel):
    """An exact passage copied from the frozen corpus for annotation review."""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    source_key: str
    version: str
    page: int = Field(ge=1)
    article: str | None
    url: str
    excerpt: str = Field(min_length=1)


class BenchmarkItem(BaseModel):
    """A synthetic standalone question with draft reference and evidence groups."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    split: Literal["development", "test"] = "development"
    origin: Literal["synthetic-assistant-authored"] = "synthetic-assistant-authored"
    question: str = Field(min_length=2, max_length=2000)
    category: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    expected_status: Literal["evidence_found", "no_evidence", "needs_clarification"]
    evidence_groups: list[list[str]] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    review_status: Literal["pending", "reviewed"] = "pending"
    reviewer_code: str | None = None
    review_notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_annotation(self) -> "BenchmarkItem":
        """Require evidence for answers and explicit human identity for approval."""
        ids = [ref.chunk_id for ref in self.evidence]
        grouped = [identifier for group in self.evidence_groups for identifier in group]
        if len(ids) != len(set(ids)) or any(
            not group for group in self.evidence_groups
        ):
            raise ValueError("Duplicate evidence or empty evidence group")
        if len(grouped) != len(set(grouped)) or set(grouped) != set(ids):
            raise ValueError("Evidence must occur in exactly one required group")
        if (self.expected_status == "evidence_found") != bool(self.evidence_groups):
            raise ValueError("Only answerable items have required retrieval evidence")
        if self.review_status == "reviewed" and not (self.reviewer_code or "").strip():
            raise ValueError("Reviewed items need an anonymous human reviewer code")
        return self


class Benchmark(BaseModel):
    """A versioned synthetic benchmark, not a collection of historical enquiries."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    name: str
    corpus_id: str
    corpus_sha256: str
    scope: str
    authorship: str
    items: list[BenchmarkItem] = Field(min_length=1)

    @model_validator(mode="after")
    def check_splits(self) -> "Benchmark":
        """Reject duplicate questions and paraphrase-family leakage across splits."""
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Duplicate question ID")
        if len({item.question.casefold() for item in self.items}) != len(self.items):
            raise ValueError("Duplicate question text")
        families: dict[str, str] = {}
        for item in self.items:
            if families.setdefault(item.family_id, item.split) != item.split:
                raise ValueError("Question family appears in both development and test")
        return self


def load_benchmark(path: Path, corpus: PreparedCorpus) -> Benchmark:
    """Fail before clients/calls if source bytes, passages or provenance have drifted."""
    benchmark = Benchmark.model_validate_json(path.read_text(encoding="utf-8"))
    if (
        benchmark.corpus_id != corpus.corpus_id
        or benchmark.corpus_sha256 != corpus_digest(corpus)
    ):
        raise ValueError(
            "Benchmark corpus mismatch; review annotations before rebasing"
        )
    chunks = {
        chunk.id: (document, chunk)
        for document in corpus.documents
        for chunk in document.chunks
    }
    for item in benchmark.items:
        for ref in item.evidence:
            if ref.chunk_id not in chunks:
                raise ValueError(f"Unknown reference chunk in {item.id}")
            doc, chunk = chunks[ref.chunk_id]
            expected = EvidenceReference(
                chunk_id=chunk.id,
                source_key=doc.source.key,
                version=chunk.version,
                page=chunk.page,
                article=chunk.article,
                url=citation_url(doc.source, chunk),
                excerpt=chunk.text,
            )
            if ref != expected:
                raise ValueError(f"Evidence provenance mismatch in {item.id}")
    return benchmark
