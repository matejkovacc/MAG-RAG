"""Import anonymized student-office cases into the existing controlled workflow."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.knowledge_base.models import PreparedCorpus

from .controlled import ControlledDataset, ControlledItem, audit_dataset, passage
from .models import corpus_digest


class OfficeCase(BaseModel):
    """One anonymized enquiry and the office's original answer, kept outside the index."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=2, max_length=2000)
    original_response: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1, max_length=12000)
    source_ids: list[str] = Field(default_factory=list)
    required_source_groups: list[list[str]] = Field(default_factory=list)
    category: str = Field(default="unspecified", min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    answerable: bool
    expected_status: Literal["evidence_found", "no_evidence", "needs_clarification"]
    family_id: str = Field(min_length=1)
    split: Literal["development", "test"]


class OfficeImport(BaseModel):
    """Declared origin and anonymization for a batch of real cases."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    data_origin: Literal["real"]
    anonymized: Literal[True]
    cases: list[OfficeCase] = Field(min_length=1)


def import_office_cases(
    source: Path, corpus: PreparedCorpus, verification: Path
) -> ControlledDataset:
    """Bind real cases to the audited snapshot and exact prepared passages."""
    raw = OfficeImport.model_validate_json(source.read_text(encoding="utf-8-sig"))
    audit = json.loads(verification.read_text(encoding="utf-8-sig"))
    if audit.get("ok") is not True or audit.get("corpus_id") != corpus.corpus_id:
        raise ValueError("Verification must confirm this corpus")
    snapshot_id = audit.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("Verification is missing a snapshot ID")
    if audit.get("mongo_chunks") != sum(len(doc.chunks) for doc in corpus.documents):
        raise ValueError("Verification chunk count differs from prepared corpus")
    expected_sources = {
        doc.source_id: (
            doc.source.title,
            str(doc.source.url) if doc.source.url else None,
            len(doc.pages),
            len(doc.chunks),
        )
        for doc in corpus.documents
    }
    audited_sources = {
        source["document_id"]: (
            source["title"],
            source["source_url"],
            source["pages"],
            source["chunks"],
        )
        for source in audit.get("sources", [])
    }
    if expected_sources != audited_sources or len(audit.get("sources", [])) != len(
        corpus.documents
    ):
        raise ValueError("Verification sources differ from prepared corpus")
    lookup = {
        chunk.id: passage(doc, chunk)
        for doc in corpus.documents
        for chunk in doc.chunks
    }
    items = []
    for case in raw.cases:
        missing = set(case.source_ids) - lookup.keys()
        if missing:
            raise ValueError(
                f"{case.id}: evidence IDs absent from selected corpus: {sorted(missing)}"
            )
        items.append(
            ControlledItem(
                id=case.id,
                question=case.question,
                original_response=case.original_response,
                expected_answer=case.reference_answer,
                source_ids=case.source_ids,
                required_source_groups=case.required_source_groups,
                supporting_passages=[lookup[key] for key in case.source_ids],
                category=case.category,
                difficulty=case.difficulty,
                answerable=case.answerable,
                expected_status=case.expected_status,
                generation_method="student_office",
                data_origin="real",
                family_id=case.family_id,
                split=case.split,
                review_notes="Reference answer, answerability and source applicability require human review.",
            )
        )
    dataset = ControlledDataset(
        name=raw.name,
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus_digest(corpus),
        corpus_snapshot_id=snapshot_id,
        generation_description="Anonymized real student enquiries and original office responses; reference annotations pending human review.",
        items=items,
    )
    result = audit_dataset(dataset, corpus)
    if not result["valid"]:
        raise ValueError("; ".join(result["errors"]))
    return dataset
