"""Controlled document-derived silver/gold datasets and human review provenance."""

import hashlib
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from src.knowledge_base.models import PreparedCorpus, citation_url
from .metrics import normalized, tokens
from .models import corpus_digest, load_benchmark


CHECKS = (
    "question_clear",
    "answer_correct",
    "source_correct",
    "sufficient_evidence",
    "answerability_correct",
)


class SupportingPassage(BaseModel):
    """Source ID is the existing stable chunk ID, not the document-level ID."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    document_id: str
    title: str
    page: int | None
    section: str | None
    article: str | None = None
    url: str
    version: str
    text: str = Field(min_length=1)


class OfficialFAQProvenance(BaseModel):
    """Publisher provenance is distinct from human approval of benchmark annotations."""

    model_config = ConfigDict(extra="forbid")
    source_url: str
    document_id: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    section_index: int = Field(ge=1)
    anchor: str = Field(min_length=1)


class ControlledItem(BaseModel):
    """A draft or human-reviewed case, never claimed to be a real student enquiry."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=2, max_length=2000)
    expected_answer: str = Field(min_length=1, max_length=12000)
    source_ids: list[str] = Field(default_factory=list)
    required_source_groups: list[list[str]] = Field(default_factory=list)
    supporting_passages: list[SupportingPassage] = Field(default_factory=list)
    category: str = Field(default="unspecified", min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    answerable: bool
    expected_status: Literal["evidence_found", "no_evidence", "needs_clarification"]
    generation_method: Literal["synthetic", "manual", "paraphrased", "official_faq"]
    official_faq: OfficialFAQProvenance | None = None
    review_status: Literal["unreviewed", "approved", "rejected", "needs_revision"] = (
        "unreviewed"
    )
    review_notes: str = ""
    family_id: str = Field(min_length=1)
    split: Literal["development", "test"] = "development"
    reviewer_code: str | None = None
    review_checks: dict[str, bool] = Field(default_factory=dict)
    reviewed_content_sha256: str | None = None

    @model_serializer(mode="wrap")
    def serialize_origin(self, handler):
        """Keep existing content hashes and human approvals unchanged for old cases."""
        data = handler(self)
        if self.official_faq is None:
            data.pop("official_faq", None)
        return data

    def content_hash(self) -> str:
        """Bind approvals to content; changes require a fresh human review."""
        data = self.model_dump(
            mode="json",
            exclude={
                "review_status",
                "review_notes",
                "reviewer_code",
                "review_checks",
                "reviewed_content_sha256",
            },
        )
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    @model_validator(mode="after")
    def validate_case(self):
        """Enforce evidence identity, answerability and explicit review provenance."""
        if (self.generation_method == "official_faq") != (
            self.official_faq is not None
        ):
            raise ValueError("Official FAQ cases require explicit publisher provenance")
        if not tokens(self.expected_answer):
            raise ValueError("Empty normalized expected answer")
        if (
            len(set(self.source_ids)) != len(self.source_ids)
            or set(self.source_ids)
            != {item.source_id for item in self.supporting_passages}
            or len(self.supporting_passages) != len(self.source_ids)
        ):
            raise ValueError("Missing or duplicate supporting sources")
        if self.answerable != bool(self.source_ids) or self.answerable != (
            self.expected_status == "evidence_found"
        ):
            raise ValueError("Answerability and evidence/status disagree")
        if not self.required_source_groups:
            self.required_source_groups = [
                [identifier] for identifier in self.source_ids
            ]
        flattened = [
            identifier for group in self.required_source_groups for identifier in group
        ]
        if (
            any(not group for group in self.required_source_groups)
            or len(flattened) != len(set(flattened))
            or set(flattened) != set(self.source_ids)
        ):
            raise ValueError("Required evidence groups must partition source IDs")
        if self.review_status == "approved":
            if (
                not self.reviewer_code
                or not self.reviewer_code.strip()
                or any(self.review_checks.get(key) is not True for key in CHECKS)
            ):
                raise ValueError(
                    "Reviewed references require a human reviewer and all five review checks"
                )
            if self.reviewed_content_sha256 != self.content_hash():
                raise ValueError(
                    "Human approval is missing or stale after content changes"
                )
        return self


class ControlledDataset(BaseModel):
    """Versioned experimental annotations kept entirely outside retrieval storage."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[2] = 2
    name: str
    domain: Literal["student_affairs"] = "student_affairs"
    corpus_id: str
    corpus_sha256: str
    seed: int = 42
    generation_description: str
    protocol: Literal["controlled", "official_faq_lookup"] = "controlled"
    items: list[ControlledItem] = Field(min_length=1)

    @model_serializer(mode="wrap")
    def serialize_protocol(self, handler):
        """Keep legacy dataset serialization stable."""
        data = handler(self)
        if self.protocol == "controlled":
            data.pop("protocol", None)
        return data

    @model_validator(mode="after")
    def check_duplicates(self):
        """Reject normalized duplicates and prevent family leakage between splits."""
        if any(
            (item.generation_method == "official_faq")
            != (self.protocol == "official_faq_lookup")
            for item in self.items
        ):
            raise ValueError(
                "Official FAQ lookup must be a separate, explicitly labelled dataset"
            )
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Duplicate case IDs")
        if len({normalized(item.question) for item in self.items}) != len(self.items):
            raise ValueError("Duplicate normalized questions")
        families = {}
        for item in self.items:
            if families.setdefault(item.family_id, item.split) != item.split:
                raise ValueError("Question family leaks across development/test splits")
        return self


def passage(document, chunk) -> SupportingPassage:
    """Copy exact reference provenance from prepared source evidence."""
    return SupportingPassage(
        source_id=chunk.id,
        document_id=document.source_id,
        title=document.source.title,
        page=None if document.source.web else chunk.page,
        section=(
            document.source.web.sections[chunk.page - 1].heading
            if document.source.web
            else None
        ),
        article=chunk.article,
        url=citation_url(document.source, chunk),
        version=chunk.version,
        text=chunk.text,
    )


def audit_dataset(dataset: ControlledDataset, corpus: PreparedCorpus) -> dict:
    """Check provenance/leakage and flag semantic support that needs human judgment."""
    errors, warnings = [], []
    if dataset.corpus_id != corpus.corpus_id or dataset.corpus_sha256 != corpus_digest(
        corpus
    ):
        errors.append("Frozen corpus mismatch")
    lookup = {
        chunk.id: passage(doc, chunk)
        for doc in corpus.documents
        for chunk in doc.chunks
    }
    texts = [
        (doc.source_id, normalized(page.text))
        for doc in corpus.documents
        for page in doc.pages
    ]
    for item in dataset.items:
        permitted_faq_document = None
        if item.official_faq:
            from .faq import validate_faq_reference

            try:
                validate_faq_reference(item, corpus)
                permitted_faq_document = item.official_faq.document_id
                warnings.append(
                    f"{item.id}: official FAQ lookup; published question and answer are intentionally in retrieval evidence, not an independent held-out test"
                )
            except ValueError as exc:
                errors.append(f"{item.id}: {exc}")
        for ref in item.supporting_passages:
            if lookup.get(ref.source_id) != ref:
                errors.append(
                    f"{item.id}: missing source or altered supporting passage"
                )
        evidence_text = " ".join(ref.text for ref in item.supporting_passages)
        if item.answerable:
            provenance_text = (
                evidence_text
                + " "
                + " ".join(ref.article or "" for ref in item.supporting_passages)
            )
            unsupported_numbers = set(
                re.findall(r"\d+(?:[.,]\d+)*", item.expected_answer)
            ) - set(re.findall(r"\d+(?:[.,]\d+)*", provenance_text))
            if unsupported_numbers:
                errors.append(
                    f"{item.id}: reference contains numbers absent from evidence"
                )
            if normalized(item.expected_answer) not in normalized(evidence_text):
                warnings.append(
                    f"{item.id}: non-extractive answer needs human entailment/completeness review"
                )
        if any(
            normalized(item.question) in text and document_id != permitted_faq_document
            for document_id, text in texts
        ):
            errors.append(
                f"{item.id}: question appears in knowledge base (possible leakage or copied FAQ; review and rephrase)"
            )
    for index, item in enumerate(dataset.items):
        for other in dataset.items[index + 1 :]:
            a, b = normalized(item.question), normalized(other.question)
            similarity = SequenceMatcher(None, a, b).ratio()
            words_a, words_b = set(tokens(a)), set(tokens(b))
            overlap = (
                len(words_a & words_b) / len(words_a | words_b)
                if words_a | words_b
                else 1
            )
            if similarity >= 0.88 or overlap >= 0.85:
                message = f"Near-duplicate questions: {item.id}, {other.id}"
                if item.family_id != other.family_id or item.split != other.split:
                    errors.append(message + "; assign the same family and split")
                else:
                    warnings.append(
                        message + "; correlated examples, not independent samples"
                    )
    return {
        "valid": not errors,
        "protocol": dataset.protocol,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            status: sum(item.review_status == status for item in dataset.items)
            for status in ("unreviewed", "approved", "rejected", "needs_revision")
        },
        "semantic_support": "heuristics plus required human review; not automatic entailment verification",
    }


def evaluation_tier(item: ControlledItem) -> str:
    """Do not describe publisher-authored, unreviewed FAQ cases as synthetic silver."""
    if item.review_status == "approved":
        return "gold"
    return "official_faq" if item.generation_method == "official_faq" else "silver"


def load_controlled(
    path: Path, corpus: PreparedCorpus
) -> tuple[ControlledDataset, dict]:
    """Validate dataset shape and frozen evidence before any provider is created."""
    dataset = ControlledDataset.model_validate_json(path.read_text(encoding="utf-8"))
    audit = audit_dataset(dataset, corpus)
    if not audit["valid"]:
        raise ValueError("; ".join(audit["errors"]))
    return dataset, audit


def write_new(path: Path, data) -> None:
    """Never overwrite a dataset, review or experiment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def review_template(dataset: ControlledDataset) -> list[dict]:
    """Prepare explicit human checklist decisions without self-approving anything."""
    return [
        {
            "id": item.id,
            "question": item.question,
            "expected_answer": item.expected_answer,
            "origin": (
                "Official FAQ"
                if item.generation_method == "official_faq"
                else "Created from documents"
            ),
            "answerable": item.answerable,
            "supporting_passages": [
                ref.model_dump(mode="json") for ref in item.supporting_passages
            ],
            "content_sha256": item.content_hash(),
            "review_status": item.review_status,
            "reviewer_code": item.reviewer_code or "",
            "review_notes": item.review_notes,
            "checks": {key: item.review_checks.get(key, False) for key in CHECKS},
        }
        for item in dataset.items
    ]


def apply_reviews(
    dataset: ControlledDataset, decisions: list[dict]
) -> ControlledDataset:
    """Apply decisions to a new dataset, retaining rejected and revision-needed cases."""
    lookup = {item.id: item for item in dataset.items}
    seen = set()
    data = dataset.model_dump()
    updates = {}
    for decision in decisions:
        key = decision["id"]
        if key not in lookup or key in seen:
            raise ValueError("Unknown or duplicate review ID")
        seen.add(key)
        if decision["content_sha256"] != lookup[key].content_hash():
            raise ValueError("Review applies to different case content")
        item = lookup[key]
        context = {
            "question": item.question,
            "expected_answer": item.expected_answer,
            "origin": (
                "Official FAQ"
                if item.generation_method == "official_faq"
                else "Created from documents"
            ),
            "answerable": item.answerable,
            "supporting_passages": [
                ref.model_dump(mode="json") for ref in item.supporting_passages
            ],
        }
        if any(
            field in decision and decision[field] != value
            for field, value in context.items()
        ):
            raise ValueError(
                "Review context was edited; revise the dataset and create a fresh review template"
            )
        status = decision["review_status"]
        if status != "unreviewed" and not decision.get("reviewer_code", "").strip():
            raise ValueError("A human reviewer code is required")
        updates[key] = {
            "review_status": status,
            "reviewer_code": decision.get("reviewer_code"),
            "review_notes": decision.get("review_notes", ""),
            "review_checks": decision.get("checks", {}),
            "reviewed_content_sha256": (
                decision["content_sha256"] if status == "approved" else None
            ),
        }
    data["items"] = [{**item, **updates.get(item["id"], {})} for item in data["items"]]
    return ControlledDataset.model_validate(data)


def make_silver(
    corpus: PreparedCorpus, *, legacy: Path | None = None, seed=42, limit=24
) -> ControlledDataset:
    """Create reproducible source-grounded drafts without a generation API call."""
    import random

    items = []
    lookup = {
        chunk.id: (doc, chunk) for doc in corpus.documents for chunk in doc.chunks
    }
    if legacy:
        pilot = load_benchmark(legacy, corpus)
        for old in pilot.items:
            refs = [passage(*lookup[ref.chunk_id]) for ref in old.evidence]
            items.append(
                ControlledItem(
                    id=old.id,
                    question=old.question,
                    expected_answer=old.reference_answer,
                    source_ids=[ref.source_id for ref in refs],
                    required_source_groups=old.evidence_groups,
                    supporting_passages=refs,
                    category=old.category,
                    difficulty="hard" if len(old.evidence_groups) > 1 else "easy",
                    answerable=old.expected_status == "evidence_found",
                    expected_status=old.expected_status,
                    generation_method="synthetic",
                    review_notes=old.review_notes,
                    family_id=old.family_id,
                    split=old.split,
                )
            )
    else:
        candidates = [
            (doc, chunk)
            for doc in corpus.documents
            for chunk in doc.chunks
            if 100 <= len(chunk.text) <= 2500
        ]
        random.Random(seed).shuffle(candidates)
        for doc, chunk in candidates[:limit]:
            ref = passage(doc, chunk)
            location = ref.section or (
                f"{chunk.article}. člen" if chunk.article else f"stran {chunk.page}"
            )
            question = f"Kaj določa dokument »{doc.source.title}«, {location}, v odlomku »{chunk.text[:70].strip()}«?"
            if any(normalized(item.question) == normalized(question) for item in items):
                continue
            items.append(
                ControlledItem(
                    id="synthetic-" + chunk.id,
                    question=question,
                    expected_answer=chunk.text,
                    source_ids=[chunk.id],
                    supporting_passages=[ref],
                    category="direct",
                    answerable=True,
                    expected_status="evidence_found",
                    generation_method="synthetic",
                    family_id=chunk.id,
                    review_notes="Deterministic extractive template; source wording in question biases retrieval. Human rewriting and review required.",
                )
            )
    if not items:
        raise ValueError("No usable evidence for a silver dataset")
    return ControlledDataset(
        name="Controlled public-document development set",
        corpus_id=corpus.corpus_id,
        corpus_sha256=corpus_digest(corpus),
        seed=seed,
        generation_description="Document-derived synthetic drafts, not actual student enquiries; existing pilot import or deterministic extractive templates.",
        items=items,
    )
