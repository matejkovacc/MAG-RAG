"""Import the official FRI prospective-student FAQ as a separate lookup benchmark."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from src.knowledge_base.models import PreparedCorpus, SourceSpec
from src.knowledge_base.prepare import stable_id
from src.knowledge_base.website import prepare_html
from .controlled import (
    ControlledDataset,
    ControlledItem,
    OfficialFAQProvenance,
    audit_dataset,
    passage,
    review_template,
    write_new,
)
from .models import corpus_digest


FAQ_URL = "https://fri.uni-lj.si/sl/pogosta-vprasanja-1"
RETURN_LINK = "<< Nazaj na vstopno stran Virtualni informativni dan"
LOOKUP_LIMITATION = (
    "The published FAQ questions and answers are intentionally available to retrieval. "
    "This measures FAQ lookup and answer reproduction, not independent regulation-based "
    "generalisation, private student correspondence, or current applicability. "
    "Official publication does not imply human benchmark approval."
)


def faq_answer(text: str, heading: str) -> str:
    """Remove only the exact heading and known return-navigation line, never rewrite answers."""
    prefix = heading + "\n"
    if not text.startswith(prefix):
        raise ValueError("FAQ section does not start with its question heading")
    answer = text[len(prefix) :]
    if answer.endswith("\n" + RETURN_LINK):
        answer = answer[: -(len(RETURN_LINK) + 1)]
    if not answer.strip():
        raise ValueError("FAQ question has no published answer")
    return answer.strip()


def validate_faq_reference(item: ControlledItem, corpus: PreparedCorpus) -> None:
    """Verify exact published Q/A and version before allowing intentional FAQ overlap."""
    origin = item.official_faq
    if origin is None:
        raise ValueError("Missing official FAQ provenance")
    doc = next(
        (doc for doc in corpus.documents if doc.source_id == origin.document_id), None
    )
    if (
        doc is None
        or doc.source.web is None
        or origin.section_index > len(doc.source.web.sections)
    ):
        raise ValueError("Missing official FAQ document or section")
    web = doc.source.web
    section = web.sections[origin.section_index - 1]
    page = next((page for page in doc.pages if page.page == origin.section_index), None)
    expected_ids = {
        chunk.id for chunk in doc.chunks if chunk.page == origin.section_index
    }
    if (
        origin.source_url != FAQ_URL
        or str(doc.source.url) != FAQ_URL
        or origin.snapshot_sha256 != doc.content_sha256
        or origin.captured_at != web.captured_at
        or origin.anchor != section.anchor
        or item.question != section.heading
        or page is None
        or item.expected_answer != faq_answer(page.text, section.heading)
        or not item.answerable
        or not expected_ids
        or set(item.source_ids) != expected_ids
    ):
        raise ValueError("Official FAQ reference differs from the captured source")


def import_faq(
    html_path: Path, output: Path, captured_at: datetime
) -> ControlledDataset:
    """Prepare a separate frozen corpus/dataset locally; never publish to a live index."""
    if output.exists():
        raise ValueError("FAQ output must be a new directory")
    if captured_at.tzinfo is None:
        raise ValueError("Capture timestamp must include a timezone")
    content = html_path.read_bytes()
    if len(content) > 20_000_000:
        raise ValueError("FAQ HTML exceeds the existing download size limit")
    corpus_id = "fri-prospective-student-faq-lookup"
    source = SourceSpec(
        key="fri-prospective-student-faq",
        title="Pogosta vprašanja – bodoči študenti FRI",
        url=FAQ_URL,
        local_path=str((output / "source.html").resolve()),
        notes=LOOKUP_LIMITATION
        + " Public institutional source; redistribution license not established. Navigation-only sections are not retrieval chunks.",
    )
    doc, links = prepare_html(source, corpus_id, content, "katedre-container")
    doc.source.web.captured_at = captured_at.astimezone(timezone.utc)
    candidates = []
    for page, section in zip(doc.pages, doc.source.web.sections):
        if section.anchor and section.heading.endswith("?"):
            answer = faq_answer(page.text, section.heading)
            candidates.append((page, section, answer))
    if not candidates:
        raise ValueError("No anchored FAQ answers found; inspect the source structure")
    if len({section.anchor for _, section, _ in candidates}) != len(candidates):
        raise ValueError("Duplicate FAQ anchors")
    linked_anchors = {
        unquote(parts.fragment)
        for link in links
        for parts in [urlsplit(link["url"])]
        if parts.path.rstrip("/") == urlsplit(FAQ_URL).path
        and parts.hostname in {"fri.uni-lj.si", "www.fri.uni-lj.si"}
        and parts.fragment
    }
    if linked_anchors and linked_anchors != {
        section.anchor for _, section, _ in candidates
    }:
        raise ValueError(
            "FAQ table of contents does not match the extracted answer sections"
        )
    candidate_pages = {page.page for page, _, _ in candidates}
    # Preserve original pages/section indexes and exact chunk offsets, but do not
    # retrieve the duplicated question-only table of contents or introductory text.
    doc.chunks = [chunk for chunk in doc.chunks if chunk.page in candidate_pages]
    corpus = PreparedCorpus(corpus_id=corpus_id, documents=[doc])
    items = []
    for page, section, answer in candidates:
        refs = [passage(doc, chunk) for chunk in doc.chunks if chunk.page == page.page]
        identifier = "fri-faq-" + stable_id(FAQ_URL, section.anchor)
        items.append(
            ControlledItem(
                id=identifier,
                question=section.heading,
                expected_answer=answer,
                source_ids=[ref.source_id for ref in refs],
                supporting_passages=refs,
                category="prospective_student_faq",
                difficulty="easy",
                answerable=True,
                expected_status="evidence_found",
                generation_method="official_faq",
                family_id=identifier,
                review_status="unreviewed",
                review_notes="Exact normalized published Q/A; extraction, applicability and difficulty need human review. "
                + LOOKUP_LIMITATION,
                official_faq=OfficialFAQProvenance(
                    source_url=FAQ_URL,
                    document_id=doc.source_id,
                    snapshot_sha256=doc.content_sha256,
                    captured_at=doc.source.web.captured_at,
                    section_index=page.page,
                    anchor=section.anchor,
                ),
            )
        )
    dataset = ControlledDataset(
        name="Official FRI prospective-student FAQ lookup",
        corpus_id=corpus_id,
        corpus_sha256=corpus_digest(corpus),
        generation_description=LOOKUP_LIMITATION,
        protocol="official_faq_lookup",
        items=items,
    )
    audit = audit_dataset(dataset, corpus)
    if not audit["valid"]:
        raise ValueError("; ".join(audit["errors"]))
    reviews = review_template(dataset)
    output.mkdir(parents=True)
    (output / "source.html").write_bytes(content)
    write_new(output / "corpus.json", corpus.model_dump(mode="json"))
    write_new(output / "dataset.json", dataset.model_dump(mode="json"))
    write_new(output / "reference-review.json", reviews)
    write_new(
        output / "import-report.json",
        {
            "source_url": FAQ_URL,
            "snapshot_sha256": hashlib.sha256(content).hexdigest(),
            "captured_at": doc.source.web.captured_at.isoformat(),
            "question_count": len(items),
            "retrieval_chunks": len(doc.chunks),
            "protocol": dataset.protocol,
            "limitations": LOOKUP_LIMITATION,
            "audit": audit,
        },
    )
    return dataset
