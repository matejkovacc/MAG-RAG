"""Offline regression tests for identity, source locations and CLI failure paths."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from PyPDF2 import PdfWriter
from PyPDF2.generic import DictionaryObject, NameObject, DecodedStreamObject

from src.knowledge_base.__main__ import inspect_evidence, write_corpus
from src.knowledge_base.models import (
    CorpusManifest,
    PageText,
    PreparedCorpus,
    SourceSpec,
)
from src.knowledge_base.prepare import (
    PreparationError,
    extract_pages,
    prepare_document,
    prepare_manifest,
    split_pages,
)


def pdf_bytes(text: str = "Public study rules for students.") -> bytes:
    """Build a tiny native-text PDF fixture without network or external assets."""
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=400)
    page = writer.pages[0]
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 350 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = stream
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def source(key: str = "rules", **changes: object) -> SourceSpec:
    """Return a synthetic source specification with explicit fixture provenance."""
    return SourceSpec.model_validate(
        {
            "key": key,
            "title": "Synthetic rules",
            "url": "https://example.org/rules.pdf",
            "local_path": "rules.pdf",
            **changes,
        }
    )


def split(pages: list[PageText], exclusions: list[int] | None = None, **options: int):
    """Chunk synthetic pages using stable fixture source/version identities."""
    return split_pages(pages, "source", "version", exclusions or [], **options)


def test_real_pdf_extraction_and_roundtrip(tmp_path: Path) -> None:
    """Native extraction produces evidence that survives JSON validation."""
    document = prepare_document(source(), "test", pdf_bytes())
    assert "Public study rules" in document.pages[0].text
    corpus = PreparedCorpus(corpus_id="test", documents=[document])
    output = tmp_path / "corpus.json"
    write_corpus(corpus, output)
    restored = PreparedCorpus.model_validate_json(output.read_text(encoding="utf-8"))
    assert restored == corpus


def test_stable_ids_and_version_changes() -> None:
    """Re-ingestion is stable; changed documents and different sources do not collide."""
    original = prepare_document(source(), "test", pdf_bytes())
    repeated = prepare_document(source(), "test", pdf_bytes())
    changed = prepare_document(source(), "test", pdf_bytes("Updated study rules."))
    other_source = prepare_document(source("other"), "test", pdf_bytes())
    other_corpus = prepare_document(source(), "other", pdf_bytes())
    assert original.chunks == repeated.chunks
    assert original.source_id == changed.source_id
    assert original.chunks[0].id != changed.chunks[0].id
    assert (
        len({doc.chunks[0].id for doc in [original, other_source, other_corpus]}) == 3
    )


def test_article_continuation_and_exact_offsets() -> None:
    """Article labels survive page breaks and text remains an exact source slice."""
    pages = [
        PageText(page=1, text="1. člen\nPrvo pravilo.\n2. člen\nZačetek."),
        PageText(page=2, text="Nadaljevanje.\n3. člen\nTretje pravilo."),
    ]
    chunks = split(pages)
    assert [(c.page, c.article) for c in chunks] == [
        (1, "1"),
        (1, "2"),
        (2, "2"),
        (2, "3"),
    ]
    assert all(c.text == pages[c.page - 1].text[c.start : c.end] for c in chunks)


def test_section_heading_resets_article() -> None:
    """A new Roman-numbered section is not attributed to the preceding article."""
    chunks = split(
        [
            PageText(
                page=1,
                text="1. člen\nPravilo.\nII. Drugo poglavje\n2. člen\nDrugo pravilo.",
            )
        ]
    )
    assert [c.article for c in chunks] == ["1", None, "2"]


def test_page_headers_do_not_create_false_article_citations() -> None:
    """Trim configured edge lines without changing raw offsets or article carryover."""
    pages = [
        PageText(page=1, text="Header\r\n1. člen\r\nPravilo.\r\nFooter\r\n"),
        PageText(page=2, text="Header\r\n2. člen\r\nDrugo pravilo.\r\nFooter\r\n"),
    ]
    chunks = split_pages(
        pages, "source", "version", [], excluded_edge_lines=["Header", "Footer"]
    )
    assert [(c.page, c.article) for c in chunks] == [(1, "1"), (2, "2")]
    assert all("Header" not in c.text and "Footer" not in c.text for c in chunks)
    assert all(c.text == pages[c.page - 1].text[c.start : c.end] for c in chunks)


def test_exclusions_reset_article_and_preserve_page_numbers() -> None:
    """Skipping cover/forms cannot shift physical citations or inherit unrelated labels."""
    pages = [
        PageText(page=1, text="1. člen\nPravilo."),
        PageText(page=2, text=""),
        PageText(page=3, text="Nova vsebina."),
    ]
    chunks = split(pages, [2])
    assert [(c.page, c.article) for c in chunks] == [(1, "1"), (3, None)]


@pytest.mark.parametrize("max_chars,overlap", [(99, 0), (100, -1), (100, 100)])
def test_invalid_chunk_options(max_chars: int, overlap: int) -> None:
    """Invalid overlap/size fails instead of looping or dropping text."""
    with pytest.raises(PreparationError):
        split([PageText(page=1, text="Text")], max_chars=max_chars, overlap=overlap)


def test_long_passage_is_bounded_and_has_no_nonspace_gaps() -> None:
    """Overlapping splits cover the complete article, including long tokens."""
    text = "1. člen\n" + "Besedilo pravila. " * 50 + "x" * 210
    chunks = split([PageText(page=1, text=text)], max_chars=100, overlap=20)
    covered = {i for c in chunks for i in range(c.start, c.end)}
    assert all(len(c.text) <= 100 and c.article == "1" for c in chunks)
    assert all(
        i in covered for i, character in enumerate(text) if not character.isspace()
    )


def test_duplicate_text_is_not_deduplicated_across_locations() -> None:
    """Identical page text keeps two distinct citations."""
    chunks = split([PageText(page=i, text="Enaka vsebina.") for i in [1, 2]])
    assert len(chunks) == 2
    assert chunks[0].id != chunks[1].id


@pytest.mark.parametrize(
    "pages,exclusions",
    [
        ([], []),
        ([PageText(page=1, text="")], []),
        ([PageText(page=1, text="Text")], [2]),
        ([PageText(page=1, text="Text")], [1]),
    ],
)
def test_unusable_evidence_fails(pages: list[PageText], exclusions: list[int]) -> None:
    """Empty/scanned or invalidly excluded inputs cannot masquerade as a successful corpus."""
    with pytest.raises(PreparationError):
        split(pages, exclusions)


def test_invalid_and_encrypted_pdf() -> None:
    """Parser failures and encryption produce actionable errors."""
    with pytest.raises(PreparationError, match="extraction failed"):
        extract_pages(b"not a PDF")
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("password")
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(PreparationError, match="Encrypted"):
        extract_pages(output.getvalue())


def test_manifest_relative_paths_and_contextual_errors(tmp_path: Path) -> None:
    """Resolve files next to the manifest, not against the shell working directory."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        CorpusManifest(corpus_id="test", sources=[source()]).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(PreparationError, match="Source rules"):
        prepare_manifest(manifest)
    (tmp_path / "rules.pdf").write_bytes(pdf_bytes())
    assert len(prepare_manifest(manifest).documents) == 1


def test_duplicate_keys_and_bad_exclusions() -> None:
    """Invalid manually maintained metadata is rejected before ingestion."""
    with pytest.raises(ValidationError):
        CorpusManifest(corpus_id="test", sources=[source(), source()])
    with pytest.raises(ValidationError):
        source(excluded_pages=[0])


def test_tampered_citation_rejected() -> None:
    """A stored chunk cannot cite different text from its recorded page."""
    corpus = PreparedCorpus(
        corpus_id="test", documents=[prepare_document(source(), "test", pdf_bytes())]
    )
    payload = corpus.model_dump()
    payload["documents"][0]["chunks"][0]["text"] = "Invented text"
    with pytest.raises(ValidationError, match="Invalid evidence"):
        PreparedCorpus.model_validate(payload)


def test_inspection_returns_sources_and_empty_results() -> None:
    """The diagnostic inspector returns evidence, never fabricates an answer."""
    corpus = PreparedCorpus(
        corpus_id="test", documents=[prepare_document(source(), "test", pdf_bytes())]
    )
    results = inspect_evidence(corpus, "STUDY")
    assert results[0]["url"] == "https://example.org/rules.pdf#page=1"
    assert results[0]["review_status"] == "pending"
    assert inspect_evidence(corpus, "zzzznonexistent") == []


def test_failed_cli_preserves_existing_output(tmp_path: Path) -> None:
    """A failed preparation must not truncate an existing usable export."""
    output = tmp_path / "corpus.json"
    output.write_text("existing corpus", encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AZURE_", "QDRANT_", "MONGO_", "OPENAI_"))
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.knowledge_base",
            "prepare",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 1
    assert "Knowledge-base operation failed" in result.stderr
    assert output.read_text(encoding="utf-8") == "existing corpus"


def test_cli_prepares_and_inspects_without_cloud_configuration(tmp_path: Path) -> None:
    """Exercise the public CLI end to end with no cloud configuration."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        CorpusManifest(corpus_id="test", sources=[source()]).model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "rules.pdf").write_bytes(pdf_bytes())
    output = tmp_path / "corpus.json"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AZURE_", "QDRANT_", "MONGO_", "OPENAI_"))
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.knowledge_base",
            "prepare",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["documents"] == 1
    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.knowledge_base",
            "inspect",
            "--corpus",
            str(output),
            "--query",
            "study",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)[0]["page"] == 1


@pytest.mark.parametrize("output_name", ["manifest.json", "rules.pdf"])
def test_cli_cannot_overwrite_source_material(tmp_path: Path, output_name: str) -> None:
    """An output-path mistake must not overwrite the manifest or original PDF."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        CorpusManifest(corpus_id="test", sources=[source()]).model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "rules.pdf").write_bytes(pdf_bytes())
    output = tmp_path / output_name
    original = output.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.knowledge_base",
            "prepare",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "must not overwrite" in result.stderr
    assert output.read_bytes() == original
