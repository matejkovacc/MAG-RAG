"""Public-source catalog preparation using the existing PDF/HTML and index contracts."""

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .models import (
    PageText,
    PreparedCorpus,
    PreparedDocument,
    SourceSpec,
    WebMetadata,
    WebSection,
)
from .prepare import prepare_document, split_pages, stable_id
from .refresh import MAX_BYTES, curl_fetch, fetch_source, urllib_fetch
from .website import prepare_html


class CatalogSource(BaseModel):
    """Explicit public-source provenance; local paths are relative to the catalog."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    institution: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl | None = None
    local_path: str | None = None
    accessed_on: date
    document_type: Literal["pdf", "html", "txt", "markdown"]
    usage_note: str = Field(min_length=1)
    public_source_confirmed: bool = False
    selector_id: str = "katedre-container"
    excluded_pages: list[int] = Field(default_factory=list)
    excluded_lines: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_origin(self):
        """Require a traceable location and explicit public-source declaration."""
        if not self.url and not self.local_path:
            raise ValueError("Source needs a URL or local path")
        if not self.public_source_confirmed:
            raise ValueError("Only confirmed public documents may enter this catalog")
        return self


class SourceCatalog(BaseModel):
    """Allowlist of source documents, never evaluation questions or correspondence."""

    model_config = ConfigDict(extra="forbid")
    corpus_id: str = Field(min_length=1)
    sources: list[CatalogSource] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self):
        """Reject source identity collisions."""
        if len({item.id for item in self.sources}) != len(self.sources):
            raise ValueError("Duplicate catalog source ID")
        return self


def prepare_text(
    source: SourceSpec,
    corpus_id: str,
    content: bytes,
    kind: str,
    excluded_lines: list[str],
) -> PreparedDocument:
    """Split UTF-8 plain/Markdown text into headings and exact normalized passages."""
    text = content.decode("utf-8-sig")
    if any(
        marker in text
        for marker in (
            '"expected_answer"',
            '"supporting_passages"',
            '"reference_answer"',
        )
    ):
        raise ValueError("Possible evaluation artifact in source documents")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"(?m)(?=^#{1,6}\s+)", text) if kind == "markdown" else [text]
    pages, sections = [], []
    for block in blocks:
        lines = [
            line.rstrip()
            for line in block.strip().splitlines()
            if line.strip() not in excluded_lines
        ]
        clean = "\n".join(lines).strip()
        if not clean:
            continue
        title = (
            re.sub(r"^#{1,6}\s+", "", lines[0]).strip()
            if kind == "markdown" and lines[0].startswith("#")
            else source.title
        )
        pages.append(PageText(page=len(pages) + 1, text=clean))
        sections.append(WebSection(heading=title))
    if not pages:
        raise ValueError("No usable text remains")
    now = datetime.now(timezone.utc)
    version = hashlib.sha256(content).hexdigest()
    source_id = stable_id(corpus_id, source.key)
    metadata = WebMetadata(
        captured_at=now, selector_id="plain-text", sections=sections, format=kind
    )
    source = source.model_copy(update={"web": metadata})
    chunks = split_pages(pages, source_id, version, [], 1400, 180)
    chunks = [
        chunk.model_copy(
            update={"id": stable_id("text-section-v1", kind, chunk.id), "article": None}
        )
        for chunk in chunks
    ]
    return PreparedDocument(
        source_id=source_id,
        source=source,
        content_sha256=version,
        extraction_version="utf8-text-v1",
        chunking_version="text-section-v1:1400:180",
        prepared_at=now,
        pages=pages,
        chunks=chunks,
    )


def prepare_catalog(
    path: Path, output: Path, *, allow_website_fetch=False, transport="urllib"
) -> PreparedCorpus:
    """Prepare all sources into a new directory; index only via the existing opt-in CLI."""
    catalog = SourceCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("Output directory already exists")
    if (
        any(not source.local_path for source in catalog.sources)
        and not allow_website_fetch
    ):
        raise ValueError(
            "URL downloads require --allow-website-fetch; model access remains disabled"
        )
    output.mkdir(parents=True)
    (output / "raw").mkdir()
    documents, hashes = [], set()
    for item in catalog.sources:
        if item.local_path:
            source_path = (path.parent / item.local_path).resolve()
            if any(
                part in {"evaluation", "source_evaluation"}
                for part in source_path.parts
            ):
                raise ValueError("Evaluation directories cannot be source documents")
            content = source_path.read_bytes()
        else:
            response = fetch_source(
                str(item.url), curl_fetch if transport == "curl" else urllib_fetch
            )
            content = response.content
            content_type = (
                response.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            permitted = {
                "pdf": {"application/pdf", "application/octet-stream"},
                "html": {"text/html", "application/xhtml+xml"},
                "txt": {"text/plain"},
                "markdown": {"text/plain", "text/markdown"},
            }
            if content_type not in permitted[item.document_type]:
                raise ValueError(
                    "Remote content type does not match the catalog document type"
                )
        if len(content) > MAX_BYTES:
            raise ValueError("Document exceeds the 20 MB size limit")
        digest = hashlib.sha256(content).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate document content in source catalog")
        hashes.add(digest)
        raw = output / "raw" / f"{item.id}-{digest}.{item.document_type}"
        raw.write_bytes(content)
        source = SourceSpec(
            key=item.id,
            title=item.title,
            url=item.url,
            local_path=str(raw.resolve()),
            issuer=item.institution,
            excluded_pages=item.excluded_pages if item.document_type == "pdf" else [],
            excluded_edge_lines=(
                item.excluded_lines if item.document_type == "pdf" else []
            ),
            notes=json.dumps(
                {
                    "accessed_on": str(item.accessed_on),
                    "usage_note": item.usage_note,
                    "public_source_confirmed": True,
                },
                ensure_ascii=False,
            ),
        )
        if item.document_type == "pdf":
            document = prepare_document(source, catalog.corpus_id, content)
        elif item.document_type == "html":
            document, _ = prepare_html(
                source, catalog.corpus_id, content, item.selector_id
            )
        else:
            document = prepare_text(
                source,
                catalog.corpus_id,
                content,
                item.document_type,
                item.excluded_lines,
            )
        if any(
            marker in page.text
            for page in document.pages
            for marker in (
                '"expected_answer"',
                '"supporting_passages"',
                '"reference_answer"',
            )
        ):
            raise ValueError("Possible evaluation artifact in source documents")
        documents.append(document)
    corpus = PreparedCorpus(corpus_id=catalog.corpus_id, documents=documents)
    (output / "catalog.json").write_text(
        catalog.model_dump_json(indent=2), encoding="utf-8"
    )
    (output / "corpus.json").write_text(
        corpus.model_dump_json(indent=2), encoding="utf-8"
    )
    return corpus
