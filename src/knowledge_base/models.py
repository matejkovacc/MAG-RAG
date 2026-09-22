"""Validated source records and portable document preparation output."""

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_serializer,
    model_validator,
)


class WebSection(BaseModel):
    """A section of extracted HTML, not a physical PDF page."""

    heading: str
    anchor: str | None = None


class WebMetadata(BaseModel):
    """Capture provenance and section locations for a website snapshot."""

    captured_at: datetime
    selector_id: str
    sections: list[WebSection] = Field(min_length=1)
    format: Literal["html", "txt", "markdown"] = "html"

    @model_serializer(mode="wrap")
    def serialize_sections(self, handler):
        """Preserve existing HTML snapshot fingerprints when extending text formats."""
        data = handler(self)
        if self.format == "html":
            data.pop("format", None)
        return data


class SourceSpec(BaseModel):
    """Human-maintained identity and provenance for a public PDF or HTML snapshot."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl | None = None
    local_path: str = Field(min_length=1)
    language: str = "sl"
    issuer: str = "UL FRI"
    excluded_pages: list[int] = Field(default_factory=list)
    excluded_edge_lines: list[str] = Field(default_factory=list)
    review_status: Literal["pending", "reviewed"] = "pending"
    notes: str = ""
    web: WebMetadata | None = None

    @model_serializer(mode="wrap")
    def serialize_source(self, handler):
        """Keep legacy PDF serialization and frozen evaluation hashes unchanged."""
        data = handler(self)
        if self.web is None:
            data.pop("web", None)
        return data

    @model_validator(mode="after")
    def validate_pages(self) -> "SourceSpec":
        """Reject ambiguous or invalid one-based page exclusions."""
        if any(page < 1 for page in self.excluded_pages):
            raise ValueError("Excluded pages must be one-based positive integers")
        if len(set(self.excluded_pages)) != len(self.excluded_pages):
            raise ValueError("Excluded page numbers must be unique")
        return self


class CorpusManifest(BaseModel):
    """Explicit list of sources; never an instruction to crawl arbitrary links."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    corpus_id: str = Field(min_length=1)
    sources: list[SourceSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> "CorpusManifest":
        """Require unique logical identities within a corpus."""
        keys = [source.key for source in self.sources]
        if len(set(keys)) != len(keys):
            raise ValueError("Source keys must be unique within the corpus")
        return self


class PageText(BaseModel):
    """Extracted PDF page or HTML section, numbered from one."""

    page: int = Field(ge=1)
    text: str


class EvidenceChunk(BaseModel):
    """Exact substring of a source page, with a stable Qdrant-compatible ID."""

    id: str
    source_id: str
    version: str
    page: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    article: str | None
    text: str = Field(min_length=1)


class PreparedDocument(BaseModel):
    """Versioned source snapshot; preparation does not assert legal currency."""

    source_id: str
    source: SourceSpec
    content_sha256: str
    extraction_version: str
    chunking_version: str
    prepared_at: datetime
    pages: list[PageText]
    chunks: list[EvidenceChunk]


class PreparedCorpus(BaseModel):
    """JSON exchange format for later MongoDB and Qdrant adapters."""

    schema_version: Literal[1] = 1
    corpus_id: str
    documents: list[PreparedDocument]

    @model_validator(mode="after")
    def verify_evidence(self) -> "PreparedCorpus":
        """Reject duplicate IDs or citations that do not match source text."""
        seen_sources: set[str] = set()
        seen_chunks: set[str] = set()
        for document in self.documents:
            if document.source_id in seen_sources:
                raise ValueError("Duplicate source in prepared corpus")
            seen_sources.add(document.source_id)
            pages = {page.page: page.text for page in document.pages}
            if document.source.web and (
                set(pages) != set(range(1, len(document.source.web.sections) + 1))
                or document.source.excluded_pages
            ):
                raise ValueError("HTML sections must match extracted text locations")
            if len(pages) != len(document.pages):
                raise ValueError("Duplicate page number")
            for chunk in document.chunks:
                if chunk.id in seen_chunks:
                    raise ValueError("Duplicate chunk ID")
                seen_chunks.add(chunk.id)
                if (
                    chunk.source_id != document.source_id
                    or chunk.version != document.content_sha256
                    or chunk.page not in pages
                    or chunk.page in document.source.excluded_pages
                    or chunk.start >= chunk.end
                    or chunk.end > len(pages[chunk.page])
                    or pages[chunk.page][chunk.start : chunk.end] != chunk.text
                ):
                    raise ValueError(f"Invalid evidence location for chunk {chunk.id}")
        return self


def citation_url(source: SourceSpec, chunk: EvidenceChunk) -> str:
    """Link PDFs to physical pages and HTML to original anchors where available."""
    from urllib.parse import quote

    if source.url is None:
        return ""
    base = str(source.url).split("#", 1)[0]
    if source.web:
        anchor = source.web.sections[chunk.page - 1].anchor
        return base + ("#" + quote(anchor, safe="-._~") if anchor else "")
    return base + f"#page={chunk.page}"
