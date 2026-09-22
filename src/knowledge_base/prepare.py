"""Extract and split public PDFs without importing cloud clients."""

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from PyPDF2 import PdfReader, __version__

from src.knowledge_base.models import (
    CorpusManifest,
    EvidenceChunk,
    PageText,
    PreparedCorpus,
    PreparedDocument,
    SourceSpec,
)

ARTICLE = re.compile(
    r"^[ \t]*(\d+[a-z]?)\.[ \t]*člen[ \t\r]*$", re.MULTILINE | re.IGNORECASE
)
SECTION = re.compile(
    r"^[ \t]*(?:[IVXLCDM]+\.[ \t]+\S.*|Končne določbe[ \t\r]*)$", re.MULTILINE
)


class PreparationError(ValueError):
    """A source could not be prepared faithfully."""


def stable_id(*parts: str) -> str:
    """Derive a UUID from an unambiguous tuple, preserving cross-source identity."""
    return str(uuid5(NAMESPACE_URL, json.dumps(parts, ensure_ascii=False)))


def extract_pages(content: bytes) -> list[PageText]:
    """Extract native PDF text; encrypted or malformed input fails explicitly."""
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise PreparationError("Encrypted PDFs are not supported")
        return [
            PageText(page=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"PDF extraction failed: {exc}") from exc


def split_pages(
    pages: list[PageText],
    source_id: str,
    version: str,
    excluded_pages: list[int],
    max_chars: int = 1400,
    overlap: int = 180,
    excluded_edge_lines: list[str] | None = None,
) -> list[EvidenceChunk]:
    """Split by page/article, then bounded characters; preserve exact offsets."""
    if max_chars < 100 or not 0 <= overlap < max_chars:
        raise PreparationError("max_chars must be >= 100 and 0 <= overlap < max_chars")
    if set(excluded_pages) - {page.page for page in pages}:
        raise PreparationError("An excluded page does not exist in this PDF")
    chunks: list[EvidenceChunk] = []
    article: str | None = None
    previous_page = 0
    recipe = f"page-article-v1:{__version__}:{max_chars}:{overlap}"
    for page in pages:
        if page.page in excluded_pages:
            article = None
            continue
        if page.page != previous_page + 1:
            article = None
        previous_page = page.page
        if not page.text.strip():
            raise PreparationError(
                f"Page {page.page} has no native text; review it for OCR or explicitly exclude it"
            )
        ignored = {" ".join(line.split()) for line in excluded_edge_lines or []}
        lines = list(re.finditer(r"[^\n]+", page.text))
        body = [
            line
            for line in lines
            if line.group().strip() and " ".join(line.group().split()) not in ignored
        ]
        if not body:
            raise PreparationError(
                f"Page {page.page} contains only excluded boilerplate"
            )
        body_start, body_end = body[0].start(), body[-1].end()
        # Roman section headings reset article inheritance. Article headings remain
        # in the evidence; citations refer to this document's articles, not quoted ones.
        headings = sorted(
            [
                (m.start(), m.group(1))
                for m in ARTICLE.finditer(page.text)
                if body_start <= m.start() < body_end
            ]
            + [
                (m.start(), None)
                for m in SECTION.finditer(page.text)
                if body_start <= m.start() < body_end
            ]
        )
        boundaries = [(body_start, article)] + headings + [(body_end, None)]
        for index in range(len(boundaries) - 1):
            start, article = boundaries[index]
            stop = boundaries[index + 1][0]
            while start < stop:
                end = min(start + max_chars, stop)
                if end < stop:
                    whitespace = list(re.finditer(r"\s+", page.text[start:end]))
                    if whitespace and whitespace[-1].start() > max_chars // 2:
                        end = start + whitespace[-1].start()
                left, right = start, end
                while left < right and page.text[left].isspace():
                    left += 1
                while right > left and page.text[right - 1].isspace():
                    right -= 1
                if left < right:
                    chunks.append(
                        EvidenceChunk(
                            id=stable_id(
                                source_id,
                                version,
                                recipe,
                                str(page.page),
                                str(left),
                                str(right),
                                hashlib.sha256(
                                    page.text[left:right].encode("utf-8")
                                ).hexdigest(),
                            ),
                            source_id=source_id,
                            version=version,
                            page=page.page,
                            start=left,
                            end=right,
                            article=article,
                            text=page.text[left:right],
                        )
                    )
                if end == stop:
                    break
                start = max(start + 1, end - overlap)
    if not chunks:
        raise PreparationError("No usable evidence remains after page exclusions")
    return chunks


def prepare_document(
    source: SourceSpec,
    corpus_id: str,
    content: bytes,
    max_chars: int = 1400,
    overlap: int = 180,
) -> PreparedDocument:
    """Prepare one immutable content version while retaining all raw page text."""
    if source.web:
        if source.web.format != "html":
            raise PreparationError(
                "TXT/Markdown sources must be prepared through their source catalog"
            )
        from src.knowledge_base.website import prepare_html

        return prepare_html(
            source, corpus_id, content, source.web.selector_id, max_chars, overlap
        )[0]
    version = hashlib.sha256(content).hexdigest()
    source_id = stable_id(corpus_id, source.key)
    pages = extract_pages(content)
    return PreparedDocument(
        source_id=source_id,
        source=source,
        content_sha256=version,
        extraction_version=f"PyPDF2/{__version__}",
        chunking_version=f"page-article-v1:{max_chars}:{overlap}",
        prepared_at=datetime.now(timezone.utc),
        pages=pages,
        chunks=split_pages(
            pages,
            source_id,
            version,
            source.excluded_pages,
            max_chars,
            overlap,
            source.excluded_edge_lines,
        ),
    )


def prepare_manifest(
    manifest_path: Path, max_chars: int = 1400, overlap: int = 180
) -> PreparedCorpus:
    """Prepare local files, resolving paths relative to the manifest location."""
    manifest = CorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    documents = []
    for source in manifest.sources:
        try:
            content = (manifest_path.parent / source.local_path).read_bytes()
            documents.append(
                prepare_document(
                    source, manifest.corpus_id, content, max_chars, overlap
                )
            )
        except (OSError, PreparationError) as exc:
            raise PreparationError(f"Source {source.key}: {exc}") from exc
    return PreparedCorpus(corpus_id=manifest.corpus_id, documents=documents)
