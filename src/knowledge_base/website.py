"""Extract versioned FRI HTML evidence without scripts, cloud clients or telemetry."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from src.knowledge_base.models import (
    PageText,
    PreparedDocument,
    SourceSpec,
    WebMetadata,
    WebSection,
)
from src.knowledge_base.prepare import PreparationError, split_pages, stable_id


@dataclass
class Element:
    """Small HTML tree used only to isolate configured content and retain structure."""

    tag: str
    attrs: dict[str, str]
    children: list = field(default_factory=list)


class ContentParser(HTMLParser):
    """Parse static HTML; no JavaScript, stylesheets or linked resources execute."""

    VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        """Initialize a bounded document tree."""
        super().__init__(convert_charrefs=True)
        self.root = Element("root", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Retain attributes for heading anchors and content selection."""
        if len(self.stack) > 200:
            raise PreparationError("HTML nesting exceeds the extraction limit")
        node = Element(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        """Handle self-closing markup without leaving an open frame."""
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        """Recover from unmatched closing tags without widening content scope."""
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        """Preserve text for deterministic whitespace normalization."""
        self.stack[-1].children.append(data)


def descendants(node: Element):
    """Yield elements in document order."""
    yield node
    for child in node.children:
        if isinstance(child, Element):
            yield from descendants(child)


def ignored(node: Element) -> bool:
    """Remove navigation and executable/hidden text, retaining collapsed FAQ answers."""
    return (
        node.tag
        in {
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "form",
            "iframe",
            "svg",
            "template",
        }
        or "hidden" in node.attrs
    )


def plain_text(node: Element) -> str:
    """Read inline text while maintaining paragraph/list/table boundaries."""
    if ignored(node):
        return ""
    parts = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        else:
            parts.append(plain_text(child))
            if child.tag in {"p", "div", "li", "br", "tr", "td", "th"}:
                parts.append("\n")
    return "".join(parts)


def normalize(text: str) -> str:
    """Normalize HTML layout whitespace without rewriting source claims."""
    return "\n".join(
        line for raw in text.splitlines() if (line := " ".join(raw.split()))
    )


def table_text(node: Element) -> str:
    """Preserve explicit row/cell boundaries instead of flattening office-hour tables."""
    rows = []
    for row in descendants(node):
        if row.tag != "tr":
            continue
        cells = []
        for cell in row.children:
            if not isinstance(cell, Element) or cell.tag not in {"td", "th"}:
                continue
            if (
                cell.attrs.get("colspan", "1") != "1"
                or cell.attrs.get("rowspan", "1") != "1"
            ):
                raise PreparationError(
                    "Merged HTML table cells require a reviewed extractor"
                )
            cells.append(" ".join(plain_text(cell).split()))
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_html(
    content: bytes, selector_id: str, title: str, url: str
) -> tuple[list[PageText], list[WebSection], list[dict]]:
    """Extract one explicitly selected body and inventory links without following them."""
    try:
        html = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PreparationError(
            "HTML must be UTF-8; review the source encoding"
        ) from exc
    parser = ContentParser()
    parser.feed(html)
    matches = [
        node for node in descendants(parser.root) if node.attrs.get("id") == selector_id
    ]
    if len(matches) != 1:
        raise PreparationError("Expected exactly one configured HTML content element")
    root = matches[0]
    sections: list[WebSection] = []
    pages: list[PageText] = []
    buffer: list[str] = []
    heading = title
    anchor: str | None = root.attrs.get("id")
    links: list[dict] = []

    def flush() -> None:
        """Publish a section only if extracted source text is present."""
        text = normalize("".join(buffer))
        if text:
            pages.append(PageText(page=len(pages) + 1, text=text))
            sections.append(WebSection(heading=heading, anchor=anchor))
        buffer.clear()

    def visit(node: Element) -> None:
        """Retain headings and their following content in the same section."""
        nonlocal heading, anchor
        if ignored(node):
            return
        if node.tag == "table":
            buffer.append("\n" + table_text(node) + "\n")
            return
        candidate_text = normalize(plain_text(node)) if node.tag in {"li", "p"} else ""
        # FRI encodes FAQ questions as list items rather than semantic headings.
        faq_question = (
            node.tag == "li"
            and candidate_text.endswith("?")
            and len(candidate_text) <= 700
        )
        if node.tag in {"h1", "h2", "h3", "h4", "h5", "h6"} or faq_question:
            text = normalize(plain_text(node))
            if text:
                flush()
                heading = text
                anchor = node.attrs.get("id") or next(
                    (
                        item.attrs.get("id") or item.attrs.get("name")
                        for item in descendants(node)
                        if item.tag == "a"
                        and (item.attrs.get("id") or item.attrs.get("name"))
                    ),
                    None,
                )
                buffer.append(text + "\n")
            return
        if node.tag == "a" and node.attrs.get("href"):
            target = urljoin(url, node.attrs["href"])
            if urlsplit(target).scheme in {"http", "https"}:
                links.append({"url": target, "title": normalize(plain_text(node))})
        for child in node.children:
            if isinstance(child, str):
                buffer.append(child)
            else:
                visit(child)
        if node.tag in {"p", "div", "li", "br", "tr", "td", "th", "ul", "ol"}:
            buffer.append("\n")

    visit(root)
    flush()
    if sum(len(page.text) for page in pages) < 100:
        raise PreparationError(
            "Too little HTML evidence; possible error or redesigned page"
        )

    # Include links inside headings/tables as well, without following any of them.
    def collect_links(node: Element) -> None:
        """Inspect only the selected visible body, excluding script/navigation subtrees."""
        if ignored(node):
            return
        if node.tag == "a" and node.attrs.get("href"):
            target = urljoin(url, node.attrs["href"])
            if urlsplit(target).scheme in {"http", "https"}:
                links.append({"url": target, "title": normalize(plain_text(node))})
        for child in node.children:
            if isinstance(child, Element):
                collect_links(child)

    collect_links(root)
    unique = {item["url"]: item for item in links}
    return pages, sections, list(unique.values())


def prepare_html(
    source: SourceSpec,
    corpus_id: str,
    content: bytes,
    selector_id: str,
    max_chars: int = 1400,
    overlap: int = 180,
) -> tuple[PreparedDocument, list[dict]]:
    """Prepare immutable HTML bytes and exact normalized-section chunk offsets."""
    if source.excluded_pages or source.excluded_edge_lines:
        raise PreparationError("PDF exclusions cannot be applied to HTML")
    pages, sections, links = extract_html(
        content, selector_id, source.title, str(source.url)
    )
    version = hashlib.sha256(content).hexdigest()
    source_id = stable_id(corpus_id, source.key)
    captured = datetime.now(timezone.utc)
    web = WebMetadata(captured_at=captured, selector_id=selector_id, sections=sections)
    source = source.model_copy(update={"web": web, "review_status": "pending"})
    chunks = split_pages(pages, source_id, version, [], max_chars, overlap)
    # HTML section indexes must never imply a regulation article or PDF page.
    chunks = [
        chunk.model_copy(
            update={"article": None, "id": stable_id("html-section-v1", chunk.id)}
        )
        for chunk in chunks
    ]
    return (
        PreparedDocument(
            source_id=source_id,
            source=source,
            content_sha256=version,
            extraction_version="stdlib-html-v1",
            chunking_version=f"html-section-v1:{max_chars}:{overlap}",
            prepared_at=captured,
            pages=pages,
            chunks=chunks,
        ),
        links,
    )
