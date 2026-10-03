"""Inventory and acquire all PDFs explicitly linked by FRI's document page."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from src.knowledge_base.discovery import (
    LANDING_PAGE,
    OFFICIAL_HOSTS,
    discover_regulations,
)
from src.knowledge_base.models import PreparedCorpus, SourceSpec
from src.knowledge_base.prepare import blank_pdf_pages, extract_pages, prepare_document
from src.knowledge_base.refresh import curl_fetch, fetch_source, refresh_sources
from src.knowledge_base.website import ContentParser, descendants, normalize, plain_text


def portal_pdfs(content: bytes, url: str) -> list[dict]:
    """List PDFs in the linked portal's main body, retaining year-specific titles."""
    parser = ContentParser()
    parser.feed(content.decode("utf-8-sig"))
    roots = [n for n in descendants(parser.root) if n.tag == "main"]
    if len(roots) != 1:
        raise ValueError("Expected one official pricing content container")
    links = {}
    for node in descendants(roots[0]):
        if node.tag != "a" or not node.attrs.get("href"):
            continue
        target = urljoin(url, node.attrs["href"])
        if urlsplit(target).path.lower().endswith(".pdf"):
            links[target] = {"url": target, "title": normalize(plain_text(node))}
    if not links:
        raise ValueError("The linked pricing page contains no PDF links")
    return list(links.values())


def sync_pdfs(
    output: Path,
    *,
    allow_website_fetch: bool = False,
    previous: Path | None = None,
    transport=curl_fetch,
) -> dict:
    """Save full coverage and extraction outcomes; never index or call a model.

    The scope is all direct PDFs plus documents resolved by the existing normative
    portal resolver and all PDFs on the explicitly linked UL pricing page. DOCX,
    XLSX and external GitHub templates remain inventoried but are not PDF content.
    """
    if not allow_website_fetch:
        raise ValueError("Website requests require --allow-website-fetch")
    if output.exists():
        raise ValueError("Use a new PDF audit directory")
    output.mkdir(parents=True)
    (output / "raw").mkdir()
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "landing_page": LANDING_PAGE,
        "documents": [],
        "discovery_errors": [],
    }

    def save():
        """Keep a partial audit available if any source is unavailable."""
        (output / "inventory.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save()
    landing = fetch_source(LANDING_PAGE, transport, OFFICIAL_HOSTS)
    (output / "landing.html").write_bytes(landing.content)
    links = discover_regulations(landing.content, LANDING_PAGE)
    report["landing_sha256"] = hashlib.sha256(landing.content).hexdigest()
    report["landing_links"] = links
    candidates = {}
    for link in links:
        if link["format"] == "pdf":
            candidates[link["url"]] = {
                "title": link["title"],
                "url": link["url"],
                "kind": (
                    "regulation_or_instruction"
                    if link["included"]
                    else "form_template_or_pricing"
                ),
                "discovery_path": [LANDING_PAGE],
            }
    report["direct_pdfs"] = len(candidates)
    normative = refresh_sources(
        Path(__file__).resolve().parents[2] / "config/fri-regulations.json",
        output / "regulations",
        allow_website_fetch=True,
        previous=previous,
        transport=transport,
        delay=0,
    )
    norm_path = output / "regulations/discovered-sources.json"
    if norm_path.exists():
        for source in json.loads(norm_path.read_text(encoding="utf-8")):
            if source["format"] == "pdf":
                candidates.setdefault(
                    source["url"],
                    {
                        "title": source["title"],
                        "url": source["url"],
                        "kind": "regulation_or_instruction",
                        "discovery_path": source["regulation"]["discovery_path"],
                    },
                )
    if normative.get("status") != "ready":
        report["discovery_errors"].append(
            "Normative refresh incomplete; inspect regulations/refresh-report.json"
        )
    pricing = [
        r
        for r in links
        if "cenik storitev" in r["title"].casefold() and r["format"] == "html"
    ]
    for entry in pricing:
        try:
            page = fetch_source(entry["url"], transport, OFFICIAL_HOSTS)
            (output / "pricing-page.html").write_bytes(page.content)
            for link in portal_pdfs(page.content, page.url):
                candidates.setdefault(
                    link["url"],
                    {
                        **link,
                        "kind": "pricing",
                        "discovery_path": [LANDING_PAGE, entry["url"], page.url],
                    },
                )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report["discovery_errors"].append(
                f"Pricing resolution failed: {type(exc).__name__}"
            )
    if len(candidates) > 80:
        raise ValueError("PDF scope exceeds 80 documents; inspect source changes")
    prior_urls = set()
    if previous is not None:
        old = PreparedCorpus.model_validate_json(previous.read_text(encoding="utf-8"))
        prior_urls = {str(d.source.url) for d in old.documents}
    prepared = []
    for url, candidate in candidates.items():
        record = {**candidate, "present_in_previous_corpus": url in prior_urls}
        try:
            response = fetch_source(url, transport, OFFICIAL_HOSTS)
            if not response.content.lstrip().startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF")
            digest = hashlib.sha256(response.content).hexdigest()
            raw = output / "raw" / f"{digest}.pdf"
            raw.write_bytes(response.content)
            record.update(
                status="downloaded",
                resolved_url=response.url,
                sha256=digest,
                bytes=len(response.content),
                local_path=str(raw),
            )
            source = SourceSpec(
                key="pdf-" + hashlib.sha256(url.encode()).hexdigest()[:20],
                title=candidate["title"],
                url=response.url,
                local_path=str(raw.resolve()),
                issuer="Official FRI/UL/PISRS source",
                notes=(
                    f"Document category: {candidate['kind']}. Retrieved from FRI's document directory. "
                    "Forms/templates are not normative rules. Check the applicable academic year for prices. "
                    "Source applicability and extraction require human review."
                ),
            )
            try:
                pages = extract_pages(response.content)
                if any(not page.text.strip() for page in pages):
                    source = source.model_copy(
                        update={
                            "excluded_pages": blank_pdf_pages(response.content, pages)
                        }
                    )
                doc = prepare_document(source, "fri-all-pdfs", response.content)
                prepared.append(doc)
                record.update(
                    extraction="prepared", pages=len(doc.pages), chunks=len(doc.chunks)
                )
            except (OSError, ValueError) as exc:
                record.update(
                    extraction="needs_review", extraction_error=type(exc).__name__
                )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            record.update(status="failed", error=type(exc).__name__)
        report["documents"].append(record)
        save()
    report.update(
        expected_pdfs=len(candidates),
        downloaded=sum(d["status"] == "downloaded" for d in report["documents"]),
        prepared=len(prepared),
        new_to_previous_corpus=sum(url not in prior_urls for url in candidates),
    )
    report["download_complete"] = not report["discovery_errors"] and report[
        "downloaded"
    ] == len(candidates)
    report["corpus_complete"] = report["download_complete"] and len(prepared) == len(
        candidates
    )
    # An incomplete extraction never masquerades as the full knowledge base.
    if report["corpus_complete"]:
        corpus = PreparedCorpus(corpus_id="fri-all-pdfs", documents=prepared)
        (output / "corpus.json").write_text(
            corpus.model_dump_json(indent=2), encoding="utf-8"
        )
    save()
    return report
