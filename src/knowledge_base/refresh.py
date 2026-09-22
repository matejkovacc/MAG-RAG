"""Explicit, bounded website refresh into immutable local corpus candidates."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from urllib.error import HTTPError
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from src.knowledge_base.models import PreparedCorpus, SourceSpec
from src.knowledge_base.prepare import prepare_document
from src.knowledge_base.website import prepare_html


ALLOWED_HOSTS = frozenset({"www.fri.uni-lj.si", "fri.uni-lj.si"})
MAX_BYTES = 20_000_000
USER_AGENT = "MagRagSourceRefresh/1.0"


class RefreshSource(BaseModel):
    """One approved source; discovery never silently expands this list."""

    model_config = ConfigDict(extra="forbid")
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,99}$")
    title: str = Field(min_length=1)
    url: HttpUrl
    format: Literal["html", "pdf"] = "html"
    selector_id: str = "katedre-container"
    excluded_pages: list[int] = Field(default_factory=list)
    excluded_edge_lines: list[str] = Field(default_factory=list)
    notes: str = ""


class RefreshManifest(BaseModel):
    """A separate website corpus, optionally including the existing pilot PDFs."""

    model_config = ConfigDict(extra="forbid")
    corpus_id: str = Field(min_length=1)
    base_corpus: str | None = None
    sources: list[RefreshSource] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_sources(self) -> "RefreshManifest":
        """Reject duplicate identities and unsupported destinations before fetching."""
        if len({source.key for source in self.sources}) != len(self.sources):
            raise ValueError("Duplicate refresh source key")
        for source in self.sources:
            validate_url(str(source.url))
        return self


def validate_url(url: str) -> str:
    """Allow only official FRI HTTPS hosts, including on every redirect."""
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in ALLOWED_HOSTS
        or parts.port not in {None, 443}
        or parts.username
        or parts.password
    ):
        raise ValueError("Website refresh only supports official FRI HTTPS hosts")
    return urldefrag(url)[0]


@dataclass
class FetchResult:
    """An HTTP response with bounded bytes and selected metadata."""

    status: int
    content: bytes
    headers: dict[str, str]
    url: str


class NoRedirect(HTTPRedirectHandler):
    """Require redirects to pass host validation before another request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Return no replacement request; callers examine Location explicitly."""
        return None


def urllib_fetch(url: str) -> FetchResult:
    """Fetch once with certificate verification and a bounded response body."""
    request = Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    )
    try:
        response = build_opener(NoRedirect()).open(request, timeout=25)
    except HTTPError as exc:
        response = exc
    with response:
        content = response.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise ValueError("Website response exceeds size limit")
        return FetchResult(
            response.code,
            content,
            {key.lower(): value for key, value in response.headers.items()},
            url,
        )


def curl_fetch(url: str) -> FetchResult:
    """Use system curl's verified TLS when Python's certificate store is unsuitable."""
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if executable is None:
        raise ValueError("System curl is unavailable; use --transport urllib")
    with tempfile.TemporaryDirectory(prefix="thesis-fetch-") as temporary:
        body = Path(temporary) / "body"
        headers = Path(temporary) / "headers"
        result = subprocess.run(
            [
                executable,
                "--disable",
                "--silent",
                "--show-error",
                "--globoff",
                "--proto",
                "=https",
                "--max-time",
                "25",
                "--max-filesize",
                str(MAX_BYTES),
                "--user-agent",
                USER_AGENT,
                "--header",
                "Accept-Encoding: identity",
                "--dump-header",
                str(headers),
                "--output",
                str(body),
                "--write-out",
                "%{http_code}",
                "--url",
                url,
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"Verified curl download failed (exit {result.returncode})"
            )
        content = body.read_bytes()
        if len(content) > MAX_BYTES:
            raise ValueError("Website response exceeds size limit")
        # A proxy may add a CONNECT response. Select the final HTTP header block.
        blocks = headers.read_text(encoding="iso-8859-1").strip().split("\n\n")
        pairs = [
            line.split(":", 1) for line in blocks[-1].splitlines()[1:] if ":" in line
        ]
        return FetchResult(
            int(result.stdout),
            content,
            {key.lower(): value.strip() for key, value in pairs},
            url,
        )


def fetch_source(url: str, transport: Callable[[str], FetchResult]) -> FetchResult:
    """Follow at most three validated redirects; never retry a failed source."""
    for _ in range(4):
        url = validate_url(url)
        response = transport(url)
        if response.status in {301, 302, 303, 307, 308}:
            if not response.headers.get("location"):
                raise ValueError("Redirect has no Location")
            url = urljoin(url, response.headers["location"])
            continue
        if response.status != 200:
            raise ValueError(f"Website HTTP status {response.status}")
        if len(response.content) > MAX_BYTES:
            raise ValueError("Website response exceeds size limit")
        response.url = url
        return response
    raise ValueError("Website redirect limit exceeded")


def refresh_sources(
    manifest_path: Path,
    output: Path,
    *,
    allow_website_fetch: bool = False,
    previous: Path | None = None,
    transport: Callable[[str], FetchResult] = urllib_fetch,
    delay: float = 1.0,
) -> dict:
    """Create a candidate only when every configured source succeeds; never index it."""
    if not allow_website_fetch:
        raise ValueError("Website requests require --allow-website-fetch")
    if not 0 <= delay <= 60:
        raise ValueError("Delay must be between 0 and 60 seconds")
    manifest = RefreshManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    sources = list(manifest.sources)
    if manifest.base_corpus:
        base = PreparedCorpus.model_validate_json(
            (manifest_path.parent / manifest.base_corpus).read_text(encoding="utf-8")
        )
        if base.corpus_id == manifest.corpus_id:
            raise ValueError("Use a separate corpus ID to preserve the frozen pilot")
        for document in base.documents:
            source = document.source
            if source.web:
                raise ValueError(
                    "Base corpus must contain PDFs; list HTML sources explicitly"
                )
            sources.append(
                RefreshSource(
                    key=source.key,
                    title=source.title,
                    url=source.url,
                    format="pdf",
                    excluded_pages=source.excluded_pages,
                    excluded_edge_lines=source.excluded_edge_lines,
                    notes=source.notes,
                )
            )
    if len(sources) > 50 or len({source.key for source in sources}) != len(sources):
        raise ValueError("Too many sources or duplicate base/website source keys")
    if len({validate_url(str(source.url)) for source in sources}) != len(sources):
        raise ValueError("Duplicate source URL")
    old = None
    if previous:
        old = PreparedCorpus.model_validate_json(previous.read_text(encoding="utf-8"))
        if old.corpus_id != manifest.corpus_id:
            raise ValueError("Previous corpus ID differs from refresh manifest")
    prior = {doc.source.key: doc for doc in old.documents} if old else {}
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    report = {
        "corpus_id": manifest.corpus_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "fetching",
        "published": False,
        "model_api_calls": 0,
        "refresh_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extractor_code_sha256": hashlib.sha256(
            Path(__file__).with_name("website.py").read_bytes()
        ).hexdigest(),
        "sources": [],
        "removed_source_keys": sorted(set(prior) - {item.key for item in sources}),
    }
    documents = []

    def save_report() -> None:
        """Keep partial-run diagnostics even if a later request fails."""
        (output / "refresh-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save_report()
    for index, item in enumerate(sources):
        if index:
            time.sleep(delay)
        record = {"key": item.key, "url": str(item.url), "format": item.format}
        try:
            response = fetch_source(str(item.url), transport)
            content_type = (
                response.headers.get("content-type", "")
                .split(";", 1)[0]
                .lower()
                .strip()
            )
            expected = (
                {"text/html", "application/xhtml+xml"}
                if item.format == "html"
                else {"application/pdf", "application/octet-stream"}
            )
            if content_type not in expected:
                raise ValueError("Unexpected response content type")
            version = hashlib.sha256(response.content).hexdigest()
            raw_path = output / "raw" / f"{item.key}-{version}.{item.format}"
            raw_path.write_bytes(response.content)
            source = SourceSpec(
                key=item.key,
                title=item.title,
                url=response.url,
                local_path=str(raw_path.resolve()),
                notes=item.notes,
                excluded_pages=item.excluded_pages,
                excluded_edge_lines=item.excluded_edge_lines,
            )
            if item.format == "html":
                document, links = prepare_html(
                    source, manifest.corpus_id, response.content, item.selector_id
                )
            else:
                if not response.content.startswith(b"%PDF-"):
                    raise ValueError("Response is not a PDF")
                document = prepare_document(
                    source, manifest.corpus_id, response.content
                )
                links = []
            previous_doc = prior.get(item.key)
            unchanged = False
            if previous_doc:
                # Ignore local archive location/capture time, but detect configuration changes.
                def comparable(doc):
                    """Compare content and extraction settings, excluding capture bookkeeping."""
                    data = doc.model_dump(mode="json", exclude={"prepared_at"})
                    data["source"].pop("local_path", None)
                    data["source"].pop("review_status", None)
                    if data["source"].get("web"):
                        data["source"]["web"].pop("captured_at", None)
                        # FRI embeds volatile scripts outside the selected body. Do
                        # not demand paid embeddings when the evidence is identical.
                        data.pop("content_sha256")
                        for chunk in data["chunks"]:
                            chunk.pop("id")
                            chunk.pop("version")
                    return data

                unchanged = comparable(previous_doc) == comparable(document)
                if unchanged:
                    # Retain exact evidence/metadata so unchanged refreshes need no new vectors.
                    document = previous_doc
            documents.append(document)
            record.update(
                status=(
                    "unchanged"
                    if unchanged
                    else ("changed" if previous_doc else "added")
                ),
                final_url=response.url,
                content_sha256=version,
                evidence_sha256=document.content_sha256,
                raw_changed_since_previous=previous_doc is not None
                and version != previous_doc.content_sha256,
                chunks=len(document.chunks),
                extracted_characters=sum(len(page.text) for page in document.pages),
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                links=links,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            # Do not publish a reduced corpus or silently represent old content as current.
            record.update(
                status="failed", error_type=type(exc).__name__, error=str(exc)[:350]
            )
        report["sources"].append(record)
        save_report()
    if any(item["status"] == "failed" for item in report["sources"]):
        report["status"] = "failed"
        report["next_step"] = (
            "Fix failed sources and run into a new directory; the published database snapshot is unchanged."
        )
    else:
        corpus = PreparedCorpus(corpus_id=manifest.corpus_id, documents=documents)
        (output / "corpus.json").write_text(
            corpus.model_dump_json(indent=2), encoding="utf-8"
        )
        report["status"] = "ready"
        report["documents"] = len(documents)
        report["chunks"] = sum(len(doc.chunks) for doc in documents)
        report["index_update_required"] = (
            old is None
            or bool(report["removed_source_keys"])
            or any(item["status"] != "unchanged" for item in report["sources"])
        )
        report["next_step"] = (
            "Inspect the candidate and approve Azure indexing separately; refresh never publishes vectors."
        )
    save_report()
    return report
