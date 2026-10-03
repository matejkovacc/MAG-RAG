"""Fetch allowlisted FRI pages per question, with request-local provenance."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
import math
import re
import subprocess

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.knowledge_base.models import PreparedDocument, SourceSpec, citation_url
from src.knowledge_base.refresh import (
    RefreshSource,
    FetchResult,
    curl_fetch,
    fetch_source,
    validate_url,
)
from src.knowledge_base.vector_store import SearchHit
from src.knowledge_base.website import prepare_html
from src.rag.answers import WebsiteCheck, Retriever
from src.rag.offline import terms


class WebsiteManifest(BaseModel):
    """A bounded public allowlist; question text cannot add destinations."""

    model_config = ConfigDict(extra="forbid")
    sources: list[RefreshSource] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_sources(self) -> "WebsiteManifest":
        """Reject non-FRI URLs and duplicate/non-HTML sources before any request."""
        urls = [validate_url(str(s.url)) for s in self.sources]
        if len(set(urls)) != len(urls) or len({s.key for s in self.sources}) != len(
            urls
        ):
            raise ValueError("Duplicate current-website source")
        if any(s.format != "html" or urlsplit(str(s.url)).query for s in self.sources):
            raise ValueError("Current website sources must be fixed HTML page URLs")
        return self


class CurrentWebsiteRetriever:
    """Combine fresh HTML evidence with a fixed PDF retriever, without indexing.

    Every request fetches the whole small allowlist (four workers maximum). No
    question or history is sent to the website. A failed page supplies no evidence;
    frozen HTML cannot silently replace a failed fresh request. Context variables
    keep checks isolated across concurrent local HTTP requests.
    """

    def __init__(
        self,
        base: Retriever,
        manifest: Path | WebsiteManifest,
        *,
        allow_website_fetch: bool = False,
        transport: Callable[[str], FetchResult] = curl_fetch,
    ) -> None:
        """Validate configuration without fetching or constructing model clients."""
        if not allow_website_fetch:
            raise ValueError("Current website access requires --allow-website-fetch")
        self.base = base
        self.manifest = (
            manifest
            if isinstance(manifest, WebsiteManifest)
            else WebsiteManifest.model_validate_json(
                manifest.read_text(encoding="utf-8")
            )
        )
        self.transport = transport
        self._checks: ContextVar[tuple[WebsiteCheck, ...]] = ContextVar(
            "website_checks", default=()
        )

    def website_checks(self) -> list[WebsiteCheck]:
        """Return only this request's source checks, without storing user questions."""
        return list(self._checks.get())

    def _capture(
        self, item: RefreshSource
    ) -> tuple[PreparedDocument | None, WebsiteCheck]:
        """Revalidate redirects and HTML structure, preserving bytes' content hash."""
        checked = datetime.now(timezone.utc).isoformat()
        try:
            response = fetch_source(str(item.url), self.transport)
            if "html" not in response.headers.get("content-type", "").lower():
                raise ValueError("Current page is not HTML")
            source = SourceSpec(
                key=item.key,
                title=item.title,
                url=response.url,
                local_path="request-memory/" + item.key,
                notes="Fetched during this question; capture time is not a validity date. "
                + item.notes,
            )
            document, _ = prepare_html(
                source, "fri-current-website", response.content, item.selector_id
            )
            return document, WebsiteCheck(
                url=response.url,
                title=item.title,
                status="fetched",
                checked_at=document.source.web.captured_at.isoformat(),
                version=document.content_sha256,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return None, WebsiteCheck(
                url=str(item.url),
                title=item.title,
                status="failed",
                checked_at=checked,
                error=type(exc).__name__,
            )

    def search(self, question: str, limit: int) -> list[SearchHit]:
        """Rank fresh chunks lexically and reserve context for relevant PDF evidence."""
        if not 1 <= limit <= 100:
            raise ValueError("Retrieval limit must be between 1 and 100")
        self._checks.set(())
        with ThreadPoolExecutor(max_workers=4) as pool:
            captures = list(pool.map(self._capture, self.manifest.sources))
        self._checks.set(tuple(check for _, check in captures))
        query = terms(question)
        query_years = set(re.findall(r"\b20\d{2}\b", question))
        candidates = []
        for doc, check in captures:
            if doc is None:
                continue
            for chunk in doc.chunks:
                heading = doc.source.web.sections[chunk.page - 1].heading
                if chunk.text.strip() == heading.strip():
                    continue  # A standalone navigation/topic heading is not an answer.
                chunk_terms = terms(chunk.text)
                overlap = query & chunk_terms
                if not overlap:
                    continue
                heading_years = set(re.findall(r"\b20\d{2}\b", heading))
                if query_years and heading_years and not query_years & heading_years:
                    continue
                # Bound length normalization so short boilerplate cannot dominate.
                score = len(overlap) / math.sqrt(max(20, len(chunk_terms)))
                score += len(query & terms(heading)) / max(1, len(query))
                if query_years & heading_years:
                    score += 0.5
                candidates.append(
                    (
                        score,
                        SearchHit(
                            rank=1,
                            score=None,
                            corpus_id="fri-current-website",
                            snapshot_id=check.checked_at,
                            point_id=chunk.id,
                            chunk=chunk,
                            source=doc.source,
                            citation_url=citation_url(doc.source, chunk),
                            retrieval_origin="current_website",
                        ),
                    )
                )
        candidates.sort(key=lambda item: (-item[0], item[1].chunk.id))
        fresh = [hit for _, hit in candidates[:limit]]
        # Published HTML is never presented as current in this mode, including on failures.
        fixed = [
            h for h in self.base.search(question, limit=limit) if h.source.web is None
        ]
        fresh_count = min(len(fresh), max(1, (limit + 1) // 2)) if fixed else len(fresh)
        selected = fresh[:fresh_count] + fixed[: limit - fresh_count]
        selected += fresh[fresh_count : limit - len(selected) + fresh_count]
        return [hit.model_copy(update={"rank": i}) for i, hit in enumerate(selected, 1)]
