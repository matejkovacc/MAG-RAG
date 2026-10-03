"""Per-question acquisition, freshness boundaries and trusted citations."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.knowledge_base.models import SourceSpec, citation_url
from src.knowledge_base.refresh import FetchResult
from src.knowledge_base.vector_store import SearchHit
from src.knowledge_base.website import prepare_html
from src.rag.answers import AnswerService, Question, SimulatedGenerator, Draft
from src.rag.current_website import CurrentWebsiteRetriever, WebsiteManifest

URL = "https://fri.uni-lj.si/sl/synthetic-fixture"


def html(value="MONDAY", year="2026/27"):
    """Synthetic office text; no real university policy."""
    return (
        f'<div id="katedre-container"><h2 id="hours">Office hours {year}</h2>'
        f"<p>Office hours registration enquiries: {value}. This is a synthetic fixture "
        "for checking freshness and source identity, not actual institutional advice.</p></div>"
    ).encode()


class Fixed:
    """A saved-corpus retriever with inspectable calls."""

    def __init__(self, hits=()):
        self.hits = hits
        self.calls = []

    def search(self, question, limit):
        self.calls.append((question, limit))
        return list(self.hits)[:limit]


def manifest(url=URL):
    return WebsiteManifest(
        sources=[{"key": "fixture", "title": "Office fixture", "url": url}]
    )


def hit(content):
    document, _ = prepare_html(
        SourceSpec(key="saved", title="Saved page", url=URL, local_path="unused"),
        "saved",
        content,
        "katedre-container",
    )
    chunk = document.chunks[0]
    return SearchHit(
        rank=1,
        score=None,
        corpus_id="saved",
        snapshot_id="saved",
        point_id=chunk.id,
        chunk=chunk,
        source=document.source,
        citation_url=citation_url(document.source, chunk),
    )


def test_new_download_and_hash_for_each_question():
    calls = []
    base = Fixed()

    def transport(url):
        calls.append(url)
        return FetchResult(
            200,
            html("MONDAY" if len(calls) == 1 else "TUESDAY"),
            {"content-type": "text/html"},
            url,
        )

    retriever = CurrentWebsiteRetriever(
        base, manifest(), allow_website_fetch=True, transport=transport
    )
    service = AnswerService(retriever, SimulatedGenerator())
    first = service.answer(Question(question="Office hours"))
    second = service.answer(Question(question="Office hours"))
    assert calls == [URL, URL]  # user text never travels to FRI
    assert len(base.calls) == 2
    assert "MONDAY" in first.parts[0].text and "TUESDAY" in second.parts[0].text
    assert first.citations[0].version != second.citations[0].version
    assert second.citations[0].retrieved_live and second.citations[0].page is None
    assert second.citations[0].url == URL + "#hours"
    assert second.website_checks[0].status == "fetched"


def test_failed_fresh_request_never_falls_back_to_old_html():
    def failed(url):
        return FetchResult(503, b"Unavailable", {}, url)

    retriever = CurrentWebsiteRetriever(
        Fixed([hit(html("STALE"))]),
        manifest(),
        allow_website_fetch=True,
        transport=failed,
    )
    answer = AnswerService(retriever, SimulatedGenerator()).answer(
        Question(question="Office hours")
    )
    assert answer.status == "no_evidence" and not answer.citations
    assert answer.website_checks[0].status == "failed" and answer.warnings


@pytest.mark.parametrize(
    "url",
    [
        "http://fri.uni-lj.si/a",
        "https://evil.example/a",
        "https://fri.uni-lj.si:444/a",
        "https://fri.uni-lj.si/a?query=private",
    ],
)
def test_manifest_rejects_unsafe_or_query_destinations(url):
    with pytest.raises(ValueError):
        manifest(url)


def test_opt_in_before_any_fetch():
    with pytest.raises(ValueError, match="allow-website-fetch"):
        CurrentWebsiteRetriever(Fixed(), manifest())


def test_redirect_cannot_escape_allowlist():
    calls = []

    def redirect(url):
        calls.append(url)
        return FetchResult(302, b"", {"location": "https://evil.example/"}, url)

    retriever = CurrentWebsiteRetriever(
        Fixed(), manifest(), allow_website_fetch=True, transport=redirect
    )
    assert retriever.search("office hours", 3) == []
    assert calls == [URL] and retriever.website_checks()[0].status == "failed"


def test_explicit_academic_year_excludes_conflicting_section():
    content = (
        html("OLD", "2025/26").replace(b"</div>", b"")
        + html("NEW", "2026/27").split(b">", 1)[1]
    )
    retriever = CurrentWebsiteRetriever(
        Fixed(),
        manifest(),
        allow_website_fetch=True,
        transport=lambda url: FetchResult(
            200, content, {"content-type": "text/html"}, url
        ),
    )
    hits = retriever.search("Office hours 2026/27", 3)
    assert hits and all(
        "NEW" in h.chunk.text and "OLD" not in h.chunk.text for h in hits
    )


def test_checks_survive_generator_abstention():
    class Abstain:
        def generate(self, question, evidence):
            return Draft(parts=[], status="no_evidence")

    retriever = CurrentWebsiteRetriever(
        Fixed(),
        manifest(),
        allow_website_fetch=True,
        transport=lambda url: FetchResult(
            200, html(), {"content-type": "text/html"}, url
        ),
    )
    answer = AnswerService(retriever, Abstain()).answer(
        Question(question="office hours")
    )
    assert (
        answer.status == "no_evidence" and answer.website_checks[0].status == "fetched"
    )


def test_concurrent_requests_have_independent_checks():
    retriever = CurrentWebsiteRetriever(
        Fixed(),
        manifest(),
        allow_website_fetch=True,
        transport=lambda url: FetchResult(
            200, html(), {"content-type": "text/html"}, url
        ),
    )

    def request(_):
        retriever.search("office hours", 3)
        return retriever.website_checks()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(request, range(2)))
    assert all(len(checks) == 1 for checks in outcomes)
    assert retriever.website_checks() == []


def test_standalone_heading_does_not_displace_substantive_evidence():
    content = b'<div id="katedre-container"><h2>Office hours</h2><h2>Another topic</h2><p>Office hours: Monday to Friday. Synthetic substantive evidence with a full explanation of how to contact the office, not real university policy.</p></div>'
    retriever = CurrentWebsiteRetriever(
        Fixed(),
        manifest(),
        allow_website_fetch=True,
        transport=lambda url: FetchResult(
            200, content, {"content-type": "text/html"}, url
        ),
    )
    hits = retriever.search("Office hours", 1)
    assert len(hits) == 1 and "Monday to Friday" in hits[0].chunk.text


def test_live_question_limit_is_checked_before_public_fetch():
    from src.rag.live import LiveAnswerService, LiveSessionLimit, Usage

    calls = []

    def transport(url):
        calls.append(url)
        return FetchResult(200, html(), {"content-type": "text/html"}, url)

    retriever = CurrentWebsiteRetriever(
        Fixed(), manifest(), allow_website_fetch=True, transport=transport
    )
    service = LiveAnswerService(retriever, SimulatedGenerator(), Usage(), 1)
    service.answer(Question(question="Office hours"))
    with pytest.raises(LiveSessionLimit):
        service.answer(Question(question="Office hours"))
    assert calls == [URL] and service.session_status()["questions_remaining"] == 0
