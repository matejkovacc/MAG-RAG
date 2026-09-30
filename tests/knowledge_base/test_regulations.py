"""Offline regulation discovery, source identity and failure-safety regressions."""

import io
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import DecodedStreamObject, NameObject

from src.knowledge_base.discovery import (
    LANDING_PAGE,
    OFFICIAL_HOSTS,
    canonical_url,
    discover_regulations,
    document_type,
    official_document_links,
    pisrs_document,
    source_key,
)
from src.knowledge_base.models import (
    PageText,
    PreparedCorpus,
    RegulationMetadata,
    SourceSpec,
)
from src.knowledge_base.prepare import PreparationError, prepare_document, split_pages
from src.knowledge_base.refresh import FetchResult, fetch_source, refresh_sources
from src.knowledge_base.website import prepare_html
from test_preparation import pdf_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "regulations-landing.html"
HTML = (
    '<div id="katedre-container"><h2>I. Pravila</h2><p>1. člen</p><p>'
    + "Sintetični primer za preizkus razčlenjevanja. " * 8
    + "</p><nav>Izloči meni</nav></div>"
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """All source responses are fixtures; tests must never contact official hosts."""

    def blocked(*args, **kwargs):
        raise AssertionError("Regulation tests cannot access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def manifest(tmp_path):
    """Use the real discovery manifest contract with synthetic linked evidence."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "corpus_id": "regulations-fixture",
                "sources": [],
                "discovery": {"landing_page": LANDING_PAGE},
            }
        ),
        encoding="utf-8",
    )
    return path


def transport(url, *, changed=False, failed=False, landing=None):
    """Return tiny native PDFs/HTML, never actual university rules."""
    if url == LANDING_PAGE:
        return FetchResult(
            200, landing or FIXTURE.read_bytes(), {"content-type": "text/html"}, url
        )
    if failed and url.endswith("rules.pdf"):
        return FetchResult(503, b"", {}, url)
    if url.endswith(".html"):
        return FetchResult(200, HTML.encode(), {"content-type": "text/html"}, url)
    data = pdf_bytes(
        "Changed fixture"
        if changed and url.endswith("rules.pdf")
        else "Synthetic fixture for " + url.rsplit("/", 1)[-1]
    )
    return FetchResult(200, data, {"content-type": "application/pdf"}, url)


def test_discovery_sections_forms_and_types():
    """Use structure and label meaning, including generic PDF/DOCX form links."""
    links = discover_regulations(FIXTURE.read_bytes())
    accepted = [r for r in links if r["included"]]
    excluded = [r for r in links if not r["included"]]
    assert len(accepted) == 6 and len(excluded) == 5
    assert {r["format"] for r in accepted} == {"pdf", "html"}
    assert all(
        "facebook" not in r["url"] and "unrelated" not in r["url"] for r in links
    )
    amendment = accepted[1]["regulation"]
    assert amendment["is_amendment"]
    assert amendment["parent_regulation"] == accepted[0]["key"]
    assert amendment["publication_date"] == "2026-09-08"
    assert amendment["effective_date"] is None
    assert accepted[0]["regulation"]["publication_date"] is None


def test_normative_docx_is_not_misclassified_as_form():
    """An unsupported normative file must be reported, never excluded by suffix."""
    html = FIXTURE.read_bytes().replace(b"/master.pdf", b"/master.docx")
    record = next(
        r for r in discover_regulations(html) if r["url"].endswith("master.docx")
    )
    assert record["included"] and record["format"] == "docx"


def test_stable_identity_survives_url_move_and_link_order():
    """Titles/section identify logical documents, while URLs remain provenance."""
    old = discover_regulations(FIXTURE.read_bytes())
    new = discover_regulations(
        FIXTURE.read_bytes().replace(b"/rules.pdf", b"/new-rules.pdf")
    )
    assert old[0]["key"] == new[0]["key"]
    assert source_key("  PRAVILNIK O DELU ", "general") == source_key(
        "Pravilnik o delu", "general"
    )
    assert canonical_url("https://fri.uni-lj.si/a b.pdf#page=2").endswith("/a%20b.pdf")
    assert document_type("https://uni-lj.si/a.PDF?download=1") == "pdf"


def test_discovery_rejects_missing_sections_and_duplicate_titles():
    """A redesigned/reduced page cannot accidentally publish a smaller corpus."""
    with pytest.raises(ValueError, match="required regulation section"):
        discover_regulations(
            FIXTURE.read_bytes().replace(
                b"/master.pdf", b"https://evil.test/master.pdf"
            )
        )
    with pytest.raises(ValueError, match="duplicate regulation title"):
        discover_regulations(
            FIXTURE.read_bytes().replace(
                b'<a href="/rules.pdf">',
                b'<a href="/other.pdf">\xc5\xa0tudijski red FRI UL</a><a href="/rules.pdf">',
            )
        )


def test_refresh_idempotency_change_failure_and_url_move(tmp_path):
    """Repeat captures are byte-identical candidates; only affected sources change."""
    config = manifest(tmp_path)
    first = tmp_path / "first"
    report = refresh_sources(
        config, first, allow_website_fetch=True, transport=transport, delay=0
    )
    assert report["status"] == "ready" and report["successful_documents"] == 6
    original = (first / "corpus.json").read_bytes()
    second = tmp_path / "second"
    report = refresh_sources(
        config,
        second,
        previous=first / "corpus.json",
        allow_website_fetch=True,
        transport=transport,
        delay=0,
    )
    assert not report["index_update_required"]
    assert (second / "corpus.json").read_bytes() == original
    third = tmp_path / "changed"
    report = refresh_sources(
        config,
        third,
        previous=first / "corpus.json",
        allow_website_fetch=True,
        transport=lambda u: transport(u, changed=True),
        delay=0,
    )
    assert sum(s["status"] == "changed" for s in report["sources"]) == 1
    changed = PreparedCorpus.model_validate_json((third / "corpus.json").read_bytes())
    old = PreparedCorpus.model_validate_json(original)
    assert changed.documents[0].source_id == old.documents[0].source_id
    assert changed.documents[0].chunks[0].id != old.documents[0].chunks[0].id
    broken = tmp_path / "failed"
    report = refresh_sources(
        config,
        broken,
        previous=first / "corpus.json",
        allow_website_fetch=True,
        transport=lambda u: transport(u, failed=True),
        delay=0,
    )
    assert report["failed_documents"] == 1 and report["successful_documents"] == 5
    assert report["sources"][0]["previous_version_retained"]
    assert not (broken / "corpus.json").exists()
    assert (first / "corpus.json").read_bytes() == original
    moved = tmp_path / "moved"
    html = FIXTURE.read_bytes().replace(b"/rules.pdf", b"/new-rules.pdf")
    refresh_sources(
        config,
        moved,
        previous=first / "corpus.json",
        allow_website_fetch=True,
        transport=lambda u: transport(u, landing=html),
        delay=0,
    )
    candidate = PreparedCorpus.model_validate_json((moved / "corpus.json").read_bytes())
    assert candidate.documents[0].source_id == old.documents[0].source_id
    assert str(candidate.documents[0].source.url).endswith("new-rules.pdf")


def test_failed_discovery_keeps_previous_and_reports(tmp_path):
    """A failed landing request writes diagnostics without an indexable candidate."""
    config = manifest(tmp_path)
    report = refresh_sources(
        config,
        tmp_path / "failed",
        allow_website_fetch=True,
        transport=lambda u: FetchResult(503, b"", {}, u),
        delay=0,
    )
    assert report["status"] == "failed" and "503" in report["discovery_error"]


def test_official_portal_excludes_archives_and_other_regulations():
    """Resolve only the FRI-linked regulation family, keeping its listed amendments."""
    html = """<main><h2>Pravni akti</h2>
    <a href="/new.pdf">Pravilnik o doktorskem študiju – prečiščeno besedilo pdf 200 KB</a>
    <a href="/change.pdf">Spremembe Pravilnika o doktorskem študiju</a>
    <a href="/award.pdf">Pravilnik o podeljevanju nagrad</a>
    <h2>Arhiv pravnih aktov</h2><a href="/old.pdf">Pravilnik o doktorskem študiju (stari)</a></main>"""
    links = official_document_links(
        html.encode(), "https://www.uni-lj.si/portal", "doctoral"
    )
    assert [r["url"] for r in links] == [
        "https://www.uni-lj.si/new.pdf",
        "https://www.uni-lj.si/change.pdf",
    ]


def test_external_redirects_remain_bounded_to_official_hosts():
    """The discovery host scope never permits loopback or unrelated redirects."""
    calls = []

    def redirect(url):
        calls.append(url)
        return FetchResult(302, b"", {"location": "https://127.0.0.1/private"}, url)

    with pytest.raises(ValueError):
        fetch_source("https://www.uni-lj.si/rules", redirect, OFFICIAL_HOSTS)
    assert len(calls) == 1


def test_pisrs_discovers_version_and_file_id_without_hardcoding():
    """Choose numerically latest NPB rather than response order or fixed PDF IDs."""
    record = {
        "data": {
            "evidencniPodatki": {"zunanjiID": "STAT999"},
            "besedilo": {
                "npbVerzije": [
                    {"id": 50, "naziv": "NPB 10"},
                    {"id": 90, "naziv": "NPB 9"},
                ]
            },
            "datoteke": [
                {
                    "npbVerzija": {"id": 50},
                    "datoteke": [{"tip": "PDF_DOCUMENT", "id": 777}],
                }
            ],
        }
    }

    def capture(url):
        data = (
            {"BACKEND_ENDPOINT": "https://pisrs.si/api"}
            if url.endswith("config.json")
            else record
        )
        return SimpleNamespace(content=json.dumps(data).encode())

    result = pisrs_document("https://pisrs.si/pregledPredpisa?id=STAT999", capture)
    assert result["url"] == "https://pisrs.si/api/datoteke/integracije/777"
    assert result["version_label"] == "NPB 10"
    record["data"]["evidencniPodatki"]["veljaDo"] = "2020-01-01"
    with pytest.raises(ValueError, match="ended"):
        pisrs_document("https://pisrs.si/pregledPredpisa?id=STAT999", capture)


def test_article_page_section_and_html_metadata():
    """Articles survive chunk/page splits; arbitrary numbered lists are not articles."""
    pages = [
        PageText(page=1, text="I. PRAVILA\n1. člen\n" + "Besedilo pravila. " * 40),
        PageText(
            page=2,
            text="Nadaljevanje.\n2. člen\nDrugo pravilo.\n1. Prva točka seznama.",
        ),
    ]
    chunks = split_pages(
        pages,
        "source",
        "version",
        [],
        max_chars=140,
        overlap=10,
        preserve_sections=True,
    )
    assert all(c.article == "1" for c in chunks if c.page == 1 and "Besedilo" in c.text)
    assert next(c for c in chunks if c.page == 2).article == "1"
    assert chunks[-1].article == "2" and chunks[-1].section == "I. PRAVILA"
    assert all(c.text == pages[c.page - 1].text[c.start : c.end] for c in chunks)
    source = regulatory_source()
    doc, _ = prepare_html(source, "fixture", HTML.encode(), "katedre-container")
    assert any(c.article == "1" for c in doc.chunks)
    assert all(c.section == "I. Pravila" for c in doc.chunks)
    assert all("Izloči" not in c.text for c in doc.chunks)


def regulatory_source():
    """Attach discovery metadata to a synthetic PDF source."""
    return SourceSpec(
        key="fixture",
        title="Synthetic regulation",
        url=LANDING_PAGE,
        local_path="unused.pdf",
        regulation=RegulationMetadata(
            landing_page=LANDING_PAGE,
            linked_url=LANDING_PAGE,
            subcategory="study_rules",
            document_type="pdf",
        ),
    )


def test_blank_page_is_preserved_but_scanned_or_painted_page_fails():
    """Skip only proven blanks, retaining physical page numbers and raw pages."""
    writer = PdfWriter()
    writer.add_page(PdfReader(io.BytesIO(pdf_bytes())).pages[0])
    writer.add_blank_page(width=400, height=400)
    stream = io.BytesIO()
    writer.write(stream)
    doc = prepare_document(regulatory_source(), "fixture", stream.getvalue())
    assert len(doc.pages) == 2 and doc.source.excluded_pages == [2]
    assert {c.page for c in doc.chunks} == {1}
    graphics = DecodedStreamObject()
    graphics.set_data(b"0 0 200 200 re f")
    writer.pages[1][NameObject("/Contents")] = graphics
    stream = io.BytesIO()
    writer.write(stream)
    with pytest.raises(PreparationError, match="no native text"):
        prepare_document(regulatory_source(), "fixture", stream.getvalue())


def test_malformed_pdf_continues_other_documents(tmp_path):
    """A 200/PDF header cannot turn corrupt evidence into an empty replacement."""

    def corrupt(url):
        return (
            FetchResult(200, b"%PDF-broken", {"content-type": "application/pdf"}, url)
            if url.endswith("rules.pdf")
            else transport(url)
        )

    report = refresh_sources(
        manifest(tmp_path),
        tmp_path / "run",
        allow_website_fetch=True,
        transport=corrupt,
        delay=0,
    )
    assert report["failed_documents"] == 1 and report["successful_documents"] == 5
