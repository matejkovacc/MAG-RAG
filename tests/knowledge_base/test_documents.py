"""PDF inventory includes forms and distinguishes acquisition from extraction."""

import json
from pathlib import Path

import pytest

from src.knowledge_base import documents
from src.knowledge_base.refresh import FetchResult
from test_preparation import pdf_bytes


def fake_sync(monkeypatch, tmp_path, bad_pdf=False):
    landing = (
        Path(__file__)
        .with_name("fixtures")
        .joinpath("regulations-landing.html")
        .read_bytes()
    )

    def normative(_manifest, output, **kwargs):
        output.mkdir()
        (output / "discovered-sources.json").write_text("[]")
        return {"status": "ready"}

    monkeypatch.setattr(documents, "refresh_sources", normative)

    def transport(url):
        content = (
            landing
            if url == documents.LANDING_PAGE
            else (
                b"%PDF-unreadable"
                if bad_pdf and url.endswith("apply.pdf")
                else pdf_bytes()
            )
        )
        return FetchResult(200, content, {}, url)

    return documents.sync_pdfs(
        tmp_path / "audit", allow_website_fetch=True, transport=transport
    )


def test_all_direct_forms_and_prices_included(monkeypatch, tmp_path):
    report = fake_sync(monkeypatch, tmp_path)
    assert report["corpus_complete"]
    urls = {item["url"] for item in report["documents"]}
    assert any(url.endswith("apply.pdf") for url in urls)
    assert any(url.endswith("template.pdf") for url in urls)
    assert any(url.endswith("prices.pdf") for url in urls)
    assert not any(url.endswith("unrelated.pdf") for url in urls)
    assert report["downloaded"] == report["prepared"] == report["direct_pdfs"]


def test_downloaded_but_unreadable_pdf_prevents_complete_corpus(monkeypatch, tmp_path):
    report = fake_sync(monkeypatch, tmp_path, bad_pdf=True)
    assert report["download_complete"] and not report["corpus_complete"]
    assert not (tmp_path / "audit/corpus.json").exists()
    assert any(item.get("extraction") == "needs_review" for item in report["documents"])


def test_pdf_opt_in_precedes_disk_or_network(tmp_path):
    with pytest.raises(ValueError, match="allow-website-fetch"):
        documents.sync_pdfs(tmp_path / "no-write")
    assert not (tmp_path / "no-write").exists()


def test_pricing_pdf_years_preserved():
    links = documents.portal_pdfs(
        b'<main><a href="/2026.pdf">Prices 2026/27</a><a href="/2025.pdf">Prices 2025/26</a><a href="/fees.xlsx">Spreadsheet</a></main>',
        "https://www.uni-lj.si/studij/",
    )
    assert [item["title"] for item in links] == ["Prices 2026/27", "Prices 2025/26"]
