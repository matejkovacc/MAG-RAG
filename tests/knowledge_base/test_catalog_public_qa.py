"""Local document formats and public benchmark adapters, with all networking blocked."""

import json
import socket
from pathlib import Path

import pytest

from src.evaluation.public_qa import import_public, score_public
from src.knowledge_base.catalog import CatalogSource, SourceCatalog, prepare_catalog
from src.knowledge_base.models import PreparedCorpus, citation_url
from src.rag.offline import offline_service
from src.rag.answers import Question


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Catalog tests operate only on public document copies and local fixtures."""

    def fail(*args, **kwargs):
        """No downloads or model requests are part of these tests."""
        raise AssertionError("Network is forbidden")

    monkeypatch.setattr(socket.socket, "connect", fail)


def source(kind, filename):
    """Use clearly labeled fixture metadata rather than invented institutional facts."""
    return {
        "id": kind,
        "institution": "Unit test",
        "title": "Invented fixture",
        "local_path": filename,
        "accessed_on": "2026-09-18",
        "document_type": kind,
        "usage_note": "Invented local unit fixture",
        "public_source_confirmed": True,
        "selector_id": "content",
        "excluded_lines": ["TEST HEADER"],
    }


@pytest.mark.parametrize(
    "kind,content",
    [
        (
            "txt",
            "TEST HEADER\nTestni obrazec je moder. To je izmišljeno besedilo za testiranje.",
        ),
        (
            "markdown",
            "# Barva\nTestni obrazec je moder.\n\n# Velikost\nTEST HEADER\nTestni obrazec je majhen.",
        ),
        (
            "html",
            "<html><nav>TEST NAVIGATION</nav><main id='content'><h1>Barva</h1><p>Testni obrazec je moder. To je izmišljeno besedilo za testiranje. Primer vsebuje dovolj besedila za preverjanje odsekov in citatov.</p></main></html>",
        ),
    ],
)
def test_local_text_formats_and_citation_rendering(tmp_path, kind, content):
    """Offsets and stable IDs survive repeat preparation; missing URL stays absent."""
    (tmp_path / "input.txt").write_text(content, encoding="utf-8")
    catalog = {"corpus_id": "test-documents", "sources": [source(kind, "input.txt")]}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    corpus = prepare_catalog(path, tmp_path / "first")
    repeat = prepare_catalog(path, tmp_path / "second")
    doc = corpus.documents[0]
    assert [chunk.id for chunk in doc.chunks] == [
        chunk.id for chunk in repeat.documents[0].chunks
    ]
    assert "TEST HEADER" not in " ".join(chunk.text for chunk in doc.chunks)
    assert "TEST NAVIGATION" not in " ".join(chunk.text for chunk in doc.chunks)
    assert citation_url(doc.source, doc.chunks[0]) == ""
    assert PreparedCorpus.model_validate_json(corpus.model_dump_json()) == corpus
    with offline_service(corpus) as service:
        answer = service.answer(Question(question="Testni obrazec moder?"))
        assert answer.citations[0].url == ""
        assert answer.citations[0].page is None
        assert answer.citations[0].section
    with pytest.raises(ValueError, match="exists"):
        prepare_catalog(path, tmp_path / "first")


def test_existing_public_pdf_catalog(tmp_path):
    """Prepare the existing official PDFs without altering their frozen pilot corpus."""
    if not Path("data/thesis/raw/fri-study-rules.pdf").exists():
        pytest.skip("Public source PDFs are intentionally not distributed")
    corpus = prepare_catalog(Path("config/evaluation-sources.json"), tmp_path / "pdf")
    assert len(corpus.documents) == 2
    assert sum(len(doc.chunks) for doc in corpus.documents) == 54
    assert all(doc.source.web is None for doc in corpus.documents)
    assert all(doc.source.notes for doc in corpus.documents)


def test_source_declarations_duplicate_content_and_leakage(tmp_path):
    """Require public-source declarations and reject duplicate/Q&A source artifacts."""
    record = source("txt", "test.txt")
    with pytest.raises(ValueError, match="confirmed public"):
        CatalogSource.model_validate({**record, "public_source_confirmed": False})
    with pytest.raises(ValueError, match="Duplicate"):
        SourceCatalog.model_validate({"corpus_id": "test", "sources": [record, record]})
    (tmp_path / "test.txt").write_text("This is an invented fixture passage.")
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {"corpus_id": "test", "sources": [record, {**record, "id": "other"}]}
        )
    )
    with pytest.raises(ValueError, match="Duplicate document"):
        prepare_catalog(path, tmp_path / "duplicate")
    (tmp_path / "test.txt").write_text(
        'Evaluation data: "expected_answer": "not source evidence"'
    )
    path.write_text(json.dumps({"corpus_id": "test", "sources": [record]}))
    with pytest.raises(ValueError, match="evaluation artifact"):
        prepare_catalog(path, tmp_path / "leak")
    record.update(local_path=None, url="https://www.fri.uni-lj.si/test")
    path.write_text(json.dumps({"corpus_id": "test", "sources": [record]}))
    with pytest.raises(ValueError, match="allow-website-fetch"):
        prepare_catalog(path, tmp_path / "network")
    assert not (tmp_path / "network").exists()


@pytest.fixture
def license_record():
    """Test-only provenance, not a claim about any real dataset license."""
    return {
        "dataset_name": "Invented test fixture",
        "dataset_url": "https://example.org/fixture",
        "license": "Unit fixture permission",
        "license_url": "https://example.org/license",
        "checked_on": "2026-09-18",
        "attribution": "Unit fixture",
        "permitted_use_confirmed": True,
    }


def test_public_squad_import_and_score(tmp_path, license_record):
    """Preserve answer alternatives and score correct refusal separately from FRI."""
    path = tmp_path / "squad.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "paragraphs": [
                            {
                                "context": "The fixture form is blue.",
                                "qas": [
                                    {
                                        "id": "one",
                                        "question": "Which colour?",
                                        "answers": [{"text": "blue"}, {"text": "blue"}],
                                        "is_impossible": False,
                                    },
                                    {
                                        "id": "two",
                                        "question": "Who won?",
                                        "answers": [],
                                        "is_impossible": True,
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        )
    )
    dataset = import_public(path, "squad2", license_record, tmp_path / "import")
    assert dataset.domain == "general_qa"
    assert dataset.items[0].answers == ["blue"]
    requests = [
        json.loads(line)
        for line in (tmp_path / "import" / "requests.jsonl").read_text().splitlines()
    ]
    assert set(requests[0]) == {"id", "input"}
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "one", "answer": "Blue!", "refused": False},
                {"id": "two", "answer": "", "refused": True},
            ]
        )
    )
    summary = score_public(
        tmp_path / "import" / "dataset.json",
        predictions,
        "fake-fixture-provider",
        tmp_path / "run",
    )
    assert summary["overall"]["metrics"]["exact_match"] == {"mean": 1, "n": 2}
    assert summary["unanswerable"]["metrics"]["unanswerable_detection"]["mean"] == 1
    assert "gold" not in summary
    assert (tmp_path / "run" / "report.md").exists()
    predictions.write_text(
        json.dumps({"id": "one", "answer": "Blue", "refused": False})
    )
    with pytest.raises(ValueError, match="exactly one"):
        score_public(
            tmp_path / "import" / "dataset.json",
            predictions,
            "fixture",
            tmp_path / "bad",
        )


def test_public_csv_license_and_empty_output(tmp_path, license_record):
    """QAslovene labels stay literal; an empty answer never silently becomes refusal."""
    path = tmp_path / "fixture.csv"
    path.write_text(
        "input,output,type,answerable\nQuestion with options,true,HT,true\nNo answer,,MT,false\n"
    )
    with pytest.raises(ValueError, match="license"):
        import_public(
            path,
            "qaslovene",
            {**license_record, "permitted_use_confirmed": False},
            tmp_path / "unlicensed",
        )
    dataset = import_public(path, "qaslovene", license_record, tmp_path / "licensed")
    assert dataset.items[0].answers == ["true"]
    assert dataset.items[1].answerable is False
    path.write_text("input,output,type\nQuestion,,MT\n")
    with pytest.raises(ValueError, match="explicit"):
        import_public(path, "qaslovene", license_record, tmp_path / "empty")
