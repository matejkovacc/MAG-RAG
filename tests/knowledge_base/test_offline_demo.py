"""Offline answer, provenance, network isolation and loopback HTTP regression tests."""

import hashlib
import http.client
import json
import socket
import threading
from datetime import datetime, timezone
from http.server import HTTPServer

import pytest

from src.knowledge_base.models import (
    PageText,
    PreparedCorpus,
    PreparedDocument,
    SourceSpec,
)
from src.knowledge_base.prepare import split_pages, stable_id
from src.rag.__main__ import make_handler
from src.rag.answers import AnswerPart, AnswerService, Draft, Question
from src.rag.offline import offline_service


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch):
    """Fail if the preview tries any outgoing connection beyond numeric loopback."""
    original = socket.socket.connect

    def connect(sock, address):
        if not isinstance(address, tuple) or address[0] != "127.0.0.1":
            raise AssertionError("External network is forbidden in offline demo tests")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def service():
    """Use a synthetic fixture; its statements are not actual FRI regulations."""
    text = "1. člen\nSintetični primer: prijava na izpit. <script>alert('fixture')</script>"
    version = hashlib.sha256(text.encode()).hexdigest()
    source_id = stable_id("demo-test", "rules")
    pages = [PageText(page=1, text=text)]
    source = SourceSpec(
        key="rules",
        title="Synthetic test fixture",
        url="https://example.org/rules.pdf",
        local_path="unused.pdf",
    )
    corpus = PreparedCorpus(
        corpus_id="demo-test",
        documents=[
            PreparedDocument(
                source_id=source_id,
                source=source,
                content_sha256=version,
                extraction_version="fixture",
                chunking_version="fixture",
                prepared_at=datetime.now(timezone.utc),
                pages=pages,
                chunks=split_pages(pages, source_id, version, []),
            )
        ],
    )
    with offline_service(corpus) as result:
        yield result


def test_offline_answer_uses_exact_evidence_and_trusted_citations(service):
    """No fabricated answer text/URLs; preserve page, version and pending status."""
    answer = service.answer(Question(question="Kako poteka prijava na izpit?"))
    assert answer.mode == "simulated"
    assert answer.status == "evidence_found"
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert answer.parts[0].text == citation.text
    assert answer.parts[0].citation_ids == [citation.id]
    assert citation.url == "https://example.org/rules.pdf#page=1"
    assert citation.article == "1"
    assert citation.review_status == "pending"
    assert len(citation.version) == 64
    assert any("pregledani" in warning for warning in answer.warnings)


@pytest.mark.parametrize("query", ["astronavti galaksija", "kaj in ali", "12345"])
def test_no_overlap_abstains_without_generation(service, monkeypatch, query):
    """Unanswerable/stopword-only enquiries cannot trigger a model or spurious citation."""

    def fail(*args, **kwargs):
        raise AssertionError("No generation without evidence")

    monkeypatch.setattr(service.generator, "generate", fail)
    answer = service.answer(Question(question=query))
    assert answer.status == "no_evidence"
    assert not answer.parts and not answer.citations


def test_generator_cannot_invent_evidence_references(service):
    """Reject provider output pointing at a non-retrieved document."""

    class InvalidGenerator:
        def generate(self, question, evidence):
            return Draft(
                parts=[AnswerPart(text="Unsupported", citation_ids=["invented"])]
            )

    invalid = AnswerService(service.retriever, InvalidGenerator())
    with pytest.raises(ValueError, match="outside the retrieved set"):
        invalid.answer(Question(question="prijava izpit"))


@pytest.mark.parametrize("text", ["", " ", "a", "x" * 2001])
def test_question_validation(text):
    """Reject whitespace-only, undersized and oversized inputs before retrieval."""
    with pytest.raises(ValueError):
        Question(question=text)


@pytest.fixture
def http_demo(service):
    """Run the real adapter against real local Qdrant with a Mongo test substitute."""
    server = HTTPServer(
        ("127.0.0.1", 0), make_handler(service, {"external_api_calls": False})
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server.server_port
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def request(port, method, path, payload=None, headers=None):
    """Contact only the test server on the loopback interface."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, payload, headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_http_question_flow_and_static_assets(http_demo):
    """Render assets and submit real HTTP JSON without any external provider."""
    status, headers, body = request(http_demo, "GET", "/")
    assert status == 200
    assert b'lang="sl"' in body
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    status, _, script = request(http_demo, "GET", "/app.js")
    assert status == 200
    assert b"textContent" in script and b"innerHTML" not in script
    status, _, body = request(
        http_demo,
        "POST",
        "/ask",
        json.dumps({"question": "prijava izpit"}),
        {"Content-Type": "application/json"},
    )
    assert status == 200
    answer = json.loads(body)
    assert answer["mode"] == "simulated"
    assert answer["citations"][0]["page"] == 1


@pytest.mark.parametrize(
    "payload,headers,expected",
    [
        ('{"question":""}', {"Content-Type": "application/json"}, 422),
        ("not-json", {"Content-Type": "application/json"}, 422),
        ("{}", {"Content-Type": "text/plain"}, 415),
        (
            "{}",
            {"Content-Type": "application/json", "Origin": "https://example.org"},
            403,
        ),
        ("{}", {"Content-Type": "application/json", "Host": "example.org"}, 403),
        pytest.param(
            "x" * 65537, {"Content-Type": "application/json"}, 413, id="oversized-body"
        ),
    ],
)
def test_http_rejects_invalid_and_cross_origin_inputs(
    http_demo, payload, headers, expected
):
    """Validate before retrieval and prevent unrelated websites from using the preview."""
    status, _, _ = request(http_demo, "POST", "/ask", payload, headers)
    assert status == expected


def test_http_does_not_serve_environment_or_source_files(http_demo):
    """The static asset allowlist cannot disclose local files."""
    for path in ["/.env", "/../runtime.py", "/corpus.json"]:
        assert request(http_demo, "GET", path)[0] == 404


def test_http_followup_accepts_bounded_history_without_persisting_it(http_demo):
    """Use real JSON requests for a follow-up, then a separate context-free request."""
    headers = {"Content-Type": "application/json"}
    status, _, body = request(
        http_demo,
        "POST",
        "/ask",
        json.dumps(
            {
                "question": "In potem?",
                "history": [
                    {
                        "question": "prijava izpit",
                        "answer": "Previous simulated answer",
                        "status": "evidence_found",
                    }
                ],
            }
        ),
        headers,
    )
    assert status == 200
    assert json.loads(body)["citations"]
    status, _, body = request(http_demo, "GET", "/status")
    assert status == 200
    assert "history" not in json.loads(body)
    status, _, body = request(
        http_demo, "POST", "/ask", json.dumps({"question": "zzzzzzzzzz"}), headers
    )
    assert status == 200
    assert json.loads(body)["status"] == "no_evidence"


def test_http_rejects_oversized_history(http_demo):
    """The byte limit alone does not replace semantic context size validation."""
    turn = {"question": "prijava izpit", "answer": "", "status": "evidence_found"}
    status, _, _ = request(
        http_demo,
        "POST",
        "/ask",
        json.dumps({"question": "And then?", "history": [turn] * 5}),
        {"Content-Type": "application/json"},
    )
    assert status == 422


def test_local_database_settings_ignore_remote_hosts():
    """Cloud/remote connection configuration cannot redirect the offline demo."""
    from src.rag.offline import local_database_settings

    settings = local_database_settings(
        {
            "MONGO_INITDB_ROOT_USERNAME": "test-user",
            "MONGO_INITDB_ROOT_PASSWORD": "test-password",
            "QDRANT_API_KEY": "test-key",
            "THESIS_MONGO_URI": "mongodb://example.org/",
            "THESIS_QDRANT_URL": "https://example.org/",
            "MONGO_URL": "example.org",
        }
    )
    assert settings["mongo_port"] == 10100
    assert settings["qdrant_port"] == 10060
    assert "example.org" not in str(settings)


def test_local_clients_use_numeric_loopback_and_close(monkeypatch):
    """Verify real-client construction arguments with doubles and no real credentials."""
    import dotenv
    import mongomock
    from qdrant_client import QdrantClient
    import src.rag.offline as offline

    for key, value in {
        "MONGO_INITDB_ROOT_USERNAME": "test-user",
        "MONGO_INITDB_ROOT_PASSWORD": "test-password",
        "QDRANT_API_KEY": "test-key",
        "MONGO_PORT": "10100",
        "QDRANT_HTTP_PORT": "10060",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *args: {})
    calls = {}

    def mongo(*args, **kwargs):
        calls["mongo"] = args
        return mongomock.MongoClient()

    def vectors(**kwargs):
        calls["qdrant"] = kwargs["url"]
        return QdrantClient(":memory:")

    monkeypatch.setattr(offline, "MongoClient", mongo)
    monkeypatch.setattr(offline, "QdrantClient", vectors)
    with offline.demo_clients(True) as (database, vector):
        assert database.admin.command("ping")["ok"]
        assert vector.get_collections().collections == []
    assert calls == {"mongo": ("127.0.0.1", 10100), "qdrant": "http://127.0.0.1:10060"}


def test_local_database_errors_hide_credentials(monkeypatch):
    import dotenv
    import src.rag.offline as offline

    monkeypatch.setattr(dotenv, "dotenv_values", lambda *args: {})

    def fail(values):
        raise RuntimeError("test-private-password")

    monkeypatch.setattr(offline, "local_database_settings", fail)
    with pytest.raises(ValueError) as error:
        with offline.demo_clients(True):
            pytest.fail("Should fail before opening clients")
    assert "test-private-password" not in str(error.value)


def test_live_gate_precedes_credentials_and_sdk(monkeypatch):
    """Importing live code is safe; activation needs the explicit per-run opt-in."""
    import src.rag.live as live

    def forbidden(*args, **kwargs):
        raise AssertionError("Credentials must not be read")

    monkeypatch.setattr(live, "dotenv_values", forbidden)
    with pytest.raises(ValueError, match="disabled"):
        with live.live_service("fixture"):
            pytest.fail("Must refuse before creating clients")


def test_grounded_generator_schema_prompt_and_citations(service):
    """Use a fake SDK response to exercise the exact production generation path."""
    from types import SimpleNamespace
    from src.rag.live import GroundedGenerator

    hits = service.retriever.search("prijava izpit", 3)

    class Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "fake-chat"
            assert kwargs["max_tokens"] == 1200
            assert kwargs["response_format"]["json_schema"]["strict"] is True
            assert "untrusted data" in kwargs["messages"][0]["content"]
            payload = json.loads(kwargs["messages"][1]["content"])
            assert payload["evidence"][0]["id"] == hits[0].chunk.id
            content = json.dumps(
                {
                    "status": "evidence_found",
                    "message": "",
                    "parts": [
                        {
                            "text": "Sintetični odgovor.",
                            "citation_ids": [hits[0].chunk.id],
                        }
                    ],
                }
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=content, refusal=None),
                    )
                ]
            )

    live = AnswerService(
        service.retriever,
        GroundedGenerator(Completions(), "fake-chat"),
        mode="generated",
    )
    answer = live.answer(Question(question="prijava izpit"))
    assert answer.mode == "generated"
    assert answer.parts[0].text == "Sintetični odgovor."
    assert answer.citations[0].text == hits[0].chunk.text
    assert answer.citations[0].url == hits[0].citation_url


@pytest.mark.parametrize(
    "finish,refusal,content",
    [
        ("length", None, "{}"),
        ("content_filter", None, "{}"),
        ("stop", "refused", None),
        ("stop", None, "invalid-json"),
        ("stop", None, None),
    ],
)
def test_incomplete_or_refused_generation_is_not_shown(
    service, finish, refusal, content
):
    """Fail closed without a second paid attempt on incomplete model responses."""
    from types import SimpleNamespace
    from src.rag.live import GroundedGenerator

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=finish,
                        message=SimpleNamespace(content=content, refusal=refusal),
                    )
                ]
            )

    generator = GroundedGenerator(Completions(), "fake-chat")
    with pytest.raises(ValueError):
        generator.generate(
            "prijava izpit", service.retriever.search("prijava izpit", 3)
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "status,message", [("no_evidence", ""), ("needs_clarification", "Kateri program?")]
)
def test_model_can_abstain_or_clarify_without_fabricated_citations(
    service, status, message
):
    """No-evidence and clarification states contain no factual answer parts."""

    class Generator:
        def generate(self, question, evidence):
            return Draft(status=status, message=message, parts=[])

    live = AnswerService(service.retriever, Generator(), mode="generated")
    answer = live.answer(Question(question="prijava izpit"))
    assert answer.status == status and answer.mode == "generated"
    assert not answer.parts and not answer.citations


def test_live_session_limit_stops_calls_before_retrieval(service):
    """The question allowance is enforced on the backend, not only the browser."""
    from src.rag.live import LiveAnswerService, LiveSessionLimit, Usage

    live = LiveAnswerService(service.retriever, service.generator, Usage(), 1)
    live.answer(Question(question="prijava izpit"))
    with pytest.raises(LiveSessionLimit):
        live.answer(Question(question="prijava izpit"))
    assert live.session_status()["questions_remaining"] == 0


def test_semantic_retriever_skips_headings(service):
    """Retrieve candidates once and skip short headings before supplying evidence."""
    from src.rag.live import SemanticRetriever

    hit = service.retriever.search("prijava izpit", 3)[0]
    heading = hit.model_copy(
        update={"chunk": hit.chunk.model_copy(update={"text": "IV. Napredovanje"})}
    )
    substantive = hit.model_copy(
        update={"chunk": hit.chunk.model_copy(update={"text": hit.chunk.text * 3})}
    )

    class Index:
        store = service.retriever.index.store

        def search(self, corpus_id, question, limit):
            assert limit == 15
            return [heading, substantive]

    assert SemanticRetriever(Index(), "fixture").search("prijava izpit", 5) == [
        substantive
    ]


def test_live_usage_counts_only_metadata():
    """Usage accounting does not retain model input, output or credentials."""
    import httpx
    from src.rag.live import Usage

    usage = Usage()
    for path, tokens in [
        ("/embeddings", {"total_tokens": 14}),
        ("/chat/completions", {"prompt_tokens": 100, "completion_tokens": 20}),
    ]:
        req = httpx.Request("POST", "https://example.org" + path)
        response = httpx.Response(
            200, json={"usage": tokens, "secret": "do-not-store"}, request=req
        )
        usage.request(req)
        usage.response(response)
    assert usage.counters["embedding_requests"] == 1
    assert usage.counters["chat_requests"] == 1
    assert usage.counters["embedding_tokens"] == 14
    assert usage.counters["chat_input_tokens"] == 100
    assert "do-not-store" not in str(usage.__dict__)
