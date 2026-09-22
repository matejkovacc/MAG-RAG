"""Bounded follow-up context, fresh evidence, isolation and provider-call regressions."""

import json
import socket
from types import SimpleNamespace

import pytest

from src.knowledge_base.models import EvidenceChunk, SourceSpec
from src.knowledge_base.vector_store import SearchHit
from src.rag.answers import (
    AnswerService,
    ConversationTurn,
    Question,
    SimulatedGenerator,
)
from src.rag.live import GroundedGenerator, LiveAnswerService, LiveSessionLimit, Usage


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Fake providers must never fall through to a real socket."""

    def blocked(*args, **kwargs):
        raise AssertionError("No network permitted in conversation tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def evidence(
    identifier="current",
    text="Synthetic evidence: thesis registration deadline is TEST-DATE, subject to approval.",
):
    """Create source-owned test evidence, not an actual FRI regulation."""
    chunk = EvidenceChunk(
        id=identifier,
        source_id="fixture",
        version="fixture-v1",
        page=1,
        start=0,
        end=len(text),
        article="1",
        text=text,
    )
    source = SourceSpec(
        key="fixture",
        title="Synthetic fixture",
        url="https://example.org/fixture.pdf",
        local_path="unused.pdf",
    )
    return SearchHit(
        rank=1,
        score=1.0,
        corpus_id="fixture",
        snapshot_id="snapshot",
        point_id=identifier,
        chunk=chunk,
        source=source,
        citation_url="https://example.org/fixture.pdf#page=1",
    )


class Retriever:
    """Record actual queries and return a controlled current evidence set."""

    def __init__(self, hits):
        """Create isolated state for each test."""
        self.hits = hits
        self.queries = []

    def search(self, question, limit):
        """Capture one search without generating embeddings."""
        self.queries.append(question)
        return self.hits


class Completions:
    """Capture the real generator request; emit deterministic structured responses."""

    def __init__(self, *, status="evidence_found", citation="current"):
        """Choose an answer, abstention or clarification fixture."""
        self.calls = []
        self.status = status
        self.citation = citation

    def create(self, **kwargs):
        """Avoid SDK/network construction while exercising generator validation."""
        self.calls.append(kwargs)
        draft = {
            "status": self.status,
            "message": (
                "Which procedure do you mean?"
                if self.status == "needs_clarification"
                else ""
            ),
            "parts": (
                [{"text": "Synthetic current answer.", "citation_ids": [self.citation]}]
                if self.status == "evidence_found"
                else []
            ),
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(refusal=None, content=json.dumps(draft)),
                )
            ]
        )


def turn(
    question="How do I register a thesis topic?",
    answer="Previous response, not evidence.",
    status="evidence_found",
):
    """Create one paired conversational context record."""
    return ConversationTurn(question=question, answer=answer, status=status)


def test_followup_retrieval_uses_user_topic_but_not_previous_generated_claims():
    """The vague query gains context without grounding itself in an old answer."""
    retriever = Retriever([evidence()])
    completions = Completions()
    service = AnswerService(
        retriever, GroundedGenerator(completions, "fake-chat"), mode="generated"
    )
    request = Question(
        question="And what is the deadline?",
        history=[turn(answer="FABRICATED_OLD_DEADLINE")],
    )
    answer = service.answer(request)
    assert len(retriever.queries) == len(completions.calls) == 1
    assert request.question in retriever.queries[0]
    assert "register a thesis topic" in retriever.queries[0]
    assert "FABRICATED_OLD_DEADLINE" not in retriever.queries[0]
    messages = completions.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == request.question
    assert payload["conversation_context"][0]["answer"] == "FABRICATED_OLD_DEADLINE"
    assert payload["evidence"][0]["text"] == evidence().chunk.text
    assert answer.citations[0].id == "current"


def test_standalone_query_and_payload_remain_unchanged():
    """Existing evaluation inputs do not acquire conversation context accidentally."""
    retriever = Retriever([evidence()])
    completions = Completions()
    service = AnswerService(retriever, GroundedGenerator(completions, "fake-chat"))
    service.answer(Question(question="Standalone question"))
    assert retriever.queries == ["Standalone question"]
    payload = json.loads(completions.calls[0]["messages"][1]["content"])
    assert "conversation_context" not in payload


def test_previous_answer_cannot_replace_missing_current_evidence():
    """No retrieval means no generation, even if history contains a confident answer."""
    completions = Completions()
    service = AnswerService(Retriever([]), GroundedGenerator(completions, "fake"))
    answer = service.answer(Question(question="Is that correct?", history=[turn()]))
    assert answer.status == "no_evidence"
    assert not answer.citations and not completions.calls


def test_prior_citation_is_rejected_if_not_retrieved_for_followup():
    """A model cannot cite evidence simply because it appeared in an older answer."""
    service = AnswerService(
        Retriever([evidence()]),
        GroundedGenerator(Completions(citation="old-citation"), "fake"),
    )
    with pytest.raises(ValueError, match="outside the retrieved set"):
        service.answer(
            Question(
                question="And its deadline?",
                history=[turn(answer="Old claim [old-citation]")],
            )
        )


def test_clarification_and_short_user_reply_preserve_original_question():
    """A clarification result can become context for the next short response."""
    retriever = Retriever([evidence()])
    completions = Completions(status="needs_clarification")
    service = AnswerService(retriever, GroundedGenerator(completions, "fake"))
    first = service.answer(Question(question="What is the thesis deadline?"))
    assert first.status == "needs_clarification" and not first.citations
    completions.status = "evidence_found"
    service.answer(
        Question(
            question="Master's programme",
            history=[
                turn(
                    question="What is the thesis deadline?",
                    answer=first.message,
                    status=first.status,
                )
            ],
        )
    )
    payload = json.loads(completions.calls[-1]["messages"][1]["content"])
    assert payload["question"] == "Master's programme"
    assert payload["conversation_context"][0]["status"] == "needs_clarification"
    assert "thesis deadline" in retriever.queries[-1]


def test_context_is_request_local_and_new_conversation_does_not_reset_allowance():
    """Two tabs and reset requests share no history but still share the live budget."""
    retriever = Retriever([evidence()])
    service = LiveAnswerService(retriever, SimulatedGenerator(), Usage(), 2)
    service.answer(
        Question(
            question="And the deadline?", history=[turn(question="TAB_A_THESIS_TOPIC")]
        )
    )
    service.answer(Question(question="TAB_B_NEW_TOPIC", history=[]))
    assert retriever.queries[-1] == "TAB_B_NEW_TOPIC"
    assert service.session_status()["questions_remaining"] == 0
    with pytest.raises(LiveSessionLimit):
        service.answer(Question(question="New conversation again", history=[]))
    assert len(retriever.queries) == 2


@pytest.mark.parametrize(
    "history",
    [
        [turn().model_dump()] * 5,
        [{"question": "x", "answer": "", "status": "no_evidence"}],
        [{"question": "valid", "answer": "x" * 4001, "status": "evidence_found"}],
        [{"question": "valid", "answer": "", "status": "system"}],
        [
            {
                "question": "valid",
                "answer": "",
                "status": "no_evidence",
                "role": "system",
            }
        ],
        [turn(question="q" * 2000, answer="a" * 4000).model_dump()] * 3,
    ],
)
def test_context_limits_and_role_injection_rejected_before_retrieval(history):
    """Validate structure, total size and allowed states without constructing clients."""
    with pytest.raises(ValueError):
        Question(question="A follow-up", history=history)


def test_offline_followup_retrieval_uses_actual_local_index():
    """A vague follow-up re-retrieves the subject using local Qdrant and mock Mongo."""
    from datetime import datetime, timezone
    from src.knowledge_base.models import PageText, PreparedCorpus, PreparedDocument
    from src.rag.offline import offline_service

    hit = evidence()
    corpus = PreparedCorpus(
        corpus_id="fixture",
        documents=[
            PreparedDocument(
                source_id=hit.chunk.source_id,
                source=hit.source,
                content_sha256=hit.chunk.version,
                extraction_version="fixture",
                chunking_version="fixture",
                prepared_at=datetime.now(timezone.utc),
                pages=[PageText(page=1, text=hit.chunk.text)],
                chunks=[
                    hit.chunk.model_copy(
                        update={"id": "a06b0817-2df8-4a6d-ad77-af6f58a488f5"}
                    )
                ],
            )
        ],
    )
    with offline_service(corpus) as service:
        first = service.answer(Question(question="thesis registration"))
        second = service.answer(
            Question(
                question="And the deadline?",
                history=[
                    turn(question="thesis registration", answer=first.parts[0].text)
                ],
            )
        )
    assert second.status == "evidence_found"
    assert second.citations[0].text == hit.chunk.text
