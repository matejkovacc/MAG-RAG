"""Explicitly approved Azure RAG with bounded requests and trusted citations."""

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from dotenv import dotenv_values

from src.knowledge_base.runtime import RetrievalSettings, configured_index
from src.knowledge_base.models import PreparedCorpus
from src.knowledge_base.vector_store import (
    RetrievalError,
    SearchHit,
    ThesisVectorIndex,
    fingerprint,
)
from src.rag.answers import Answer, AnswerService, ConversationTurn, Draft, Question
from src.rag.diagnostics import (
    FailureDiagnostic,
    LiveAnswerFailure,
    Operation,
    exception_failure,
    http_failure,
)


class LiveSessionLimit(ValueError):
    """The approved development session has used its question allowance."""


class Usage:
    """Track cloud HTTP requests and returned tokens without logging prompts or keys."""

    def __init__(self) -> None:
        """Initialize non-secret counters for this server process only."""
        self.counters = dict(
            embedding_requests=0,
            chat_requests=0,
            embedding_tokens=0,
            chat_input_tokens=0,
            chat_output_tokens=0,
            failed_requests=0,
        )
        self.last_failure: FailureDiagnostic | None = None
        self.operation: Operation | None = None

    def begin_question(self) -> None:
        """Clear per-question metadata without resetting allowance or usage counters."""
        self.last_failure = None
        self.operation = None

    def request(self, request: Any) -> None:
        """Count actual SDK HTTP attempts, including any failed response."""
        self.operation = (
            "embedding" if request.url.path.endswith("/embeddings") else "chat"
        )
        key = (
            "embedding_requests"
            if request.url.path.endswith("/embeddings")
            else "chat_requests"
        )
        self.counters[key] += 1

    def response(self, response: Any) -> None:
        """Read only usage metadata; never retain or display response bodies here."""
        response.read()
        if response.status_code != 200:
            self.counters["failed_requests"] += 1
            self.last_failure = http_failure(
                response.status_code,
                response.headers,
                (
                    "embedding"
                    if response.request.url.path.endswith("/embeddings")
                    else "chat"
                ),
            )
            return
        usage = response.json().get("usage", {})
        if response.request.url.path.endswith("/embeddings"):
            self.counters["embedding_tokens"] += usage.get("total_tokens", 0)
        else:
            self.counters["chat_input_tokens"] += usage.get("prompt_tokens", 0)
            self.counters["chat_output_tokens"] += usage.get("completion_tokens", 0)


class SemanticRetriever:
    """Search the published Azure vector space and omit heading-only passages."""

    def __init__(self, index: ThesisVectorIndex, corpus_id: str) -> None:
        """Fix the corpus for all enquiries during this live session."""
        self.index = index
        self.corpus_id = corpus_id

    def search(self, question: str, limit: int) -> list[SearchHit]:
        """Overretrieve once and reserve at most one slot for a split-article neighbor."""
        hits = self.index.search(self.corpus_id, question, limit=min(100, limit * 3))
        primary = [hit for hit in hits if substantive(hit)][:limit]
        primary_ids = {hit.chunk.id for hit in primary}
        selected = []
        expanded = False
        for hit in primary:
            if len(selected) == limit:
                break
            selected.append(hit)
            if not expanded and len(selected) < limit and hit.chunk.article:
                for neighbor in self.index.store.article_neighbors(hit):
                    if neighbor.chunk.id not in primary_ids and substantive(neighbor):
                        selected.append(neighbor)
                        expanded = True
                        break
        return [
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(selected, 1)
        ]


def substantive(hit: SearchHit) -> bool:
    """Apply the existing heading heuristic equally to ranked and neighboring chunks."""
    return (
        len(hit.chunk.text.strip()) >= 80
        and len(re.findall(r"\w+", hit.chunk.text)) >= 12
    )


class GroundedGenerator:
    """Request structured paragraphs; all source identities are validated downstream."""

    def __init__(self, completions: Any, deployment: str) -> None:
        """Accept an already-approved SDK boundary, which can be faked in tests."""
        self.completions = completions
        self.deployment = deployment
        self.prompt = (Path(__file__).parent / "grounded-answer.txt").read_text(
            encoding="utf-8"
        )

    def generate(
        self,
        question: str,
        evidence: list[SearchHit],
        *,
        history: list[ConversationTurn] | None = None,
    ) -> Draft:
        """Generate once; refuse truncated, refused, empty or invalid structured output."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["evidence_found", "no_evidence", "needs_clarification"],
                },
                "message": {"type": "string"},
                "parts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["text", "citation_ids"],
                    },
                },
            },
            "required": ["status", "message", "parts"],
        }
        payload = {
            "question": question,
            "evidence": [
                {
                    "id": hit.chunk.id,
                    "title": hit.source.title,
                    "page": None if hit.source.web else hit.chunk.page,
                    "section": (
                        hit.source.web.sections[hit.chunk.page - 1].heading
                        if hit.source.web
                        else None
                    ),
                    "captured_at": (
                        hit.source.web.captured_at.isoformat()
                        if hit.source.web
                        else None
                    ),
                    "article": hit.chunk.article,
                    "review_status": hit.source.review_status,
                    "source_notes": hit.source.notes,
                    "text": hit.chunk.text,
                }
                for hit in evidence
            ],
        }
        if history:
            payload["conversation_context"] = [turn.model_dump() for turn in history]
        response = self.completions.create(
            model=self.deployment,
            temperature=0,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "student_affairs_answer",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        if not response.choices:
            raise RetrievalError("Chat model returned no answer")
        choice = response.choices[0]
        if (
            choice.finish_reason != "stop"
            or choice.message.refusal
            or not choice.message.content
        ):
            raise RetrievalError("Chat model response was incomplete or refused")
        return Draft.model_validate_json(choice.message.content)


class LiveAnswerService(AnswerService):
    """Serialize live calls and limit paid enquiries for a development session."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: GroundedGenerator,
        usage: Usage,
        max_questions: int,
    ) -> None:
        """Keep failed attempts within the limit; never retry a whole answer silently."""
        super().__init__(retriever, generator, mode="generated", retrieval_limit=5)
        if not 1 <= max_questions <= 100:
            raise ValueError("Live session limit must be between 1 and 100 questions")
        self.usage = usage
        self.max_questions = max_questions
        self.questions_used = 0
        self.lock = RLock()

    def session_status(self) -> dict:
        """Expose remaining enquiries and token usage, not raw questions or secrets."""
        with self.lock:
            return {
                "questions_remaining": self.max_questions - self.questions_used,
                "questions_used": self.questions_used,
                **self.usage.counters,
            }

    def answer(self, request: Question) -> Answer:
        """Reserve a question slot before the first possible external request."""
        with self.lock:
            if self.questions_used >= self.max_questions:
                raise LiveSessionLimit(
                    "Omejitev vprašanj v tej seji je dosežena. Za nadaljevanje je potreben nov odobren zagon."
                )
            self.questions_used += 1
            self.usage.begin_question()
            try:
                return super().answer(request)
            except Exception as exc:
                diagnostic = self.usage.last_failure or exception_failure(
                    exc, self.usage.operation
                )
                raise LiveAnswerFailure(diagnostic) from None


@contextmanager
def live_service(
    corpus_id: str,
    *,
    allow_external_api: bool = False,
    max_questions: int = 10,
    expected_corpus: PreparedCorpus | None = None,
) -> Iterator[LiveAnswerService]:
    """Create only the configured resource's clients after per-run approval."""
    if not allow_external_api:
        raise RetrievalError(
            "External API calls are disabled; explicit approval is required"
        )
    if not 1 <= max_questions <= 100:
        raise ValueError("Live session limit must be between 1 and 100 questions")
    values = {**dotenv_values(".env"), **os.environ}
    settings = RetrievalSettings.from_values(values)
    deployment = (values.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or "").strip()
    if not deployment:
        raise RetrievalError("Missing AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    import httpx
    from openai import AzureOpenAI

    usage = Usage()
    with httpx.Client(
        event_hooks={"request": [usage.request], "response": [usage.response]}
    ) as http:
        with configured_index(
            allow_external_api=True, http_client=http, max_retries=0
        ) as index:
            active = index.store.get_active(corpus_id)
            if (
                not active
                or active["embedding"] != settings.profile.model_dump()
                or active.get("vector_collection") != settings.collection
            ):
                raise RetrievalError(
                    "No matching published Azure snapshot; index the corpus first"
                )
            index.check_collection()
            if expected_corpus is not None:
                require_published_corpus(index, expected_corpus)
            with AzureOpenAI(
                azure_endpoint=settings.profile.endpoint,
                api_key=settings.azure_key,
                api_version=settings.profile.api_version,
                http_client=http,
                max_retries=0,
                timeout=45,
            ) as chat:
                yield LiveAnswerService(
                    SemanticRetriever(index, corpus_id),
                    GroundedGenerator(chat.chat.completions, deployment),
                    usage,
                    max_questions,
                )


def require_published_corpus(index: ThesisVectorIndex, corpus: PreparedCorpus) -> None:
    """Reject unindexed refresh candidates before serving misleading live metadata."""
    active = index.store.get_active(corpus.corpus_id)
    if not active or active["fingerprint"] != fingerprint(corpus, index.profile):
        raise RetrievalError(
            "Selected corpus is not the published snapshot; approve and run indexing first"
        )
