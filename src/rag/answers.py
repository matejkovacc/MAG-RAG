"""Typed answer orchestration with injected retrieval and generation providers."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.knowledge_base.vector_store import SearchHit


MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARACTERS = 12000


class ConversationTurn(BaseModel):
    """Untrusted conversational context, without reusable source/citation claims."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=2000)
    answer: str = Field(max_length=4000)
    status: Literal["evidence_found", "no_evidence", "needs_clarification"]


class Question(BaseModel):
    """Current enquiry with optional bounded context; standalone calls stay valid."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=2000)
    history: list[ConversationTurn] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS
    )

    @model_validator(mode="after")
    def bound_history(self) -> "Question":
        """Reject excessive context before retrieval or paid requests."""
        if (
            sum(len(turn.question) + len(turn.answer) for turn in self.history)
            > MAX_HISTORY_CHARACTERS
        ):
            raise ValueError("Conversation context exceeds the character allowance")
        return self

    def retrieval_query(self) -> str:
        """Include prior user topics without embedding previous generated claims."""
        if not self.history:
            return self.question
        previous = "\n".join(turn.question for turn in self.history)
        return f"Aktualno vprašanje: {self.question}\nPrejšnja vprašanja (kontekst pogovora):\n{previous}"


class AnswerPart(BaseModel):
    """An answer paragraph and identifiers of the evidence supporting it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=4000)
    citation_ids: list[str] = Field(min_length=1)


class Draft(BaseModel):
    """Provider output before evidence-reference validation."""

    parts: list[AnswerPart]
    status: Literal["evidence_found", "no_evidence", "needs_clarification"] = (
        "evidence_found"
    )
    message: str = Field(default="", max_length=1000)
    model_config = ConfigDict(extra="forbid")


class Citation(BaseModel):
    """Server-owned provenance, never a URL invented by a generator."""

    id: str
    title: str
    url: str
    page: int | None
    section: str | None = None
    captured_at: str | None = None
    article: str | None
    version: str
    review_status: str
    text: str


class Answer(BaseModel):
    """Preview response with explicit simulation and applicability limitations."""

    mode: Literal["simulated", "generated"] = "simulated"
    status: Literal["evidence_found", "no_evidence", "needs_clarification"]
    message: str
    parts: list[AnswerPart] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Retriever(Protocol):
    """Evidence retrieval boundary compatible with a wrapped ThesisVectorIndex."""

    def search(self, question: str, limit: int) -> list[SearchHit]:
        """Return ordered evidence for one fixed corpus."""
        ...


class Generator(Protocol):
    """Generation boundary shared by simulated and explicitly approved live modes."""

    def generate(
        self,
        question: str,
        evidence: list[SearchHit],
        *,
        history: list[ConversationTurn] | None = None,
    ) -> Draft:
        """Treat evidence as data and return only references from the supplied set."""
        ...


class SimulatedGenerator:
    """Quote actual passages verbatim instead of pretending to generate advice."""

    def generate(
        self,
        question: str,
        evidence: list[SearchHit],
        *,
        history: list[ConversationTurn] | None = None,
    ) -> Draft:
        """Produce deterministic excerpts; the question is not executed or interpreted."""
        return Draft(
            parts=[
                AnswerPart(text=hit.chunk.text, citation_ids=[hit.chunk.id])
                for hit in evidence
            ]
        )


class AnswerService:
    """Retrieve, generate, validate references and attach trusted source metadata."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        mode: Literal["simulated", "generated"] = "simulated",
        retrieval_limit: int = 3,
    ) -> None:
        """Inject providers without constructing clients or reading credentials."""
        self.retriever = retriever
        self.generator = generator
        self.mode = mode
        self.retrieval_limit = retrieval_limit

    def answer(self, request: Question) -> Answer:
        """Abstain on empty evidence and reject unknown or duplicate references."""
        evidence = self.retriever.search(
            request.retrieval_query(), limit=self.retrieval_limit
        )
        if not evidence:
            return Answer(
                mode=self.mode,
                status="no_evidence",
                message=(
                    "Ni ujemajočih se odlomkov. Poskusite z izrazi iz pravilnika ali se obrnite na referat."
                ),
            )
        identifiers = {hit.chunk.id for hit in evidence}
        if len(identifiers) != len(evidence):
            raise ValueError("Retriever returned duplicate evidence identifiers")
        generated = (
            self.generator.generate(request.question, evidence, history=request.history)
            if request.history
            else self.generator.generate(request.question, evidence)
        )
        draft = Draft.model_validate(generated.model_dump())
        if draft.status != "evidence_found":
            if draft.parts:
                raise ValueError(
                    "An abstention must not contain unsupported answer parts"
                )
            if draft.status == "needs_clarification" and not draft.message.strip():
                raise ValueError("A clarification must include a question")
            return Answer(
                mode=self.mode,
                status=draft.status,
                message=(
                    draft.message
                    if draft.status == "needs_clarification"
                    else "V razpoložljivih odlomkih ni dovolj podatkov za zanesljiv odgovor. Za pojasnilo se obrnite na referat."
                ),
            )
        if not draft.parts:
            raise ValueError("Generator returned no answer parts")
        used: set[str] = set()
        for part in draft.parts:
            if not set(part.citation_ids) <= identifiers:
                raise ValueError(
                    "Generator referenced evidence outside the retrieved set"
                )
            used.update(part.citation_ids)
        citations = [
            Citation(
                id=hit.chunk.id,
                title=hit.source.title,
                url=hit.citation_url,
                page=None if hit.source.web else hit.chunk.page,
                section=(
                    hit.source.web.sections[hit.chunk.page - 1].heading
                    if hit.source.web
                    else None
                ),
                captured_at=(
                    hit.source.web.captured_at.isoformat() if hit.source.web else None
                ),
                article=hit.chunk.article,
                version=hit.chunk.version,
                review_status=hit.source.review_status,
                text=hit.chunk.text,
            )
            for hit in evidence
            if hit.chunk.id in used
        ]
        warnings = [
            "Spremembe pravilnikov niso samodejno usklajene. Odgovor preverite v navedenih virih."
        ]
        if self.mode == "simulated":
            warnings.insert(
                0,
                "Simulacija prikazuje izvirne odlomke, ne odgovora jezikovnega modela.",
            )
        if any(item.review_status != "reviewed" for item in citations):
            warnings.append("Uporabnost in veljavnost teh virov še nista pregledani.")
        return Answer(
            mode=self.mode,
            status="evidence_found",
            message=(
                "Odgovor na podlagi virov"
                if self.mode == "generated"
                else "Najdeni odlomki za pregled"
            ),
            parts=draft.parts,
            citations=citations,
            warnings=warnings,
        )
