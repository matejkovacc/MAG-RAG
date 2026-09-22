"""Capture one real retrieval per question and compute transparent pilot metrics."""

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from src.knowledge_base.vector_store import SearchHit
from src.rag.answers import AnswerService, Question, Retriever
from src.rag.diagnostics import LiveAnswerFailure

from .models import Benchmark, BenchmarkItem


class CapturingRetriever:
    """Record evidence actually sent to generation without a second embedding call."""

    def __init__(self, delegate: Retriever, snapshot_id: str | None = None) -> None:
        """Optionally enforce the approved live snapshot throughout a batch."""
        self.delegate = delegate
        self.snapshot_id = snapshot_id
        self.hits: list[SearchHit] = []

    def search(self, question: str, limit: int) -> list[SearchHit]:
        """Abort before generation if a concurrent publisher changes the evidence."""
        self.hits = self.delegate.search(question, limit)
        if self.snapshot_id and any(
            hit.snapshot_id != self.snapshot_id for hit in self.hits
        ):
            raise ValueError("Published snapshot changed during evaluation")
        return self.hits


def retrieval_metrics(groups: list[list[str]], ranked_ids: list[str]) -> dict:
    """Score required evidence groups; alternatives within one group are equivalent."""
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Duplicate retrieved chunk IDs")
    result: dict[str, float | None] = {}
    for k in (1, 3, 5):
        found = set(ranked_ids[:k])
        covered = sum(bool(found.intersection(group)) for group in groups)
        result[f"evidence_recall_at_{k}"] = covered / len(groups) if groups else None
        result[f"all_evidence_at_{k}"] = (
            float(covered == len(groups)) if groups else None
        )
    relevant = {identifier for group in groups for identifier in group}
    first = next(
        (i for i, identifier in enumerate(ranked_ids[:5], 1) if identifier in relevant),
        None,
    )
    result["reciprocal_rank_at_5"] = (1 / first if first else 0.0) if groups else None
    return result


def aggregate(rows: list[dict]) -> dict:
    """Include failures as status misses; retain any retrieval completed before failure."""
    keys = [*retrieval_metrics([], []), "status_match", "citation_id_validity"]
    scores = {}
    for key in keys:
        values = [
            row["metrics"][key] for row in rows if row["metrics"][key] is not None
        ]
        scores[key] = {"mean": mean(values) if values else None, "n": len(values)}
    latencies = sorted(row["latency_seconds"] for row in rows)
    return {
        "attempted": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "metrics": scores,
        "latency_seconds": {
            "p50": median(latencies) if latencies else None,
            "p95_nearest_rank": (
                latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None
            ),
        },
    }


def usage_counts(service: AnswerService) -> dict:
    """Read counters if live; offline evaluation performs no provider requests."""
    usage = getattr(service, "usage", None)
    return (
        dict(usage.counters) if usage else {"embedding_requests": 0, "chat_requests": 0}
    )


def run_benchmark(
    service: AnswerService,
    benchmark: Benchmark,
    items: list[BenchmarkItem],
    output: Path,
    metadata: dict,
    *,
    snapshot_id: str | None = None,
) -> dict:
    """Persist every attempt, stop on failure, and leave correctness scores unfilled."""
    output.mkdir(parents=True, exist_ok=False)
    dependencies = {}
    for package in (
        "pydantic",
        "qdrant-client",
        "pymongo",
        "openai",
        "langchain-openai",
    ):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not installed"
    metadata = {
        **metadata,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": dependencies,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("src/evaluation", "src/rag", "src/knowledge_base")
            for path in sorted(Path(folder).glob("*.py"))
        },
        "corpus_sha256": benchmark.corpus_sha256,
        "benchmark_sha256": hashlib.sha256(
            benchmark.model_dump_json().encode()
        ).hexdigest(),
        "mode": service.mode,
        "retrieval_limit": service.retrieval_limit,
        "selected_ids": [item.id for item in items],
        "annotation_status": (
            "draft"
            if any(item.review_status != "reviewed" for item in items)
            else "reviewed"
        ),
        "limitations": [
            "Synthetic development pilot; not representative historical student enquiries.",
            "ID overlap measures retrieval coverage, not factual correctness or entailment.",
            "Human correctness, faithfulness, citation support and semantic similarity are unmeasured.",
            "Source applicability remains subject to human review.",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "benchmark.json").write_text(
        benchmark.model_dump_json(indent=2), encoding="utf-8"
    )
    original = service.retriever
    recorder = CapturingRetriever(original, snapshot_id)
    service.retriever = recorder
    rows = []
    try:
        with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
            for item in items:
                recorder.hits = []
                started = perf_counter()
                answer = None
                error = None
                diagnostic = None
                try:
                    # Only the question goes to RAG, never reference answers or gold IDs.
                    answer = service.answer(Question(question=item.question))
                except Exception as exc:
                    error = type(exc).__name__  # Provider messages can contain secrets.
                    if isinstance(exc, LiveAnswerFailure):
                        diagnostic = exc.diagnostic.model_dump(mode="json")
                elapsed = perf_counter() - started
                hits = recorder.hits
                metrics = retrieval_metrics(
                    item.evidence_groups, [hit.chunk.id for hit in hits]
                )
                metrics["status_match"] = float(
                    answer is not None and answer.status == item.expected_status
                )
                cited = (
                    {
                        identifier
                        for part in answer.parts
                        for identifier in part.citation_ids
                    }
                    if answer
                    else set()
                )
                metrics["citation_id_validity"] = (
                    len(cited.intersection(hit.chunk.id for hit in hits)) / len(cited)
                    if cited
                    else None
                )
                row = {
                    "id": item.id,
                    "category": item.category,
                    "expected_status": item.expected_status,
                    "question": item.question,
                    "reference_answer": item.reference_answer,
                    "review_status": item.review_status,
                    "answer": answer.model_dump(mode="json") if answer else None,
                    "retrieved": [hit.model_dump(mode="json") for hit in hits],
                    "metrics": metrics,
                    "error": error,
                    "error_diagnostic": diagnostic,
                    "latency_seconds": elapsed,
                    "cumulative_usage": usage_counts(service),
                    "human_review": {
                        "reviewer_code": None,
                        "correctness": None,
                        "faithfulness": None,
                        "citation_support": None,
                        "notes": None,
                    },
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if error:
                    break  # No automatic retry or continued spending on a broken setup.
    finally:
        service.retriever = original
    summary = {
        "metadata": metadata,
        "overall": aggregate(rows),
        "by_category": {
            category: aggregate([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "by_expected_status": {
            status: aggregate([row for row in rows if row["expected_status"] == status])
            for status in sorted({row["expected_status"] for row in rows})
        },
        "unattempted_ids": [item.id for item in items[len(rows) :]],
        "usage": usage_counts(service),
        "ragas": "not run",
        "semantic_similarity": None,
        "human_answer_correctness": None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "judge-inputs.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            if row["answer"] is None:
                continue
            response = (
                "\n\n".join(part["text"] for part in row["answer"]["parts"])
                or row["answer"]["message"]
            )
            stream.write(
                json.dumps(
                    {
                        "user_input": row["question"],
                        "response": response,
                        "reference": row["reference_answer"],
                        "retrieved_contexts": [
                            hit["chunk"]["text"] for hit in row["retrieved"]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    lines = [
        "# Synthetic pilot review",
        "",
        f"Mode: **{service.mode}**. Annotations: **{metadata['annotation_status']}**.",
        "",
        "Fill human scores only after reviewing sources. See docs/evaluation.md for the rubric.",
        "",
    ]
    lookup = {item.id: item for item in items}
    for row in rows:
        item = lookup[row["id"]]
        lines.extend(
            [
                f"## {item.id}: {item.question}",
                "",
                f"Expected status: {item.expected_status}",
                "",
                "Draft reference: " + item.reference_answer,
                "",
                "Review notes: " + item.review_notes,
                "",
                "### Reference evidence",
                "",
            ]
        )
        for ref in item.evidence:
            lines.extend(
                [
                    f"[{ref.source_key}, page {ref.page}, article {ref.article}]({ref.url})",
                    "",
                    "> " + ref.excerpt.replace("\n", "\n> "),
                    "",
                ]
            )
        lines.extend(
            [
                "### System output",
                "",
                "```json",
                json.dumps(row["answer"], ensure_ascii=False, indent=2),
                "```",
                "",
                "Reviewer code: ___; correctness (0/1/2): ___; faithfulness (0/1/2): ___; citation support (0/1/2/N/A): ___",
                "",
                "Notes: ___",
                "",
            ]
        )
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
