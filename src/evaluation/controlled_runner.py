"""Reproducible controlled evaluation around the existing AnswerService boundary."""

import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

from src.rag.answers import Question
from src.rag.diagnostics import LiveAnswerFailure
from .controlled import ControlledDataset, write_new
from .reporting import question_origin, review_state, write_report_files
from .metrics import answer_scores, retrieval_scores
from .runner import CapturingRetriever, usage_counts


RATING_FIELDS = (
    "correctness",
    "relevance",
    "completeness",
    "groundedness",
    "language_clarity",
)
ERROR_TAGS = {
    "hallucination",
    "wrong_source",
    "incomplete",
    "unsupported_claim",
    "should_have_refused",
    "correct_refusal",
}


def summarize(rows: list[dict]) -> dict:
    """Aggregate only defined metrics and expose denominator/failed-call counts."""
    names = sorted({name for row in rows for name in row["metrics"]})
    metrics = {}
    for name in names:
        values = [
            row["metrics"][name] for row in rows if row["metrics"].get(name) is not None
        ]
        metrics[name] = {"mean": mean(values) if values else None, "n": len(values)}
    return {
        "n": len(rows),
        "errors": sum(row["error"] is not None for row in rows),
        "metrics": metrics,
    }


def result_groups(rows: list[dict]) -> dict:
    """Group only by question origin, reference review and answerability."""
    return {
        "overall": summarize(rows),
        "by_origin": {
            key: summarize(
                [
                    row
                    for row in rows
                    if question_origin(row["generation_method"]) == key
                ]
            )
            for key in sorted(
                {question_origin(row["generation_method"]) for row in rows}
            )
        },
        "by_review_status": {
            key: summarize(
                [row for row in rows if review_state(row["review_status"]) == key]
            )
            for key in sorted({review_state(row["review_status"]) for row in rows})
        },
        "answerable": summarize([row for row in rows if row["answerable"]]),
        "unanswerable": summarize([row for row in rows if not row["answerable"]]),
    }


def response_hash(answer) -> str:
    """Bind manual ratings to the exact saved output, including failures."""
    return hashlib.sha256(
        json.dumps(answer, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def run_controlled(
    service,
    dataset: ControlledDataset,
    items,
    output: Path,
    metadata: dict,
    *,
    snapshot_id=None,
    delay=0,
) -> dict:
    """Capture one retrieval, never pass references to the model, and stop on errors."""
    if not 0 <= delay <= 60:
        raise ValueError("Delay must be between 0 and 60 seconds")
    if (
        not items
        or any(
            item not in dataset.items
            or item.review_status not in {"unreviewed", "approved"}
            for item in items
        )
        or len({item.id for item in items}) != len(items)
    ):
        raise ValueError("Select unique eligible cases from the dataset")
    output.mkdir(parents=True, exist_ok=False)
    meta = {
        **metadata,
        "mode": service.mode,
        "seed": dataset.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "corpus_id": dataset.corpus_id,
        "protocol": dataset.protocol,
        "corpus_sha256": dataset.corpus_sha256,
        "dataset_sha256": hashlib.sha256(
            dataset.model_dump_json().encode()
        ).hexdigest(),
        "retrieval_limit": service.retrieval_limit,
        "selected_ids": [item.id for item in items],
        "delay_seconds": delay,
        "model_version": metadata.get("model_version", "not verified"),
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("src/evaluation", "src/rag", "src/knowledge_base")
            for path in sorted(Path(folder).glob("*.py"))
        },
    }
    meta["packages"] = {}
    for package in (
        "pydantic",
        "qdrant-client",
        "pymongo",
        "mongomock",
        "openai",
        "langchain-openai",
    ):
        try:
            meta["packages"][package] = version(package)
        except PackageNotFoundError:
            meta["packages"][package] = "not installed"
    write_new(output / "metadata.json", meta)
    write_new(output / "dataset.json", dataset.model_dump(mode="json"))
    original = service.retriever
    recorder = CapturingRetriever(original, snapshot_id)
    service.retriever = recorder
    rows = []
    run_start = time.perf_counter()
    try:
        with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
            for index, item in enumerate(items):
                if index:
                    time.sleep(delay)
                recorder.hits = []
                started = time.perf_counter()
                answer, error, diagnostic = None, None, None
                try:
                    answer = service.answer(Question(question=item.question))
                except Exception as exc:
                    error = type(exc).__name__
                    if isinstance(exc, LiveAnswerFailure):
                        diagnostic = exc.diagnostic.model_dump(mode="json")
                metrics = {
                    **retrieval_scores(
                        item.source_ids,
                        [hit.chunk.id for hit in recorder.hits],
                        item.required_source_groups,
                    ),
                    **answer_scores(item, answer),
                }
                serialized = answer.model_dump(mode="json") if answer else None
                row = {
                    "id": item.id,
                    "question": item.question,
                    "expected_answer": item.expected_answer,
                    "category": item.category,
                    "difficulty": item.difficulty,
                    "answerable": item.answerable,
                    "expected_status": item.expected_status,
                    "review_status": item.review_status,
                    "generation_method": item.generation_method,
                    "origin": question_origin(item.generation_method),
                    "review_notes": item.review_notes,
                    "reviewer_code": item.reviewer_code,
                    "source_ids": item.source_ids,
                    "supporting_passages": [
                        ref.model_dump() for ref in item.supporting_passages
                    ],
                    "response": (
                        "\n".join(part.text for part in answer.parts) or answer.message
                        if answer
                        else ""
                    ),
                    "answer": serialized,
                    "retrieved": [hit.model_dump(mode="json") for hit in recorder.hits],
                    "metrics": metrics,
                    "error": error,
                    "error_diagnostic": diagnostic,
                    "latency_seconds": time.perf_counter() - started,
                    "response_sha256": response_hash(serialized),
                    "cumulative_usage": usage_counts(service),
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if error:
                    break
    finally:
        service.retriever = original
    summary = {
        "metadata": meta,
        "schema_version": 2,
        **result_groups(rows),
        "unattempted_ids": [item.id for item in items[len(rows) :]],
        "elapsed_seconds": time.perf_counter() - run_start,
        "usage": usage_counts(service),
        "ragas": "not run",
        "semantic_similarity": None,
        "human_ratings": "not collected",
    }
    write_new(output / "summary.json", summary)
    write_new(
        output / "answer-review.json",
        [
            {
                "id": row["id"],
                "response_sha256": row["response_sha256"],
                "reviewer_code": "",
                **{key: None for key in RATING_FIELDS},
                "tags": [],
                "notes": "",
            }
            for row in rows
        ],
    )
    write_report_files(output, summary, rows)
    return summary


def score_human_reviews(run: Path, review_path: Path) -> dict:
    """Aggregate only complete 1–5 human reviews bound to saved outputs."""
    rows = [
        json.loads(line)
        for line in (run / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    lookup = {row["id"]: row for row in rows}
    seen, reviews = set(), []
    for review in json.loads(review_path.read_text(encoding="utf-8")):
        key = review["id"]
        if (
            key not in lookup
            or key in seen
            or review["response_sha256"] != lookup[key]["response_sha256"]
        ):
            raise ValueError("Unknown, duplicate or stale answer review")
        seen.add(key)
        if not review.get("reviewer_code", "").strip():
            if any(
                review.get(field) is not None for field in RATING_FIELDS
            ) or review.get("tags"):
                raise ValueError("Scores require a human reviewer code")
            continue
        if any(
            type(review.get(field)) is not int or not 1 <= review[field] <= 5
            for field in RATING_FIELDS
        ):
            raise ValueError("Complete human scores must be integers from 1 to 5")
        if set(review.get("tags", [])) - ERROR_TAGS:
            raise ValueError("Unknown review tag")
        reviews.append(
            {
                **review,
                "origin": question_origin(lookup[key]["generation_method"]),
                "reference_review": review_state(lookup[key]["review_status"]),
            }
        )
    return {
        "reviewed_n": len(reviews),
        "unreviewed_n": len(rows) - len(reviews),
        "by_origin": rating_groups(reviews, "origin"),
        "by_review_status": rating_groups(reviews, "reference_review"),
        "tag_counts": {
            tag: sum(tag in item["tags"] for item in reviews)
            for tag in sorted(ERROR_TAGS)
        },
        "reviews": reviews,
    }


def rating_groups(reviews: list[dict], field: str) -> dict:
    """Summarize collected human ratings without manufacturing empty-group scores."""
    return {
        key: {
            "n": len(selected),
            "scores": {
                rating: mean(item[rating] for item in selected)
                for rating in RATING_FIELDS
            },
        }
        for key in sorted({item[field] for item in reviews})
        for selected in [[item for item in reviews if item[field] == key]]
    }


def export_saved_run(run: Path, output: Path) -> dict:
    """Reformat a saved controlled run without calls, rescoring or modifying history."""
    rows = [
        json.loads(line)
        for line in (run / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    saved = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    dataset = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 2:
        raise ValueError("Report export requires a saved controlled evaluation run")
    metadata = saved["metadata"]
    annotations = {item["id"]: item for item in dataset["items"]}
    if len({row["id"] for row in rows}) != len(rows) or any(
        row["id"] not in annotations for row in rows
    ):
        raise ValueError("Saved attempts contain duplicate or unknown question IDs")
    if [row["id"] for row in rows] + saved["unattempted_ids"] != metadata[
        "selected_ids"
    ]:
        raise ValueError(
            "Saved attempt order or coverage differs from the recorded selection"
        )
    for row in rows:
        item = annotations[row["id"]]
        for key in (
            "question",
            "expected_answer",
            "review_status",
            "answerable",
            "generation_method",
        ):
            if row[key] != item[key]:
                raise ValueError("Saved attempt differs from the saved dataset")
        row.setdefault("review_notes", item.get("review_notes", ""))
        row.setdefault("reviewer_code", item.get("reviewer_code"))
    summary = {
        "schema_version": 2,
        "metadata": metadata,
        **result_groups(rows),
        **{
            key: saved[key]
            for key in (
                "unattempted_ids",
                "elapsed_seconds",
                "usage",
                "ragas",
                "semantic_similarity",
                "human_ratings",
            )
        },
        "reformatted_from": str(run.resolve()),
        "saved_attempts_sha256": hashlib.sha256(
            (run / "attempts.jsonl").read_bytes()
        ).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "metadata.json",
        "dataset.json",
        "attempts.jsonl",
        "answer-review.json",
    ):
        shutil.copyfile(run / name, output / name)
    write_new(output / "summary.json", summary)
    write_report_files(output, summary, rows)
    return summary
