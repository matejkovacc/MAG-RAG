"""Plain-language evaluation views, separate from frozen annotations and audit records."""

import csv
from pathlib import Path


ORIGINS = {"official_faq": "Official FAQ", "document_derived": "Created from documents"}
REVIEWS = {
    "approved": "Reviewed",
    "unreviewed": "Needs review",
    "needs_revision": "Needs revision",
    "rejected": "Excluded",
}
METRICS = {
    "hit_rate_at_1": "Correct source in first result",
    "hit_rate_at_3": "Correct source in first 3 results",
    "hit_rate_at_5": "Correct source in first 5 results",
    "recall_at_5": "Required sources found (Recall@5)",
    "reciprocal_rank": "Source ranking (MRR)",
    "exact_match": "Exact answer match",
    "token_f1": "Word overlap (F1)",
    "rouge_l": "Word sequence overlap (ROUGE-L)",
    "unanswerable_detection": "Correct refusal when information is missing",
    "false_refusal": "Refusal despite available information",
    "answer_with_correct_source": "Answers citing a correct source",
}


def question_origin(generation_method: str) -> str:
    """Keep institutional FAQ publication separate from document-derived examples."""
    return "official_faq" if generation_method == "official_faq" else "document_derived"


def review_state(status: str) -> str:
    """Use ordinary review labels while preserving revision and exclusion decisions."""
    return {"approved": "reviewed", "unreviewed": "needs_review"}.get(status, status)


def display_sources(passages: list[dict]) -> list[str]:
    """Show document locations without dumping source hashes into the reading view."""
    sources = []
    for source in passages:
        location = source.get("section") or (
            f"page {source['page']}" if source.get("page") else ""
        )
        title = source["title"] + (" — " + location if location else "")
        if source.get("article"):
            title += f", article {source['article']}"
        label = f"[{title}]({source['url']})" if source.get("url") else title
        if label not in sources:
            sources.append(label)
    return sources or ["No supporting passage is expected for this question."]


def render_questions(items: list[dict], *, answers: bool = False) -> str:
    """Readable questions, references and optional outputs; technical fields stay in JSON."""
    lines = ["# Answers" if answers else "# Questions for review", ""]
    for number, item in enumerate(items, 1):
        lines.extend(
            [
                f"## {number}. {item['question']}",
                "",
                f"Origin: {ORIGINS[question_origin(item['generation_method'])]} · Reference: {REVIEWS[item['review_status']]} · Answerable: {'Yes' if item['answerable'] else 'No'}",
                "",
            ]
        )
        if item.get("expected_status") == "needs_clarification":
            lines.extend(["A clarifying question is expected.", ""])
        lines.extend(
            [
                "**Expected answer**",
                "",
                item["expected_answer"],
                "",
                "**Supporting sources**",
                "",
            ]
        )
        lines.extend(
            "- " + label
            for label in display_sources(item.get("supporting_passages", []))
        )
        if answers:
            lines.extend(
                [
                    "",
                    "**Actual answer**",
                    "",
                    item.get("response") or "No answer was produced.",
                    "",
                ]
            )
            citations = (item.get("answer") or {}).get("citations", [])
            if citations:
                lines.extend(["**Sources cited in the answer**", ""])
                lines.extend("- " + label for label in display_sources(citations))
                lines.append("")
            if item.get("error"):
                lines.extend([f"Execution error: {item['error']}", ""])
            lines.extend(["| Score | Value |", "| --- | ---: |"])
            for key in (
                "exact_match",
                "token_f1",
                "rouge_l",
                "answer_with_correct_source",
                "unanswerable_detection",
            ):
                value = item["metrics"].get(key)
                lines.append(
                    f"| {METRICS[key]} | {'N/A' if value is None else f'{value:.3f}'} |"
                )
        lines.extend(
            [
                "",
                "**Reviewer notes**",
                "",
                (
                    (item.get("review_notes") or "No notes recorded.")
                    if item.get("reviewer_code")
                    else "Not reviewed yet."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def render_report(summary: dict, rows: list[dict]) -> str:
    """Keep the overview short, with origin, review and answerability as the only groups."""
    metadata = summary["metadata"]
    overall = summary["overall"]
    lines = [
        "# Evaluation results",
        "",
        f"Questions evaluated: **{overall['n']}** · Execution errors: **{overall['errors']}**",
        "",
    ]
    if metadata["mode"] == "simulated":
        lines.extend(
            [
                "This run used local word-based retrieval and quoted source passages. It does not measure Azure-generated answer quality.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "This run evaluated generated answers against the saved reference answers.",
                "",
            ]
        )
    if metadata.get("protocol") == "official_faq_lookup":
        lines.extend(
            [
                "**FAQ lookup:** the published questions and answers are available to retrieval. This measures finding FAQ information, not independent regulation-based generalisation. Publication does not establish current applicability or human approval of the evaluation references.",
                "",
            ]
        )
    lines.extend(["## Questions used", "", "| Origin | Questions |", "| --- | ---: |"])
    for origin, group in summary["by_origin"].items():
        lines.append(f"| {ORIGINS[origin]} | {group['n']} |")
    reviewed = summary["by_review_status"].get("reviewed", {}).get("n", 0)
    lines.extend(
        [
            "",
            f"References reviewed: **{reviewed}/{overall['n']}**. Answerable: **{summary['answerable']['n']}**; not answerable as stated: **{summary['unanswerable']['n']}**.",
            "",
        ]
    )
    if reviewed < overall["n"]:
        lines.extend(
            ["Results using references that still need review are provisional.", ""]
        )
    lines.extend(
        [
            "## Scores",
            "",
            "| Measure | Average | Questions scored |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, label in METRICS.items():
        result = overall["metrics"].get(key, {"mean": None, "n": 0})
        value = result["mean"]
        lines.append(
            f"| {label} | {'N/A' if value is None else f'{value:.3f}'} | {result['n']} |"
        )
    lines.extend(
        [
            "",
            "Scores range from 0 to 1; a lower false-refusal score is better. N/A means there were no applicable cases. Word overlap and citation IDs do not prove factual correctness.",
            "",
        ]
    )
    for title, groups in (
        ("By question origin", summary["by_origin"]),
        ("By reference review", summary["by_review_status"]),
    ):
        if len(groups) < 2:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Questions | Word overlap (F1) | Correct source cited |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, group in groups.items():
            label = ORIGINS.get(
                key,
                {"reviewed": "Reviewed", "needs_review": "Needs review"}.get(key, key),
            )
            values = [
                group["metrics"].get(metric, {}).get("mean")
                for metric in ("token_f1", "answer_with_correct_source")
            ]
            lines.append(
                f"| {label} | {group['n']} | "
                + " | ".join(
                    "N/A" if value is None else f"{value:.3f}" for value in values
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Questions to inspect",
            "",
            "These are selected by execution errors, unexpected answer status and lowest word overlap. This is a review aid, not a judgment of factual correctness.",
            "",
        ]
    )
    ordered = sorted(
        rows,
        key=lambda row: (
            row["error"] is None,
            row["metrics"].get("status_match", 0),
            row["metrics"].get("token_f1") or 0,
        ),
    )
    lines.extend(f"- {row['question']}" for row in ordered[:5])
    lines.extend(
        [
            "",
            "[Read every question, reference, source and actual answer](answers.md). Numeric results are in `results.csv`; reproducibility details and all metric denominators are in `summary.json` and `metadata.json`.",
            "",
            "Human answer ratings have not been inferred from automatic scores. Record them separately in `answer-review.json`.",
            "",
        ]
    )
    if summary.get("unattempted_ids"):
        lines.extend(
            [
                "Not attempted after interruption: "
                + ", ".join(summary["unattempted_ids"]),
                "",
            ]
        )
    if summary.get("reformatted_from"):
        lines.extend(
            [
                "This is a new presentation of a saved run. No model or retrieval was rerun.",
                "",
            ]
        )
    return "\n".join(lines)


def write_report_files(output: Path, summary: dict, rows: list[dict]) -> None:
    """Write human-facing reports and a flat CSV without optional taxonomies."""
    (output / "report.md").write_text(render_report(summary, rows), encoding="utf-8")
    (output / "answers.md").write_text(
        render_questions(rows, answers=True), encoding="utf-8"
    )
    fields = [
        "id",
        "origin",
        "review_status",
        "answerable",
        "error",
        "latency_seconds",
        *sorted({key for row in rows for key in row["metrics"]}),
    ]
    with (output / "results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = {key: row.get(key, row["metrics"].get(key)) for key in fields}
            data.update(
                origin=ORIGINS[question_origin(row["generation_method"])],
                review_status=REVIEWS[row["review_status"]],
            )
            writer.writerow(data)
