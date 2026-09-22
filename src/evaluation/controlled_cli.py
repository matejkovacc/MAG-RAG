"""Controlled evaluation commands; providers are created only after validation."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from src.knowledge_base.models import PreparedCorpus
from .controlled import (
    ControlledDataset,
    ControlledItem,
    evaluation_tier,
    apply_reviews,
    audit_dataset,
    load_controlled,
    make_silver,
    review_template,
    write_new,
)
from .controlled_runner import export_saved_run, run_controlled, score_human_reviews
from .reporting import question_origin, render_questions


def read_json(path: Path):
    """Read UTF-8 JSON, including files saved by Windows editors with a BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parser() -> argparse.ArgumentParser:
    """Expose separate preparation, reference review, execution and scoring steps."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    report = commands.add_parser(
        "report", help="Simplify a saved report without rerunning evaluation"
    )
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    faq = commands.add_parser("import-faq")
    faq.add_argument("--html", type=Path, required=True)
    faq.add_argument("--captured-at", type=datetime.fromisoformat, required=True)
    faq.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-sources")
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--allow-website-fetch", action="store_true")
    prepare.add_argument("--transport", choices=["urllib", "curl"], default="urllib")
    silver = commands.add_parser("make-questions")
    silver.add_argument("--corpus", type=Path, required=True)
    silver.add_argument("--legacy", type=Path)
    silver.add_argument("--variants", type=Path)
    silver.add_argument("--seed", type=int, default=42)
    silver.add_argument("--limit", type=int, default=24)
    silver.add_argument("--output", type=Path, required=True)
    for name in (
        "validate-set",
        "review-template",
        "export-questions",
        "apply-review",
        "evaluate",
    ):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--corpus", type=Path, required=True)
        if name != "validate-set":
            command.add_argument("--output", type=Path, required=True)
        if name == "apply-review":
            command.add_argument("--reviews", type=Path, required=True)
        if name == "evaluate":
            command.add_argument(
                "--mode", choices=["offline", "live"], default="offline"
            )
            command.add_argument(
                "--split", choices=["development", "test"], default="development"
            )
            command.add_argument(
                "--tier",
                choices=["all", "silver", "gold", "official_faq"],
                default="all",
                help=argparse.SUPPRESS,
            )
            command.add_argument(
                "--reviewed-only",
                action="store_true",
                help="Use only human-reviewed reference answers",
            )
            command.add_argument(
                "--origin",
                choices=["all", "official_faq", "document_derived"],
                default="all",
            )
            command.add_argument("--limit", type=int, default=5)
            command.add_argument("--allow-unreviewed", action="store_true")
            command.add_argument("--allow-external-api", action="store_true")
            command.add_argument(
                "--model-version", default="not independently verified"
            )
            command.add_argument("--delay", type=float, default=0)
    ratings = commands.add_parser("score-review")
    ratings.add_argument("--run", type=Path, required=True)
    ratings.add_argument("--reviews", type=Path, required=True)
    ratings.add_argument("--output", type=Path, required=True)
    public = commands.add_parser("public-import")
    public.add_argument("--format", choices=["squad2", "qaslovene"], required=True)
    public.add_argument("--input", type=Path, required=True)
    public.add_argument("--license", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("public-score")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--model-version", required=True)
    score.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None) -> int:
    """Keep every offline command free of cloud clients and never overwrite runs."""
    cli = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "make-silver":
        arguments[0] = "make-questions"
    args = cli.parse_args(arguments)
    if hasattr(args, "output") and args.output.exists():
        cli.error("Output must be a new path")
    if args.command == "report":
        summary = export_saved_run(args.run, args.output)
        print(
            f"Reformatted {summary['overall']['n']} saved results; no model calls or retrieval rerun"
        )
        return 0
    if args.command == "import-faq":
        from .faq import import_faq

        dataset = import_faq(args.html, args.output, args.captured_at)
        print(
            f"Imported {len(dataset.items)} official FAQ cases; protocol: {dataset.protocol}; human approval pending"
        )
        return 0
    if args.command == "prepare-sources":
        from src.knowledge_base.catalog import prepare_catalog

        corpus = prepare_catalog(
            args.catalog,
            args.output,
            allow_website_fetch=args.allow_website_fetch,
            transport=args.transport,
        )
        print(
            f"Prepared {len(corpus.documents)} documents: {args.output / 'corpus.json'}"
        )
        return 0
    if args.command == "score-review":
        write_new(args.output, score_human_reviews(args.run, args.reviews))
        return 0
    if args.command in {"public-import", "public-score"}:
        from .public_qa import import_public, score_public

        if args.command == "public-import":
            import_public(args.input, args.format, read_json(args.license), args.output)
        else:
            score_public(
                args.dataset, args.predictions, args.model_version, args.output
            )
        return 0
    corpus = PreparedCorpus.model_validate_json(args.corpus.read_text(encoding="utf-8"))
    if args.command == "make-questions":
        if not 1 <= args.limit <= 10000:
            cli.error("--limit must be between 1 and 10000")
        dataset = make_silver(
            corpus, legacy=args.legacy, seed=args.seed, limit=args.limit
        )
        if args.variants:
            lookup = {item.id: item for item in dataset.items}
            additions = []
            for variant in read_json(args.variants):
                original = lookup[variant["base_id"]]
                additions.append(
                    ControlledItem.model_validate(
                        {
                            **original.model_dump(),
                            "id": variant["id"],
                            "question": variant["question"],
                            "category": variant["category"],
                            "generation_method": "paraphrased",
                            "review_status": "unreviewed",
                            "reviewer_code": None,
                            "review_checks": {},
                            "reviewed_content_sha256": None,
                            "review_notes": "Manually written synthetic variant; reference and applicability need human review.",
                        }
                    )
                )
            dataset = ControlledDataset.model_validate(
                {
                    **dataset.model_dump(),
                    "items": [item.model_dump() for item in dataset.items + additions],
                }
            )
        audit = audit_dataset(dataset, corpus)
        if not audit["valid"]:
            cli.error("; ".join(audit["errors"]))
        write_new(args.output, dataset.model_dump(mode="json"))
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    dataset, audit = load_controlled(args.dataset, corpus)
    if args.command == "validate-set":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    if args.command == "review-template":
        write_new(args.output, review_template(dataset))
        return 0
    if args.command == "export-questions":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(
                render_questions(
                    [item.model_dump(mode="json") for item in dataset.items]
                )
            )
        return 0
    if args.command == "apply-review":
        updated = apply_reviews(dataset, read_json(args.reviews))
        write_new(args.output, updated.model_dump(mode="json"))
        return 0
    if not 1 <= args.limit <= 100 or not 0 <= args.delay <= 60:
        cli.error("--limit must be 1..100 and --delay 0..60 seconds")
    if args.mode == "live" and not args.allow_external_api:
        cli.error(
            "Live evaluation needs explicit batch approval and --allow-external-api"
        )
    if args.mode != "live" and args.allow_external_api:
        cli.error("--allow-external-api is valid only for live evaluation")
    items = [
        item
        for item in dataset.items
        if item.split == args.split
        and item.review_status in {"approved", "unreviewed"}
        and (not args.reviewed_only or item.review_status == "approved")
        and (
            args.origin == "all"
            or question_origin(item.generation_method) == args.origin
        )
        and (args.tier == "all" or evaluation_tier(item) == args.tier)
    ][: args.limit]
    if not items:
        cli.error(
            "No eligible questions match the selection; rejected and needs_revision are excluded"
        )
    if any(item.review_status != "approved" for item in items) and (
        not args.allow_unreviewed or args.split == "test"
    ):
        cli.error(
            "Unreviewed references require --allow-unreviewed and development split"
        )
    metadata = {
        "split": args.split,
        "origin": args.origin,
        "reviewed_only": args.reviewed_only,
        "limit": args.limit,
        "model_version": args.model_version,
        "audit": audit,
        "excluded_review_states": ["rejected", "needs_revision"],
        "dataset_file_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    }
    if args.mode == "offline":
        from src.rag.offline import offline_service

        metadata.update(
            model_version="offline-prefix5-binary-v1/simulated-excerpts",
            generator="verbatim source excerpts; not an LLM",
        )
        with offline_service(corpus) as service:
            service.retrieval_limit = 5
            summary = run_controlled(
                service, dataset, items, args.output, metadata, delay=args.delay
            )
    else:
        from src.rag.live import live_service

        with live_service(
            corpus.corpus_id,
            allow_external_api=True,
            max_questions=len(items),
            expected_corpus=corpus,
        ) as service:
            service.retrieval_limit = 5
            index = service.retriever.index
            active = index.store.get_active(corpus.corpus_id)
            metadata.update(
                snapshot_id=active["snapshot_id"],
                embedding=index.profile.model_dump(exclude={"endpoint"}),
                chat_deployment=service.generator.deployment,
                prompt_sha256=hashlib.sha256(
                    service.generator.prompt.encode()
                ).hexdigest(),
                temperature=0,
                max_output_tokens=1200,
                retries=0,
            )
            summary = run_controlled(
                service,
                dataset,
                items,
                args.output,
                metadata,
                snapshot_id=active["snapshot_id"],
                delay=args.delay,
            )
    print(
        f"Attempted {summary['overall']['n']}; errors {summary['overall']['errors']}; report: {args.output / 'report.md'}"
    )
    return int(bool(summary["overall"]["errors"]))
