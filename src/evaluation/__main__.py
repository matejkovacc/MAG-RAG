"""Run a synthetic pilot offline by default; live batches require explicit opt-in."""

import argparse
import hashlib
import sys
from pathlib import Path

from src.knowledge_base.models import PreparedCorpus
from .models import load_benchmark
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> int:
    """Validate annotations before creating providers and never reindex live data."""
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] not in {"validate", "run"}:
        from .controlled_cli import main as controlled_main

        return controlled_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run"])
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/pilot.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/thesis/corpus.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument(
        "--split", choices=["development", "test"], default="development"
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--allow-external-api", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    if args.mode == "live" and not args.allow_external_api:
        parser.error("Live evaluation needs user approval and --allow-external-api")
    if args.allow_external_api and (args.mode != "live" or args.command != "run"):
        parser.error("External API opt-in is only valid for a live run")
    corpus = PreparedCorpus.model_validate_json(args.corpus.read_text(encoding="utf-8"))
    benchmark = load_benchmark(args.benchmark, corpus)
    items = [item for item in benchmark.items if item.split == args.split][: args.limit]
    if args.command == "validate":
        print(
            f"Valid provenance: {len(benchmark.items)} synthetic items; human reference review remains separate."
        )
        return 0
    if not items:
        parser.error("No items in the selected split")
    if any(item.review_status != "reviewed" for item in items) and (
        not args.allow_draft or args.split == "test"
    ):
        parser.error("Unreviewed items require --allow-draft and development split")
    if args.output is None or args.output.exists():
        parser.error(
            "--output must name a new directory; existing runs are never overwritten"
        )
    metadata = {
        "split": args.split,
        "benchmark_file_sha256": hashlib.sha256(
            args.benchmark.read_bytes()
        ).hexdigest(),
    }
    if args.mode == "offline":
        from src.rag.offline import offline_service

        with offline_service(corpus) as service:
            service.retrieval_limit = 5
            summary = run_benchmark(
                service,
                benchmark,
                items,
                args.output,
                {
                    **metadata,
                    "retrieval": "prefix5-binary-v1; temporary Qdrant/mongomock",
                    "generator": "verbatim simulated excerpts",
                },
            )
    else:
        from src.knowledge_base.vector_store import fingerprint
        from src.rag.live import live_service

        with live_service(
            corpus.corpus_id, allow_external_api=True, max_questions=len(items)
        ) as service:
            index = service.retriever.index
            active = index.store.get_active(corpus.corpus_id)
            if not active or active["fingerprint"] != fingerprint(
                corpus, index.profile
            ):
                raise ValueError(
                    "Published corpus differs from the benchmark; no model calls made"
                )
            metadata.update(
                {
                    "snapshot_id": active["snapshot_id"],
                    "embedding": index.profile.model_dump(exclude={"endpoint"}),
                    "chat_deployment": service.generator.deployment,
                    "chat_model_version": "not independently resolved from Azure deployment",
                    "prompt_sha256": hashlib.sha256(
                        service.generator.prompt.encode()
                    ).hexdigest(),
                    "temperature": 0,
                    "max_output_tokens": 1200,
                    "retries": 0,
                    "retrieval": "SemanticRetriever v2: overretrieve 15, length filter, at most one same-article neighbor within five context slots",
                }
            )
            summary = run_benchmark(
                service,
                benchmark,
                items,
                args.output,
                metadata,
                snapshot_id=active["snapshot_id"],
            )
    print(
        f"Attempted {summary['overall']['attempted']}; errors {summary['overall']['errors']}; report: {args.output / 'summary.json'}"
    )
    return 1 if summary["overall"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
