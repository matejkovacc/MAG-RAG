"""CLI for offline PDF preparation and exact keyword evidence inspection."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from src.knowledge_base.models import CorpusManifest, PreparedCorpus, citation_url
from src.knowledge_base.prepare import prepare_manifest


def write_corpus(corpus: PreparedCorpus, output: Path) -> None:
    """Atomically replace a completed corpus export; preserve it on preparation failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(corpus.model_dump_json(indent=2))
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


def inspect_evidence(
    corpus: PreparedCorpus, query: str, limit: int = 5
) -> list[dict[str, object]]:
    """Rank literal keyword matches for inspection, not semantic RAG evaluation."""
    terms = set(re.findall(r"\w+", query.casefold()))
    if not terms or limit < 1:
        raise ValueError("Supply at least one search term and a positive limit")
    results: list[dict[str, object]] = []
    for document in corpus.documents:
        for chunk in document.chunks:
            score = len(terms & set(re.findall(r"\w+", chunk.text.casefold())))
            if score:
                results.append(
                    {
                        "matched_terms": score,
                        "source_title": document.source.title,
                        "source_id": document.source_id,
                        "chunk_id": chunk.id,
                        "version": chunk.version,
                        "page": chunk.page,
                        "article": chunk.article,
                        "url": citation_url(document.source, chunk),
                        "location_kind": "section" if document.source.web else "page",
                        "section": (
                            document.source.web.sections[chunk.page - 1].heading
                            if document.source.web
                            else None
                        ),
                        "review_status": document.source.review_status,
                        "text": chunk.text,
                    }
                )
    return sorted(results, key=lambda item: -int(item["matched_terms"]))[:limit]


def main() -> None:
    """Run a preparation or evidence-inspection operation with contextual errors."""
    parser = argparse.ArgumentParser(
        description="Prepare FRI evidence locally; no LLM or database required"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    refresh = commands.add_parser(
        "refresh",
        help="Fetch official website/PDF snapshots; no model calls or indexing",
    )
    refresh.add_argument(
        "--manifest", type=Path, default=Path("config/thesis-web-sources.json")
    )
    refresh.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New snapshot directory; never overwrite a run",
    )
    refresh.add_argument(
        "--previous",
        type=Path,
        help="Previous expanded corpus.json for change detection",
    )
    refresh.add_argument(
        "--allow-website-fetch",
        action="store_true",
        help="Permit public FRI HTTP downloads only",
    )
    refresh.add_argument("--transport", choices=("urllib", "curl"), default="urllib")
    refresh.add_argument(
        "--delay", type=float, default=1.0, help="Seconds between sources (0-60)"
    )
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-chars", type=int, default=1400)
    prepare.add_argument("--overlap", type=int, default=180)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--corpus", type=Path, required=True)
    inspect.add_argument("--query", required=True)
    inspect.add_argument("--limit", type=int, default=5)
    index = commands.add_parser(
        "index", help="Embed and publish a MongoDB/Qdrant snapshot"
    )
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Enable cloud calls for this run only, after explicit approval",
    )
    search = commands.add_parser(
        "search", help="Search the published corpus using embeddings"
    )
    search.add_argument("--corpus-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Enable cloud calls for this run only, after explicit approval",
    )
    args = parser.parse_args()
    try:
        if args.command == "refresh":
            from src.knowledge_base.refresh import (
                curl_fetch,
                refresh_sources,
                urllib_fetch,
            )

            report = refresh_sources(
                args.manifest,
                args.output,
                allow_website_fetch=args.allow_website_fetch,
                previous=args.previous,
                transport=curl_fetch if args.transport == "curl" else urllib_fetch,
                delay=args.delay,
            )
            print(
                json.dumps(
                    {key: value for key, value in report.items() if key != "sources"},
                    ensure_ascii=True,
                    indent=2,
                )
            )
            if report["status"] == "failed":
                parser.exit(
                    1,
                    f"Refresh incomplete; inspect {args.output / 'refresh-report.json'}\n",
                )
        elif args.command == "prepare":
            manifest = CorpusManifest.model_validate_json(
                args.manifest.read_text(encoding="utf-8")
            )
            inputs = {args.manifest.resolve()} | {
                (args.manifest.parent / source.local_path).resolve()
                for source in manifest.sources
            }
            if args.output.resolve() in inputs:
                raise ValueError(
                    "Output must not overwrite the manifest or a source PDF"
                )
            corpus = prepare_manifest(args.manifest, args.max_chars, args.overlap)
            write_corpus(corpus, args.output)
            print(
                json.dumps(
                    {
                        "documents": len(corpus.documents),
                        "chunks": sum(len(doc.chunks) for doc in corpus.documents),
                        "output": str(args.output),
                    }
                )
            )
        elif args.command == "inspect":
            corpus = PreparedCorpus.model_validate_json(
                args.corpus.read_text(encoding="utf-8")
            )
            print(
                json.dumps(
                    inspect_evidence(corpus, args.query, args.limit),
                    ensure_ascii=True,
                    indent=2,
                )
            )
        else:
            # Optional dependencies and service clients stay out of offline commands.
            from src.knowledge_base.runtime import configured_index

            corpus = None
            if args.command == "index":
                corpus = PreparedCorpus.model_validate_json(
                    args.corpus.read_text(encoding="utf-8")
                )
            with configured_index(
                allow_external_api=args.allow_external_api
            ) as service:
                if corpus is not None:
                    result = service.index(corpus)
                else:
                    result = [
                        hit.model_dump(mode="json")
                        for hit in service.search(
                            args.corpus_id, args.query, args.limit
                        )
                    ]
            print(json.dumps(result, ensure_ascii=True, indent=2))
    except ImportError:
        parser.exit(
            1,
            "Install optional dependencies: python -m pip install -r requirements-live.txt\n",
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Knowledge-base operation failed: {exc}\n")


if __name__ == "__main__":
    main()
