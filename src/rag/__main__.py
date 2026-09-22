"""Loopback thesis preview with offline defaults and an explicitly approved live mode."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from src.knowledge_base.models import PreparedCorpus
from src.rag.answers import AnswerService, Question
from src.rag.offline import offline_service
from src.rag.live import LiveSessionLimit, live_service


ASSETS = Path(__file__).parent / "static"


def make_handler(
    service: AnswerService, metadata: dict
) -> type[BaseHTTPRequestHandler]:
    """Create a bounded local HTTP adapter around the typed answer service."""

    class Handler(BaseHTTPRequestHandler):
        """Serve fixed assets and local questions without telemetry or access logs."""

        def setup(self) -> None:
            """Bound incomplete local requests so an idle connection cannot hang a worker."""
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format: str, *args: object) -> None:
            """Do not persist questions, IPs or access history in demo logs."""

        def send_payload(self, status: int, body: bytes, content_type: str) -> None:
            """Constrain browser requests to this origin and disallow embedded content."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, data: dict) -> None:
            """Serialize UTF-8 responses without interpolating HTML."""
            self.send_payload(
                status,
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def local_request(self) -> bool:
            """Reject unexpected hosts and cross-origin requests to the local preview."""
            host = f"127.0.0.1:{self.server.server_port}"
            return (
                self.headers.get("Host") == host
                and self.headers.get("Origin", f"http://{host}") == f"http://{host}"
            )

        def do_GET(self) -> None:
            """Expose only the three fixed assets and non-secret demo status."""
            if not self.local_request():
                self.send_json(403, {"error": "Local origin required"})
                return
            if self.path == "/status":
                session = (
                    service.session_status()
                    if hasattr(service, "session_status")
                    else {}
                )
                self.send_json(200, {**metadata, **session})
                return
            files = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if self.path not in files:
                self.send_json(404, {"error": "Not found"})
                return
            name, kind = files[self.path]
            self.send_payload(
                200, (ASSETS / name).read_bytes(), kind + "; charset=utf-8"
            )

        def do_POST(self) -> None:
            """Validate local JSON questions before running offline orchestration."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(422, {"error": "Invalid content length"})
                return
            # Drain bounded request bodies before rejecting headers. On Windows,
            # closing with unread input can reset the connection and lose the error.
            body = self.rfile.read(min(max(length, 0), 65536))
            if not 0 < length <= 65536:
                self.send_json(413, {"error": "Question payload too large or empty"})
                return
            if not self.local_request():
                self.send_json(403, {"error": "Local origin required"})
                return
            if self.path != "/ask":
                self.send_json(404, {"error": "Not found"})
                return
            if (
                self.headers.get("Content-Type", "").split(";", 1)[0]
                != "application/json"
            ):
                self.send_json(415, {"error": "Expected application/json"})
                return
            try:
                request = Question.model_validate_json(body)
            except (ValueError, ValidationError):
                self.send_json(
                    422,
                    {
                        "error": "Vprašanje mora imeti 2–2000 znakov; kontekst največ 4 izmenjave in skupaj 12000 znakov."
                    },
                )
                return
            try:
                self.send_json(200, service.answer(request).model_dump(mode="json"))
            except LiveSessionLimit as exc:
                self.send_json(429, {"error": str(exc)})
            except Exception:
                self.send_json(
                    502,
                    {
                        "error": "Odgovora ni bilo mogoče pripraviti. Preverite storitve, konfiguracijo in vire; odgovor ni bil prikazan."
                    },
                )

    return Handler


def main() -> None:
    """Default to offline mode; live mode requires explicit per-run cloud opt-in."""
    parser = argparse.ArgumentParser(
        description="Offline student-affairs preview; no external API calls"
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/thesis/corpus.json"))
    parser.add_argument("--port", type=int, default=10130)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the published Azure index and chat deployment",
    )
    parser.add_argument(
        "--allow-external-api",
        action="store_true",
        help="Enable cloud calls only after explicit user approval",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Live question allowance for this run (1-100)",
    )
    parser.add_argument(
        "--local-databases",
        action="store_true",
        help="Use real MongoDB/Qdrant on fixed loopback hosts; models remain disabled",
    )
    args = parser.parse_args()
    if args.live and (not args.allow_external_api or args.local_databases):
        parser.error(
            "Live mode requires --allow-external-api and cannot combine with --local-databases"
        )
    if args.allow_external_api and not args.live:
        parser.error("--allow-external-api requires --live")
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    try:
        corpus = PreparedCorpus.model_validate_json(
            args.corpus.read_text(encoding="utf-8")
        )
        context = (
            live_service(
                corpus.corpus_id,
                allow_external_api=True,
                max_questions=args.max_questions,
                expected_corpus=corpus,
            )
            if args.live
            else offline_service(corpus, local_databases=args.local_databases)
        )
        with context as service:
            metadata = {
                "mode": "generated" if args.live else "simulated",
                "external_api_calls": args.live,
                "documents": len(corpus.documents),
                "website_sources": sum(
                    doc.source.web is not None for doc in corpus.documents
                ),
                "website_capture_oldest": min(
                    (
                        doc.source.web.captured_at.isoformat()
                        for doc in corpus.documents
                        if doc.source.web
                    ),
                    default=None,
                ),
                "website_capture_newest": max(
                    (
                        doc.source.web.captured_at.isoformat()
                        for doc in corpus.documents
                        if doc.source.web
                    ),
                    default=None,
                ),
                "chunks": sum(len(doc.chunks) for doc in corpus.documents),
                "storage": (
                    "MongoDB + Qdrant (local containers)"
                    if args.local_databases or args.live
                    else "Qdrant local engine + mongomock (temporary)"
                ),
                "persistent_storage": args.local_databases or args.live,
                "retrieval": (
                    "azure-semantic-vectors" if args.live else "lexical-prefix-vectors"
                ),
            }
            with ThreadingHTTPServer(
                ("127.0.0.1", args.port), make_handler(service, metadata)
            ) as server:
                server.timeout = 5
                print(
                    f"Thesis preview: http://127.0.0.1:{args.port} | mode={metadata['mode']} | {metadata['storage']}",
                    flush=True,
                )
                server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as exc:
        parser.exit(
            1,
            f"Offline preview could not start ({type(exc).__name__}). Check the corpus and local port.\n",
        )


if __name__ == "__main__":
    main()
