# Offline student-affairs preview

An explicitly approved [live answer mode](../../docs/live-mode.md)
now uses the Azure semantic index and `gpt-4o` to answer with source citations:
`python -m src.rag --live --allow-external-api --max-questions 10 --corpus data/thesis/regulations/2026-09-23-ready/corpus.json --port 10131`.
The default commands below remain offline. The per-run flag is not a substitute
for user approval; submitting questions in live mode incurs model API calls.

Run from the repository root with the prepared corpus present:

```powershell
.venv/Scripts/python.exe -m src.rag --corpus data/thesis/regulations/2026-09-23-ready/corpus.json
```

With the local containers running, add `--local-databases` to persist demo evidence
in real MongoDB/Qdrant. Both hosts are fixed to `127.0.0.1`; models remain disabled.
Start containers with `docker compose up -d`.

Open `http://127.0.0.1:10130`. The server binds only to numeric loopback; use that
address rather than `localhost`. Stop a foreground run with Ctrl+C. `--port` and
`--corpus` can override the defaults. Only explicit live mode enables cloud access.

- `answers.py`: validated questions, injected retrieval/generation contracts,
  bounded conversational context, deterministic excerpt responses, server-owned citation metadata.
- `diagnostics.py`: allowlisted HTTP/error categories and request IDs carried to
  evaluation artifacts without provider error text, payloads or credentials.
- `live.py`: approved generation and bounded article-neighbor retrieval. At most
  one missing neighbor uses a slot within the existing five-passage budget.
- `offline.py`: disposable local Qdrant and a `mongomock` MongoDB substitute,
  or opt-in persistent loopback database clients,
  reusing the existing snapshot/index/hydration implementation. Five-character
  word prefixes produce lexical vectors locally, not Azure semantic embeddings.
- `__main__.py`: bounded development HTTP adapter and fixed static asset routes.
- `static/`: Slovenian preview interface, with no external fonts, scripts,
  analytics or automatic source downloads. Live mode clearly indicates Azure use.

The Python standard-library adapter keeps the standalone preview small and avoids
requiring FastAPI or React.

Offline dependencies are listed in `requirements.txt`.

See [architecture and limitations](../../docs/architecture.md).

The chat now retains recent turns in browser-tab memory and supports contextual
follow-ups. Every turn performs a new retrieval; earlier answers are context, not
evidence. "Nov pogovor" clears history without changing the live request allowance.
The server does not persist conversation history.
