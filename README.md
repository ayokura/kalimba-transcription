# Kalimba Score

Kalimba performance to sheet music MVP.

## Structure

- `apps/web`: Next.js web application for recording, tuning, notation display, and light editing
- `apps/api`: FastAPI transcription API with kalimba-focused analysis pipeline
- `docs/`: project documentation
  - [architecture.md](docs/architecture.md): パイプライン構成（Stage 1-9）
  - [recognition-roadmap.md](docs/recognition-roadmap.md): 認識精度の現状とロードマップ
  - [testing.md](docs/testing.md): テスト手順・fixture 管理
  - [recognizer-local-rules.md](docs/recognizer-local-rules.md): fixture-specific ルール一覧
  - [free-performance-readiness.md](docs/free-performance-readiness.md): Free Performance 適合度評価

## Run

### Web

```bash
npm install
npm run dev:web
```

### API

```bash
# WSL/Linux (primary)
uv sync
uv run uvicorn app.main:app --reload --app-dir apps/api
```

```bash
# Windows (legacy) — Python 3.13
py -3.13 -m pip install -r apps/api/requirements.txt
py -3.13 -m uvicorn app.main:app --reload --app-dir apps/api
```

Set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` for the web app if needed.

### Tests

```bash
uv run pytest apps/api/tests -q
```

## Manual test capture export

After a successful analysis in the web UI, you can download a capture zip from the workflow panel.

The zip includes:

- `audio.wav`
- `request.json`
- `response.json`
- `notes.md`

Use this pack to keep reproducible manual test evidence and feed regression fixtures later.
