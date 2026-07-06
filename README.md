# Kalimba Score

カリンバの演奏を録音すると自動で楽譜になるソフト (個人開発 MVP)。最終目標は、事前に何を弾くか知らせなくても自由演奏がそのまま楽譜になること。開発は人間と AI エージェントの共同作業で進んでいる。

Kalimba performance to sheet music MVP — recording in the browser, kalimba-focused analysis on the server, editable notation out.

## 一般向けの文書 (技術知識を前提にしない読み物)

技術に詳しくない方 (カリンバ講師・テスター・録音協力者) 向けに、専門用語を使わない文書を用意しています:

- [録音で協力してくださる方へ](docs/guide-for-recording-helpers.md) — なぜ録音が一番の貢献なのか・録り方・著作権と機材を聞かれる理由・提供後の流れ
- [録音はどうやって楽譜になるのか](docs/guide-how-it-works.md) — 仕組みと「なぜ間違えるのか」のやさしい解説
- [開発計画のやさしい版](docs/sprint-plan-2026-07d-plain.md) — 現行計画 (第 4 期) の全内容を用語辞書付きで

## Structure

- `apps/web`: Next.js web application for recording, tuning, notation display, and light editing
- `apps/api`: FastAPI transcription API with kalimba-focused analysis pipeline
- `docs/`: project documentation — 索引は [docs/README.md](docs/README.md)
  - [architecture.md](docs/architecture.md): パイプライン構成（Stage 1-9）
  - [sprint-plan-2026-07d.md](docs/sprint-plan-2026-07d.md): 現行の中期計画 (第 4 期)
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
uv sync
uv run uvicorn app.main:app --reload --app-dir apps/api
```

`uv` is the cross-platform source of truth (it manages the Python 3.14 toolchain
from `pyproject.toml` / `uv.lock`); the same commands work on Windows.

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
