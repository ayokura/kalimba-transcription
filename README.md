# Kalimba Score

Kalimba performance to sheet music MVP.

## Structure

- `apps/web`: Next.js web application for recording, tuning, notation display, and light editing
- `apps/api`: FastAPI transcription API with kalimba-focused analysis pipeline
- `docs/testing.md`: local verification steps and manual test checklist

## Run

### Web

```bash
npm install
npm run dev:web
```

### API

Use Python 3.13 for the API.

```bash
py -3.13 -m pip install -r apps/api/requirements.txt
py -3.13 -m uvicorn app.main:app --reload --app-dir apps/api
```

Set `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` for the web app if needed.

## Manual test capture export

After a successful analysis in the web UI, you can download a capture zip from the workflow panel.

The zip includes:

- `audio.wav`
- `request.json`
- `response.json`
- `notes.md`

Use this pack to keep reproducible manual test evidence and feed regression fixtures later.
