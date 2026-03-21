# Architecture

The MVP uses browser recording and server-side analysis.

1. The browser records audio with `MediaRecorder`.
2. The UI uploads the recorded file and the selected tuning to `POST /api/transcriptions`.
3. The FastAPI service normalizes audio, detects onsets, estimates peak frequencies, snaps them to kalimba keys, quantizes events to beat grid positions, and returns a shared score model.
4. The UI renders three notation views from the shared model and allows lightweight event-level editing.