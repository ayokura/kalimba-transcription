"use client";

import { useState } from "react";

import { detectOnsetsInBrowser } from "@/lib/wasm/onset";
import { identifyNotesInBrowser, type IdentifiedNote } from "@/lib/wasm/pitch";

type DemoState = {
  info: string;
  notes: IdentifiedNote[];
};

export default function WasmDemoPage() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DemoState | null>(null);

  async function handleFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    let audioContext: AudioContext | null = null;
    try {
      const arrayBuffer = await file.arrayBuffer();
      // Request 48 kHz: a sample-rate sweep showed note identification is 100%
      // identical to 96 kHz down to 32 kHz, then degrades (22 kHz ~97%, ≤16 kHz
      // breaks as high-note partials fall below Nyquist). 48 kHz is the modern
      // hardware default (no resample on live capture) with ample partial
      // headroom; this is only a hint — the recognizer is sample-rate-robust, so
      // whatever rate the browser actually provides is fine.
      audioContext = new AudioContext({ sampleRate: 48000 });
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const samples = audioBuffer.getChannelData(0); // mono (first channel)
      const { onsetTimesSec, frameCount, elapsedMs } = await detectOnsetsInBrowser(
        samples,
        audioBuffer.sampleRate,
      );
      const { notes, elapsedMs: pitchMs } = await identifyNotesInBrowser(
        samples,
        audioBuffer.sampleRate,
        onsetTimesSec,
      );
      setResult({
        info:
          `${audioBuffer.duration.toFixed(1)}s @ ${audioBuffer.sampleRate} Hz, ` +
          `${samples.length.toLocaleString()} samples, ${frameCount} frames → ` +
          `${onsetTimesSec.length} onsets (${elapsedMs.toFixed(0)} ms) → ` +
          `${notes.length} notes (${pitchMs.toFixed(0)} ms), tuning kalimba-17-c — ` +
          `fully in-browser, no server round-trip`,
        notes,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      await audioContext?.close();
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 640, margin: "0 auto", padding: "2rem 1.5rem" }}>
      <h1 style={{ fontSize: "1.4rem", marginBottom: "0.5rem" }}>
        WASM onset + pitch identification (browser-only)
      </h1>
      <p style={{ color: "#555", lineHeight: 1.6, marginBottom: "1.25rem" }}>
        Proof of the browser-side pipeline: the kalimba-dsp Rust core runs as
        WebAssembly on a WebAudio-decoded <code>Float32Array</code> — the same
        onset detection <em>and</em> tuning-candidate ranking the server uses, with
        zero server round-trip. Each onset is matched to the strongest note in the
        kalimba-17-c tuning. Pick an audio file (WAV/MP3/etc.).
      </p>
      <p style={{ color: "#999", fontSize: "0.85rem", lineHeight: 1.5, marginBottom: "1.25rem" }}>
        Note: pitch ID is experimental. The tuning is fixed to kalimba-17-c and the
        per-onset analysis window is a simple heuristic (top-1, monophonic) — the
        Rust↔WASM numerics are parity-checked against the server, but the window
        choice is not yet tuned for accuracy.
      </p>

      <input type="file" accept="audio/*" onChange={handleFile} disabled={busy} />

      {busy && <p style={{ marginTop: "1rem" }}>Decoding + detecting…</p>}
      {error && (
        <p style={{ marginTop: "1rem", color: "#b00020" }}>Error: {error}</p>
      )}

      {result && (
        <section style={{ marginTop: "1.5rem" }}>
          <p style={{ color: "#333", marginBottom: "0.75rem" }}>{result.info}</p>
          <ol style={{ columns: 3, fontVariantNumeric: "tabular-nums", color: "#222" }}>
            {result.notes.map((note, i) => (
              <li key={i}>
                <strong>{note.noteName}</strong>{" "}
                <span style={{ color: "#777" }}>@ {note.onsetTimeSec.toFixed(3)}s</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
