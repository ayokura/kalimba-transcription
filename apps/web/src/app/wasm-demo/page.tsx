"use client";

import { useState } from "react";

import { detectOnsetsInBrowser } from "@/lib/wasm/onset";

type DemoState = {
  info: string;
  onsetTimesSec: number[];
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
      audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const samples = audioBuffer.getChannelData(0); // mono (first channel)
      const { onsetTimesSec, frameCount, elapsedMs } = await detectOnsetsInBrowser(
        samples,
        audioBuffer.sampleRate,
      );
      setResult({
        info:
          `${audioBuffer.duration.toFixed(1)}s @ ${audioBuffer.sampleRate} Hz, ` +
          `${samples.length.toLocaleString()} samples, ${frameCount} frames → ` +
          `${onsetTimesSec.length} onsets in ${elapsedMs.toFixed(0)} ms ` +
          `(fully in-browser, no server round-trip)`,
        onsetTimesSec,
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
        WASM onset detection (browser-only)
      </h1>
      <p style={{ color: "#555", lineHeight: 1.6, marginBottom: "1.25rem" }}>
        Proof of the browser-side pipeline: the kalimba-dsp Rust core runs as
        WebAssembly on a WebAudio-decoded <code>Float32Array</code> — the same
        onset detection the server uses, with zero server round-trip. Pick an
        audio file (WAV/MP3/etc.).
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
            {result.onsetTimesSec.map((sec, i) => (
              <li key={i}>{sec.toFixed(3)}s</li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
