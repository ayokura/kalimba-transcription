"use client";

// gt-review の per-note band energy トレースパネル (#205 v1)。
// 再生位置 ±1s の窓で、プレイヘッドに最も近い行のノート (+任意で隣接 tine) の
// 帯域エネルギーをスパークライン表示する。計算はブラウザ内 WASM
// (lib/wasm/energy.ts) で完結し、サーバー往復しない。
//
// 60fps 再レンダーを避けるため、audio.currentTime は interval ポーリング +
// canvas 直描画 (gt-review のプレイヘッド表示と同じ流儀)。

import { RefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { GtDraftRow, GtDraftRowVerdict } from "@/lib/api";
import { traceNoteBandEnergies } from "@/lib/wasm/energy";
import { KALIMBA_17C_TUNING } from "@/lib/wasm/tuning-17c";

const HALF_WINDOW_SEC = 1.0;
const STEP_SEC = 0.04;
const POLL_MS = 300;
const RECOMPUTE_MIN_DELTA_SEC = 0.15;
const MAX_TRACE_NOTES = 8;

// 17-C の周波数昇順テーブル (隣接 tine = このスケール上の隣)。
const TUNING_BY_FREQ = [...KALIMBA_17C_TUNING].sort((a, b) => a.frequency - b.frequency);

function noteFrequency(noteName: string): number | null {
  const hit = KALIMBA_17C_TUNING.find((n) => n.noteName === noteName);
  if (hit) return hit.frequency;
  // 17-C 外 (半音など) は平均律 A4=440 で代用
  const m = noteName.match(/^([A-G])(#?)(\d)$/);
  if (!m) return null;
  const base: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  const midi = 12 * (parseInt(m[3], 10) + 1) + base[m[1]] + (m[2] === "#" ? 1 : 0);
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function scaleNeighbors(noteName: string): string[] {
  const i = TUNING_BY_FREQ.findIndex((n) => n.noteName === noteName);
  if (i < 0) return [];
  const out: string[] = [];
  if (i > 0) out.push(TUNING_BY_FREQ[i - 1].noteName);
  if (i < TUNING_BY_FREQ.length - 1) out.push(TUNING_BY_FREQ[i + 1].noteName);
  return out;
}

type Props = {
  txId: string;
  audioRef: RefObject<HTMLAudioElement | null>;
  rows: GtDraftRow[];
  verdictRows: Record<string, GtDraftRowVerdict>;
};

type DecodedAudio = { samples: Float32Array; sampleRate: number; durationSec: number };

export function GtEnergyTrace({ txId, audioRef, rows, verdictRows }: Props) {
  const [enabled, setEnabled] = useState(false);
  const [withNeighbors, setWithNeighbors] = useState(false);
  const [status, setStatus] = useState<string>("");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const decodedRef = useRef<{ txId: string; audio: DecodedAudio } | null>(null);
  const lastCenterRef = useRef<number>(-999);
  const lastNotesKeyRef = useRef<string>("");
  const computingRef = useRef(false);

  // 行の時刻列 (プレイヘッド近傍行の決定に使用)
  const rowTimes = useMemo(() => rows.map((r) => r.timeSec), [rows]);

  const notesForCenter = useCallback(
    (centerSec: number): string[] => {
      if (rows.length === 0) return [];
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < rows.length; i++) {
        const d = Math.abs(rowTimes[i] - centerSec);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      const row = rows[best];
      const rv = verdictRows[String(row.index)];
      const base = rv?.notes && rv.notes.length > 0 ? rv.notes : row.draftNotes;
      const set = new Set<string>(base);
      if (withNeighbors) {
        for (const n of base) for (const nb of scaleNeighbors(n)) set.add(nb);
      }
      return [...set]
        .filter((n) => noteFrequency(n) !== null)
        .sort((a, b) => (noteFrequency(b) ?? 0) - (noteFrequency(a) ?? 0))
        .slice(0, MAX_TRACE_NOTES);
    },
    [rows, rowTimes, verdictRows, withNeighbors],
  );

  const ensureDecoded = useCallback(async (): Promise<DecodedAudio | null> => {
    if (decodedRef.current?.txId === txId) return decodedRef.current.audio;
    setStatus("音声を読み込み中…");
    try {
      const res = await fetch(`/api/transcriptions/${txId}/audio`);
      if (!res.ok) throw new Error(`audio fetch ${res.status}`);
      const buf = await res.arrayBuffer();
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctx();
      try {
        const decoded = await ctx.decodeAudioData(buf);
        // 片チャンネル無音ステレオ録音があるため、全チャンネル平均ではなく
        // RMS の大きいチャンネルを採用する
        let bestCh = 0;
        let bestRms = -1;
        for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
          const data = decoded.getChannelData(ch);
          let acc = 0;
          const stride = Math.max(1, Math.floor(data.length / 5000));
          for (let i = 0; i < data.length; i += stride) acc += data[i] * data[i];
          if (acc > bestRms) {
            bestRms = acc;
            bestCh = ch;
          }
        }
        const audio: DecodedAudio = {
          samples: decoded.getChannelData(bestCh).slice(),
          sampleRate: decoded.sampleRate,
          durationSec: decoded.duration,
        };
        decodedRef.current = { txId, audio };
        return audio;
      } finally {
        void ctx.close();
      }
    } catch (err) {
      setStatus(`音声読み込み失敗: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  }, [txId]);

  const draw = useCallback(
    (
      notes: string[],
      trace: { startSec: number; stepSec: number; steps: number; values: Float32Array },
      centerSec: number,
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = canvas.clientWidth || 600;
      const rowH = 34;
      const cssHeight = Math.max(1, notes.length) * rowH + 18;
      canvas.width = Math.round(cssWidth * dpr);
      canvas.height = Math.round(cssHeight * dpr);
      canvas.style.height = `${cssHeight}px`;
      const g = canvas.getContext("2d");
      if (!g) return;
      g.scale(dpr, dpr);
      g.clearRect(0, 0, cssWidth, cssHeight);

      const labelW = 52;
      const plotW = cssWidth - labelW - 6;
      const { startSec, stepSec, steps, values } = trace;
      const windowDur = steps * stepSec;
      let max = 0;
      for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
      if (max <= 0) max = 1;

      const xForSec = (sec: number) => labelW + ((sec - startSec) / windowDur) * plotW;

      // 行 onset の縦ティック (窓内のみ)
      g.strokeStyle = "rgba(30,31,31,0.18)";
      g.lineWidth = 1;
      for (const t of rowTimes) {
        if (t < startSec || t > startSec + windowDur) continue;
        const x = xForSec(t);
        g.beginPath();
        g.moveTo(x, 0);
        g.lineTo(x, cssHeight - 16);
        g.stroke();
      }

      // プレイヘッド (窓中央)
      g.strokeStyle = "rgba(23,126,137,0.9)";
      g.lineWidth = 1.5;
      const cx = xForSec(centerSec);
      g.beginPath();
      g.moveTo(cx, 0);
      g.lineTo(cx, cssHeight - 16);
      g.stroke();

      g.font = "11px ui-monospace, Menlo, monospace";
      for (let n = 0; n < notes.length; n++) {
        const y0 = n * rowH + 4;
        const base = y0 + rowH - 8;
        g.fillStyle = "#63615d";
        g.fillText(notes[n], 4, y0 + rowH / 2 + 3);
        g.strokeStyle = "rgba(30,31,31,0.1)";
        g.beginPath();
        g.moveTo(labelW, base);
        g.lineTo(labelW + plotW, base);
        g.stroke();
        // sqrt 圧縮で減衰尾を可視化 (窓内グローバル max で正規化 = 行間の
        // 相対強度が読める)
        g.strokeStyle = "#177e89";
        g.lineWidth = 1.4;
        g.beginPath();
        for (let s = 0; s < steps; s++) {
          const v = Math.sqrt(values[n * steps + s] / max);
          const x = labelW + (s / Math.max(1, steps - 1)) * plotW;
          const y = base - v * (rowH - 12);
          if (s === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        g.stroke();
      }

      // 時間軸ラベル (窓端)
      g.fillStyle = "#63615d";
      g.fillText(`${startSec.toFixed(2)}s`, labelW, cssHeight - 4);
      const endLabel = `${(startSec + windowDur).toFixed(2)}s`;
      g.fillText(endLabel, labelW + plotW - g.measureText(endLabel).width, cssHeight - 4);
    },
    [rowTimes],
  );

  const recompute = useCallback(
    async (force: boolean) => {
      if (!enabled || computingRef.current) return;
      const audioEl = audioRef.current;
      if (!audioEl) return;
      const centerSec = audioEl.currentTime;
      const notes = notesForCenter(centerSec);
      const notesKey = notes.join(",");
      if (
        !force &&
        Math.abs(centerSec - lastCenterRef.current) < RECOMPUTE_MIN_DELTA_SEC &&
        notesKey === lastNotesKeyRef.current
      ) {
        return;
      }
      if (notes.length === 0) return;
      computingRef.current = true;
      try {
        const audio = await ensureDecoded();
        if (!audio) return;
        const startSec = Math.max(0, centerSec - HALF_WINDOW_SEC);
        const endSec = Math.min(audio.durationSec, centerSec + HALF_WINDOW_SEC);
        if (endSec - startSec < STEP_SEC * 4) return;
        const freqs = notes.map((n) => noteFrequency(n) ?? 0);
        const trace = await traceNoteBandEnergies(
          audio.samples,
          audio.sampleRate,
          freqs,
          startSec,
          endSec - startSec,
          STEP_SEC,
        );
        lastCenterRef.current = centerSec;
        lastNotesKeyRef.current = notesKey;
        draw(notes, trace, centerSec);
        setStatus(`${notes.length} 音 × ${trace.steps} step (${trace.elapsedMs.toFixed(0)}ms)`);
      } finally {
        computingRef.current = false;
      }
    },
    [enabled, audioRef, notesForCenter, ensureDecoded, draw],
  );

  // 再生位置ポーリング → 閾値超えの移動で再計算
  useEffect(() => {
    if (!enabled) return;
    const timer = window.setInterval(() => void recompute(false), POLL_MS);
    void recompute(true);
    return () => window.clearInterval(timer);
  }, [enabled, recompute]);

  // tx 切替でキャッシュ・表示をリセット
  useEffect(() => {
    lastCenterRef.current = -999;
    lastNotesKeyRef.current = "";
  }, [txId, withNeighbors]);

  return (
    <div className="gt-energy-trace">
      <div className="gt-energy-trace-controls">
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
          />{" "}
          energy trace (再生位置 ±{HALF_WINDOW_SEC}s、近傍行のノート)
        </label>
        {enabled ? (
          <label>
            <input
              type="checkbox"
              checked={withNeighbors}
              onChange={(e) => setWithNeighbors(e.target.checked)}
            />{" "}
            隣接 tine も表示
          </label>
        ) : null}
        {enabled && status ? <span className="muted">{status}</span> : null}
      </div>
      {enabled ? <canvas ref={canvasRef} className="gt-energy-trace-canvas" /> : null}
    </div>
  );
}
