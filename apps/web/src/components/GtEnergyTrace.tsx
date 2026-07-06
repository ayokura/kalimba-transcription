"use client";

// gt-review の per-note band energy トレースパネル (#205 v1)。
// 再生位置 ±1s の窓で、プレイヘッドに最も近い行のノート (+任意で隣接 tine) の
// 帯域エネルギーをスパークライン表示する。計算はブラウザ内 WASM
// (lib/wasm/energy.ts) で完結し、サーバー往復しない。
//
// 60fps 再レンダーを避けるため、audio.currentTime は interval ポーリング +
// canvas 直描画 (gt-review のプレイヘッド表示と同じ流儀)。

import { RefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { traceNoteBandEnergies } from "@/lib/wasm/energy";
import { KALIMBA_17C_TUNING } from "@/lib/wasm/tuning-17c";

const HALF_WINDOW_SEC = 1.0;
const STEP_SEC = 0.04;
const POLL_MS = 300;
const RECOMPUTE_MIN_DELTA_SEC = 0.15;
const MAX_ROW_MODE_NOTES = 8;

// tuning テーブル (周波数昇順)。既定は 17-C。録音の instrumentTuning が
// 17 音以下 (dedup 後) ならそれに差し替える — G-low 等の 17 鍵別スケール対応
// (2026-07-06)。34/21 鍵は行数が画面に収まらないため対象外 (ユーザー判断):
// 従来どおり 17-C fallback + 平均律代用のまま。
type TuningNoteLite = { noteName: string; frequency: number };

const FALLBACK_TABLE: TuningNoteLite[] = [...KALIMBA_17C_TUNING]
  .sort((a, b) => a.frequency - b.frequency)
  .map((n) => ({ noteName: n.noteName, frequency: n.frequency }));
const MAX_TUNING_TABLE_NOTES = 17;

function noteFrequency(table: TuningNoteLite[], noteName: string): number | null {
  const hit = table.find((n) => n.noteName === noteName);
  if (hit) return hit.frequency;
  // テーブル外 (半音など) は平均律 A4=440 で代用
  const m = noteName.match(/^([A-G])(#?)(\d)$/);
  if (!m) return null;
  const base: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  const midi = 12 * (parseInt(m[3], 10) + 1) + base[m[1]] + (m[2] === "#" ? 1 : 0);
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function scaleNeighbors(table: TuningNoteLite[], noteName: string): string[] {
  const i = table.findIndex((n) => n.noteName === noteName);
  if (i < 0) return [];
  const out: string[] = [];
  if (i > 0) out.push(table[i - 1].noteName);
  if (i < table.length - 1) out.push(table[i + 1].noteName);
  return out;
}

/** 時刻アンカー: その時刻で注目すべきノート群 (gt-review 行 / bp-verify 行など)。
 * 呼び出し側スキーマへの依存を切るための最小インターフェース。 */
export type EnergyTraceAnchor = {
  timeSec: number;
  notes: string[];
};

type Props = {
  txId: string;
  audioRef: RefObject<HTMLAudioElement | null>;
  anchors: EnergyTraceAnchor[];
};

type DecodedAudio = { samples: Float32Array; sampleRate: number; durationSec: number };

export function GtEnergyTrace({ txId, audioRef, anchors }: Props) {
  const [enabled, setEnabled] = useState(false);
  // "all": 全 17 tine を表示 (伴奏として鳴っている未認識ノーツの探索が主用途)。
  // "row": 近傍行のノート (+隣接 tine) のみ — 行が多い時の縮約表示。
  const [mode, setMode] = useState<"all" | "row">("all");
  const [withNeighbors, setWithNeighbors] = useState(false);
  const [status, setStatus] = useState<string>("");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const decodedRef = useRef<{ txId: string; audio: DecodedAudio } | null>(null);
  const lastCenterRef = useRef<number>(-999);
  const lastNotesKeyRef = useRef<string>("");
  const computingRef = useRef(false);

  // アンカーの時刻列 (プレイヘッド近傍アンカーの決定に使用)
  const rowTimes = useMemo(() => anchors.map((a) => a.timeSec), [anchors]);

  // 録音の tuning を取得 (取得失敗・17 音超は 17-C fallback のまま)
  const [tuningNotes, setTuningNotes] = useState<TuningNoteLite[] | null>(null);
  useEffect(() => {
    let alive = true;
    setTuningNotes(null);
    fetch(`/api/transcriptions/${txId}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`tuning ${r.status}`))))
      .then((doc: { instrumentTuning?: { notes?: TuningNoteLite[] } }) => {
        if (!alive) return;
        const notes = doc?.instrumentTuning?.notes;
        if (Array.isArray(notes) && notes.length > 0) {
          setTuningNotes(notes.map((n) => ({ noteName: n.noteName, frequency: n.frequency })));
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [txId]);
  const tuningByFreq = useMemo(() => {
    if (!tuningNotes) return FALLBACK_TABLE;
    const seen = new Set<string>();
    const out: TuningNoteLite[] = [];
    for (const n of [...tuningNotes].sort((a, b) => a.frequency - b.frequency)) {
      if (seen.has(n.noteName)) continue;
      seen.add(n.noteName);
      out.push(n);
    }
    // 34/21 鍵: 全 tine 表示が画面に収まらないため差し替えない (冒頭コメント参照)
    return out.length <= MAX_TUNING_TABLE_NOTES ? out : FALLBACK_TABLE;
  }, [tuningNotes]);

  const notesForCenter = useCallback(
    (centerSec: number): { notes: string[]; highlight: Set<string> } => {
      // プレイヘッドに最も近いアンカーのノートをハイライト対象とする
      let base: string[] = [];
      if (anchors.length > 0) {
        let best = 0;
        let bestDist = Infinity;
        for (let i = 0; i < anchors.length; i++) {
          const d = Math.abs(rowTimes[i] - centerSec);
          if (d < bestDist) {
            bestDist = d;
            best = i;
          }
        }
        base = anchors[best].notes;
      }
      const highlight = new Set<string>(base);
      if (mode === "all") {
        // 全 tine 表示: 未認識の伴奏ノーツがどこで鳴っているかの探索用。
        // 1 音 3ms 実測 (48kHz/±1s/40ms step) なので 17 tine でも ~50ms
        const notes = [...tuningByFreq].reverse().map((n) => n.noteName);
        return { notes, highlight };
      }
      const set = new Set<string>(base);
      if (withNeighbors) {
        for (const n of base) for (const nb of scaleNeighbors(tuningByFreq, n)) set.add(nb);
      }
      const notes = [...set]
        .filter((n) => noteFrequency(tuningByFreq, n) !== null)
        .sort((a, b) => (noteFrequency(tuningByFreq, b) ?? 0) - (noteFrequency(tuningByFreq, a) ?? 0))
        .slice(0, MAX_ROW_MODE_NOTES);
      return { notes, highlight };
    },
    [anchors, rowTimes, withNeighbors, mode, tuningByFreq],
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
      highlight: Set<string>,
      trace: { startSec: number; stepSec: number; steps: number; values: Float32Array },
      centerSec: number,
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = canvas.clientWidth || 600;
      const rowH = notes.length >= 10 ? 24 : 34;
      const cssHeight = Math.max(1, notes.length) * rowH + 18;
      canvas.width = Math.round(cssWidth * dpr);
      canvas.height = Math.round(cssHeight * dpr);
      canvas.style.height = `${cssHeight}px`;
      const g = canvas.getContext("2d");
      if (!g) return;
      g.scale(dpr, dpr);
      g.clearRect(0, 0, cssWidth, cssHeight);

      const labelW = 76;
      const plotW = cssWidth - labelW - 6;
      const { startSec, stepSec, steps, values } = trace;
      const windowDur = steps * stepSec;
      let globalMax = 0;
      for (let i = 0; i < values.length; i++) if (values[i] > globalMax) globalMax = values[i];
      if (globalMax <= 0) globalMax = 1;

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
        const isHi = highlight.has(notes[n]);
        // 行ごと正規化 (波形形状の可視化 — 弱い伴奏でも山が見える)。
        // 相対強度は % ラベル (行 max / 窓内グローバル max) で補う
        let rowMax = 0;
        for (let s = 0; s < steps; s++) {
          const v = values[n * steps + s];
          if (v > rowMax) rowMax = v;
        }
        const pct = Math.round((rowMax / globalMax) * 100);
        g.fillStyle = isHi ? "#0f5f67" : "#63615d";
        if (isHi) g.font = "bold 11px ui-monospace, Menlo, monospace";
        g.fillText(notes[n], 4, y0 + rowH / 2 + 3);
        if (isHi) g.font = "11px ui-monospace, Menlo, monospace";
        g.fillStyle = pct >= 20 ? "#177e89" : "#9b9893";
        g.fillText(`${pct}%`, 34, y0 + rowH / 2 + 3);
        g.strokeStyle = "rgba(30,31,31,0.1)";
        g.beginPath();
        g.moveTo(labelW, base);
        g.lineTo(labelW + plotW, base);
        g.stroke();
        g.strokeStyle = isHi ? "#0f5f67" : pct >= 20 ? "#177e89" : "#a8a5a0";
        g.lineWidth = isHi ? 1.8 : 1.2;
        g.beginPath();
        // 共通 dB スケール (窓内グローバル max 基準、floor -48dB):
        // 全行の最大値が同じ物差しに載り、行間の相対強度が形で読める
        // (2026-07-05 フィードバック — 行別正規化は 1% の音が 100% と同じ
        // 高さに見えて誤読を招いた。弱音の形は dB が担保する)
        const DB_FLOOR = -48;
        for (let s = 0; s < steps; s++) {
          const raw = values[n * steps + s] / globalMax;
          const db = raw > 0 ? 20 * Math.log10(raw) : DB_FLOOR;
          const v = Math.max(0, 1 - Math.max(db, DB_FLOOR) / DB_FLOOR);
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
      const { notes, highlight } = notesForCenter(centerSec);
      const notesKey = `${mode}:${notes.join(",")}|${[...highlight].join(",")}`;
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
        const freqs = notes.map((n) => noteFrequency(tuningByFreq, n) ?? 0);
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
        draw(notes, highlight, trace, centerSec);
        setStatus(`${notes.length} 音 × ${trace.steps} step (${trace.elapsedMs.toFixed(0)}ms)`);
      } finally {
        computingRef.current = false;
      }
    },
    [enabled, mode, audioRef, notesForCenter, ensureDecoded, draw, tuningByFreq],
  );

  // 再生位置ポーリング → 閾値超えの移動で再計算
  useEffect(() => {
    if (!enabled) return;
    const timer = window.setInterval(() => void recompute(false), POLL_MS);
    void recompute(true);
    return () => window.clearInterval(timer);
  }, [enabled, recompute]);

  // tx / 表示モード切替でキャッシュ・表示をリセット
  useEffect(() => {
    lastCenterRef.current = -999;
    lastNotesKeyRef.current = "";
  }, [txId, withNeighbors, mode]);

  return (
    <div className="gt-energy-trace">
      <div className="gt-energy-trace-controls">
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
          />{" "}
          energy trace (再生位置 ±{HALF_WINDOW_SEC}s)
        </label>
        {enabled ? (
          <label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value === "row" ? "row" : "all")}
            >
              <option value="all">全 tine (伴奏探し)</option>
              <option value="row">近傍行のノートのみ</option>
            </select>
          </label>
        ) : null}
        {enabled && mode === "row" ? (
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
