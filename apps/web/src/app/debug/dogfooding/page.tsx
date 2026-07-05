"use client";

// 開発用 dogfooding (用途検証) 計測ページ (第 3 期 S2、docs/sprint-plan-2026-07c.md)。
// 判定基準は docs/usage-validation-criteria.md (2026-07-05 固定)、記入票は
// docs/dogfooding-protocol.md。修正操作の 7 分類カウントを手動 (正の字) で数える
// のは非現実的なため、review UI の操作ログ (lib/opLog.ts, localStorage 由来) を
// 自動集計し、諦め箇所・主観負荷・曖昧性カタログ・弾き戻し結果だけを手動記入する。
// R1-R5 / A1-A5 は usage-validation-criteria.md の閾値をそのままコード内定数に
// 落とし、事後裁量なしに自動判定する。
// main nav からはリンクしない。撤去条件: 用途検証の運用が落ち着いた時点で
// /api/dev/dogfooding と一緒に削除する。

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";

import {
  fetchCorrections,
  fetchDogfoodingRecord,
  fetchReviewQueue,
  fetchTranscription,
  saveDogfoodingRecord,
  emptyDogfoodingManual,
  type DogfoodingAmbiguityRow,
  type DogfoodingManual,
} from "@/lib/api";
import { loadOpLog, summarizeOpLog, type OpClass, type OpLogEntry } from "@/lib/opLog";
import { CorrectionsPayload, ReviewQueueEntry, TranscriptionResult } from "@/lib/types";

// usage-validation-criteria.md の 7 分類 (制御系 undo/redo/other を除く)。
// この順序で表・レポートに並べる
const CORRECTION_OP_CLASSES: OpClass[] = [
  "candidate-remove",
  "event-remove",
  "restrike-judgment",
  "chord-note-remove",
  "chord-note-add",
  "onset-insert-single",
  "onset-insert-multi",
];

const CONTROL_OP_CLASSES: OpClass[] = ["undo", "redo", "other"];

const OP_CLASS_INFO: Record<OpClass, { no: string; label: string; weightLabel: string }> = {
  "candidate-remove": { no: "1", label: "偽の候補の除去 (1タップ)", weightLabel: "0.5" },
  "event-remove": { no: "2", label: "偽の認識の除去", weightLabel: "1" },
  "restrike-judgment": { no: "3", label: "近接同一音のシングル/弾き直し判定", weightLabel: "1" },
  "chord-note-remove": { no: "4", label: "和音扱いからの一部削除", weightLabel: "1.5" },
  "chord-note-add": { no: "5", label: "和音扱いへの一部追加", weightLabel: "2" },
  "onset-insert-single": { no: "6", label: "候補に無い onset の追加 (単音)", weightLabel: "3" },
  "onset-insert-multi": { no: "7", label: "候補に無い onset の追加 (複数音)", weightLabel: "3 + 追加音数×1" },
  undo: { no: "-", label: "元に戻す (undo)", weightLabel: "—" },
  redo: { no: "-", label: "やり直す (redo)", weightLabel: "—" },
  other: { no: "-", label: "その他 (7分類非該当)", weightLabel: "—" },
};

// docs/usage-validation-criteria.md (2026-07-05 fixed) の数値をそのまま定数化。
// 変更する場合は docs/decision-log.md に理由付きで追記した上で行うこと。
const R1_CORRECTION_RATE = 0.1;
const R2_GIVE_UP_COUNT = 2;
const R3_PLAYBACK_SUCCESS_RATE = 0.6;
const R4_TIME_MULTIPLIER = 5;
const R5_TOTAL_OPS = 20;
const A1_PLAYBACK_SUCCESS_RATE = 0.8;
const A2_CORRECTION_RATE = 0.05;
const A4_TIME_MULTIPLIER = 2;
const A5_TOTAL_OPS = 10;

function formatMinSec(sec: number | null): string {
  if (sec === null || !Number.isFinite(sec)) return "—";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatPercent(ratio: number | null): string {
  if (ratio === null || !Number.isFinite(ratio)) return "—";
  return `${(ratio * 100).toFixed(1)}%`;
}

type Verdict3 = "精度が律速" | "粗い転写で足りる" | "中間帯 (保留)";

function emptyAmbiguityRow(): DogfoodingAmbiguityRow {
  return { timeSec: "", judgment: "", resolution: "" };
}

export default function DebugDogfoodingPage() {
  const [queue, setQueue] = useState<ReviewQueueEntry[]>([]);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [selectedTxId, setSelectedTxId] = useState<string | null>(null);

  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [corrections, setCorrections] = useState<CorrectionsPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [opEntries, setOpEntries] = useState<OpLogEntry[]>([]);
  const [sourceDurationSec, setSourceDurationSec] = useState<number | null>(null);

  const [manual, setManual] = useState<DogfoodingManual>(emptyDogfoodingManual());
  const [done, setDone] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");

  const [showReport, setShowReport] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">("idle");

  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    fetchReviewQueue({ limit: 100 })
      .then((entries) => {
        setQueue(entries);
        setSelectedTxId((prev) => prev ?? entries[0]?.transactionId ?? null);
      })
      .catch((err) => setQueueError(err instanceof Error ? err.message : "読み込み失敗"));
  }, []);

  useEffect(() => {
    if (!selectedTxId) return;
    let cancelled = false;
    setLoadError(null);
    setResult(null);
    setCorrections(null);
    setSourceDurationSec(null);
    setSaveState("idle");
    setShowReport(false);

    Promise.all([
      fetchTranscription(selectedTxId),
      fetchCorrections(selectedTxId).catch(() => null),
      fetchDogfoodingRecord(selectedTxId).catch(() => null),
    ])
      .then(([res, corr, record]) => {
        if (cancelled) return;
        setResult(res);
        setCorrections(corr);
        setManual(record?.manual ?? emptyDogfoodingManual());
        setDone(record?.done ?? false);
        setSavedAt(record?.updatedAt ?? null);
        setOpEntries(loadOpLog(selectedTxId));
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : "読み込み失敗");
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTxId]);

  const refreshOpLog = useCallback(() => {
    if (selectedTxId) setOpEntries(loadOpLog(selectedTxId));
  }, [selectedTxId]);

  const summary = useMemo(() => summarizeOpLog(opEntries), [opEntries]);

  const totalNotes = useMemo(() => {
    if (corrections && corrections.events.length > 0) {
      return {
        count: corrections.events.reduce((sum, e) => sum + e.notes.length, 0),
        basis: "修正済み譜面 (corrections)" as const,
      };
    }
    if (result) {
      return {
        count: result.events.reduce((sum, e) => sum + e.notes.length, 0),
        basis: "認識結果 baseline (未保存)" as const,
      };
    }
    return { count: 0, basis: "—" as const };
  }, [result, corrections]);

  const correctionRate = totalNotes.count > 0 ? summary.touchedNoteCount / totalNotes.count : null;

  const totalCorrectionOps = CORRECTION_OP_CLASSES.reduce(
    (sum, cls) => sum + summary.countsByClass[cls],
    0,
  );

  const reproducedRate = useMemo(() => {
    const { phraseCount, reproducedPhraseCount } = manual.playback;
    if (!phraseCount || phraseCount <= 0 || reproducedPhraseCount === null) return null;
    return reproducedPhraseCount / phraseCount;
  }, [manual.playback]);

  const r1 = correctionRate !== null && correctionRate > R1_CORRECTION_RATE;
  const r2 = manual.giveUpCount >= R2_GIVE_UP_COUNT;
  const r3 = reproducedRate !== null && reproducedRate < R3_PLAYBACK_SUCCESS_RATE;
  const r4 = sourceDurationSec !== null && summary.activeTimeSec > sourceDurationSec * R4_TIME_MULTIPLIER;
  const r5 = totalCorrectionOps > R5_TOTAL_OPS;

  const a1 = reproducedRate !== null && reproducedRate >= A1_PLAYBACK_SUCCESS_RATE;
  const a2 = correctionRate !== null && correctionRate <= A2_CORRECTION_RATE;
  const a3 = manual.giveUpCount === 0;
  const a4 = sourceDurationSec !== null && summary.activeTimeSec <= sourceDurationSec * A4_TIME_MULTIPLIER;
  const a5 = totalCorrectionOps <= A5_TOTAL_OPS;

  const verdict: Verdict3 = useMemo(() => {
    if (r1 || r2 || r3 || r4 || r5) return "精度が律速";
    if (a1 && a2 && a3 && a4 && a5) return "粗い転写で足りる";
    return "中間帯 (保留)";
  }, [r1, r2, r3, r4, r5, a1, a2, a3, a4, a5]);

  const updateManual = useCallback((patch: Partial<DogfoodingManual>) => {
    setManual((prev) => ({ ...prev, ...patch }));
  }, []);

  const updatePlayback = useCallback((patch: Partial<DogfoodingManual["playback"]>) => {
    setManual((prev) => ({ ...prev, playback: { ...prev.playback, ...patch } }));
  }, []);

  const addAmbiguityRow = useCallback(() => {
    setManual((prev) => ({ ...prev, ambiguityLog: [...prev.ambiguityLog, emptyAmbiguityRow()] }));
  }, []);

  const updateAmbiguityRow = useCallback(
    (index: number, patch: Partial<DogfoodingAmbiguityRow>) => {
      setManual((prev) => ({
        ...prev,
        ambiguityLog: prev.ambiguityLog.map((row, i) => (i === index ? { ...row, ...patch } : row)),
      }));
    },
    [],
  );

  const removeAmbiguityRow = useCallback((index: number) => {
    setManual((prev) => ({
      ...prev,
      ambiguityLog: prev.ambiguityLog.filter((_, i) => i !== index),
    }));
  }, []);

  const handleSave = useCallback(async () => {
    if (!selectedTxId) return;
    setSaveState("saving");
    try {
      const saved = await saveDogfoodingRecord(selectedTxId, { manual, done });
      setSavedAt(saved.updatedAt ?? null);
      setSaveState("saved");
    } catch {
      setSaveState("error");
    }
  }, [selectedTxId, manual, done]);

  const reportMarkdown = useMemo(() => {
    if (!selectedTxId) return "";
    const tx8 = selectedTxId.slice(0, 8);
    const rows = CORRECTION_OP_CLASSES.map((cls) => {
      const info = OP_CLASS_INFO[cls];
      return `| ${info.no} | ${info.label} | ${summary.countsByClass[cls]} |`;
    }).join("\n");
    const ambiguityRows =
      manual.ambiguityLog.length > 0
        ? manual.ambiguityLog
            .map((row) => `| ${row.timeSec} | ${row.judgment} | ${row.resolution} |`)
            .join("\n")
        : "| | | |";
    const calibrationRows = [...CORRECTION_OP_CLASSES, ...CONTROL_OP_CLASSES]
      .map((cls) => {
        const avg = summary.avgGapSecByClass[cls];
        return `- ${OP_CLASS_INFO[cls].label}: 平均 ${avg !== undefined ? formatMinSec(avg) : "—"} (n=${summary.countsByClass[cls]})`;
      })
      .join("\n");

    return `## 実施記録 (自動生成 — /debug/dogfooding)

### 基本情報

| 項目 | 記入 |
|------|------|
| 実施日 | ${manual.sessionDate || "(未記入)"} |
| tx8 | ${tx8} |
| 曲 / 内容 | ${manual.pieceInfo || "(未記入)"} |
| 音源長 (秒) | ${sourceDurationSec !== null ? sourceDurationSec.toFixed(1) : "(未取得)"} |
| GT/最終譜面の音数 | ${totalNotes.count} (${totalNotes.basis}) |

### 修正セッション

| 項目 | 記入 |
|------|------|
| 修正時間 (分, active time) | ${(summary.activeTimeSec / 60).toFixed(1)} |
| 修正時間 (分, wall time 参考) | ${(summary.wallTimeSec / 60).toFixed(1)} |
| 主観負荷 (1=楽 〜 5=苦行) | ${manual.subjectiveLoad ?? "(未記入)"} |

修正操作の 7 分類カウント (opLog 自動集計):

| # | 操作 | 件数 |
|---|------|------|
${rows}

| 項目 | 記入 |
|------|------|
| 「直すより諦めた」箇所数 | ${manual.giveUpCount} |
| 諦めた内容 (簡潔に) | ${manual.giveUpNotes || "(未記入)"} |

### 音楽的曖昧性カタログ

| 時刻付近 | 曖昧だった判断 | どう裁いたか |
|----------|----------------|--------------|
${ambiguityRows}

### 弾き戻し検証 (未修正譜面で実施 — 別日可)

| 項目 | 記入 |
|------|------|
| フレーズ分割数 | ${manual.playback.phraseCount ?? "(未記入)"} |
| 再現できたフレーズ数 | ${manual.playback.reproducedPhraseCount ?? "(未記入)"} |
| 成功率 (%) | ${formatPercent(reproducedRate)} |
| つまずき分類 (音違い / リズム / 表記) と件数 | ${manual.playback.stumblePitch} / ${manual.playback.stumbleRhythm} / ${manual.playback.stumbleNotation} |

### 判定 (基準 doc の R/A 条件に機械的に当てる — 事後裁量なし)

| 条件 | 値 | 成立? |
|------|-----|-------|
| R1 修正必要音率 > 0.10 | ${formatPercent(correctionRate)} | ${r1 ? "○" : "×"} |
| R2 諦め ≥ 2 | ${manual.giveUpCount} | ${r2 ? "○" : "×"} |
| R3 未修正弾き戻し < 60% | ${formatPercent(reproducedRate)} | ${reproducedRate === null ? "未計測" : r3 ? "○" : "×"} |
| R4 修正時間 > 音源長 ×5 | ${formatMinSec(summary.activeTimeSec)} vs ${formatMinSec(sourceDurationSec !== null ? sourceDurationSec * R4_TIME_MULTIPLIER : null)} | ${sourceDurationSec === null ? "未計測" : r4 ? "○" : "×"} |
| R5 総修正箇所 > 20 | ${totalCorrectionOps} | ${r5 ? "○" : "×"} |
| A1 弾き戻し ≥ 80% | ${formatPercent(reproducedRate)} | ${reproducedRate === null ? "未計測" : a1 ? "○" : "×"} |
| A2 修正必要音率 ≤ 0.05 | ${formatPercent(correctionRate)} | ${a2 ? "○" : "×"} |
| A3 諦め 0 | ${manual.giveUpCount} | ${a3 ? "○" : "×"} |
| A4 修正時間 ≤ 音源長 ×2 | ${formatMinSec(summary.activeTimeSec)} vs ${formatMinSec(sourceDurationSec !== null ? sourceDurationSec * A4_TIME_MULTIPLIER : null)} | ${sourceDurationSec === null ? "未計測" : a4 ? "○" : "×"} |
| A5 総修正箇所 ≤ 10 | ${totalCorrectionOps} | ${a5 ? "○" : "×"} |

**判定**: ${verdict}

### 重み較正メモ (分類別平均所要時間、直前操作からの経過を対象クラスごとに平均。離席等の長い空白は 120s で頭打ち)

${calibrationRows}
`;
  }, [
    selectedTxId,
    manual,
    summary,
    sourceDurationSec,
    totalNotes,
    correctionRate,
    reproducedRate,
    totalCorrectionOps,
    r1,
    r2,
    r3,
    r4,
    r5,
    a1,
    a2,
    a3,
    a4,
    a5,
    verdict,
  ]);

  const handleCopyReport = useCallback(async () => {
    setShowReport(true);
    try {
      await navigator.clipboard.writeText(reportMarkdown);
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
  }, [reportMarkdown]);

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Dev dogfooding (temporary)</p>
          <h1>用途検証 (dogfooding) 計測</h1>
          <p className="hero-copy">
            判定基準:{" "}
            <a href="/docs/usage-validation-criteria.md" target="_blank" rel="noreferrer">
              usage-validation-criteria.md
            </a>{" "}
            (2026-07-05 固定)。7 分類カウントは review UI の操作ログから自動集計、
            諦め箇所・主観負荷・曖昧性カタログ・弾き戻しだけを手動記入する。
          </p>
        </div>
      </section>

      {queueError ? (
        <section className="panel">
          <div className="warning-box">
            <p>{queueError}</p>
          </div>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <h2>録音を選ぶ</h2>
          <span className="muted">{queue.length} 件</span>
        </div>
        <div className="row wrap" style={{ gap: 8 }}>
          {queue.map((entry) => (
            <button
              key={entry.transactionId}
              type="button"
              className={`review-btn review-btn-small${
                entry.transactionId === selectedTxId ? " review-btn-primary" : ""
              }`}
              onClick={() => setSelectedTxId(entry.transactionId)}
              title={entry.transactionId}
            >
              {entry.transactionId.slice(0, 8)}
              {entry.tuningName ? ` · ${entry.tuningName}` : ""} ({entry.eventCount})
            </button>
          ))}
        </div>
      </section>

      {loadError ? (
        <section className="panel">
          <div className="warning-box">
            <p>{loadError}</p>
          </div>
        </section>
      ) : null}

      {selectedTxId && result ? (
        <>
          <section className="panel">
            <div className="panel-header">
              <h2>音源</h2>
              <span className="muted">tx8: {selectedTxId.slice(0, 8)}</span>
            </div>
            <audio
              ref={audioRef}
              controls
              preload="metadata"
              src={`/api/transcriptions/${selectedTxId}/audio`}
              style={{ width: "100%" }}
              onLoadedMetadata={() => {
                const dur = audioRef.current?.duration;
                setSourceDurationSec(Number.isFinite(dur) ? (dur as number) : null);
              }}
            />
            <p className="muted">
              音源長: {sourceDurationSec !== null ? `${sourceDurationSec.toFixed(1)}s` : "読み込み中…"} /
              GT・最終音数: {totalNotes.count} ({totalNotes.basis})
            </p>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>自動計測 (opLog 集計)</h2>
              <button type="button" className="review-btn review-btn-small" onClick={refreshOpLog}>
                ↻ 再読込
              </button>
            </div>
            <div className="row wrap" style={{ gap: 16 }}>
              <span className="pill">Active time: {formatMinSec(summary.activeTimeSec)}</span>
              <span className="pill">Wall time: {formatMinSec(summary.wallTimeSec)}</span>
              <span className="pill">総修正箇所: {totalCorrectionOps}</span>
              <span className="pill">修正必要音率 (近似): {formatPercent(correctionRate)}</span>
              <span className="pill">touched notes: {summary.touchedNoteCount}</span>
            </div>

            <div style={{ overflowX: "auto", marginTop: 10 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.92rem" }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>操作</th>
                    <th style={thStyle}>暫定重み</th>
                    <th style={thStyle}>件数</th>
                    <th style={thStyle}>平均間隔 (較正用)</th>
                  </tr>
                </thead>
                <tbody>
                  {[...CORRECTION_OP_CLASSES, ...CONTROL_OP_CLASSES].map((cls) => {
                    const info = OP_CLASS_INFO[cls];
                    const avg = summary.avgGapSecByClass[cls];
                    return (
                      <tr key={cls}>
                        <td style={tdStyle}>{info.no}</td>
                        <td style={tdStyle}>{info.label}</td>
                        <td style={tdStyle}>{info.weightLabel}</td>
                        <td style={tdStyle}>{summary.countsByClass[cls]}</td>
                        <td style={tdStyle}>{avg !== undefined ? formatMinSec(avg) : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {opEntries.length === 0 ? (
              <p className="empty">
                この tx8 の操作ログはまだありません。/review/{selectedTxId} で修正してから戻ってください。
              </p>
            ) : null}
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>手動記入</h2>
            </div>
            <div className="stack">
              <div className="row wrap" style={{ gap: 12 }}>
                <label className="muted">
                  実施日
                  <br />
                  <input
                    type="date"
                    value={manual.sessionDate}
                    onChange={(e) => updateManual({ sessionDate: e.target.value })}
                  />
                </label>
                <label className="muted" style={{ flex: 1, minWidth: 220 }}>
                  曲 / 内容
                  <br />
                  <input
                    type="text"
                    style={{ width: "100%" }}
                    value={manual.pieceInfo}
                    onChange={(e) => updateManual({ pieceInfo: e.target.value })}
                  />
                </label>
              </div>

              <div className="row wrap" style={{ gap: 12, alignItems: "center" }}>
                <span className="muted">主観負荷 (1=楽 〜 5=苦行):</span>
                {[1, 2, 3, 4, 5].map((v) => (
                  <button
                    key={v}
                    type="button"
                    className={`review-btn review-btn-small${manual.subjectiveLoad === v ? " review-btn-primary" : ""}`}
                    onClick={() => updateManual({ subjectiveLoad: v })}
                  >
                    {v}
                  </button>
                ))}
              </div>

              <div className="row wrap" style={{ gap: 12, alignItems: "center" }}>
                <label className="muted">
                  「直すより諦めた」箇所数
                  <br />
                  <input
                    type="number"
                    min={0}
                    value={manual.giveUpCount}
                    onChange={(e) => updateManual({ giveUpCount: Math.max(0, Number(e.target.value) || 0) })}
                    style={{ width: 90 }}
                  />
                </label>
                <label className="muted" style={{ flex: 1, minWidth: 220 }}>
                  諦めた内容 (簡潔に)
                  <br />
                  <input
                    type="text"
                    style={{ width: "100%" }}
                    value={manual.giveUpNotes}
                    onChange={(e) => updateManual({ giveUpNotes: e.target.value })}
                  />
                </label>
              </div>

              <div>
                <div className="panel-header compact">
                  <h3 style={{ margin: 0 }}>音楽的曖昧性カタログ (副産物 — 気づいたら書く)</h3>
                  <button type="button" className="review-btn review-btn-small" onClick={addAmbiguityRow}>
                    ＋ 行を追加
                  </button>
                </div>
                {manual.ambiguityLog.length === 0 ? (
                  <p className="muted">まだ記録がありません。</p>
                ) : (
                  <div className="stack" style={{ gap: 6 }}>
                    {manual.ambiguityLog.map((row, i) => (
                      <div key={i} className="row wrap" style={{ gap: 6 }}>
                        <input
                          type="text"
                          placeholder="時刻付近 (例 12.3s)"
                          value={row.timeSec}
                          onChange={(e) => updateAmbiguityRow(i, { timeSec: e.target.value })}
                          style={{ width: 130 }}
                        />
                        <input
                          type="text"
                          placeholder="曖昧だった判断"
                          value={row.judgment}
                          onChange={(e) => updateAmbiguityRow(i, { judgment: e.target.value })}
                          style={{ flex: 1, minWidth: 160 }}
                        />
                        <input
                          type="text"
                          placeholder="どう裁いたか"
                          value={row.resolution}
                          onChange={(e) => updateAmbiguityRow(i, { resolution: e.target.value })}
                          style={{ flex: 1, minWidth: 160 }}
                        />
                        <button
                          type="button"
                          className="review-btn review-btn-small review-btn-danger"
                          onClick={() => removeAmbiguityRow(i)}
                        >
                          削除
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <h3 style={{ margin: "8px 0" }}>弾き戻し検証 (未修正譜面で実施 — 別日可)</h3>
                <div className="row wrap" style={{ gap: 12 }}>
                  <label className="muted">
                    フレーズ分割数
                    <br />
                    <input
                      type="number"
                      min={0}
                      value={manual.playback.phraseCount ?? ""}
                      onChange={(e) =>
                        updatePlayback({ phraseCount: e.target.value === "" ? null : Number(e.target.value) })
                      }
                      style={{ width: 90 }}
                    />
                  </label>
                  <label className="muted">
                    再現できたフレーズ数
                    <br />
                    <input
                      type="number"
                      min={0}
                      value={manual.playback.reproducedPhraseCount ?? ""}
                      onChange={(e) =>
                        updatePlayback({
                          reproducedPhraseCount: e.target.value === "" ? null : Number(e.target.value),
                        })
                      }
                      style={{ width: 90 }}
                    />
                  </label>
                  <span className="pill">成功率: {formatPercent(reproducedRate)}</span>
                </div>
                <div className="row wrap" style={{ gap: 12, marginTop: 6 }}>
                  <label className="muted">
                    つまずき: 音違い
                    <br />
                    <input
                      type="number"
                      min={0}
                      value={manual.playback.stumblePitch}
                      onChange={(e) => updatePlayback({ stumblePitch: Math.max(0, Number(e.target.value) || 0) })}
                      style={{ width: 90 }}
                    />
                  </label>
                  <label className="muted">
                    つまずき: リズム
                    <br />
                    <input
                      type="number"
                      min={0}
                      value={manual.playback.stumbleRhythm}
                      onChange={(e) => updatePlayback({ stumbleRhythm: Math.max(0, Number(e.target.value) || 0) })}
                      style={{ width: 90 }}
                    />
                  </label>
                  <label className="muted">
                    つまずき: 表記
                    <br />
                    <input
                      type="number"
                      min={0}
                      value={manual.playback.stumbleNotation}
                      onChange={(e) =>
                        updatePlayback({ stumbleNotation: Math.max(0, Number(e.target.value) || 0) })
                      }
                      style={{ width: 90 }}
                    />
                  </label>
                </div>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>判定</h2>
              <span className={`pill${verdict === "精度が律速" ? " live" : ""}`}>{verdict}</span>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.92rem" }}>
                <thead>
                  <tr>
                    <th style={thStyle}>条件</th>
                    <th style={thStyle}>値</th>
                    <th style={thStyle}>成立?</th>
                  </tr>
                </thead>
                <tbody>
                  <VerdictRow label="R1 修正必要音率 > 0.10" value={formatPercent(correctionRate)} hit={r1} />
                  <VerdictRow label="R2 諦め ≥ 2" value={String(manual.giveUpCount)} hit={r2} />
                  <VerdictRow
                    label="R3 未修正弾き戻し < 60%"
                    value={formatPercent(reproducedRate)}
                    hit={reproducedRate === null ? null : r3}
                  />
                  <VerdictRow
                    label="R4 修正時間 > 音源長×5"
                    value={`${formatMinSec(summary.activeTimeSec)} / ${formatMinSec(
                      sourceDurationSec !== null ? sourceDurationSec * R4_TIME_MULTIPLIER : null,
                    )}`}
                    hit={sourceDurationSec === null ? null : r4}
                  />
                  <VerdictRow label="R5 総修正箇所 > 20" value={String(totalCorrectionOps)} hit={r5} />
                  <VerdictRow
                    label="A1 弾き戻し ≥ 80%"
                    value={formatPercent(reproducedRate)}
                    hit={reproducedRate === null ? null : a1}
                  />
                  <VerdictRow label="A2 修正必要音率 ≤ 0.05" value={formatPercent(correctionRate)} hit={a2} />
                  <VerdictRow label="A3 諦め 0" value={String(manual.giveUpCount)} hit={a3} />
                  <VerdictRow
                    label="A4 修正時間 ≤ 音源長×2"
                    value={`${formatMinSec(summary.activeTimeSec)} / ${formatMinSec(
                      sourceDurationSec !== null ? sourceDurationSec * A4_TIME_MULTIPLIER : null,
                    )}`}
                    hit={sourceDurationSec === null ? null : a4}
                  />
                  <VerdictRow label="A5 総修正箇所 ≤ 10" value={String(totalCorrectionOps)} hit={a5} />
                </tbody>
              </table>
            </div>
            <p className="muted" style={{ marginTop: 8 }}>
              R はいずれか 1 つ成立で「精度が律速」。A は全成立で「粗い転写で足りる」。
              どちらでもなければ「中間帯 (保留)」(docs/usage-validation-criteria.md)。
            </p>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>保存 / レポート</h2>
            </div>
            <div className="row wrap" style={{ gap: 12, alignItems: "center" }}>
              <label className="muted" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={done} onChange={(e) => setDone(e.target.checked)} />
                記入完了
              </label>
              <button type="button" className="review-btn review-btn-primary" onClick={handleSave}>
                {saveState === "saving" ? "保存中…" : "保存"}
              </button>
              <span className="muted">
                {saveState === "saved"
                  ? "保存しました"
                  : saveState === "error"
                    ? "⚠保存失敗 (再操作で再試行)"
                    : savedAt
                      ? `前回保存: ${savedAt}`
                      : " "}
              </span>
              <button
                type="button"
                className="review-btn review-btn-small"
                onClick={handleCopyReport}
              >
                📋 レポート markdown をコピー
              </button>
              {copyState === "copied" ? <span className="muted">コピーしました</span> : null}
              {copyState === "error" ? (
                <span className="muted">自動コピー失敗 — 下のテキストを手動選択してください</span>
              ) : null}
            </div>
            {showReport ? (
              <textarea
                readOnly
                value={reportMarkdown}
                style={{ width: "100%", minHeight: 360, marginTop: 10, fontFamily: "monospace", fontSize: "0.85rem" }}
                onFocus={(e) => e.currentTarget.select()}
              />
            ) : null}
          </section>
        </>
      ) : null}
    </main>
  );
}

const thStyle: CSSProperties = {
  textAlign: "left",
  padding: "4px 8px",
  borderBottom: "2px solid var(--line)",
  whiteSpace: "nowrap",
};

const tdStyle: CSSProperties = {
  padding: "4px 8px",
  borderBottom: "1px solid var(--line)",
};

function VerdictRow({ label, value, hit }: { label: string; value: string; hit: boolean | null }) {
  return (
    <tr>
      <td style={tdStyle}>{label}</td>
      <td style={tdStyle}>{value}</td>
      <td style={tdStyle}>{hit === null ? "未計測" : hit ? "○" : "×"}</td>
    </tr>
  );
}
