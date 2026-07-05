"use client";

// 開発用 bp-only 裁定ページ (第 3 期 S2、docs/sprint-plan-2026-07c.md の GT 除染)。
// bp_verify_prep.py が出力した bp-only 行 (GT にあるが recognizer が検出せず、
// Basic Pitch は検出した note) を試聴し、「聞こえる (実在)」「聞こえない (GT から
// 除去候補)」「不明瞭」のワンタップ裁定を自動保存する。/debug/gt-review と同じ
// 部品・保存方式を踏襲した姉妹ページ。
// 撤去条件: GT 除染の運用が落ち着いた時点で /api/dev/bp-verify と一緒に削除する。

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  bpVerifyRowKey,
  fetchBpVerify,
  saveBpVerifyVerdict,
  type BpVerifyData,
  type BpVerifyDecision,
  type BpVerifyRow,
  type BpVerifyRowVerdict,
  type BpVerifyVerdict,
} from "@/lib/api";
import { GtEnergyTrace } from "@/components/GtEnergyTrace";
import { computeAudioLevels } from "@/lib/audio";
import {
  boostDbForPeak,
  closeAudioBoost,
  ensureAudioBoost,
  type AudioBoostChain,
} from "@/lib/audioBoost";

const PLAY_LEAD_SEC = 1.0;
const PLAY_SNIPPET_SEC = 2.5;

const DECISION_LABEL: Record<BpVerifyDecision, string> = {
  real: "聞こえる (実在)",
  absent: "聞こえない (GT から除去候補)",
  unclear: "不明瞭",
};

function emptyVerdict(): BpVerifyVerdict {
  return { rows: {}, done: false };
}

export default function DebugBpVerifyPage() {
  const [data, setData] = useState<BpVerifyData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedTx8, setSelectedTx8] = useState<string | null>(null);
  const [verdict, setVerdict] = useState<BpVerifyVerdict>(emptyVerdict());
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [commentRow, setCommentRow] = useState<string | null>(null);
  const [playbackRate, setPlaybackRate] = useState(1);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playheadRef = useRef<HTMLSpanElement | null>(null);
  const pauseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const audio = audioRef.current;
      if (audio && playheadRef.current) {
        playheadRef.current.textContent = audio.currentTime.toFixed(3);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const applyPlaybackRate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.preservesPitch = true;
    audio.playbackRate = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    applyPlaybackRate();
  }, [applyPlaybackRate, selectedTx8]);

  useEffect(() => {
    fetchBpVerify()
      .then((doc) => {
        setData(doc);
        const groups = [...new Set(doc.rows.map((r) => r.tx8))];
        setSelectedTx8((prev) => prev ?? groups[0] ?? null);
        setVerdict(doc.verdict ?? emptyVerdict());
        setLoadError(null);
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : "読み込み失敗"));
  }, []);

  const rowsByTx = useMemo(() => {
    const groups = new Map<string, BpVerifyRow[]>();
    for (const row of data?.rows ?? []) {
      const list = groups.get(row.tx8) ?? [];
      list.push(row);
      groups.set(row.tx8, list);
    }
    return groups;
  }, [data]);

  const tx8List = useMemo(() => [...rowsByTx.keys()], [rowsByTx]);
  const currentRows = useMemo(
    () => (selectedTx8 ? (rowsByTx.get(selectedTx8) ?? []) : []),
    [rowsByTx, selectedTx8],
  );
  const currentTxId = currentRows[0]?.txId ?? null;
  // energy trace 用アンカー: bp-only 行の時刻 + 検証対象 note
  const energyAnchors = useMemo(
    () => currentRows.map((row) => ({ timeSec: row.timeSec, notes: [row.note] })),
    [currentRows],
  );

  // 静音録音のブースト + 片チャンネル無音ステレオの両耳化 (lib/audioBoost)。
  // peak はクライアント側で測る (audio は同一 URL なのでブラウザキャッシュが効く)
  const boostChainRef = useRef<AudioBoostChain | null>(null);
  const [peakDb, setPeakDb] = useState<number | null>(null);
  useEffect(() => {
    setPeakDb(null);
    if (!currentTxId) return;
    let alive = true;
    fetch(`/api/transcriptions/${currentTxId}/audio`)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`audio ${r.status}`))))
      .then((blob) => computeAudioLevels(blob))
      .then((levels) => {
        if (alive) setPeakDb(levels.peakDb);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [currentTxId]);
  const boostDb = boostDbForPeak(peakDb);
  const ensureBoost = useCallback(() => {
    ensureAudioBoost(audioRef.current, boostChainRef, boostDb);
  }, [boostDb]);
  useEffect(() => {
    return () => closeAudioBoost(boostChainRef);
  }, []);

  const scheduleSave = useCallback((next: BpVerifyVerdict) => {
    setSaveState("saving");
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      saveBpVerifyVerdict(next)
        .then(() => setSaveState("saved"))
        .catch(() => setSaveState("error"));
    }, 600);
  }, []);

  const updateVerdict = useCallback(
    (mutate: (v: BpVerifyVerdict) => BpVerifyVerdict) => {
      setVerdict((prev) => {
        const next = mutate(prev);
        scheduleSave(next);
        return next;
      });
    },
    [scheduleSave],
  );

  const setRowVerdict = useCallback(
    (key: string, patch: BpVerifyRowVerdict) => {
      updateVerdict((v) => {
        const merged = { ...(v.rows[key] ?? {}), ...patch };
        if (merged.comment === "") delete merged.comment;
        const rows = { ...v.rows };
        if (Object.keys(merged).length === 0) {
          delete rows[key];
        } else {
          rows[key] = merged;
        }
        return { ...v, rows };
      });
    },
    [updateVerdict],
  );

  const playAt = useCallback((timeSec: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    if (pauseTimer.current) clearTimeout(pauseTimer.current);
    audio.currentTime = Math.max(0, timeSec - PLAY_LEAD_SEC);
    void audio.play();
    pauseTimer.current = setTimeout(
      () => audio.pause(),
      (PLAY_LEAD_SEC + PLAY_SNIPPET_SEC) * 1000,
    );
  }, []);

  const totalRows = data?.rows.length ?? 0;
  const decidedCount = data?.rows.filter((r) => verdict.rows[bpVerifyRowKey(r)]?.decision).length ?? 0;
  const allDecided = totalRows > 0 && decidedCount === totalRows;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Dev bp-verify (temporary)</p>
          <h1>bp-only 23 件のワンタップ裁定</h1>
          <p className="hero-copy">
            GT にはあるが recognizer が検出せず、Basic Pitch は検出した note です。
            ▶ で該当時刻を再生し、聞こえるかどうかを裁定してください。裁定は自動保存されます。
            全件終わったらチャットで知らせてもらえれば、GT 反映はエージェント側で行います。
          </p>
        </div>
      </section>

      {loadError ? (
        <section className="panel">
          <div className="warning-box">
            <p>{loadError}</p>
            <p className="muted">
              先に `uv run python scripts/audio-analysis/research/bp_verify_prep.py`
              を実行してください。
            </p>
          </div>
        </section>
      ) : null}

      {data ? (
        <section className="panel">
          <div className="row wrap" style={{ gap: 8, alignItems: "center" }}>
            <span className="muted">
              全体進捗: {decidedCount}/{totalRows}
              {saveState === "saving" ? " / 保存中…" : ""}
              {saveState === "saved" ? " / 保存済み" : ""}
              {saveState === "error" ? " / ⚠保存失敗 (再操作で再試行)" : ""}
            </span>
            <label
              className="muted"
              style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 12 }}
            >
              <input
                type="checkbox"
                checked={verdict.done}
                disabled={!allDecided && !verdict.done}
                onChange={(e) => updateVerdict((v) => ({ ...v, done: e.target.checked }))}
              />
              全裁定完了 {allDecided ? "" : "(全行の裁定が終わると有効になります)"}
            </label>
          </div>
        </section>
      ) : null}

      {tx8List.length > 0 ? (
        <section className="panel">
          <div className="row wrap" style={{ gap: 8 }}>
            {tx8List.map((tx8) => {
              const rows = rowsByTx.get(tx8) ?? [];
              const decided = rows.filter((r) => verdict.rows[bpVerifyRowKey(r)]?.decision).length;
              return (
                <button
                  key={tx8}
                  type="button"
                  className={`review-btn review-btn-small${tx8 === selectedTx8 ? " review-btn-primary" : ""}`}
                  onClick={() => setSelectedTx8(tx8)}
                >
                  {decided === rows.length ? "✔ " : ""}
                  {tx8} ({decided}/{rows.length})
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      {currentTxId ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>{selectedTx8}</h2>
            </div>
            <span className="muted">{currentRows.length} 件の bp-only 候補</span>
          </div>

          <div style={{ position: "sticky", top: 0, zIndex: 5, background: "var(--background, #fff)", padding: "6px 0" }}>
            <audio
              ref={audioRef}
              controls
              preload="auto"
              src={`/api/transcriptions/${currentTxId}/audio`}
              style={{ width: "100%" }}
              onLoadedMetadata={applyPlaybackRate}
              onPlay={ensureBoost}
            />
            <div className="row wrap" style={{ gap: 6, alignItems: "center", marginTop: 4 }}>
              <span style={{ fontFamily: "monospace", fontSize: "1.05rem" }}>
                再生位置: <span ref={playheadRef}>0.000</span>s
              </span>
              {boostDb > 0 ? (
                <span className="muted">
                  試聴 +{boostDb.toFixed(0)}dB ブースト (元 peak {peakDb?.toFixed(1)}dB)
                </span>
              ) : null}
              <span className="muted" style={{ marginLeft: 8 }}>速度:</span>
              {[0.25, 0.5, 0.75, 1].map((rate) => (
                <button
                  key={rate}
                  type="button"
                  className={`review-btn review-btn-small${playbackRate === rate ? " active" : ""}`}
                  title={`再生速度 ${rate}x (音高は保持)`}
                  onClick={() => setPlaybackRate(rate)}
                  style={playbackRate === rate ? { fontWeight: 700, borderColor: "var(--accent)" } : undefined}
                >
                  {rate}x
                </button>
              ))}
            </div>
            <GtEnergyTrace
              txId={currentTxId}
              audioRef={audioRef}
              anchors={energyAnchors}
            />
          </div>

          <div className="stack" style={{ gap: 6, marginTop: 8 }}>
            {currentRows.map((row) => {
              const key = bpVerifyRowKey(row);
              const rv = verdict.rows[key];
              return (
                <div
                  key={key}
                  className="review-card"
                  style={{
                    padding: "6px 10px",
                    borderLeft:
                      rv?.decision === "real"
                        ? "3px solid #2e9e44"
                        : rv?.decision === "absent"
                          ? "3px solid #c0392b"
                          : rv?.decision === "unclear"
                            ? "3px solid #e0a800"
                            : "3px solid #999",
                  }}
                >
                  <div className="row wrap" style={{ gap: 10, alignItems: "baseline" }}>
                    <button
                      type="button"
                      className="review-btn review-btn-small"
                      onClick={() => playAt(row.timeSec)}
                      title={`${row.timeSec.toFixed(2)}s から再生 (${PLAY_LEAD_SEC}s 前から)`}
                    >
                      ▶ {row.timeSec.toFixed(2)}s
                    </button>
                    <span style={{ fontFamily: "monospace" }}>
                      <strong>{row.note}</strong>
                    </span>
                    <span
                      className="pill"
                      title={`bandEnergy=${row.bandEnergy.toFixed(1)} / noiseFloor=${row.noiseFloor.toFixed(1)} / ratio=${row.energyRatio.toFixed(2)}`}
                    >
                      {row.likelyAudible ? "即断候補" : "要精聴"}
                    </span>
                    {rv?.decision ? (
                      <span className="pill">裁定: {DECISION_LABEL[rv.decision]}</span>
                    ) : null}
                  </div>
                  <div className="row wrap" style={{ gap: 6, marginTop: 4 }}>
                    {(["real", "absent", "unclear"] as BpVerifyDecision[]).map((decision) => (
                      <button
                        key={decision}
                        type="button"
                        className={`review-btn review-btn-small${rv?.decision === decision ? " review-btn-primary" : ""}`}
                        onClick={() => setRowVerdict(key, { decision })}
                      >
                        {DECISION_LABEL[decision]}
                      </button>
                    ))}
                    <button
                      type="button"
                      className={`review-btn review-btn-small${rv?.comment ? " review-btn-primary" : ""}`}
                      onClick={() => setCommentRow(commentRow === key ? null : key)}
                    >
                      💬{rv?.comment ? " あり" : ""}
                    </button>
                  </div>
                  {commentRow === key ? (
                    <div style={{ marginTop: 6 }}>
                      <input
                        type="text"
                        value={rv?.comment ?? ""}
                        placeholder="この行へのコメント (例: 微かに聞こえるが確信なし)"
                        style={{ width: "100%", padding: "4px 8px" }}
                        onChange={(e) => setRowVerdict(key, { comment: e.target.value })}
                      />
                    </div>
                  ) : rv?.comment ? (
                    <p className="muted" style={{ margin: "4px 0 0" }}>
                      💬 {rv.comment}
                    </p>
                  ) : null}
                </div>
              );
            })}
          </div>
        </section>
      ) : null}
    </main>
  );
}
