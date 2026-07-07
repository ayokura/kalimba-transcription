"use client";

// 開発用 day-2 弾き戻しページ (第 4 期 B0、docs/usage-validation-criteria.md の
// 弾き戻し protocol 精緻化)。記憶交絡フリーの弾き戻し検証を回すための画面:
// 指定 tx の【認識結果の譜面だけ】を表示し、それを見ながら弾き戻しを録音する。
// 元の楽譜は読み込まない。音源プレイヤーも出さない (原演奏を聴くと記憶補助に
// なるため)。弾き戻し録音は force:true で必ず別 tx 化し、memo に元 tx を埋めて
// 後から辿れるようにする (API スキーマ変更なし)。
// main nav からはリンクしない。撤去条件: B0 の用途検証が済んだ時点で削除。

import { useCallback, useEffect, useState } from "react";

import {
  createTranscriptionWithCapture,
  fetchMemo,
  fetchReviewQueue,
  fetchTranscription,
} from "@/lib/api";
import { NotationPanel } from "@/components/NotationPanel";
import { RecorderPanel } from "@/components/RecorderPanel";
import { NotationMode, ReviewQueueEntry, TranscriptionResult } from "@/lib/types";

type SubmitState = "idle" | "submitting" | "done" | "error";

export default function DebugDay2Page() {
  const [queue, setQueue] = useState<ReviewQueueEntry[]>([]);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [selectedTxId, setSelectedTxId] = useState<string | null>(null);

  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [sourceMemo, setSourceMemo] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mode, setMode] = useState<NotationMode>("score");

  const [blob, setBlob] = useState<Blob | null>(null);
  const [submitState, setSubmitState] = useState<SubmitState>("idle");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [playbackTxId, setPlaybackTxId] = useState<string | null>(null);

  useEffect(() => {
    // ?tx=... があれば初期選択に使う (useSearchParams は Suspense 要求があるため
    // client 限定で location から直接読む)
    const txParam =
      typeof window !== "undefined"
        ? new URLSearchParams(window.location.search).get("tx")
        : null;
    fetchReviewQueue({ limit: 100 })
      .then((entries) => {
        setQueue(entries);
        const firstId: string | null = entries[0]?.transactionId ?? null;
        setSelectedTxId((prev) => prev ?? txParam ?? firstId);
      })
      .catch((err) => setQueueError(err instanceof Error ? err.message : "読み込み失敗"));
  }, []);

  useEffect(() => {
    if (!selectedTxId) return;
    let cancelled = false;
    setLoadError(null);
    setResult(null);
    setSourceMemo(null);
    setBlob(null);
    setSubmitState("idle");
    setSubmitError(null);
    setPlaybackTxId(null);

    Promise.all([
      fetchTranscription(selectedTxId),
      fetchMemo(selectedTxId).catch(() => ""),
    ])
      .then(([res, memo]) => {
        if (cancelled) return;
        setResult(res);
        setSourceMemo(memo || null);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : "読み込み失敗");
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTxId]);

  const handleSubmit = useCallback(async () => {
    if (!blob || !result || !selectedTxId) return;
    setSubmitState("submitting");
    setSubmitError(null);
    try {
      const capture = await createTranscriptionWithCapture(blob, result.instrumentTuning, {
        force: true,
        memo: `day2-playback-of=${selectedTxId}`,
        sourceProfile: "acoustic_real",
      });
      setPlaybackTxId(capture.responsePayload.transactionId ?? null);
      setSubmitState("done");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "送信失敗");
      setSubmitState("error");
    }
  }, [blob, result, selectedTxId]);

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Dev day-2 playback (temporary)</p>
          <h1>day-2 弾き戻し (記憶交絡フリー)</h1>
          <p className="hero-copy">
            下の<strong>【認識結果の譜面だけ】</strong>を見て弾き、その場で録音します。
            <strong>元の楽譜は見ないこと。</strong>原演奏の音源はこのページに出していません
            (聴くと記憶補助になるため)。弾き戻し録音は元 tx に紐付けて別 tx として保存します。
            この後の順番は「弾き戻し (このページ) → /debug/dogfooding で修正・判定 → review UI で GT 化」。
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
          <h2>元の録音を選ぶ (認識結果を読む対象)</h2>
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
        {selectedTxId ? (
          <p className="muted" style={{ marginTop: 8 }}>
            対象 tx8: {selectedTxId.slice(0, 8)}
            {sourceMemo ? ` · ${sourceMemo}` : ""}
          </p>
        ) : null}
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
          <NotationPanel result={result} mode={mode} onModeChange={setMode} />

          <RecorderPanel onRecordingReady={setBlob} hasRecording={Boolean(blob)} />

          <section className="panel">
            <div className="panel-header">
              <h2>弾き戻しを保存</h2>
              {playbackTxId ? (
                <span className="pill live">保存済み</span>
              ) : blob ? (
                <span className="pill">録音待機</span>
              ) : null}
            </div>
            <div className="row wrap" style={{ gap: 12, alignItems: "center" }}>
              <button
                type="button"
                className="review-btn review-btn-primary"
                onClick={handleSubmit}
                disabled={!blob || submitState === "submitting"}
              >
                {submitState === "submitting" ? "送信中…" : "弾き戻し録音を送信"}
              </button>
              {submitState === "done" && playbackTxId ? (
                <span className="muted">
                  保存しました — 弾き戻し tx8:{" "}
                  <a href={`/score/${playbackTxId}`} target="_blank" rel="noreferrer">
                    {playbackTxId.slice(0, 8)}
                  </a>{" "}
                  (元 tx8: {selectedTxId.slice(0, 8)})
                </span>
              ) : null}
              {submitState === "error" ? (
                <span className="error">⚠ 送信失敗: {submitError}</span>
              ) : null}
            </div>
            <p className="muted" style={{ marginTop: 8 }}>
              録音は「認識結果を見て弾いたもの」です。送信すると認識まで走り、元 tx を
              memo に埋めた別 tx として保存されます。もう 1 曲あるときは上で別の tx を
              選んで繰り返してください。
            </p>
          </section>
        </>
      ) : null}
    </main>
  );
}
