"use client";

// 開発用 試聴トリアージページ (第 2 期 S1、sprint-plan-2026-07b)。
// CLI では音声再生ができない摩擦を解消するための temporary な dev ページ:
// - data/transactions の unique 録音を試聴 + ワンタップ review-status 判定
// - 破壊メニュー票を見ながらのセルフ録音 → 即採譜 → 一覧へ反映
// main nav からはリンクしない。撤去条件: 手持ちバックログの消化が完了し、
// 敵対的録音の運用が落ち着いた時点 (または本番運用フェーズ移行時) に
// /api/dev/triage と一緒に削除する。

import { useCallback, useEffect, useMemo, useState } from "react";

import { RecorderPanel } from "@/components/RecorderPanel";
import { ADVERSARIAL_MENU, type AdversarialMenuItem } from "@/lib/adversarialMenu";
import {
  createTranscriptionWithCapture,
  fetchDevTriage,
  fetchTunings,
  saveReviewStatus,
  type DevTriageSummary,
  type ManualCaptureExpectedPerformance,
} from "@/lib/api";
import { REVIEW_STATUS_OPTIONS } from "@/lib/reviewStatus";
import { InstrumentTuning, ReviewStatusValue } from "@/lib/types";

function buildExpectedPerformance(
  item: AdversarialMenuItem,
  tuning: InstrumentTuning,
): ManualCaptureExpectedPerformance | null {
  if (!item.events) return null;
  const byName = new Map(tuning.notes.map((note) => [note.noteName, note]));
  const events = [];
  for (const [index, names] of item.events.entries()) {
    const keys = [];
    for (const name of names) {
      const note = byName.get(name);
      if (!note) return null; // 選択 tuning に無い音 → 期待列は添付しない
      keys.push({ key: note.key, noteName: note.noteName });
    }
    events.push({
      index: index + 1,
      keys,
      display: names.join(" + "),
      intent: item.intent === "unknown" ? null : item.intent,
    });
  }
  return {
    source: "adversarial-menu",
    version: 1,
    summary: item.title,
    defaultCaptureIntent: item.intent === "unknown" ? null : item.intent,
    events,
  };
}

export default function DebugTriagePage() {
  const [summary, setSummary] = useState<DevTriageSummary | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [menuId, setMenuId] = useState<string>(ADVERSARIAL_MENU[0].id);
  const [recording, setRecording] = useState<Blob | null>(null);
  const [busy, setBusy] = useState(false);
  const [captureNote, setCaptureNote] = useState<string | null>(null);
  const [lastCaptureEvents, setLastCaptureEvents] = useState<string[] | null>(null);
  const [savingTx, setSavingTx] = useState<string | null>(null);

  // 録音の聞き返し用 (提出前に自分の演奏を確認できる)
  const recordingUrl = useMemo(
    () => (recording ? URL.createObjectURL(recording) : null),
    [recording],
  );
  useEffect(() => {
    return () => {
      if (recordingUrl) URL.revokeObjectURL(recordingUrl);
    };
  }, [recordingUrl]);

  const loadSummary = useCallback(() => {
    fetchDevTriage()
      .then((data) => {
        setSummary(data);
        setLoadError(null);
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : "読み込み失敗"));
  }, []);

  useEffect(() => {
    loadSummary();
    fetchTunings()
      .then(setTunings)
      .catch(() => setTunings([]));
  }, [loadSummary]);

  const tuning = tunings[0] ?? null;
  const menuItem = useMemo(
    () => ADVERSARIAL_MENU.find((item) => item.id === menuId) ?? ADVERSARIAL_MENU[0],
    [menuId],
  );

  const handleAnalyze = async () => {
    if (!recording || !tuning) return;
    setBusy(true);
    setCaptureNote(null);
    try {
      const capture = await createTranscriptionWithCapture(recording, tuning, {
        scenario: `adversarial-${menuItem.id}`,
        expectedNote: menuItem.title,
        expectedPerformance: buildExpectedPerformance(menuItem, tuning),
        memo: `adversarial-menu: ${menuItem.id} — ${menuItem.expectedFailure}`,
        captureIntent: menuItem.intent,
        force: true,
      });
      const txId = capture.responsePayload.transactionId;
      const recognized = capture.responsePayload.events.map((event) =>
        event.notes.map((note) => `${note.pitchClass}${note.octave}`).join("+"),
      );
      setCaptureNote(
        `登録しました: ${txId} (${recognized.length} events)。一覧は triage スクリプト再実行後に更新されます。`,
      );
      setLastCaptureEvents(recognized);
      setRecording(null);
    } catch (err) {
      setCaptureNote(err instanceof Error ? err.message : "採譜に失敗しました");
    } finally {
      setBusy(false);
    }
  };

  const handleStatus = async (txId: string, value: ReviewStatusValue) => {
    setSavingTx(txId);
    try {
      await saveReviewStatus(txId, value);
      loadSummary();
    } finally {
      setSavingTx(null);
    }
  };

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Dev Triage (temporary)</p>
          <h1>試聴トリアージ + 敵対的録音</h1>
          <p className="hero-copy">
            手持ち録音の usable/unusable 判断と、認識器を意図的に壊す録音セッションのための開発用ページ。
            バックログ消化と敵対的録音の運用が落ち着いたら /api/dev/triage ごと撤去する。
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Adversarial Recording</p>
            <h2>破壊メニュー票 → その場で録音</h2>
          </div>
          <span className="muted">{tuning ? tuning.name : "調律を読込中…"}</span>
        </div>
        <div className="stack gap-lg">
          <div className="row wrap" style={{ gap: 8 }}>
            {ADVERSARIAL_MENU.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`review-btn review-btn-small${item.id === menuId ? " review-btn-primary" : ""}`}
                onClick={() => setMenuId(item.id)}
              >
                {item.title}
              </button>
            ))}
          </div>
          <div className="warning-box">
            <p>
              <strong>狙う機構</strong>: {menuItem.target}
            </p>
            <p>
              <strong>予想される失敗</strong>: {menuItem.expectedFailure}
            </p>
            <p>
              <strong>演奏指示</strong>: {menuItem.instructions}
            </p>
            {menuItem.events ? (
              <p className="muted">
                期待列 ({menuItem.events.length} events):{" "}
                {menuItem.events.map((names) => names.join("+")).join(" / ")} — 録音に
                expectedPerformance として添付され、GT 化は自動整列 + 差分確認だけで済む
              </p>
            ) : (
              <p className="muted">期待列なし (自由演奏 — 事後の耳確認で GT 化)</p>
            )}
          </div>
          <RecorderPanel
            disabled={busy}
            hasRecording={Boolean(recording)}
            onRecordingReady={(blob) => setRecording(blob)}
          />
          {recordingUrl ? (
            <div className="stack" style={{ gap: 4 }}>
              <span className="muted">録音の聞き返し (提出前確認):</span>
              <audio controls src={recordingUrl} style={{ width: "100%" }} />
            </div>
          ) : null}
          <div className="row wrap" style={{ gap: 8 }}>
            <button
              type="button"
              className="review-btn review-btn-primary"
              onClick={handleAnalyze}
              disabled={!recording || !tuning || busy}
            >
              {busy ? "採譜中…" : "採譜して登録"}
            </button>
            {captureNote ? <span className="muted">{captureNote}</span> : null}
          </div>
          {lastCaptureEvents ? (
            <div className="warning-box">
              <p>
                <strong>認識結果</strong> ({lastCaptureEvents.length} events):{" "}
                {lastCaptureEvents.join(" ")}
              </p>
              {menuItem.events ? (
                <p className="muted">
                  期待列 ({menuItem.events.length} events):{" "}
                  {menuItem.events.map((names) => names.join("+")).join(" ")} —
                  予想どおり壊れていれば成功 (非飽和 GT 候補)
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Backlog Triage</p>
            <h2>unique 録音の試聴と判定</h2>
          </div>
          <span className="muted">
            {summary
              ? `${summary.totals.uniqueRecordings} unique / GT ${summary.totals.withGt} / recognizer ${summary.recognizerFingerprint}`
              : "…"}
          </span>
        </div>
        {loadError ? (
          <div className="warning-box">
            <p>{loadError}</p>
            <p className="muted">
              先に `uv run python scripts/audio-analysis/transactions_triage.py` を実行してください。
            </p>
          </div>
        ) : null}
        <div className="stack gap-lg">
          {(summary?.recordings ?? []).map((rec) => {
            const status = rec.reviewStatuses[rec.primaryTx] ?? null;
            return (
              <article key={rec.sha16} className="review-card" style={{ padding: 12 }}>
                <div className="row wrap" style={{ gap: 10, alignItems: "baseline" }}>
                  <strong>score {rec.score}</strong>
                  <a href={`/score/${rec.primaryTx}/review`}>{rec.primaryTx.slice(0, 8)}</a>
                  <span className="muted">
                    {rec.durationSec}s @ {rec.sampleRate}Hz / peak {rec.peakDbfs ?? "-"} dBFS /{" "}
                    {rec.storedEvents ?? "-"} events
                    {rec.gtLayer ? ` / GT: ${rec.gtLayer}` : ""}
                    {rec.duplicateTxs.length > 0 ? ` / 重複 ${rec.duplicateTxs.length} 件` : ""}
                  </span>
                  {status ? <span className="pill">{status}</span> : <span className="pill">未判定</span>}
                </div>
                {rec.signals.length > 0 ? (
                  <p className="muted" style={{ margin: "4px 0" }}>
                    {rec.signals.join("; ")}
                  </p>
                ) : null}
                {rec.memo ? (
                  <p className="muted" style={{ margin: "4px 0" }}>
                    memo: {rec.memo}
                  </p>
                ) : null}
                {rec.recognizedEvents && rec.recognizedEvents.length > 0 ? (
                  <p className="muted" style={{ margin: "4px 0", fontFamily: "monospace", fontSize: "0.8rem" }}>
                    認識: {rec.recognizedEvents.join(" ")}
                    {rec.storedEvents != null && rec.storedEvents > rec.recognizedEvents.length
                      ? ` … (先頭 ${rec.recognizedEvents.length}/${rec.storedEvents})`
                      : ""}
                  </p>
                ) : null}
                {rec.expectedEvents && rec.expectedEvents.length > 0 ? (
                  <p className="muted" style={{ margin: "4px 0", fontFamily: "monospace", fontSize: "0.8rem" }}>
                    期待: {rec.expectedEvents.join(" ")}
                  </p>
                ) : null}
                <audio
                  controls
                  preload="none"
                  src={`/api/transcriptions/${rec.primaryTx}/audio`}
                  style={{ width: "100%", margin: "6px 0" }}
                />
                <div className="row wrap" style={{ gap: 6 }}>
                  {REVIEW_STATUS_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      className={`review-btn review-btn-small${status === option.value ? " review-btn-primary" : ""}`}
                      disabled={savingTx === rec.primaryTx}
                      onClick={() => handleStatus(rec.primaryTx, option.value)}
                      title={option.description}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </main>
  );
}
