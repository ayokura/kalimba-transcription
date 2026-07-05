"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useRouter } from "next/navigation";

import { BeatGridScore } from "@/components/BeatGridScore";
import { DoReMiScore } from "@/components/DoReMiScore";
import { ScoreExportButtons } from "@/components/ScoreExportButtons";
import { TxIdBadge } from "@/components/TxIdBadge";
import {
  boostDbForPeak,
  closeAudioBoost,
  ensureAudioBoost,
  type AudioBoostChain,
} from "@/lib/audioBoost";
import { computeAudioLevels } from "@/lib/audio";
import {
  createTranscriptionWithCapture,
  fetchMemo,
  fetchTranscription,
  fetchTranscriptionAudioBlob,
  fetchTunings,
  fetchCorrections,
  saveMemo,
} from "@/lib/api";
import { findEventById, findEventIdAtSec } from "@/lib/eventTiming";
import { pushRecentTranscription } from "@/lib/recentTranscriptions";
import {
  isMovableNumberApplicable,
  movableDoLabelFn,
  movableNumberLabelFn,
  noteLabelFromScoreNote,
  tonicReferenceOctave,
} from "@/lib/scoreLayout";
import { restoreStateFromCorrections, toDisplayScoreEvents } from "@/lib/reviewCorrections";
import { InstrumentTuning, ScoreEvent, TranscriptionResult, TuningMismatch } from "@/lib/types";

type LabelMode = "fixed" | "movable" | "movableNumber";
const LABEL_MODE_STORAGE_KEY = "kalimba:score-label-mode";

function isLabelMode(value: string | null): value is LabelMode {
  return value === "fixed" || value === "movable" || value === "movableNumber";
}

type LoadState =
  | { kind: "loading" }
  | {
      kind: "ready";
      result: TranscriptionResult;
      audioUrl: string;
      initialMemo: string;
      correctedEvents: ScoreEvent[] | null;
      peakDb: number | null;
    }
  | { kind: "error"; message: string };

const MEMO_SAVE_DEBOUNCE_MS = 800;

export function ScoreViewer({ transactionId }: { transactionId: string }) {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    async function load() {
      try {
        const [result, audioBlob, memo, corrections] = await Promise.all([
          fetchTranscription(transactionId),
          fetchTranscriptionAudioBlob(transactionId),
          fetchMemo(transactionId).catch(() => ""),
          fetchCorrections(transactionId).catch(() => null),
        ]);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(audioBlob);
        // 静音録音の試聴ブースト用 (lib/audioBoost)。失敗しても再生には影響しない
        const levels = await computeAudioLevels(audioBlob).catch(() => null);
        if (cancelled) return;
        // #202: corrections が保存済みなら編集後イベント列を導出して既定表示に
        // する (表記トグル / export は表示中の列に対して機能する)。
        let correctedEvents: ScoreEvent[] | null = null;
        if (corrections && corrections.events.length > 0) {
          try {
            correctedEvents = toDisplayScoreEvents(
              restoreStateFromCorrections(result, corrections),
            );
          } catch {
            correctedEvents = null;
          }
        }
        setState({
          kind: "ready",
          result,
          audioUrl: objectUrl,
          initialMemo: memo,
          correctedEvents,
          peakDb: levels?.peakDb ?? null,
        });
      } catch (err) {
        if (cancelled) return;
        setState({
          kind: "error",
          message: err instanceof Error ? err.message : "読み込みに失敗しました。",
        });
      }
    }

    load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [transactionId]);

  if (state.kind === "loading") {
    return (
      <main className="score-viewer-shell">
        <p className="muted">読み込み中…</p>
      </main>
    );
  }

  if (state.kind === "error") {
    return (
      <main className="score-viewer-shell">
        <p className="empty">読み込めませんでした: {state.message}</p>
      </main>
    );
  }

  return (
    <ScoreViewerReady
      transactionId={transactionId}
      result={state.result}
      audioUrl={state.audioUrl}
      initialMemo={state.initialMemo}
      correctedEvents={state.correctedEvents}
      peakDb={state.peakDb}
    />
  );
}

type ReadyProps = {
  transactionId: string;
  result: TranscriptionResult;
  audioUrl: string;
  initialMemo: string;
  correctedEvents: ScoreEvent[] | null;
  peakDb: number | null;
};

function ScoreViewerReady({ transactionId, result, audioUrl, initialMemo, correctedEvents, peakDb }: ReadyProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeEventId, setActiveEventId] = useState<string | null>(null);

  // 静音録音のブースト + 片チャンネル無音ステレオの両耳化 (lib/audioBoost)
  const boostChainRef = useRef<AudioBoostChain | null>(null);
  const boostDb = boostDbForPeak(peakDb);
  const ensureBoost = useCallback(() => {
    ensureAudioBoost(audioRef.current, boostChainRef, boostDb);
  }, [boostDb]);
  useEffect(() => {
    return () => closeAudioBoost(boostChainRef);
  }, []);

  const tonic = result.instrumentTuning.tonic ?? null;
  const movableAvailable = Boolean(tonic);
  const tonicRefOctave = useMemo(
    () => tonicReferenceOctave(result.instrumentTuning, tonic),
    [result.instrumentTuning, tonic],
  );

  // #202: 修正済み版があれば既定でそれを表示。認識結果へは切替可能。
  const [viewSource, setViewSource] = useState<"corrected" | "recognized">(
    correctedEvents ? "corrected" : "recognized",
  );
  const events = viewSource === "corrected" && correctedEvents ? correctedEvents : result.events;

  const allNotes = useMemo(
    () => events.flatMap((e) => e.notes),
    [events],
  );
  const movableNumberAvailable = useMemo(
    () => isMovableNumberApplicable(allNotes, tonic),
    [allNotes, tonic],
  );

  const [labelMode, setLabelMode] = useState<LabelMode>("fixed");
  const scoreAreaRef = useRef<HTMLElement | null>(null);
  // #202 案 A' プロトタイプの dev フラグ (?notation=beatgrid)。判定材料用で
  // UI には切替を出さない。
  const [useBeatGrid, setUseBeatGrid] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    setUseBeatGrid(new URLSearchParams(window.location.search).get("notation") === "beatgrid");
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(LABEL_MODE_STORAGE_KEY);
    if (!isLabelMode(stored)) return;
    if (stored === "movable" && movableAvailable) setLabelMode("movable");
    else if (stored === "movableNumber" && movableNumberAvailable) setLabelMode("movableNumber");
  }, [movableAvailable, movableNumberAvailable]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LABEL_MODE_STORAGE_KEY, labelMode);
  }, [labelMode]);

  const labelFn = useMemo(() => {
    if (labelMode === "movable" && tonic) return movableDoLabelFn(tonic, tonicRefOctave);
    if (labelMode === "movableNumber" && tonic) return movableNumberLabelFn(tonic, tonicRefOctave);
    return noteLabelFromScoreNote;
  }, [labelMode, tonic, tonicRefOctave]);

  const shareUrl = useMemo(() => {
    if (typeof window === "undefined") return "";
    return window.location.href;
  }, []);

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const next = findEventIdAtSec(events, audio.currentTime);
    if (next !== activeEventId) {
      setActiveEventId(next);
    }
  }, [events, activeEventId]);

  const handleScoreEventTap = useCallback(
    (eventId: string) => {
      const audio = audioRef.current;
      const event = findEventById(events, eventId);
      if (!audio || !event) return;
      audio.currentTime = event.startTimeSec;
      setActiveEventId(eventId);
    },
    [events],
  );

  return (
    <main className="score-viewer-shell">
      <header className="score-viewer-header">
        <div className="score-viewer-header-row">
          <Link href="/" className="score-viewer-home-link">
            ← トップへ
          </Link>
          <h1 className="score-viewer-title">カリンバ譜面</h1>
          <TxIdBadge id={transactionId} />
        </div>
        <ShareUrlRow url={shareUrl} />
      </header>

      {result.tuningMismatch ? (
        <TuningMismatchBanner
          transactionId={transactionId}
          mismatch={result.tuningMismatch}
          currentTuningName={result.instrumentTuning.name}
        />
      ) : null}

      {result.warnings.length > 0 ? (
        <div className="warning-box">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}

      <MemoEditor transactionId={transactionId} initialMemo={initialMemo} />

      <section className="score-viewer-playback">
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          onTimeUpdate={handleTimeUpdate}
          onPlay={ensureBoost}
          className="score-viewer-audio"
        />
        {boostDb > 0 ? (
          <p className="muted score-viewer-boost-note">
            試聴 +{boostDb.toFixed(0)}dB ブースト中 (元 peak {peakDb?.toFixed(1)}dB の静音録音)
          </p>
        ) : null}
      </section>

      <section className="score-viewer-score" ref={scoreAreaRef}>
        {correctedEvents ? (
          <div className="score-viewer-source-row">
            <span className={`score-source-badge${viewSource === "corrected" ? " corrected" : ""}`}>
              {viewSource === "corrected" ? "修正済み版を表示中" : "認識結果 (未修正) を表示中"}
            </span>
            <button
              type="button"
              className="score-export-btn"
              onClick={() => setViewSource((v) => (v === "corrected" ? "recognized" : "corrected"))}
            >
              {viewSource === "corrected" ? "認識結果を表示" : "修正済み版を表示"}
            </button>
          </div>
        ) : null}
        <div className="score-viewer-mode-toggle" role="group" aria-label="ドレミ表記">
          <button
            type="button"
            className={`score-viewer-mode-btn${labelMode === "fixed" ? " active" : ""}`}
            onClick={() => setLabelMode("fixed")}
          >
            固定ド
          </button>
          <button
            type="button"
            className={`score-viewer-mode-btn${labelMode === "movable" ? " active" : ""}`}
            onClick={() => movableAvailable && setLabelMode("movable")}
            disabled={!movableAvailable}
            title={movableAvailable ? undefined : "この調律には tonic が設定されていません"}
          >
            移動ド{tonic ? ` (${tonic})` : ""}
          </button>
          <button
            type="button"
            className={`score-viewer-mode-btn${labelMode === "movableNumber" ? " active" : ""}`}
            onClick={() => movableNumberAvailable && setLabelMode("movableNumber")}
            disabled={!movableNumberAvailable}
            title={
              movableNumberAvailable
                ? undefined
                : tonic
                ? "スケール外の音が含まれているため使用できません"
                : "この調律には tonic が設定されていません"
            }
          >
            数字{tonic ? ` (${tonic}=1)` : ""}
          </button>
        </div>
        {useBeatGrid ? (
          <BeatGridScore
            events={events}
            activeEventId={activeEventId}
            onActiveEventIdChange={handleScoreEventTap}
            labelFn={labelFn}
          />
        ) : (
          <DoReMiScore
            events={events}
            activeEventId={activeEventId}
            onActiveEventIdChange={handleScoreEventTap}
            labelFn={labelFn}
          />
        )}
        <ScoreExportButtons
          scoreAreaRef={scoreAreaRef}
          fileBaseName={`kalimba-score-${transactionId.slice(0, 8)}`}
        />
      </section>

      <section className="score-viewer-review-link-row">
        <Link href={`/score/${transactionId}/review`} className="score-viewer-review-link">
          結果を確認・修正する →
        </Link>
      </section>

      <RetranscribeSection
        transactionId={transactionId}
        currentTuningId={result.instrumentTuning.id}
      />

      <footer className="score-viewer-footer">
        <p className="muted">
          {result.instrumentTuning.name} · Tempo {result.tempo.toFixed(1)} BPM · {events.length} events
        </p>
      </footer>
    </main>
  );
}

function useRetranscribe(transactionId: string) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const retranscribe = useCallback(
    async (tuning: InstrumentTuning) => {
      setBusy(true);
      setError(null);
      try {
        const audioBlob = await fetchTranscriptionAudioBlob(transactionId);
        const capture = await createTranscriptionWithCapture(audioBlob, tuning, {
          force: true,
        });
        const newId = capture.responsePayload.transactionId;
        if (!newId) throw new Error("新しい transactionId が返されませんでした。");
        pushRecentTranscription({
          transactionId: newId,
          createdAt: new Date().toISOString(),
          tuningName: tuning.name,
          eventCount: capture.responsePayload.events.length,
        });
        router.push(`/score/${newId}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : "再採譜に失敗しました。");
        setBusy(false);
      }
    },
    [transactionId, router],
  );

  return { busy, error, retranscribe };
}

function TuningMismatchBanner({
  transactionId,
  mismatch,
  currentTuningName,
}: {
  transactionId: string;
  mismatch: TuningMismatch;
  currentTuningName: string;
}) {
  const { busy, error, retranscribe } = useRetranscribe(transactionId);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const outside = mismatch.outsidePitchClasses.join(", ");

  async function handleSuggested() {
    if (!mismatch.suggestedTuningId) return;
    setFetchError(null);
    try {
      const tunings = await fetchTunings();
      const suggested = tunings.find((t) => t.id === mismatch.suggestedTuningId);
      if (!suggested) throw new Error("提案された調律が見つかりませんでした。");
      await retranscribe(suggested);
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : "再採譜に失敗しました。");
    }
  }

  return (
    <section className="score-viewer-mismatch" role="alert">
      <p className="score-viewer-mismatch-text">
        この録音には選択した調律 ({currentTuningName}) にない音
        {outside ? ` (${outside})` : ""} が強く含まれています。
        {mismatch.suggestedTuningName
          ? ` ${mismatch.suggestedTuningName} の演奏かもしれません。`
          : " 調律の選択が合っているか確認してください。"}
      </p>
      {mismatch.suggestedTuningId ? (
        <button
          type="button"
          className="score-viewer-mismatch-btn"
          onClick={handleSuggested}
          disabled={busy}
        >
          {busy ? "再採譜中…" : `${mismatch.suggestedTuningName} で再採譜`}
        </button>
      ) : null}
      {error || fetchError ? (
        <p className="score-viewer-retranscribe-error">{error ?? fetchError}</p>
      ) : null}
    </section>
  );
}

function RetranscribeSection({
  transactionId,
  currentTuningId,
}: {
  transactionId: string;
  currentTuningId: string;
}) {
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [selectedTuningId, setSelectedTuningId] = useState<string>(currentTuningId);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const { busy, error, retranscribe } = useRetranscribe(transactionId);

  useEffect(() => {
    fetchTunings()
      .then((list) => setTunings(list))
      .catch(() => setFetchError("調律一覧の取得に失敗しました。"));
  }, []);

  const selectedTuning = tunings.find((t) => t.id === selectedTuningId) ?? null;

  async function handleRetranscribe() {
    if (!selectedTuning) return;
    await retranscribe(selectedTuning);
  }

  return (
    <section className="score-viewer-retranscribe">
      <p className="score-viewer-retranscribe-label">この録音を別の条件で再採譜</p>
      <div className="score-viewer-retranscribe-row">
        <select
          className="score-viewer-retranscribe-select"
          value={selectedTuningId}
          onChange={(e) => setSelectedTuningId(e.target.value)}
          disabled={busy || tunings.length === 0}
        >
          {tunings.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="score-viewer-retranscribe-btn"
          onClick={handleRetranscribe}
          disabled={busy || !selectedTuning}
        >
          {busy ? "再採譜中…" : "再採譜"}
        </button>
      </div>
      {error || fetchError ? (
        <p className="score-viewer-retranscribe-error">{error ?? fetchError}</p>
      ) : null}
    </section>
  );
}

function ShareUrlRow({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // ignore
    }
  };

  return (
    <div className="score-viewer-share-row">
      <input
        className="score-viewer-url"
        type="text"
        value={url}
        readOnly
        onFocus={(e) => e.currentTarget.select()}
      />
      <button type="button" className="score-viewer-copy-btn" onClick={handleCopy}>
        {copied ? "コピーしました" : "URL をコピー"}
      </button>
    </div>
  );
}

function MemoEditor({
  transactionId,
  initialMemo,
}: {
  transactionId: string;
  initialMemo: string;
}) {
  const [memo, setMemo] = useState(initialMemo);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savedRef = useRef(initialMemo);

  useEffect(() => {
    if (memo === savedRef.current) {
      setSaveState("idle");
      return;
    }
    setSaveState("saving");
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        await saveMemo(transactionId, memo);
        savedRef.current = memo;
        setSaveState("saved");
      } catch {
        setSaveState("error");
      }
    }, MEMO_SAVE_DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [memo, transactionId]);

  return (
    <section className="score-viewer-memo">
      <label className="score-viewer-memo-label" htmlFor="score-viewer-memo-input">
        メモ
      </label>
      <textarea
        id="score-viewer-memo-input"
        className="score-viewer-memo-input"
        value={memo}
        onChange={(e) => setMemo(e.target.value)}
        placeholder="演奏の気づきやノートをここに…"
        rows={2}
      />
      <p className="score-viewer-memo-status muted">
        {saveState === "saving" && "保存中…"}
        {saveState === "saved" && "保存しました"}
        {saveState === "error" && "保存できませんでした"}
        {saveState === "idle" && "\u00a0"}
      </p>
    </section>
  );
}
