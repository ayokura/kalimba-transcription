"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DoReMiScore } from "@/components/DoReMiScore";
import { fetchMemo, fetchTranscription, fetchTranscriptionAudioBlob, saveMemo } from "@/lib/api";
import { findEventById, findEventIdAtSec } from "@/lib/eventTiming";
import { movableDoLabelFn, noteLabelFromScoreNote } from "@/lib/scoreLayout";
import { TranscriptionResult } from "@/lib/types";

type LabelMode = "fixed" | "movable";
const LABEL_MODE_STORAGE_KEY = "kalimba:score-label-mode";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; result: TranscriptionResult; audioUrl: string; initialMemo: string }
  | { kind: "error"; message: string };

const MEMO_SAVE_DEBOUNCE_MS = 800;

export function ScoreViewer({ transactionId }: { transactionId: string }) {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    async function load() {
      try {
        const [result, audioBlob, memo] = await Promise.all([
          fetchTranscription(transactionId),
          fetchTranscriptionAudioBlob(transactionId),
          fetchMemo(transactionId).catch(() => ""),
        ]);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(audioBlob);
        setState({ kind: "ready", result, audioUrl: objectUrl, initialMemo: memo });
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
    />
  );
}

type ReadyProps = {
  transactionId: string;
  result: TranscriptionResult;
  audioUrl: string;
  initialMemo: string;
};

function ScoreViewerReady({ transactionId, result, audioUrl, initialMemo }: ReadyProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeEventId, setActiveEventId] = useState<string | null>(null);

  const tonic = result.instrumentTuning.tonic ?? null;
  const movableAvailable = Boolean(tonic);
  const [labelMode, setLabelMode] = useState<LabelMode>("fixed");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(LABEL_MODE_STORAGE_KEY);
    if (stored === "movable" && movableAvailable) {
      setLabelMode("movable");
    }
  }, [movableAvailable]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LABEL_MODE_STORAGE_KEY, labelMode);
  }, [labelMode]);

  const labelFn = useMemo(
    () => (labelMode === "movable" && tonic ? movableDoLabelFn(tonic) : noteLabelFromScoreNote),
    [labelMode, tonic],
  );

  const events = result.events;
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
        <h1 className="score-viewer-title">カリンバ譜面</h1>
        <ShareUrlRow url={shareUrl} />
      </header>

      <MemoEditor transactionId={transactionId} initialMemo={initialMemo} />

      <section className="score-viewer-playback">
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          onTimeUpdate={handleTimeUpdate}
          className="score-viewer-audio"
        />
      </section>

      <section className="score-viewer-score">
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
        </div>
        <DoReMiScore
          events={events}
          activeEventId={activeEventId}
          onActiveEventIdChange={handleScoreEventTap}
          labelFn={labelFn}
        />
      </section>

      <footer className="score-viewer-footer">
        <p className="muted">
          {result.instrumentTuning.name} · Tempo {result.tempo.toFixed(1)} BPM · {events.length} events
        </p>
      </footer>
    </main>
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
