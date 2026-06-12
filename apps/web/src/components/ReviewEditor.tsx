"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DoReMiScore } from "@/components/DoReMiScore";
import {
  fetchCorrections,
  fetchTranscription,
  fetchTranscriptionAudioBlob,
  saveCorrections,
} from "@/lib/api";
import {
  activeEvents,
  hasActiveEventAt,
  addNote,
  buildInitialState,
  buildKnownNoteIndex,
  insertEvent,
  noteName,
  removeNote,
  resolveScoreNote,
  restoreStateFromCorrections,
  toCorrectionsPayload,
  toDisplayScoreEvents,
  toggleRemoved,
  type ReviewEvent,
  type ReviewState,
} from "@/lib/reviewCorrections";
import {
  CandidateSlot,
  CorrectionsPayload,
  ReviewOrigin,
  ScoreNote,
  TranscriptionResult,
} from "@/lib/types";

const AUDITION_LEAD_SEC = 0.15;
const AUDITION_MAX_SEC = 4.0;

const ORIGIN_LABELS: Record<ReviewOrigin, string> = {
  recognizer: "認識",
  edited: "修正済",
  "inserted-slot": "候補から追加",
  "inserted-manual": "手動追加",
};

const DROP_REASON_LABELS: Record<string, string> = {
  "residual-decay-no-reattack": "残響の可能性",
  "orphan-onset-no-segment": "onset のみ検出",
  low_register_sparse_gap_tail: "低域の弱い尾部",
};

type LoadState =
  | { kind: "loading" }
  | {
      kind: "ready";
      result: TranscriptionResult;
      audioUrl: string;
      corrections: CorrectionsPayload | null;
    }
  | { kind: "error"; message: string };

export function ReviewEditor({ transactionId }: { transactionId: string }) {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    async function load() {
      try {
        // fetchCorrections の失敗を握りつぶさない: 保存済み修正が見えないまま
        // baseline で開くと、次の保存で既存修正を上書きしてしまう。
        // 失敗時はページ全体をエラー表示にする (404 は api.ts 側で null になる)。
        const [result, audioBlob, corrections] = await Promise.all([
          fetchTranscription(transactionId),
          fetchTranscriptionAudioBlob(transactionId),
          fetchCorrections(transactionId),
        ]);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(audioBlob);
        setState({ kind: "ready", result, audioUrl: objectUrl, corrections });
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
      <main className="review-shell">
        <p className="muted">読み込み中…</p>
      </main>
    );
  }

  if (state.kind === "error") {
    return (
      <main className="review-shell">
        <p className="empty">読み込めませんでした: {state.message}</p>
      </main>
    );
  }

  return (
    // key で transaction ごとにコンポーネントを作り直す。client-side 遷移で
    // 前の transaction の編集 state が持ち越され、別 transaction に保存される事故を防ぐ
    <ReviewEditorReady
      key={transactionId}
      transactionId={transactionId}
      result={state.result}
      audioUrl={state.audioUrl}
      initialCorrections={state.corrections}
    />
  );
}

type ReadyProps = {
  transactionId: string;
  result: TranscriptionResult;
  audioUrl: string;
  initialCorrections: CorrectionsPayload | null;
};

type TimelineItem =
  | { kind: "event"; timeSec: number; event: ReviewEvent }
  | { kind: "slot"; timeSec: number; slot: CandidateSlot; slotIndex: number };

function ReviewEditorReady({ transactionId, result, audioUrl, initialCorrections }: ReadyProps) {
  const [reviewState, setReviewState] = useState<ReviewState>(() =>
    initialCorrections
      ? restoreStateFromCorrections(result, initialCorrections)
      : buildInitialState(result),
  );
  const [history, setHistory] = useState<ReviewState[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const lastSavedRef = useRef<string>(
    JSON.stringify(
      initialCorrections
        ? toCorrectionsPayload(restoreStateFromCorrections(result, initialCorrections))
        : null,
    ),
  );

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const programmaticSeekRef = useRef(false);

  const knownNotes = useMemo(() => buildKnownNoteIndex(result), [result]);
  const sourceEventById = useMemo(
    () => new Map(result.events.map((event) => [event.id, event])),
    [result.events],
  );

  const apply = useCallback(
    (next: (state: ReviewState) => ReviewState) => {
      setReviewState((current) => {
        const updated = next(current);
        if (updated === current) return current;
        setHistory((stack) => [...stack, current]);
        return updated;
      });
    },
    [],
  );

  const undo = useCallback(() => {
    setHistory((stack) => {
      if (stack.length === 0) return stack;
      const previous = stack[stack.length - 1];
      setReviewState(previous);
      return stack.slice(0, -1);
    });
  }, []);

  const resetAll = useCallback(() => {
    apply(() => buildInitialState(result));
  }, [apply, result]);

  const payload = useMemo(() => toCorrectionsPayload(reviewState), [reviewState]);
  const payloadJson = useMemo(() => JSON.stringify(payload), [payload]);
  const initialJson = useMemo(
    () => JSON.stringify(toCorrectionsPayload(buildInitialState(result))),
    [result],
  );
  const dirty = payloadJson !== lastSavedRef.current && !(lastSavedRef.current === "null" && payloadJson === initialJson);

  useEffect(() => {
    if (!dirty) return;
    const handler = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      // returnValue をセットしないと確認ダイアログを出さないブラウザがある
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);

  const handleSave = useCallback(async () => {
    setSaveState("saving");
    try {
      await saveCorrections(transactionId, payload);
      lastSavedRef.current = payloadJson;
      setSaveState("saved");
    } catch {
      setSaveState("error");
    }
  }, [transactionId, payload, payloadJson]);

  const visibleEvents = useMemo(() => activeEvents(reviewState), [reviewState]);

  const auditionEvent = useCallback(
    (event: ReviewEvent) => {
      const audio = audioRef.current;
      if (!audio) return;
      const followers = visibleEvents.filter((e) => e.timeSec > event.timeSec + 0.01);
      const nextStart = followers.length > 0 ? followers[0].timeSec : event.timeSec + AUDITION_MAX_SEC;
      stopAtRef.current = Math.min(nextStart, event.timeSec + AUDITION_MAX_SEC);
      // この currentTime 代入も onSeeking を発火させるため、programmatic seek を
      // マークして stopAt の解除対象から除外する (ユーザー操作の seek のみ解除)
      programmaticSeekRef.current = true;
      audio.currentTime = Math.max(0, event.timeSec - AUDITION_LEAD_SEC);
      void audio.play();
    },
    [visibleEvents],
  );

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (stopAtRef.current !== null && audio.currentTime >= stopAtRef.current) {
      audio.pause();
      stopAtRef.current = null;
    }
  }, []);

  const handleSelect = useCallback(
    (eventId: string) => {
      setSelectedId(eventId);
      const event = reviewState.events.find((e) => e.id === eventId);
      if (event && !event.removed) auditionEvent(event);
    },
    [reviewState.events, auditionEvent],
  );

  const handleInsertSlot = useCallback(
    (slot: CandidateSlot) => {
      apply((state) => insertEvent(state, slot.startTime, [slot.primaryNote], "inserted-slot"));
    },
    [apply],
  );

  const handleInsertAtPlayhead = useCallback(
    (note: ScoreNote) => {
      const audio = audioRef.current;
      const timeSec = audio ? audio.currentTime : 0;
      apply((state) => insertEvent(state, timeSec, [note], "inserted-manual"));
    },
    [apply],
  );

  const timeline = useMemo<TimelineItem[]>(() => {
    const items: TimelineItem[] = reviewState.events.map((event) => ({
      kind: "event" as const,
      timeSec: event.timeSec,
      event,
    }));
    (result.candidateSlots ?? []).forEach((slot, slotIndex) => {
      // 同時刻にアクティブなイベントがある slot は表示しない。挿入記録ではなく
      // state で判定することで、保存済み corrections から復元された
      // inserted-slot イベントとの重複表示 (=重複挿入) を防ぐ
      if (hasActiveEventAt(reviewState, slot.startTime)) return;
      items.push({ kind: "slot", timeSec: slot.startTime, slot, slotIndex });
    });
    return items.sort((a, b) => a.timeSec - b.timeSec);
  }, [reviewState, result.candidateSlots]);

  const displayEvents = useMemo(() => toDisplayScoreEvents(reviewState), [reviewState]);

  const pickerNotes = useMemo(() => {
    return result.instrumentTuning.notes
      .map((tuningNote) =>
        resolveScoreNote(tuningNote.noteName, knownNotes, result.instrumentTuning),
      )
      .filter((note): note is ScoreNote => note !== null)
      .sort((a, b) => a.frequency - b.frequency);
  }, [result.instrumentTuning, knownNotes]);

  return (
    <main className="review-shell">
      <header className="review-header">
        <div className="review-header-row">
          <Link
            href={`/score/${transactionId}`}
            className="review-back-link"
            onClick={(e) => {
              // beforeunload はアプリ内 route 遷移には効かないため、ここでも守る
              if (dirty && !window.confirm("未保存の修正があります。保存せずに移動しますか?")) {
                e.preventDefault();
              }
            }}
          >
            ← 譜面へ戻る
          </Link>
          <h1 className="review-title">確認と修正</h1>
        </div>
        <p className="review-subtitle muted">
          イベントを選ぶとその部分を再生します。音の過不足はカードから直せます。
        </p>
      </header>

      <section className="review-playback">
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          onTimeUpdate={handleTimeUpdate}
          onSeeking={() => {
            // auditionEvent 由来の programmatic seek では stopAt を維持し、
            // ユーザーが自分でシークした時だけ区間再生を解除する
            if (programmaticSeekRef.current) {
              programmaticSeekRef.current = false;
              return;
            }
            stopAtRef.current = null;
          }}
          className="review-audio"
        />
        <InsertAtPlayheadControl notes={pickerNotes} onInsert={handleInsertAtPlayhead} />
      </section>

      <section className="review-score">
        <DoReMiScore
          events={displayEvents}
          activeEventId={selectedId}
          onActiveEventIdChange={handleSelect}
        />
      </section>

      <section className="review-timeline" aria-label="イベント一覧">
        {timeline.map((item) =>
          item.kind === "event" ? (
            <EventCard
              key={item.event.id}
              event={item.event}
              selected={selectedId === item.event.id}
              suggestions={
                item.event.sourceEventId
                  ? sourceEventById.get(item.event.sourceEventId)?.alternateGroupings ?? null
                  : null
              }
              pickerNotes={pickerNotes}
              onSelect={() => handleSelect(item.event.id)}
              onAudition={() => auditionEvent(item.event)}
              onRemoveNote={(name) => apply((s) => removeNote(s, item.event.id, name))}
              onAddNote={(note) => apply((s) => addNote(s, item.event.id, note))}
              onToggleRemoved={() => apply((s) => toggleRemoved(s, item.event.id))}
            />
          ) : (
            <SlotCard
              key={`slot-${item.slotIndex}`}
              slot={item.slot}
              onInsert={() => handleInsertSlot(item.slot)}
            />
          ),
        )}
      </section>

      <footer className="review-footer">
        <div className="review-footer-actions">
          <button
            type="button"
            className="review-btn"
            onClick={undo}
            disabled={history.length === 0}
          >
            元に戻す
          </button>
          <button type="button" className="review-btn" onClick={resetAll}>
            認識結果にリセット
          </button>
          <button
            type="button"
            className="review-btn review-btn-primary"
            onClick={handleSave}
            disabled={saveState === "saving" || !dirty}
          >
            {saveState === "saving" ? "保存中…" : "修正を保存"}
          </button>
        </div>
        <p className="review-save-status muted" role="status">
          {saveState === "error"
            ? "保存できませんでした — 未保存の修正があります"
            : saveState === "saving"
            ? "保存中…"
            : dirty
            ? "未保存の修正があります"
            : saveState === "saved"
            ? "保存しました"
            : "\u00a0"}
        </p>
      </footer>
    </main>
  );
}

function formatTime(sec: number): string {
  return `${sec.toFixed(2)}s`;
}

function EventCard({
  event,
  selected,
  suggestions,
  pickerNotes,
  onSelect,
  onAudition,
  onRemoveNote,
  onAddNote,
  onToggleRemoved,
}: {
  event: ReviewEvent;
  selected: boolean;
  suggestions: TranscriptionResult["events"][number]["alternateGroupings"];
  pickerNotes: ScoreNote[];
  onSelect: () => void;
  onAudition: () => void;
  onRemoveNote: (name: string) => void;
  onAddNote: (note: ScoreNote) => void;
  onToggleRemoved: () => void;
}) {
  const noteSuggestions = (suggestions ?? [])
    .filter((alt) => alt.alternateNote !== null)
    .filter(
      (alt) => !event.notes.some((n) => noteName(n) === noteName(alt.alternateNote as ScoreNote)),
    );

  const addableNotes = pickerNotes.filter(
    (note) => !event.notes.some((existing) => noteName(existing) === noteName(note)),
  );

  const classNames = [
    "review-card",
    `review-card-${event.origin}`,
    selected ? "selected" : "",
    event.removed ? "removed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article className={classNames}>
      <button type="button" className="review-card-head" onClick={onSelect}>
        <span className="review-card-time">{formatTime(event.timeSec)}</span>
        <span className="review-card-notes">
          {event.notes.map((note) => (
            <span key={noteName(note)} className="review-note-chip-static">
              {note.labelDoReMi}
              <small>{noteName(note)}</small>
            </span>
          ))}
        </span>
        <span className={`review-origin-badge origin-${event.origin}`}>
          {event.removed ? "削除済" : ORIGIN_LABELS[event.origin]}
        </span>
      </button>

      {selected && !event.removed ? (
        <div className="review-card-editor">
          <div className="review-chip-row">
            {event.notes.map((note) => (
              <span key={noteName(note)} className="review-note-chip">
                {note.labelDoReMi} <small>{noteName(note)}</small>
                {event.notes.length > 1 ? (
                  <button
                    type="button"
                    className="review-chip-x"
                    aria-label={`${noteName(note)} を外す`}
                    onClick={() => onRemoveNote(noteName(note))}
                  >
                    ×
                  </button>
                ) : null}
              </span>
            ))}
            {addableNotes.length > 0 ? (
              <select
                className="review-note-add"
                value=""
                aria-label="音を追加"
                onChange={(e) => {
                  const note = addableNotes.find((n) => noteName(n) === e.target.value);
                  if (note) onAddNote(note);
                }}
              >
                <option value="">＋ 音を追加</option>
                {addableNotes.map((note) => (
                  <option key={noteName(note)} value={noteName(note)}>
                    {note.labelDoReMi} ({noteName(note)})
                  </option>
                ))}
              </select>
            ) : null}
          </div>

          {noteSuggestions.length > 0 ? (
            <div className="review-suggestions">
              <span className="review-suggestions-label">認識器の次候補:</span>
              {noteSuggestions.map((alt) => {
                const note = alt.alternateNote as ScoreNote;
                return (
                  <button
                    key={noteName(note)}
                    type="button"
                    className="review-suggestion-chip"
                    onClick={() => onAddNote(note)}
                  >
                    ＋{note.labelDoReMi} <small>{Math.round(alt.confidence * 100)}%</small>
                  </button>
                );
              })}
            </div>
          ) : null}

          <div className="review-card-actions">
            <button type="button" className="review-btn review-btn-small" onClick={onAudition}>
              ▶ この部分を再生
            </button>
            <button
              type="button"
              className="review-btn review-btn-small review-btn-danger"
              onClick={onToggleRemoved}
            >
              このイベントを削除
            </button>
          </div>
        </div>
      ) : null}

      {selected && event.removed ? (
        <div className="review-card-editor">
          <div className="review-card-actions">
            <button type="button" className="review-btn review-btn-small" onClick={onToggleRemoved}>
              復元する
            </button>
          </div>
        </div>
      ) : null}
    </article>
  );
}

function SlotCard({ slot, onInsert }: { slot: CandidateSlot; onInsert: () => void }) {
  const reasonLabel = DROP_REASON_LABELS[slot.dropReason] ?? slot.dropReason;
  return (
    <article className="review-card review-card-slot">
      <div className="review-card-head review-card-head-static">
        <span className="review-card-time">{formatTime(slot.startTime)}</span>
        <span className="review-card-notes">
          <span className="review-note-chip-static ghost">
            {slot.primaryNote.labelDoReMi}
            <small>{noteName(slot.primaryNote)}?</small>
          </span>
        </span>
        <span className="review-slot-meta">
          {reasonLabel} · {Math.round(slot.confidence * 100)}%
        </span>
        <button type="button" className="review-btn review-btn-small" onClick={onInsert}>
          ＋ 追加
        </button>
      </div>
    </article>
  );
}

function InsertAtPlayheadControl({
  notes,
  onInsert,
}: {
  notes: ScoreNote[];
  onInsert: (note: ScoreNote) => void;
}) {
  return (
    <div className="review-insert-control">
      <select
        className="review-note-add"
        value=""
        aria-label="再生位置に音を挿入"
        onChange={(e) => {
          const note = notes.find((n) => noteName(n) === e.target.value);
          if (note) onInsert(note);
        }}
      >
        <option value="">＋ 再生位置に音を挿入…</option>
        {notes.map((note) => (
          <option key={noteName(note)} value={noteName(note)}>
            {note.labelDoReMi} ({noteName(note)})
          </option>
        ))}
      </select>
    </div>
  );
}
