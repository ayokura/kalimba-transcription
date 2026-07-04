"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DoReMiScore } from "@/components/DoReMiScore";
import { KalimbaNotePicker } from "@/components/KalimbaNotePicker";
import {
  fetchCorrections,
  fetchTranscription,
  fetchTranscriptionAudioBlob,
  fetchReviewStatus,
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
  replaceNote,
  resolveScoreNote,
  restoreStateFromCorrections,
  setEventTime,
  toCorrectionsPayload,
  toDisplayScoreEvents,
  toggleRemoved,
  type ReviewEvent,
  type ReviewState,
} from "@/lib/reviewCorrections";
import { needsReviewReasons, type NeedsReviewReason } from "@/lib/needsReview";
import {
  CandidateSlot,
  CorrectionsPayload,
  ReviewOrigin,
  ReviewStatusPayload,
  ScoreNote,
  TranscriptionResult,
} from "@/lib/types";
import { ReviewStatusPanel } from "@/components/ReviewStatusPanel";

const AUDITION_LEAD_SEC = 0.15;
const AUDITION_MAX_SEC = 4.0;

const ORIGIN_LABELS: Record<ReviewOrigin, string> = {
  recognizer: "認識",
  edited: "修正済",
  "inserted-slot": "候補から追加",
  "inserted-manual": "手動追加",
};

// pipeline.py の _DROP_REASON_BASE_CONFIDENCE と対で維持する (現 8 種)
const DROP_REASON_LABELS: Record<string, string> = {
  "sub-onset-mute-dip-reattack": "ミュート後の再打鍵",
  "orphan-onset-no-segment": "onset のみ検出",
  "boundary-consumed-onset": "境界に消えた onset",
  "sub-onset-unselected-candidate": "連打/gliss 内の未選択音",
  "residual-decay-no-reattack": "残響の可能性",
  low_register_sparse_gap_tail: "低域の弱い尾部",
  "primary-score-too-low": "スコア不足で棄却",
  "onset-gate-no-evidence": "attack 証拠なし",
};

type LoadState =
  | { kind: "loading" }
  | {
      kind: "ready";
      result: TranscriptionResult;
      audioUrl: string;
      corrections: CorrectionsPayload | null;
      reviewStatus: ReviewStatusPayload | null;
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
        const [result, audioBlob, corrections, reviewStatus] = await Promise.all([
          fetchTranscription(transactionId),
          fetchTranscriptionAudioBlob(transactionId),
          fetchCorrections(transactionId),
          fetchReviewStatus(transactionId).catch(() => null),
        ]);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(audioBlob);
        setState({ kind: "ready", result, audioUrl: objectUrl, corrections, reviewStatus });
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
      initialReviewStatus={state.reviewStatus}
    />
  );
}

type ReadyProps = {
  transactionId: string;
  result: TranscriptionResult;
  audioUrl: string;
  initialCorrections: CorrectionsPayload | null;
  initialReviewStatus: ReviewStatusPayload | null;
};

type TimelineItem =
  | { kind: "event"; timeSec: number; event: ReviewEvent }
  | { kind: "slot"; timeSec: number; slot: CandidateSlot; slotIndex: number };

// undo/redo を 1 つの純粋 state で管理する (updater 内での別 setState への
// 副作用を避け、StrictMode の updater 二重呼び出しでも壊れない)
type EditHistory = {
  past: ReviewState[];
  present: ReviewState;
  future: ReviewState[];
};

function ReviewEditorReady({
  transactionId,
  result,
  audioUrl,
  initialCorrections,
  initialReviewStatus,
}: ReadyProps) {
  const [editHistory, setEditHistory] = useState<EditHistory>(() => ({
    past: [],
    present: initialCorrections
      ? restoreStateFromCorrections(result, initialCorrections)
      : buildInitialState(result),
    future: [],
  }));
  const reviewState = editHistory.present;
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

  const apply = useCallback((next: (state: ReviewState) => ReviewState) => {
    setEditHistory((current) => {
      const updated = next(current.present);
      if (updated === current.present) return current;
      // 新しい編集で redo 先は無効になる (通常の undo/redo 意味論)
      return { past: [...current.past, current.present], present: updated, future: [] };
    });
  }, []);

  const undo = useCallback(() => {
    setEditHistory((current) => {
      if (current.past.length === 0) return current;
      return {
        past: current.past.slice(0, -1),
        present: current.past[current.past.length - 1],
        future: [...current.future, current.present],
      };
    });
  }, []);

  const redo = useCallback(() => {
    setEditHistory((current) => {
      if (current.future.length === 0) return current;
      return {
        past: [...current.past, current.present],
        present: current.future[current.future.length - 1],
        future: current.future.slice(0, -1),
      };
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
    (slot: CandidateSlot, note?: ScoreNote) => {
      const chosen = note ?? slot.primaryNote;
      apply((state) => insertEvent(state, slot.startTime, [chosen], "inserted-slot"));
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

  // per-event triage 信号 (S3): recognizer 由来イベントの要確認理由。
  // ユーザー挿入イベント (sourceEventId なし) は対象外
  const reviewReasonsByEventId = useMemo(() => {
    const slots = result.candidateSlots ?? [];
    return new Map(
      result.events.map((event) => [event.id, needsReviewReasons(event, slots)]),
    );
  }, [result.events, result.candidateSlots]);

  const [showOnlyNeedsReview, setShowOnlyNeedsReview] = useState(false);

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

  const needsReviewCount = useMemo(
    () =>
      timeline.filter(
        (item) =>
          item.kind === "slot" ||
          ((item.event.sourceEventId
            ? reviewReasonsByEventId.get(item.event.sourceEventId)
            : null) ?? []).length > 0,
      ).length,
    [timeline, reviewReasonsByEventId],
  );

  // 「要確認のみ」: 棄却候補 slot と、triage 信号のあるイベントに絞る
  const visibleTimeline = useMemo(() => {
    if (!showOnlyNeedsReview) return timeline;
    return timeline.filter(
      (item) =>
        item.kind === "slot" ||
        ((item.event.sourceEventId
          ? reviewReasonsByEventId.get(item.event.sourceEventId)
          : null) ?? []).length > 0,
    );
  }, [timeline, showOnlyNeedsReview, reviewReasonsByEventId]);

  const displayEvents = useMemo(() => toDisplayScoreEvents(reviewState), [reviewState]);

  // KalimbaNotePicker は物理配置順が前提なので tuning.notes の並びを保つ
  // (frequency ソートすると実機の鍵盤レイアウトが崩れる)
  const pickerNotes = useMemo(() => {
    return result.instrumentTuning.notes
      .map((tuningNote) =>
        resolveScoreNote(tuningNote.noteName, knownNotes, result.instrumentTuning),
      )
      .filter((note): note is ScoreNote => note !== null);
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

      <ReviewStatusPanel
        transactionId={transactionId}
        initialStatus={initialReviewStatus}
        hasUnsavedCorrections={dirty}
      />

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
        <div className="review-triage-bar" role="group" aria-label="要確認でしぼり込む">
          <button
            type="button"
            className="review-mode-btn"
            aria-pressed={showOnlyNeedsReview}
            onClick={() => setShowOnlyNeedsReview((prev) => !prev)}
          >
            要確認のみ ({needsReviewCount})
          </button>
          <span className="muted">
            {showOnlyNeedsReview
              ? `要確認 ${needsReviewCount} 件を表示中 (全 ${timeline.length} 件)`
              : `全 ${timeline.length} 件を表示中`}
          </span>
        </div>
        {showOnlyNeedsReview && visibleTimeline.length === 0 ? (
          <p className="empty">要確認のイベントはありません。</p>
        ) : null}
        {visibleTimeline.map((item) =>
          item.kind === "event" ? (
            <EventCard
              key={item.event.id}
              event={item.event}
              selected={selectedId === item.event.id}
              reviewReasons={
                (item.event.sourceEventId
                  ? reviewReasonsByEventId.get(item.event.sourceEventId)
                  : null) ?? []
              }
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
              onReplaceNote={(name, note) =>
                apply((s) => replaceNote(s, item.event.id, name, note))
              }
              onNudgeTime={(deltaSec) =>
                apply((s) => setEventTime(s, item.event.id, item.event.timeSec + deltaSec))
              }
              onToggleRemoved={() => apply((s) => toggleRemoved(s, item.event.id))}
            />
          ) : (
            <SlotCard
              key={`slot-${item.slotIndex}`}
              slot={item.slot}
              onInsert={(note) => handleInsertSlot(item.slot, note)}
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
            disabled={editHistory.past.length === 0}
          >
            元に戻す
          </button>
          <button
            type="button"
            className="review-btn"
            onClick={redo}
            disabled={editHistory.future.length === 0}
          >
            やり直す
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
  reviewReasons,
  suggestions,
  pickerNotes,
  onSelect,
  onAudition,
  onRemoveNote,
  onAddNote,
  onReplaceNote,
  onNudgeTime,
  onToggleRemoved,
}: {
  event: ReviewEvent;
  selected: boolean;
  reviewReasons: NeedsReviewReason[];
  suggestions: TranscriptionResult["events"][number]["alternateGroupings"];
  pickerNotes: ScoreNote[];
  onSelect: () => void;
  onAudition: () => void;
  onRemoveNote: (name: string) => void;
  onAddNote: (note: ScoreNote) => void;
  onReplaceNote: (name: string, note: ScoreNote) => void;
  onNudgeTime: (deltaSec: number) => void;
  onToggleRemoved: () => void;
}) {
  // 鍵盤タップの意味: 置換 (armed な既存音と入れ替え) or 追加。
  // 最頻の修正は単音イベントの音高間違いなので、単音では置換を既定にする。
  const [pickMode, setPickMode] = useState<"replace" | "add">(
    event.notes.length === 1 ? "replace" : "add",
  );
  const [replaceTargetName, setReplaceTargetName] = useState<string | null>(null);

  const soleName = event.notes.length === 1 ? noteName(event.notes[0]) : null;
  const armedTarget =
    replaceTargetName && event.notes.some((n) => noteName(n) === replaceTargetName)
      ? replaceTargetName
      : null;
  const effectiveTarget = soleName ?? armedTarget;

  const handlePick = (note: ScoreNote) => {
    if (pickMode === "replace" && effectiveTarget) {
      onReplaceNote(effectiveTarget, note);
      setReplaceTargetName(null);
    } else {
      // 置換対象が未選択の場合も破壊的でない追加に倒す (ヒント文で案内)
      onAddNote(note);
    }
  };

  const noteSuggestions = (suggestions ?? [])
    .filter((alt) => alt.alternateNote !== null)
    .filter(
      (alt) => !event.notes.some((n) => noteName(n) === noteName(alt.alternateNote as ScoreNote)),
    );

  const existingNames = new Set(event.notes.map(noteName));

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
        {reviewReasons.length > 0 && !event.removed ? (
          <span
            className="review-needs-review-badge"
            title={reviewReasons.map((r) => r.label).join(" / ")}
          >
            要確認: {reviewReasons.map((r) => r.label).join(" / ")}
          </span>
        ) : null}
        <span className={`review-origin-badge origin-${event.origin}`}>
          {event.removed ? "削除済" : ORIGIN_LABELS[event.origin]}
        </span>
      </button>

      {selected && !event.removed ? (
        <div className="review-card-editor">
          <div className="review-chip-row">
            {event.notes.map((note) => {
              const name = noteName(note);
              const isTarget = pickMode === "replace" && effectiveTarget === name;
              return (
                <span
                  key={name}
                  className={`review-note-chip${isTarget ? " replace-target" : ""}`}
                  onClick={
                    pickMode === "replace" && event.notes.length > 1
                      ? () => setReplaceTargetName(name)
                      : undefined
                  }
                >
                  {note.labelDoReMi} <small>{name}</small>
                  {event.notes.length > 1 ? (
                    <button
                      type="button"
                      className="review-chip-x"
                      aria-label={`${name} を外す`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveNote(name);
                      }}
                    >
                      ×
                    </button>
                  ) : null}
                </span>
              );
            })}
          </div>

          <div className="review-pick-mode" role="group" aria-label="鍵盤タップ時の動作">
            <button
              type="button"
              className="review-mode-btn"
              aria-pressed={pickMode === "replace"}
              onClick={() => setPickMode("replace")}
            >
              置換
            </button>
            <button
              type="button"
              className="review-mode-btn"
              aria-pressed={pickMode === "add"}
              onClick={() => setPickMode("add")}
            >
              追加
            </button>
            <span className="review-mode-hint">
              {pickMode === "replace"
                ? event.notes.length > 1
                  ? effectiveTarget
                    ? `鍵盤をタップすると ${effectiveTarget} と入れ替えます`
                    : "置換する音のチップを選んでから鍵盤をタップ (未選択のタップは追加)"
                  : "鍵盤をタップすると音を入れ替えます"
                : "鍵盤をタップすると和音に追加します"}
            </span>
          </div>

          <KalimbaNotePicker
            notes={pickerNotes}
            disabledNames={existingNames}
            onPick={handlePick}
            label="音を選択"
          />

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
                    onClick={() => handlePick(note)}
                  >
                    ＋{note.labelDoReMi} <small>{Math.round(alt.confidence * 100)}%</small>
                  </button>
                );
              })}
            </div>
          ) : null}

          <div className="review-time-row" role="group" aria-label="タイミング微調整">
            <span className="review-time-label">タイミング</span>
            <button
              type="button"
              className="review-btn review-btn-small"
              onClick={() => onNudgeTime(-0.05)}
            >
              −50ms
            </button>
            <button
              type="button"
              className="review-btn review-btn-small"
              onClick={() => onNudgeTime(-0.01)}
            >
              −10ms
            </button>
            <span className="review-time-value">{formatTime(event.timeSec)}</span>
            <button
              type="button"
              className="review-btn review-btn-small"
              onClick={() => onNudgeTime(0.01)}
            >
              +10ms
            </button>
            <button
              type="button"
              className="review-btn review-btn-small"
              onClick={() => onNudgeTime(0.05)}
            >
              +50ms
            </button>
          </div>

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

function SlotCard({
  slot,
  onInsert,
}: {
  slot: CandidateSlot;
  onInsert: (note?: ScoreNote) => void;
}) {
  const reasonLabel = DROP_REASON_LABELS[slot.dropReason] ?? slot.dropReason;
  // Candidate-first: the recognizer dropped this segment but kept candidates.
  // Adopting a candidate is one tap (Candidate Recall → low correction burden);
  // typing a note by hand stays available as a fallback (InsertAtPlayhead).
  const alternates = slot.candidates.filter(
    (cand) => noteName(cand) !== noteName(slot.primaryNote),
  );
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
      </div>
      <div className="review-slot-candidates">
        <span className="review-suggestions-label">候補をそのまま採用:</span>
        <button
          type="button"
          className="review-suggestion-chip review-suggestion-chip-primary"
          onClick={() => onInsert(slot.primaryNote)}
        >
          ＋{slot.primaryNote.labelDoReMi} <small>{noteName(slot.primaryNote)}</small>
        </button>
        {alternates.map((cand) => (
          <button
            key={noteName(cand)}
            type="button"
            className="review-suggestion-chip"
            onClick={() => onInsert(cand)}
          >
            ＋{cand.labelDoReMi} <small>{noteName(cand)}</small>
          </button>
        ))}
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
  const [open, setOpen] = useState(false);
  return (
    <div className="review-insert-control">
      <button
        type="button"
        className="review-btn review-btn-small"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        ＋ 再生位置に音を挿入…
      </button>
      {open ? (
        <KalimbaNotePicker notes={notes} onPick={onInsert} label="再生位置に挿入する音を選択" />
      ) : null}
    </div>
  );
}
