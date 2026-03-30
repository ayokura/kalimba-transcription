"use client";

import { InstrumentTuning, ScoreEvent, TranscriptionResult } from "@/lib/types";
import { cloneResultWithEvents, noteFromName } from "@/lib/notation";

type EditorPanelProps = {
  result: TranscriptionResult | null;
  activeEventId: string | null;
  onActiveEventIdChange: (value: string | null) => void;
  tuning: InstrumentTuning | null;
  hasManualEdits?: boolean;
  onResetToAnalysis?: () => void;
  onResultChange: (result: TranscriptionResult) => void;
};

function sanitizeEditedEvent(event: ScoreEvent): ScoreEvent {
  return { ...event, gesture: "ambiguous", isGlissLike: false };
}

export function EditorPanel({
  result,
  activeEventId,
  onActiveEventIdChange,
  tuning,
  hasManualEdits = false,
  onResetToAnalysis,
  onResultChange,
}: EditorPanelProps) {
  const activeEvent = result?.events.find((event) => event.id === activeEventId) ?? result?.events[0] ?? null;

  function confirmDestructiveEdit(actionLabel: string) {
    return window.confirm(`${actionLabel}と現在の解析結果の近似編集になります。続行しますか？`);
  }

  function updateEvents(nextEvents: ScoreEvent[]) {
    if (!result) {
      return;
    }
    onResultChange(cloneResultWithEvents(result, nextEvents));
  }

  function removeLastNote() {
    if (!result || !activeEvent) {
      return;
    }
    if (!confirmDestructiveEdit("最後の音を削除する")) {
      return;
    }

    const nextEvents = result.events.flatMap((event) => {
      if (event.id !== activeEvent.id) {
        return [event];
      }

      const nextNotes = event.notes.slice(0, -1);
      return nextNotes.length > 0 ? [sanitizeEditedEvent({ ...event, notes: nextNotes })] : [];
    });

    updateEvents(nextEvents);
    onActiveEventIdChange(nextEvents[0]?.id ?? null);
  }

  function addFirstTuningNote() {
    if (!result || !activeEvent || !tuning || tuning.notes.length === 0) {
      return;
    }
    if (!confirmDestructiveEdit("音を追加する")) {
      return;
    }
    const baseNote = tuning.notes[0];
    const nextEvents = result.events.map((event) =>
      event.id === activeEvent.id
        ? sanitizeEditedEvent({ ...event, notes: [...event.notes, noteFromName(baseNote.noteName, baseNote.key)] })
        : event,
    );
    updateEvents(nextEvents);
  }

  function mergeWithNext() {
    if (!result || !activeEvent) {
      return;
    }
    if (!confirmDestructiveEdit("次のイベントと結合する")) {
      return;
    }
    const index = result.events.findIndex((event) => event.id === activeEvent.id);
    if (index < 0 || index === result.events.length - 1) {
      return;
    }
    const current = result.events[index];
    const next = result.events[index + 1];
    const merged: ScoreEvent = sanitizeEditedEvent({
      ...current,
      durationBeat: current.durationBeat + next.durationBeat,
      notes: [...current.notes, ...next.notes],
    });
    const nextEvents = [...result.events.slice(0, index), merged, ...result.events.slice(index + 2)];
    updateEvents(nextEvents);
    onActiveEventIdChange(merged.id);
  }

  function splitEvent() {
    if (!result || !activeEvent || activeEvent.notes.length < 2) {
      return;
    }
    if (!confirmDestructiveEdit("イベントを2つに分割する")) {
      return;
    }
    const index = result.events.findIndex((event) => event.id === activeEvent.id);
    const midpoint = Math.ceil(activeEvent.notes.length / 2);
    const first: ScoreEvent = sanitizeEditedEvent({
      ...activeEvent,
      notes: activeEvent.notes.slice(0, midpoint),
      durationBeat: Math.max(activeEvent.durationBeat / 2, 0.25),
    });
    const second: ScoreEvent = sanitizeEditedEvent({
      ...activeEvent,
      id: `${activeEvent.id}-b`,
      startBeat: activeEvent.startBeat + first.durationBeat,
      durationBeat: Math.max(activeEvent.durationBeat - first.durationBeat, 0.25),
      notes: activeEvent.notes.slice(midpoint),
    });
    const nextEvents = [...result.events.slice(0, index), first, second, ...result.events.slice(index + 1)];
    updateEvents(nextEvents);
    onActiveEventIdChange(first.id);
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Editor</p>
          <h2>簡易修正</h2>
        </div>
      </div>
      {!result ? (
        <p className="empty">解析後にイベント単位で修正できます。</p>
      ) : (
        <div className="editor-layout">
          <div className="event-list">
            <div className="warning-box">
              <p>手動修正モードです。ここでの変更は解析結果そのものではなく、確認用の近似編集として扱われます。</p>
              <div className="row wrap">
                <span className={`pill ${hasManualEdits ? "review-pill status-review_needed" : ""}`}>{hasManualEdits ? "Edited Result" : "Analysis Snapshot"}</span>
                {hasManualEdits && onResetToAnalysis ? (
                  <button className="ghost" onClick={onResetToAnalysis}>
                    解析結果に戻す
                  </button>
                ) : null}
              </div>
            </div>
            {result.events.map((event) => (
              <button
                key={event.id}
                className={`event-list-item ${event.id === (activeEvent?.id ?? "") ? "selected" : ""}`}
                onClick={() => onActiveEventIdChange(event.id)}
              >
                <strong>{event.id}</strong>
                <span>{event.notes.map((note) => note.labelDoReMi).join(" / ")}</span>
              </button>
            ))}
          </div>
          <div className="stack">
            <div className="event-meta">
              <span>開始拍: {activeEvent?.startBeat ?? "-"}</span>
              <span>長さ: {activeEvent?.durationBeat ?? "-"}</span>
            </div>
            <div className="row wrap">
              <button className="secondary" onClick={addFirstTuningNote}>
                音を追加
              </button>
              <button className="secondary" onClick={removeLastNote}>
                最後の音を削除
              </button>
              <button className="secondary" onClick={mergeWithNext}>
                次と結合
              </button>
              <button className="secondary" onClick={splitEvent}>
                2つに分割
              </button>
            </div>
            <div className="note-chip-row">
              {activeEvent?.notes.map((note, index) => (
                <span key={`${note.pitchClass}-${index}`} className="note-chip">
                  {note.labelDoReMi}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
