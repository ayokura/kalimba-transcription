"use client";

import { InstrumentTuning, ScoreEvent, TranscriptionResult } from "@/lib/types";
import { cloneResultWithEvents, noteFromName } from "@/lib/notation";

type EditorPanelProps = {
  result: TranscriptionResult | null;
  activeEventId: string | null;
  onActiveEventIdChange: (value: string | null) => void;
  tuning: InstrumentTuning | null;
  onResultChange: (result: TranscriptionResult) => void;
};

export function EditorPanel({
  result,
  activeEventId,
  onActiveEventIdChange,
  tuning,
  onResultChange,
}: EditorPanelProps) {
  const activeEvent = result?.events.find((event) => event.id === activeEventId) ?? result?.events[0] ?? null;

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

    const nextEvents = result.events.flatMap((event) => {
      if (event.id !== activeEvent.id) {
        return [event];
      }

      const nextNotes = event.notes.slice(0, -1);
      return nextNotes.length > 0 ? [{ ...event, notes: nextNotes }] : [];
    });

    updateEvents(nextEvents);
    onActiveEventIdChange(nextEvents[0]?.id ?? null);
  }

  function addFirstTuningNote() {
    if (!result || !activeEvent || !tuning || tuning.notes.length === 0) {
      return;
    }
    const baseNote = tuning.notes[0];
    const nextEvents = result.events.map((event) =>
      event.id === activeEvent.id
        ? { ...event, notes: [...event.notes, noteFromName(baseNote.noteName, baseNote.key)] }
        : event,
    );
    updateEvents(nextEvents);
  }

  function mergeWithNext() {
    if (!result || !activeEvent) {
      return;
    }
    const index = result.events.findIndex((event) => event.id === activeEvent.id);
    if (index < 0 || index === result.events.length - 1) {
      return;
    }
    const current = result.events[index];
    const next = result.events[index + 1];
    const merged: ScoreEvent = {
      ...current,
      durationBeat: current.durationBeat + next.durationBeat,
      notes: [...current.notes, ...next.notes],
      isGlissLike: current.isGlissLike || next.isGlissLike,
    };
    const nextEvents = [...result.events.slice(0, index), merged, ...result.events.slice(index + 2)];
    updateEvents(nextEvents);
    onActiveEventIdChange(merged.id);
  }

  function splitEvent() {
    if (!result || !activeEvent || activeEvent.notes.length < 2) {
      return;
    }
    const index = result.events.findIndex((event) => event.id === activeEvent.id);
    const midpoint = Math.ceil(activeEvent.notes.length / 2);
    const first: ScoreEvent = {
      ...activeEvent,
      notes: activeEvent.notes.slice(0, midpoint),
      durationBeat: Math.max(activeEvent.durationBeat / 2, 0.25),
    };
    const second: ScoreEvent = {
      ...activeEvent,
      id: `${activeEvent.id}-b`,
      startBeat: activeEvent.startBeat + first.durationBeat,
      durationBeat: Math.max(activeEvent.durationBeat - first.durationBeat, 0.25),
      notes: activeEvent.notes.slice(midpoint),
    };
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