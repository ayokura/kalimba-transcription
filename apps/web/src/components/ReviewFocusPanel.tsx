"use client";

import { buildGestureLabel } from "@/lib/scoreEventPresentation";
import { ScoreEvent } from "@/lib/types";

type ReviewFocusPanelProps = {
  events: ScoreEvent[];
  activeEventId: string | null;
  onActiveEventIdChange: (eventId: string) => void;
};

function buildEventLine(event: ScoreEvent | null) {
  if (!event) {
    return "該当なし";
  }

  return event.notes.map((note) => note.labelDoReMi).join(" / ");
}

function buildNoteDetailLine(event: ScoreEvent, index: number) {
  const note = event.notes[index];
  if (!note) {
    return "";
  }

  return `Key ${note.key} · ${note.pitchClass}${note.octave} · ${note.labelDoReMi}`;
}

function buildContextButtonLabel(position: "前" | "次", event: ScoreEvent | null) {
  if (!event) {
    return `${position}の文脈 event はありません`;
  }

  return `${position}の文脈 event ${event.id} へ移動`;
}

export function ReviewFocusPanel({
  events,
  activeEventId,
  onActiveEventIdChange,
}: ReviewFocusPanelProps) {
  const activeIndex = Math.max(
    events.findIndex((event) => event.id === activeEventId),
    0,
  );
  const activeEvent = events[activeIndex] ?? null;
  const previousEvent = activeIndex > 0 ? events[activeIndex - 1] ?? null : null;
  const nextEvent = activeIndex < events.length - 1 ? events[activeIndex + 1] ?? null : null;

  if (!activeEvent) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Review Focus</p>
            <h2>選択中の event</h2>
          </div>
        </div>
        <p className="empty">event がありません。</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Review Focus</p>
          <h2>選択中の event</h2>
        </div>
        <div className="row wrap review-focus-nav">
          <button
            type="button"
            className="secondary"
            aria-label="前の event へ移動"
            onClick={() => previousEvent && onActiveEventIdChange(previousEvent.id)}
            disabled={!previousEvent}
          >
            前の event
          </button>
          <button
            type="button"
            className="secondary"
            aria-label="次の event へ移動"
            onClick={() => nextEvent && onActiveEventIdChange(nextEvent.id)}
            disabled={!nextEvent}
          >
            次の event
          </button>
        </div>
      </div>
      <div className="summary-strip">
        <span>{activeEvent.id}</span>
        <span>{activeIndex + 1} / {events.length}</span>
        <span>{activeEvent.startBeat}拍</span>
        <span>{activeEvent.durationBeat}拍</span>
        <span>{buildGestureLabel(activeEvent.gesture)}</span>
      </div>
      <p className="muted">
        左の event list と譜面上の selection は同期しています。ここでは前後の文脈を見ながら確認できます。
      </p>
      <div className="note-chip-row">
        {activeEvent.notes.map((note, index) => (
          <span key={`${note.pitchClass}-${note.octave}-${index}`} className="note-chip">
            {note.labelDoReMi}
          </span>
        ))}
      </div>
      <div className="review-focus-detail-grid">
        <div className="review-focus-detail-card">
          <small>Event</small>
          <strong>{activeEvent.id}</strong>
          <span>{activeEvent.startBeat}拍目から {activeEvent.durationBeat}拍</span>
          <span>{buildGestureLabel(activeEvent.gesture)}</span>
        </div>
        <div className="review-focus-detail-card">
          <small>Notes</small>
          {activeEvent.notes.map((_, index) => (
            <span key={`${activeEvent.id}-detail-${index}`}>{buildNoteDetailLine(activeEvent, index)}</span>
          ))}
        </div>
      </div>
      <div className="review-context-grid">
        <button
          type="button"
          className={`review-context-card ${previousEvent ? "" : "empty"}`}
          aria-label={buildContextButtonLabel("前", previousEvent)}
          onClick={() => previousEvent && onActiveEventIdChange(previousEvent.id)}
          disabled={!previousEvent}
        >
          <small>前</small>
          <strong>{previousEvent?.id ?? "なし"}</strong>
          <span>{buildEventLine(previousEvent)}</span>
        </button>
        <div className="review-context-card current">
          <small>現在</small>
          <strong>{activeEvent.id}</strong>
          <span>{buildEventLine(activeEvent)}</span>
        </div>
        <button
          type="button"
          className={`review-context-card ${nextEvent ? "" : "empty"}`}
          aria-label={buildContextButtonLabel("次", nextEvent)}
          onClick={() => nextEvent && onActiveEventIdChange(nextEvent.id)}
          disabled={!nextEvent}
        >
          <small>次</small>
          <strong>{nextEvent?.id ?? "なし"}</strong>
          <span>{buildEventLine(nextEvent)}</span>
        </button>
      </div>
    </section>
  );
}
