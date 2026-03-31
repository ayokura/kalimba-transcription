"use client";

import { KeyboardEvent, useEffect, useRef } from "react";

import { buildGestureLabel } from "@/lib/scoreEventPresentation";
import { ScoreEvent } from "@/lib/types";

type ReviewEventListPanelProps = {
  events: ScoreEvent[];
  activeEventId: string | null;
  onActiveEventIdChange: (eventId: string) => void;
};

export function ReviewEventListPanel({
  events,
  activeEventId,
  onActiveEventIdChange,
}: ReviewEventListPanelProps) {
  const eventItemRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  useEffect(() => {
    if (!activeEventId) {
      return;
    }

    eventItemRefs.current[activeEventId]?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
    });
  }, [activeEventId]);

  function focusEventByIndex(index: number) {
    const nextEvent = events[index] ?? null;
    if (!nextEvent) {
      return;
    }

    onActiveEventIdChange(nextEvent.id);
    eventItemRefs.current[nextEvent.id]?.focus();
  }

  function handleEventListKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) {
    if (event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      focusEventByIndex(Math.min(index + 1, events.length - 1));
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      focusEventByIndex(Math.max(index - 1, 0));
      return;
    }

    if (event.key === "Home") {
      event.preventDefault();
      focusEventByIndex(0);
      return;
    }

    if (event.key === "End") {
      event.preventDefault();
      focusEventByIndex(events.length - 1);
    }
  }

  return (
    <div className="event-list">
      {events.map((event, index) => (
        <button
          key={event.id}
          type="button"
          className={`event-list-item ${event.id === activeEventId ? "selected" : ""}`}
          aria-current={event.id === activeEventId ? "true" : undefined}
          ref={(element) => {
            eventItemRefs.current[event.id] = element;
          }}
          onClick={() => onActiveEventIdChange(event.id)}
          onKeyDown={(nextEvent) => handleEventListKeyDown(nextEvent, index)}
        >
          <div className="event-list-item-header">
            <strong>{event.id}</strong>
            <span className="event-list-index">#{index + 1}</span>
          </div>
          <span>{event.notes.map((note) => note.labelDoReMi).join(" / ")}</span>
          <div className="event-list-meta">
            <span>{event.startBeat}拍</span>
            <span>{event.durationBeat}拍</span>
            <span>{buildGestureLabel(event.gesture)}</span>
          </div>
        </button>
      ))}
    </div>
  );
}
