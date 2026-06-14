"use client";

import { KeyboardEvent, useEffect, useRef } from "react";

import { buildGestureLabel } from "@/lib/scoreEventPresentation";
import { ScoreEvent } from "@/lib/types";

type ReviewEventListPanelProps = {
  events: ScoreEvent[];
  activeEventId: string | null;
  onActiveEventIdChange: (eventId: string) => void;
  /**
   * 指定 event の位置から再生 (audition) する。audio が利用可能なときだけ渡され、
   * 渡されたときは event クリックで頭出し+区間再生し、各 event に再生ボタンを表示する。
   */
  onAuditionEvent?: (eventId: string) => void;
};

export function ReviewEventListPanel({
  events,
  activeEventId,
  onActiveEventIdChange,
  onAuditionEvent,
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

  function handleSelect(eventId: string) {
    onActiveEventIdChange(eventId);
    // audio がある時は選択と同時に頭出し+区間再生する (キーボードナビゲーションは
    // onActiveEventIdChange のみ呼ぶので、矢印キーで再生が走ることはない)。
    onAuditionEvent?.(eventId);
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
          onClick={() => handleSelect(event.id)}
          onKeyDown={(nextEvent) => handleEventListKeyDown(nextEvent, index)}
        >
          <div className="event-list-item-header">
            <strong>{event.id}</strong>
            <span className="event-list-item-header-right">
              {onAuditionEvent ? (
                <span
                  role="button"
                  tabIndex={-1}
                  className="event-list-audition"
                  aria-label={`${event.id} をここから再生`}
                  onClick={(clickEvent) => {
                    // 親 button の onClick (選択) と二重発火しないよう伝播を止める。
                    clickEvent.stopPropagation();
                    onAuditionEvent(event.id);
                  }}
                >
                  ▶
                </span>
              ) : null}
              <span className="event-list-index">#{index + 1}</span>
            </span>
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
