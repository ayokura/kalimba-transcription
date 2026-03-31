"use client";

import Link from "next/link";
import { KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import { NotationPanel } from "@/components/NotationPanel";
import { ReviewFocusPanel } from "@/components/ReviewFocusPanel";
import { loadReviewAudio } from "@/lib/reviewAudioStore";
import { isReviewSessionStorageAvailable, loadReviewSession, updateReviewSessionUiState } from "@/lib/reviewSession";
import { buildGestureLabel } from "@/lib/scoreEventPresentation";
import { NotationMode } from "@/lib/types";

export function ReviewWorkspace() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session") ?? "";
  const storageAvailable = isReviewSessionStorageAvailable();
  const session = useMemo(
    () => (storageAvailable && sessionId ? loadReviewSession(sessionId) : null),
    [sessionId, storageAvailable],
  );
  const audioBlob = useMemo(
    () => (storageAvailable && sessionId ? loadReviewAudio(sessionId) : null),
    [sessionId, storageAvailable],
  );
  const sourceProfile = typeof session?.requestSnapshot.sourceProfile === "string"
    ? session.requestSnapshot.sourceProfile
    : "unknown-profile";
  const [notationMode, setNotationMode] = useState<NotationMode>(session?.notationMode ?? "vertical");
  const [activeEventId, setActiveEventId] = useState<string | null>(session?.activeEventId ?? null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const eventItemRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const activeEvent = session?.responseSnapshot.events.find((event) => event.id === activeEventId)
    ?? session?.responseSnapshot.events[0]
    ?? null;
  const events = session?.responseSnapshot.events ?? [];

  useEffect(() => {
    if (!audioBlob) {
      setAudioUrl(null);
      return;
    }

    const nextUrl = URL.createObjectURL(audioBlob);
    setAudioUrl(nextUrl);
    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [audioBlob]);

  useEffect(() => {
    setNotationMode(session?.notationMode ?? "vertical");
    setActiveEventId(session?.activeEventId ?? null);
  }, [sessionId, session?.notationMode, session?.activeEventId]);

  useEffect(() => {
    if (!session || !storageAvailable || !sessionId) {
      return;
    }

    if (notationMode === session.notationMode && activeEventId === session.activeEventId) {
      return;
    }

    updateReviewSessionUiState(sessionId, {
      notationMode,
      activeEventId,
    });
  }, [activeEventId, notationMode, session, sessionId, storageAvailable]);

  useEffect(() => {
    if (!activeEvent?.id) {
      return;
    }

    eventItemRefs.current[activeEvent.id]?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
    });
  }, [activeEvent?.id]);

  function focusEventByIndex(index: number) {
    const nextEvent = events[index] ?? null;
    if (!nextEvent) {
      return;
    }

    setActiveEventId(nextEvent.id);
    eventItemRefs.current[nextEvent.id]?.focus();
  }

  function handleEventListKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
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

  if (!storageAvailable) {
    return (
      <main className="shell">
        <section className="hero">
          <div>
            <p className="eyebrow">Review Workspace</p>
            <h1>この環境では review session を開けません。</h1>
            <p className="hero-copy">sessionStorage が利用できないため、解析結果の handoff に失敗しました。</p>
          </div>
          <div className="hero-card">
            <p>Next Action</p>
            <ul>
              <li><Link href="/">利用者向け画面へ戻る</Link></li>
            </ul>
          </div>
        </section>
      </main>
    );
  }

  if (!session) {
    return (
      <main className="shell">
        <section className="hero">
          <div>
            <p className="eyebrow">Review Workspace</p>
            <h1>review session が見つかりません。</h1>
            <p className="hero-copy">解析後に `/review` へ進むか、最初から解析をやり直してください。</p>
          </div>
          <div className="hero-card">
            <p>Next Action</p>
            <ul>
              <li><Link href="/">利用者向け画面へ戻る</Link></li>
            </ul>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Review Workspace</p>
          <h1>解析結果を確認する。</h1>
          <p className="hero-copy">
            ここでは read-only で解析結果を見返します。repair と playback は後続 issue で追加します。
          </p>
        </div>
        <div className="hero-card">
          <p>Session</p>
          <ul>
            <li>{session.tuning.name}</li>
            <li>{session.responseSnapshot.tempo} BPM</li>
            <li>{session.responseSnapshot.events.length} events</li>
            <li>{audioBlob ? "audio ready" : "audio unavailable"}</li>
            <li><Link href="/">利用者向け画面へ戻る</Link></li>
          </ul>
        </div>
      </section>

      <div className="workspace-grid">
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Session</p>
              <h2>読み込み済み snapshot</h2>
            </div>
          </div>
          <div className="summary-strip">
            <span>{session.acquisitionMode}</span>
            <span>{sourceProfile}</span>
            <span>{session.responseSnapshot.events.length} events</span>
          </div>
          <div className="event-list">
            {session.responseSnapshot.events.map((event, index) => (
              <button
                key={event.id}
                type="button"
                className={`event-list-item ${event.id === (activeEvent?.id ?? "") ? "selected" : ""}`}
                aria-current={event.id === (activeEvent?.id ?? "") ? "true" : undefined}
                ref={(element) => {
                  eventItemRefs.current[event.id] = element;
                }}
                onClick={() => setActiveEventId(event.id)}
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
        </section>

        <div className="stack gap-xl">
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Playback</p>
                <h2>録音全体を再生</h2>
              </div>
            </div>
            {audioUrl ? (
              <audio controls preload="metadata" src={audioUrl} className="wide" />
            ) : (
              <div className="warning-box">
                <p>この review session には audio が残っていません。same-tab の解析直後に `/review` を開き直してください。</p>
              </div>
            )}
          </section>
          <NotationPanel
            result={session.responseSnapshot}
            mode={notationMode}
            onModeChange={setNotationMode}
            activeEventId={activeEvent?.id ?? null}
            onActiveEventIdChange={setActiveEventId}
          />
          <ReviewFocusPanel
            events={session.responseSnapshot.events}
            activeEventId={activeEvent?.id ?? null}
            onActiveEventIdChange={setActiveEventId}
          />
        </div>
      </div>
    </main>
  );
}
