"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";

import { NotationPanel } from "@/components/NotationPanel";
import { isReviewSessionStorageAvailable, loadReviewSession } from "@/lib/reviewSession";
import { NotationMode } from "@/lib/types";

export function ReviewWorkspace() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session") ?? "";
  const storageAvailable = isReviewSessionStorageAvailable();
  const session = useMemo(
    () => (storageAvailable && sessionId ? loadReviewSession(sessionId) : null),
    [sessionId, storageAvailable],
  );
  const sourceProfile = typeof session?.requestSnapshot.sourceProfile === "string"
    ? session.requestSnapshot.sourceProfile
    : "unknown-profile";
  const [notationMode, setNotationMode] = useState<NotationMode>(session?.notationMode ?? "vertical");
  const [activeEventId, setActiveEventId] = useState<string | null>(session?.activeEventId ?? null);
  const activeEvent = session?.responseSnapshot.events.find((event) => event.id === activeEventId)
    ?? session?.responseSnapshot.events[0]
    ?? null;

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
            {session.responseSnapshot.events.map((event) => (
              <button
                key={event.id}
                className={`event-list-item ${event.id === (activeEvent?.id ?? "") ? "selected" : ""}`}
                onClick={() => setActiveEventId(event.id)}
              >
                <strong>{event.id}</strong>
                <span>{event.notes.map((note) => note.labelDoReMi).join(" / ")}</span>
              </button>
            ))}
          </div>
        </section>

        <div className="stack gap-xl">
          <NotationPanel result={session.responseSnapshot} mode={notationMode} onModeChange={setNotationMode} />
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Active Event</p>
                <h2>選択中の event</h2>
              </div>
            </div>
            {!activeEvent ? (
              <p className="empty">event がありません。</p>
            ) : (
              <div className="stack">
                <div className="summary-strip">
                  <span>{activeEvent.id}</span>
                  <span>{activeEvent.startBeat}拍</span>
                  <span>{activeEvent.durationBeat}拍</span>
                </div>
                <div className="note-chip-row">
                  {activeEvent.notes.map((note, index) => (
                    <span key={`${note.pitchClass}-${index}`} className="note-chip">
                      {note.labelDoReMi}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
