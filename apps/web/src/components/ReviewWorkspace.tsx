"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { NotationPanel } from "@/components/NotationPanel";
import { ReviewEventListPanel } from "@/components/ReviewEventListPanel";
import { ReviewFocusPanel } from "@/components/ReviewFocusPanel";
import { useAudioPlayback } from "@/hooks/useAudioPlayback";
import { useReviewWorkspaceSession } from "@/lib/useReviewWorkspaceSession";

export function ReviewWorkspace() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session") ?? "";
  const {
    activeEvent,
    activeEventId,
    audioBlob,
    audioUrl,
    notationMode,
    session,
    sourceProfile,
    storageAvailable,
    setActiveEventId,
    setNotationMode,
  } = useReviewWorkspaceSession(sessionId);

  // hooks は early return より前で無条件に呼ぶ必要があるため、session 未取得時は
  // 空配列で初期化しておく (audioRef は描画される <audio> が無ければ no-op)。
  const events = session?.responseSnapshot.events ?? [];
  const { audioRef, seekToEvent, handleTimeUpdate, handleSeeking } = useAudioPlayback(
    events,
    setActiveEventId,
  );

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
            ここでは解析結果を見返します。event を選ぶとその位置から再生し、再生に合わせて選択もハイライトされます。repair は後続 issue で追加します。
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
          <ReviewEventListPanel
            events={session.responseSnapshot.events}
            activeEventId={activeEvent?.id ?? null}
            onActiveEventIdChange={setActiveEventId}
            onAuditionEvent={audioUrl ? (eventId) => seekToEvent(eventId) : undefined}
          />
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
              <audio
                ref={audioRef}
                controls
                preload="metadata"
                src={audioUrl}
                onTimeUpdate={handleTimeUpdate}
                onSeeking={handleSeeking}
                className="wide"
              />
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
            onAuditionEvent={audioUrl ? (eventId) => seekToEvent(eventId) : undefined}
          />
        </div>
      </div>
    </main>
  );
}
