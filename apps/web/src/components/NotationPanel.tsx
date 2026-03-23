"use client";

import { CaptureAssessmentDetails } from "@/lib/api";
import { NotationMode, TranscriptionResult } from "@/lib/types";

type NotationPanelProps = {
  result: TranscriptionResult | null;
  mode: NotationMode;
  onModeChange: (mode: NotationMode) => void;
  review?: CaptureAssessmentDetails | null;
};

function buildGestureLabel(gesture: string) {
  if (gesture === "strict_chord") return "同時和音";
  if (gesture === "slide_chord") return "スライド和音";
  if (gesture === "separated_notes") return "単音列";
  return "要確認";
}

function buildReviewLine(reviewEvent: CaptureAssessmentDetails["events"][number] | undefined) {
  if (!reviewEvent) {
    return null;
  }

  if (reviewEvent.matches) {
    return `expected = detected: ${reviewEvent.detected ?? "-"}`;
  }

  return `expected: ${reviewEvent.expected ?? "-"} / detected: ${reviewEvent.detected ?? "-"}`;
}

export function NotationPanel({ result, mode, onModeChange, review }: NotationPanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Notation</p>
          <h2>解析結果</h2>
        </div>
        <div className="tabs">
          <button className={mode === "vertical" ? "active" : ""} onClick={() => onModeChange("vertical")}>
            ドレミ縦並び
          </button>
          <button className={mode === "numbered" ? "active" : ""} onClick={() => onModeChange("numbered")}>
            数字譜
          </button>
          <button className={mode === "western" ? "active" : ""} onClick={() => onModeChange("western")}>
            通常表記
          </button>
        </div>
      </div>

      {!result ? (
        <p className="empty">録音して解析すると、ここに譜面が表示されます。</p>
      ) : (
        <div className="stack gap-lg">
          <div className="summary-strip">
            <span>Tempo {result.tempo} BPM</span>
            <span>{result.instrumentTuning.name}</span>
            <span>{result.events.length} events</span>
            {review ? <span className={`pill review-pill status-${review.status}`}>{review.label}</span> : null}
          </div>
          {review ? (
            <div className={`review-box status-${review.status}`}>
              <div className="panel-header compact">
                <div>
                  <p className="eyebrow">Review</p>
                  <strong>{review.summary}</strong>
                </div>
                <span className={`pill review-pill status-${review.status}`}>{review.label}</span>
              </div>
              <p>{review.reason}</p>
              <div className="summary-strip">
                <span>expected {review.expectedEventCount}</span>
                <span>detected {review.detectedEventCount}</span>
                <span>mismatch {review.mismatchCount}</span>
                {review.extraEventCount > 0 ? <span>extra {review.extraEventCount}</span> : null}
                {review.missingEventCount > 0 ? <span>missing {review.missingEventCount}</span> : null}
              </div>
            </div>
          ) : null}
          {result.warnings.length > 0 ? (
            <div className="warning-box">
              {result.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
          {mode === "vertical" ? (
            <div className="notation-grid">
              {result.events.map((event, index) => {
                const reviewEvent = review?.events[index];
                const reviewLine = buildReviewLine(reviewEvent);
                return (
                  <article key={event.id} className={`event-card ${event.isGlissLike ? "gliss" : ""} ${reviewEvent && !reviewEvent.matches ? "mismatch" : ""}`}>
                    <small>#{index + 1}</small>
                    <div className="vertical-stack">
                      {result.notationViews.verticalDoReMi[index].map((note) => (
                        <span key={`${event.id}-${note}`}>{note}</span>
                      ))}
                    </div>
                    <div className="event-meta-row">
                      <span className={`pill gesture-pill gesture-${event.gesture}`}>{buildGestureLabel(event.gesture)}</span>
                    </div>
                    {reviewLine ? <div className={`event-review ${reviewEvent?.matches ? "match" : "mismatch"}`}>{reviewLine}</div> : null}
                    <footer>
                      <span>{event.startBeat}拍</span>
                      <span>{event.durationBeat}拍</span>
                    </footer>
                  </article>
                );
              })}
            </div>
          ) : (
            <div className="line-notation">
              {(mode === "numbered" ? result.notationViews.numbered : result.notationViews.western).map((entry, index) => {
                const reviewEvent = review?.events[index];
                const reviewLine = buildReviewLine(reviewEvent);
                return (
                  <div key={`${mode}-${index}`} className={`line-row-block ${reviewEvent && !reviewEvent.matches ? "mismatch" : ""}`}>
                    <div className="line-row">
                      <span className="line-index">{index + 1}</span>
                      <code>{entry}</code>
                    </div>
                    <div className="event-meta-row line">
                      <span className={`pill gesture-pill gesture-${result.events[index]?.gesture ?? "ambiguous"}`}>{buildGestureLabel(result.events[index]?.gesture ?? "ambiguous")}</span>
                    </div>
                    {reviewLine ? <div className={`event-review line ${reviewEvent?.matches ? "match" : "mismatch"}`}>{reviewLine}</div> : null}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
