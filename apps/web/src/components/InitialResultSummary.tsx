"use client";

import Link from "next/link";

import { CaptureAssessmentDetails } from "@/lib/api";
import { TranscriptionResult } from "@/lib/types";

type InitialResultSummaryProps = {
  result: TranscriptionResult | null;
  review: CaptureAssessmentDetails | null;
  reviewSessionId: string | null;
};

export function InitialResultSummary({ result, review, reviewSessionId }: InitialResultSummaryProps) {
  if (!result) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Initial Result</p>
            <h2>解析結果の概要</h2>
          </div>
        </div>
        <p className="empty">解析後にここへ概要を表示し、必要なら `/review` へ進めます。</p>
      </section>
    );
  }

  const previewEvents = result.events.slice(0, 6);
  const remainingCount = Math.max(0, result.events.length - previewEvents.length);

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Initial Result</p>
          <h2>解析結果の概要</h2>
        </div>
        {review ? <span className={`pill review-pill status-${review.status}`}>{review.label}</span> : null}
      </div>

      <div className="stack gap-lg">
        <div className="summary-strip">
          <span>{result.instrumentTuning.name}</span>
          <span>Tempo {result.tempo} BPM</span>
          <span>{result.events.length} events</span>
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
          </div>
        ) : null}

        {result.warnings.length > 0 ? (
          <div className="warning-box">
            {result.warnings.map((warning) => (
              <p key={warning}>{warning}</p>
            ))}
          </div>
        ) : null}

        <div className="compact-notation-preview">
          <p className="muted">先頭の event だけを簡易表示しています。詳細確認と編集は `/review` で行います。</p>
          <div className="stack">
            {previewEvents.map((event, index) => (
              <div key={event.id} className="compact-notation-line">
                <strong>#{index + 1}</strong>
                <span>{event.notes.map((note) => note.labelDoReMi).join(" / ")}</span>
              </div>
            ))}
          </div>
          {remainingCount > 0 ? <p className="muted">…ほか {remainingCount} events</p> : null}
        </div>

        <div className="action-link-row">
          {reviewSessionId ? (
            <Link href={`/review?session=${reviewSessionId}`} className="button-link-primary">
              `/review` で詳細確認する
            </Link>
          ) : (
            <span className="muted">review session を準備できなかったため、再解析が必要です。</span>
          )}
        </div>
      </div>
    </section>
  );
}
