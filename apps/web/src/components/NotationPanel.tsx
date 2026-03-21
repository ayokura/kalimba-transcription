"use client";

import { NotationMode, TranscriptionResult } from "@/lib/types";

type NotationPanelProps = {
  result: TranscriptionResult | null;
  mode: NotationMode;
  onModeChange: (mode: NotationMode) => void;
};

export function NotationPanel({ result, mode, onModeChange }: NotationPanelProps) {
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
          </div>
          {result.warnings.length > 0 ? (
            <div className="warning-box">
              {result.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
          {mode === "vertical" ? (
            <div className="notation-grid">
              {result.events.map((event, index) => (
                <article key={event.id} className={`event-card ${event.isGlissLike ? "gliss" : ""}`}>
                  <small>#{index + 1}</small>
                  <div className="vertical-stack">
                    {result.notationViews.verticalDoReMi[index].map((note) => (
                      <span key={`${event.id}-${note}`}>{note}</span>
                    ))}
                  </div>
                  <footer>
                    <span>{event.startBeat}拍</span>
                    <span>{event.durationBeat}拍</span>
                  </footer>
                </article>
              ))}
            </div>
          ) : (
            <div className="line-notation">
              {(mode === "numbered" ? result.notationViews.numbered : result.notationViews.western).map((entry, index) => (
                <div key={`${mode}-${index}`} className="line-row">
                  <span className="line-index">{index + 1}</span>
                  <code>{entry}</code>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}