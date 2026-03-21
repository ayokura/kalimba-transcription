"use client";

import type { CSSProperties } from "react";

import { InstrumentTuning } from "@/lib/types";

type ExpectedKeySelectorProps = {
  tuning: InstrumentTuning | null;
  selectedKeys: number[];
  repeatCount: string;
  onToggleKey: (key: number) => void;
  onRepeatCountChange: (value: string) => void;
  onCommitSelection: () => void;
  onClearSelection: () => void;
};

export function ExpectedKeySelector(props: ExpectedKeySelectorProps) {
  const { tuning, selectedKeys, repeatCount, onToggleKey, onRepeatCountChange, onCommitSelection, onClearSelection } = props;

  if (!tuning || tuning.notes.length === 0) {
    return <p className="muted">調律を選ぶと、ここで期待演奏のキーを選択できます。</p>;
  }

  const centerIndex = Math.floor(tuning.notes.length / 2);
  const maxDistance = Math.max(centerIndex, tuning.notes.length - centerIndex - 1);
  const selectedNotes = tuning.notes.filter((note) => selectedKeys.includes(note.key));
  const summary = selectedNotes.length > 0 ? selectedNotes.map((note) => note.noteName).join(" + ") : "キーを選択してください";
  const normalizedRepeatCount = repeatCount.trim().length > 0 ? repeatCount.trim() : "1";
  const railStyle = { "--kalimba-key-count": String(tuning.notes.length) } as CSSProperties;

  return (
    <div className="expected-composer stack gap-lg">
      <div className="phrase-rail-card">
        <div className="phrase-rail-copy">
          <span className="eyebrow">Phrase Rail</span>
          <strong>いま作っている期待イベント</strong>
          <p className="muted">複数キーを選ぶと和音として扱います。追加回数を設定してからレールへ確定します。</p>
        </div>
        <div className={`phrase-rail-preview${selectedNotes.length > 0 ? " active" : ""}`}>
          <div className="phrase-rail-chip">
            <span className="phrase-rail-label">Current Event</span>
            <strong>{summary}</strong>
          </div>
          <div className="phrase-rail-repeat">
            <span>Repeat</span>
            <strong>x {normalizedRepeatCount}</strong>
          </div>
        </div>
      </div>

      <div className="composer-toolbar row wrap">
        <label className="stack expected-repeat-field compact-field">
          <span>追加回数</span>
          <input
            inputMode="numeric"
            pattern="[0-9]*"
            value={repeatCount}
            onChange={(event) => onRepeatCountChange(event.target.value.replace(/[^0-9]/g, ""))}
            placeholder="1"
          />
        </label>
        <button type="button" className="primary" onClick={onCommitSelection} disabled={selectedNotes.length === 0}>
          イベントを追加
        </button>
        <button type="button" className="ghost" onClick={onClearSelection} disabled={selectedNotes.length === 0}>
          選択をクリア
        </button>
      </div>

      <div className="kalimba-selector enhanced">
        <div className="kalimba-selector-header">
          <div>
            <span className="eyebrow">Range Map</span>
            <strong>カリンバのキーから選択</strong>
          </div>
          <span className="muted">物理配置順 / wide では横レール表示</span>
        </div>

        <div className="kalimba-rail-viewport">
          <div className="kalimba-keys rail physical" role="group" aria-label="Expected kalimba keys" style={railStyle}>
            {tuning.notes.map((note, index) => {
              const distance = Math.abs(index - centerIndex);
              const heightWeight = maxDistance - distance;
              const tineHeight = 126 + heightWeight * 18;
              const isSelected = selectedKeys.includes(note.key);
              return (
                <button
                  key={note.key}
                  type="button"
                  className={`kalimba-key rail physical${isSelected ? " selected" : ""}${distance === 0 ? " center" : ""}`}
                  style={{ "--tine-height": `${tineHeight}px` } as CSSProperties}
                  onClick={() => onToggleKey(note.key)}
                  aria-pressed={isSelected}
                  title={`Key ${note.key}: ${note.noteName}`}
                >
                  <span className="kalimba-key-number">#{note.key}</span>
                  <span className="kalimba-key-spacer" />
                  <span className="kalimba-key-note horizontal">{note.noteName}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
