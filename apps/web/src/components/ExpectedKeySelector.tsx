"use client";

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

  const centerIndex = tuning.notes.reduce((bestIndex, note, index, notes) => {
    if (note.frequency <= notes[bestIndex].frequency) {
      return index;
    }
    return bestIndex;
  }, 0);

  const selectedNotes = tuning.notes.filter((note) => selectedKeys.includes(note.key));
  const summary = selectedNotes.length > 0 ? selectedNotes.map((note) => note.noteName).join(" + ") : "未選択";
  const normalizedRepeatCount = repeatCount.trim().length > 0 ? repeatCount.trim() : "1";

  return (
    <div className="stack gap-lg">
      <div className="kalimba-selector">
        <div className="kalimba-keys" role="group" aria-label="Expected kalimba keys">
          {tuning.notes.map((note, index) => {
            const distance = Math.abs(index - centerIndex);
            const depth = Math.max(0, 7 - distance);
            const isSelected = selectedKeys.includes(note.key);
            return (
              <button
                key={note.key}
                type="button"
                className={`kalimba-key${isSelected ? " selected" : ""}`}
                style={{ height: `${92 + depth * 12}px` }}
                onClick={() => onToggleKey(note.key)}
                aria-pressed={isSelected}
                title={`Key ${note.key}: ${note.noteName}`}
              >
                <span className="kalimba-key-number">{note.key}</span>
                <span className="kalimba-key-note">{note.noteName}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="row wrap expected-builder-row">
        <label className="stack expected-repeat-field">
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
          {`${summary} を ${normalizedRepeatCount} 回追加`}
        </button>
        <button type="button" className="ghost" onClick={onClearSelection} disabled={selectedNotes.length === 0}>
          選択をクリア
        </button>
      </div>

      <div className="stack">
        <span>選択中の和音</span>
        <div className="expected-summary-box">
          <strong>{summary}</strong>
          <span>{` x ${normalizedRepeatCount}`}</span>
        </div>
        <p className="muted">クリックしたキーを 1 イベントとして期待演奏シーケンスへ追加します。</p>
      </div>
    </div>
  );
}
