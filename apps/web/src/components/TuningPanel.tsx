"use client";

import { InstrumentTuning, TuningNote } from "@/lib/types";

type TuningPanelProps = {
  tunings: InstrumentTuning[];
  selectedId: string;
  onSelect: (id: string) => void;
  customName: string;
  customNotes: string;
  hasProtectedDraft?: boolean;
  onCustomNameChange: (value: string) => void;
  onCustomNotesChange: (value: string) => void;
};

export function TuningPanel(props: TuningPanelProps) {
  const {
    tunings,
    selectedId,
    onSelect,
    customName,
    customNotes,
    hasProtectedDraft = false,
    onCustomNameChange,
    onCustomNotesChange,
  } = props;

  const selected = tunings.find((tuning) => tuning.id === selectedId);

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Tuning</p>
          <h2>調律設定</h2>
        </div>
      </div>
      <label className="stack">
        <span>プリセット</span>
        <select value={selectedId} onChange={(event) => onSelect(event.target.value)}>
          {tunings.map((tuning) => (
            <option key={tuning.id} value={tuning.id}>
              {tuning.name}
            </option>
          ))}
          <option value="custom">Custom</option>
        </select>
      </label>
      {hasProtectedDraft ? (
        <p className="muted">
          expected performance の下書きがあります。調律を変更すると、確認のうえで下書きと解析 snapshot を破棄します。
        </p>
      ) : null}

      {selectedId === "custom" ? (
        <div className="stack gap-lg">
          <label className="stack">
            <span>カスタム名</span>
            <input value={customName} onChange={(event) => onCustomNameChange(event.target.value)} />
          </label>
          <label className="stack">
            <span>音名をカンマ区切りで入力</span>
            <textarea
              rows={4}
              value={customNotes}
              onChange={(event) => onCustomNotesChange(event.target.value)}
              placeholder="C4,D4,E4,G4,A4,C5,E5"
            />
          </label>
        </div>
      ) : (
        <div className="tuning-grid">
          {selected?.notes.map((note: TuningNote) => (
            <div key={note.key} className="tuning-card">
              <strong>{note.key}</strong>
              <span>{note.noteName}</span>
              <small>{note.frequency.toFixed(1)} Hz</small>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
