"use client";

import type { CSSProperties } from "react";

import { noteName } from "@/lib/reviewCorrections";
import { ScoreNote } from "@/lib/types";

type KalimbaNotePickerProps = {
  /** 物理配置順 (tuning.notes の並びそのまま) の音一覧 */
  notes: ScoreNote[];
  /** タップ不可にする noteName の集合 (イベント内の既存音など) */
  disabledNames?: Set<string>;
  onPick: (note: ScoreNote) => void;
  /** role="group" の aria-label */
  label: string;
};

/**
 * カリンバ実機の鍵盤レイアウト (中央最長 = 最低音、左右交互に音階が上がる)
 * を再現した note picker。ExpectedKeySelector の Range Map と同じ視覚言語
 * (kalimba-key rail physical) の compact 版で、review 編集 UI 向け。
 */
export function KalimbaNotePicker({ notes, disabledNames, onPick, label }: KalimbaNotePickerProps) {
  if (notes.length === 0) return null;

  const centerIndex = Math.floor(notes.length / 2);
  const maxDistance = Math.max(centerIndex, notes.length - centerIndex - 1);
  const railStyle = { "--kalimba-key-count": String(notes.length) } as CSSProperties;

  return (
    <div className="review-note-picker kalimba-rail-viewport">
      <div
        className="kalimba-keys rail physical compact"
        role="group"
        aria-label={label}
        style={railStyle}
      >
        {notes.map((note, index) => {
          const distance = Math.abs(index - centerIndex);
          const tineHeight = 58 + (maxDistance - distance) * 7;
          const name = noteName(note);
          const disabled = disabledNames?.has(name) ?? false;
          return (
            <button
              key={`${note.key}-${name}`}
              type="button"
              className={`kalimba-key rail physical compact${distance === 0 ? " center" : ""}`}
              style={{ "--tine-height": `${tineHeight}px` } as CSSProperties}
              onClick={() => onPick(note)}
              disabled={disabled}
              title={`Key ${note.key}: ${name}`}
            >
              <span className="kalimba-key-doremi">{note.labelDoReMi}</span>
              <span className="kalimba-key-spacer" />
              <span className="kalimba-key-note horizontal compact">{name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
