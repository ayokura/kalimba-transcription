"use client";

import type { ScoreEvent, ScoreNote } from "@/lib/types";
import { noteLabelFromScoreNote, type NoteLabel } from "@/lib/scoreLayout";
import { useMemo } from "react";

// #202 案 A' (拍グリッド) のプロトタイプ。ユーザーの手書き譜規約:
//   - 紙面は基本音長 (= 1 拍) の等間隔グリッド。1 拍 = 1 カラム
//   - 伸ばし・休符は空白カラム (空白そのものが情報)
//   - 小節線 (4 拍ごと、当面 4/4 固定) が拍の集計を保証する
//   - 基本より短い音は例外としてカラム内に詰める (ラップ)
// dev 確認用 (/score?notation=beatgrid)。判定材料であり最終実装ではない。

const BEATS_PER_MEASURE = 4;

type Cell = {
  beat: number;
  events: ScoreEvent[]; // 0 = 空白 (伸ばし/休符)、2+ = ラップ
};

function buildCells(events: ScoreEvent[]): Cell[] {
  if (events.length === 0) return [];
  const maxBeat = Math.max(...events.map((e) => e.startBeat + Math.max(e.durationBeat, 0)));
  const totalBeats = Math.max(1, Math.ceil(maxBeat));
  const cells: Cell[] = Array.from({ length: totalBeats }, (_, beat) => ({ beat, events: [] }));
  for (const event of events) {
    const idx = Math.min(Math.max(Math.floor(event.startBeat), 0), totalBeats - 1);
    cells[idx].events.push(event);
  }
  return cells;
}

function labelText(label: NoteLabel): string {
  // プロトタイプの簡易オクターブ表現: 上点 = ゛相当の「・」前置/後置ではなく
  // 手書き規約と同じ上下点。ここではテキスト 1 行で「ド˙」(上) /「ド̣」(下)。
  const dots = label.octave >= 6 ? "˙˙" : label.octave === 5 ? "˙" : "";
  const unders = label.octave <= 2 ? "̣̣" : label.octave === 3 ? "̣" : "";
  return `${label.baseName}${dots}${unders}`;
}

export function BeatGridScore({
  events,
  activeEventId = null,
  onActiveEventIdChange,
  labelFn = noteLabelFromScoreNote,
}: {
  events: ScoreEvent[];
  activeEventId?: string | null;
  onActiveEventIdChange?: (eventId: string) => void;
  labelFn?: (note: ScoreNote) => NoteLabel;
}) {
  const cells = useMemo(() => buildCells(events), [events]);
  if (cells.length === 0) {
    return <p className="empty">イベントがありません。</p>;
  }
  const interactive = typeof onActiveEventIdChange === "function";
  return (
    <div className="beatgrid-score" role="img" aria-label="拍グリッド譜 (プロトタイプ)">
      {cells.map((cell) => {
        const measureStart = cell.beat % BEATS_PER_MEASURE === 0;
        const wrap = cell.events.length > 1;
        return (
          <span key={cell.beat} className="beatgrid-cell-wrap">
            {measureStart && <span className="beatgrid-barline" aria-hidden />}
            <span
              className={`beatgrid-cell${wrap ? " wrap" : ""}${cell.events.length === 0 ? " empty" : ""}`}
            >
              {cell.events.map((event) => (
                <span
                  key={event.id}
                  className={`beatgrid-event${event.id === activeEventId ? " active" : ""}`}
                  onClick={interactive ? () => onActiveEventIdChange?.(event.id) : undefined}
                  style={interactive ? { cursor: "pointer" } : undefined}
                >
                  {[...event.notes]
                    .sort((a, b) => b.frequency - a.frequency)
                    .map((note, i) => (
                      <span key={`${note.pitchClass}${note.octave}-${i}`} className="beatgrid-note">
                        {labelText(labelFn(note))}
                      </span>
                    ))}
                </span>
              ))}
            </span>
          </span>
        );
      })}
      <span className="beatgrid-barline" aria-hidden />
    </div>
  );
}
