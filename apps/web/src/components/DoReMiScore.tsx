"use client";

import type { ScoreEvent, ScoreNote } from "@/lib/types";
import {
  buildScoreLayout,
  estimateLabelWidth,
  eventHeight,
  groupHeight,
  LAYOUT,
  noteLabelFromScoreNote,
  type NoteGroupLayout,
  type NoteLabel,
} from "@/lib/scoreLayout";
import { useMemo, useRef } from "react";

type DoReMiScoreProps = {
  events: ScoreEvent[];
  eventsPerLine?: number;
  activeEventId?: string | null;
  onActiveEventIdChange?: (eventId: string) => void;
  labelFn?: (note: ScoreNote) => NoteLabel;
};

export function DoReMiScore({
  events,
  eventsPerLine = 15,
  activeEventId = null,
  onActiveEventIdChange,
  labelFn,
}: DoReMiScoreProps) {
  const layout = useMemo(
    () => buildScoreLayout(events, eventsPerLine, labelFn),
    [events, eventsPerLine, labelFn],
  );

  const L = LAYOUT;
  const isInteractive = typeof onActiveEventIdChange === "function";
  const containerRef = useRef<HTMLDivElement | null>(null);

  if (events.length === 0) {
    return <p className="empty">イベントがありません。</p>;
  }

  return (
    <div className="doremi-score-container" ref={(el) => { containerRef.current = el; }}>
      <svg
        viewBox={`0 0 ${layout.viewBoxWidth} ${layout.viewBoxHeight}`}
        xmlns="http://www.w3.org/2000/svg"
        style={{
          width: "100%",
          height: "auto",
          fontFamily: "var(--font-heading), Georgia, serif",
          fontSize: L.fontSize,
          color: "var(--ink)",
        }}
      >
        {/* Debug: lineSpacing areas */}
        {layout.lines.map((line, idx) => {
          if (idx === 0) return null;
          const prevLine = layout.lines[idx - 1];
          const gapTop = prevLine.y + prevLine.height;
          const gapHeight = line.y - gapTop;
          return (
            <rect
              key={`gap-${idx}`}
              className="score-debug-overlay"
              x={0} y={gapTop}
              width={layout.viewBoxWidth} height={gapHeight}
              fill="rgba(255, 100, 100, 0.15)"
            />
          );
        })}
        {layout.lines.map((line) => {
          // Debug: find the tallest event in this line
          let maxH = 0;
          let maxIdx = -1;
          line.events.forEach((evt, i) => {
            const h = eventHeight(
              { topGroup: evt.topGroup, bottomGroup: evt.bottomGroup },
              LAYOUT,
            );
            if (h > maxH) { maxH = h; maxIdx = i; }
          });

          return (
          <g key={line.lineIndex} transform={`translate(0, ${line.y})`}>
            {line.lineIndex > 0 && (
              <line
                x1={0}
                y1={-L.lineSpacing / 2}
                x2={layout.viewBoxWidth}
                y2={-L.lineSpacing / 2}
                stroke="var(--muted, #888)"
                strokeWidth={1}
              />
            )}

            {line.events.map((evt, evtIdx) => {
              const isActive = evt.eventId === activeEventId;
              const isDominant = evtIdx === maxIdx;
              const hasBottom = evt.bottomGroup !== null && evt.bottomGroup.noteLabels.length > 0;

              // Per-event connector positioning (symmetric clearance)
              const topCount = evt.topGroup.noteLabels.length;
              const topIsChord = topCount > 1;
              const topLastNote = evt.topGroup.noteLabels[topCount - 1];
              const topLastOctaveBelow = topLastNote ? octaveLinesBelow(topLastNote.octave) : 0;
              // Bottom group: align as if rect exists (consistent positioning)
              const bottomLabels = evt.bottomGroup?.noteLabels ?? [];
              const bottomBelowBaseline = bottomLabels.length > 0
                ? groupStackHeight(bottomLabels) + Math.max(1, octaveLinesBelow(bottomLabels[bottomLabels.length - 1].octave)) * (L.octaveLineGap + 1) + 3
                : 0;
              const bottomStartY = line.height - bottomBelowBaseline;
              const bottomIsChord = bottomLabels.length > 1;

              // Connector Y1: from actual top group rendering position
              const topGroupLastY = L.noteBaselineY + groupStackHeight(evt.topGroup.noteLabels);
              let connectorY1: number;
              if (topIsChord) {
                const { rectBottom } = chordRectBounds(L.noteBaselineY, evt.topGroup.noteLabels, true);
                connectorY1 = rectBottom + L.connectorOvalGap;
              } else {
                connectorY1 = topGroupLastY + L.connectorPadding + topLastOctaveBelow * (L.octaveLineGap + 2);
              }

              // Connector Y2: always use rect-style positioning
              const connectorY2 = hasBottom
                ? chordRectBounds(bottomStartY, bottomLabels, false).rectTop - L.connectorOvalGap
                : 0;

              // eventHeight guarantees connectorY2 >= connectorY1 + connectorMinLength

              return (
                <g
                  key={evt.eventId}
                  data-event-id={evt.eventId}
                  transform={`translate(${evt.x}, 0)`}
                  onClick={isInteractive ? () => onActiveEventIdChange(evt.eventId) : undefined}
                  style={isInteractive ? { cursor: "pointer" } : undefined}
                >
                  {isInteractive && (
                    <rect
                      x={0}
                      y={-4}
                      width={evt.columnWidth}
                      height={line.height + 4}
                      fill="transparent"
                      pointerEvents="all"
                    />
                  )}
                  {isDominant && (
                    <rect
                      className="score-debug-overlay"
                      x={0}
                      y={0}
                      width={evt.columnWidth}
                      height={line.height}
                      fill="rgba(100, 100, 255, 0.10)"
                      pointerEvents="none"
                    />
                  )}
                  {isActive && (
                    <rect
                      x={0}
                      y={-4}
                      width={evt.columnWidth}
                      height={line.height + 4}
                      rx={4}
                      fill="var(--accent-soft, rgba(23, 126, 137, 0.12))"
                      pointerEvents="none"
                    />
                  )}

                  {/* Top group */}
                  <NoteGroup
                    group={evt.topGroup}
                    centerX={evt.columnWidth / 2}
                    startY={L.noteBaselineY}
                    isMelody
                  />

                  {/* Vertical connector */}
                  {hasBottom && (
                    <line
                      x1={evt.columnWidth / 2}
                      y1={connectorY1}
                      x2={evt.columnWidth / 2}
                      y2={connectorY2}
                      stroke="currentColor"
                      strokeWidth={L.connectorStrokeWidth}
                    />
                  )}

                  {/* Bottom group */}
                  {hasBottom && evt.bottomGroup && (
                    <NoteGroup
                      group={evt.bottomGroup}
                      centerX={evt.columnWidth / 2}
                      startY={bottomStartY}
                      isMelody={false}
                    />
                  )}
                </g>
              );
            })}
          </g>
        );})}
      </svg>
      <label className="score-debug-toggle">
        <input
          type="checkbox"
          onChange={(e) => {
            containerRef.current?.classList.toggle("show-debug", e.target.checked);
          }}
        />
        debug overlay
      </label>
    </div>
  );
}

// --- Octave line helpers ---

function octaveLinesAbove(octave: number): number {
  return octave >= 6 ? 2 : octave === 5 ? 1 : 0;
}

function octaveLinesBelow(octave: number): number {
  return octave <= 2 ? 2 : octave === 3 ? 1 : 0;
}

/** Calculate per-note Y offsets within a group (variable spacing per pair) */
function noteYOffsets(labels: NoteLabel[]): number[] {
  if (labels.length === 0) return [];
  const offsets = [0];
  for (let i = 1; i < labels.length; i++) {
    const below = octaveLinesBelow(labels[i - 1].octave);
    const above = octaveLinesAbove(labels[i].octave);
    const extra = below * (LAYOUT.octaveLineGap + 1) + above * (LAYOUT.octaveLineGap + 6);
    const spacing = extra > 0 ? LAYOUT.noteSpacingTight + extra : LAYOUT.noteSpacingNoOctave;
    offsets.push(offsets[i - 1] + spacing);
  }
  return offsets;
}

/** Total stack height for a group */
function groupStackHeight(labels: NoteLabel[]): number {
  if (labels.length <= 1) return 0;
  const offsets = noteYOffsets(labels);
  return offsets[offsets.length - 1];
}

// --- Note group: labels with rounded-rect capsule ---

/** Calculate the bounding rect top/bottom for a chord group.
 *  isMelody: true = top group (octave lines + 1 reserve), false = bottom group (octave lines only) */
function chordRectBounds(startY: number, noteLabels: NoteLabel[], isMelody: boolean) {
  const L = LAYOUT;
  const topOctave = octaveLinesAbove(noteLabels[0].octave);
  const topOctaveReserve = (isMelody ? topOctave + 1 : topOctave) * (L.octaveLineGap + 1);
  const bottomOctaveReserve = Math.max(1, octaveLinesBelow(noteLabels[noteLabels.length - 1].octave)) * (L.octaveLineGap + 1);

  const rectTop = startY - L.fontSize - topOctaveReserve - 1;
  const lastNoteY = startY + groupStackHeight(noteLabels);
  const rectBottom = lastNoteY + bottomOctaveReserve + 3;

  return { rectTop, rectBottom };
}

function NoteGroup({
  group,
  centerX,
  startY,
  isMelody = true,
}: {
  group: NoteGroupLayout;
  centerX: number;
  startY: number;
  isMelody?: boolean;
}) {
  const L = LAYOUT;
  const { noteLabels } = group;
  if (noteLabels.length === 0) return null;

  const isChord = noteLabels.length > 1;
  const offsets = noteYOffsets(noteLabels);
  const maxLabelPx = Math.max(...noteLabels.map((nl) => estimateLabelWidth(nl.baseName))) * L.fontSizeUnit;

  return (
    <g>
      {isChord && (() => {
        const { rectTop, rectBottom } = chordRectBounds(startY, noteLabels, isMelody);
        const rectW = maxLabelPx + L.ovalPaddingX * 2;
        const rectH = rectBottom - rectTop;
        const rx = Math.min(rectW / 2, 10);
        return (
          <rect
            x={centerX - rectW / 2}
            y={rectTop}
            width={rectW}
            height={rectH}
            rx={rx}
            ry={rx}
            fill="none"
            stroke="currentColor"
            strokeWidth={1}
          />
        );
      })()}

      {noteLabels.map((nl, i) => (
        <NoteLabelWithOctave
          key={`${nl.baseName}-${nl.octave}-${i}`}
          noteLabel={nl}
          x={centerX}
          y={startY + offsets[i]}
        />
      ))}
    </g>
  );
}

// --- Single note label with octave line markers ---

function NoteLabelWithOctave({
  noteLabel,
  x,
  y,
}: {
  noteLabel: NoteLabel;
  x: number;
  y: number;
}) {
  const L = LAYOUT;
  const { baseName, octave } = noteLabel;

  const linesAbove = octaveLinesAbove(octave);
  const linesBelow = octaveLinesBelow(octave);
  const halfW = L.octaveLineWidth / 2;

  return (
    <g>
      <CompactNoteText
        baseName={baseName}
        x={x}
        y={y}
        fontSize={L.fontSize}
      />

      {/* Octave lines above */}
      {Array.from({ length: linesAbove }, (_, i) => (
        <line
          key={`above-${i}`}
          x1={x - halfW}
          y1={y - L.fontSize - L.octaveLineOffset - i * L.octaveLineGap}
          x2={x + halfW}
          y2={y - L.fontSize - L.octaveLineOffset - i * L.octaveLineGap}
          stroke="currentColor"
          strokeWidth={1}
        />
      ))}

      {/* Octave lines below */}
      {Array.from({ length: linesBelow }, (_, i) => (
        <line
          key={`below-${i}`}
          x1={x - halfW}
          y1={y + L.octaveLineOffset + 2 + i * L.octaveLineGap}
          x2={x + halfW}
          y2={y + L.octaveLineOffset + 2 + i * L.octaveLineGap}
          stroke="currentColor"
          strokeWidth={1}
        />
      ))}
    </g>
  );
}

// --- Compact note name rendering (ligature-style for ファ etc.) ---

const SMALL_KANA: Record<string, string> = { ァ: "ァ", ィ: "ィ", ゥ: "ゥ", ェ: "ェ", ォ: "ォ" };

function CompactNoteText({
  baseName,
  x,
  y,
  fontSize,
}: {
  baseName: string;
  x: number;
  y: number;
  fontSize: number;
}) {
  const parts = splitLigature(baseName);

  if (!parts) {
    return (
      <text x={x} y={y} textAnchor="middle" fill="currentColor" fontSize={fontSize}>
        {baseName}
      </text>
    );
  }

  const { main, smallKana, accidental } = parts;
  const mainOffset = accidental ? -2 : 0;

  return (
    <text x={x + mainOffset} y={y} textAnchor="middle" fill="currentColor" fontSize={fontSize}>
      {main}
      <tspan fontSize={fontSize * 0.6} dy={2} dx={-2}>
        {smallKana}
      </tspan>
      {accidental && (
        <tspan fontSize={fontSize * 0.8} dy={-2} dx={0}>
          {accidental}
        </tspan>
      )}
    </text>
  );
}

function splitLigature(name: string): { main: string; smallKana: string; accidental: string } | null {
  for (let i = 0; i < name.length; i++) {
    if (SMALL_KANA[name[i]]) {
      const main = name.slice(0, i);
      const smallKana = name[i];
      const accidental = name.slice(i + 1);
      if (main.length > 0) {
        return { main, smallKana, accidental };
      }
    }
  }
  return null;
}
