import type { ScoreEvent, ScoreNote } from "@/lib/types";

// --- Pitch class to do-re-mi base name ---

const PITCH_TO_DOREMI: Record<string, string> = {
  C: "ド", "C#": "ド#", Db: "レb", D: "レ", "D#": "レ#",
  Eb: "ミb", E: "ミ", F: "ファ", "F#": "ファ#", Gb: "ソb",
  G: "ソ", "G#": "ソ#", Ab: "ラb", A: "ラ", "A#": "ラ#",
  Bb: "シb", B: "シ",
};

const PITCH_CLASS_TO_SEMITONE: Record<string, number> = {
  C: 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3,
  E: 4, F: 5, "F#": 6, Gb: 6, G: 7, "G#": 8,
  Ab: 8, A: 9, "A#": 10, Bb: 10, B: 11,
};

// Movable-do syllables indexed by semitone distance from tonic.
// Non-diatonic notes use sharp forms (e.g. ド# = raised tonic).
const MOVABLE_DO_SYLLABLES = [
  "ド", "ド#", "レ", "レ#", "ミ", "ファ",
  "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ",
];

export type NoteLabel = {
  baseName: string;   // e.g. "ド", "ファ#", "シb"
  octave: number;     // 3, 4, 5, 6 — determines octave line markers
};

export function noteLabelFromScoreNote(note: ScoreNote): NoteLabel {
  return {
    baseName: PITCH_TO_DOREMI[note.pitchClass] ?? note.pitchClass,
    octave: note.octave,
  };
}

// Returns a labelFn that renders notes in movable-do relative to the given tonic.
// Octave dots are anchored to the tonic's octave-4 reference: the octave that
// contains the tonic at octave 4 shows no dots; each octave above adds `.`,
// each below adds `_`.  Falls back to fixed-do if the tonic is unknown.
export function movableDoLabelFn(
  tonicPitchClass: string | null | undefined,
): (note: ScoreNote) => NoteLabel {
  if (!tonicPitchClass) return noteLabelFromScoreNote;
  const tonicPc = PITCH_CLASS_TO_SEMITONE[tonicPitchClass];
  if (tonicPc == null) return noteLabelFromScoreNote;
  return (note: ScoreNote) => {
    const pc = PITCH_CLASS_TO_SEMITONE[note.pitchClass];
    if (pc == null) {
      return { baseName: note.pitchClass, octave: note.octave };
    }
    const semitonesFromTonic4 = (note.octave - 4) * 12 + (pc - tonicPc);
    const scaleOctave = Math.floor(semitonesFromTonic4 / 12);
    const interval = ((semitonesFromTonic4 % 12) + 12) % 12;
    return {
      baseName: MOVABLE_DO_SYLLABLES[interval],
      octave: 4 + scaleOctave,
    };
  };
}

// --- Label width estimation ---

const WIDE_CHARS = new Set(["フ", "ァ"]);

export function estimateLabelWidth(label: string): number {
  let w = 0;
  for (const ch of label) {
    if (ch === "#" || ch === "b") w += 0.5;
    else if (WIDE_CHARS.has(ch)) w += 0.6;
    else w += 1.0;
  }
  return w;
}

// --- Adjacency-based note grouping ---

export type NoteGroupLayout = {
  noteLabels: NoteLabel[];
};

export type EventGrouping = {
  topGroup: NoteGroupLayout;
  bottomGroup: NoteGroupLayout | null;
};

/**
 * Group notes by physical key adjacency.
 * The group containing the primary (highest-frequency) note goes on top.
 * Non-adjacent notes form the bottom group.
 */
export function groupNotesByAdjacency(
  notes: ScoreNote[],
  labelFn: (note: ScoreNote) => NoteLabel,
): EventGrouping {
  if (notes.length === 0) {
    return { topGroup: { noteLabels: [] }, bottomGroup: null };
  }
  if (notes.length === 1) {
    return { topGroup: { noteLabels: [labelFn(notes[0])] }, bottomGroup: null };
  }

  // Find primary note (highest frequency)
  const primary = notes.reduce((a, b) => (b.frequency > a.frequency ? b : a));

  // Sort by key position and group by adjacency
  const sorted = [...notes].sort((a, b) => a.key - b.key);
  const groups: ScoreNote[][] = [];
  let currentGroup: ScoreNote[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].key - sorted[i - 1].key === 1) {
      currentGroup.push(sorted[i]);
    } else {
      groups.push(currentGroup);
      currentGroup = [sorted[i]];
    }
  }
  groups.push(currentGroup);

  // Find which group contains the primary note
  const primaryGroupIdx = groups.findIndex((g) => g.some((n) => n.key === primary.key));
  const primaryGroup = groups[primaryGroupIdx];
  const otherNotes = groups.filter((_, i) => i !== primaryGroupIdx).flat();

  // Labels sorted by frequency descending (highest on top)
  const topLabels = [...primaryGroup].sort((a, b) => b.frequency - a.frequency).map(labelFn);
  const bottomLabels = otherNotes.length > 0
    ? [...otherNotes].sort((a, b) => b.frequency - a.frequency).map(labelFn)
    : null;

  return {
    topGroup: { noteLabels: topLabels },
    bottomGroup: bottomLabels ? { noteLabels: bottomLabels } : null,
  };
}

// --- Layout constants ---

export const LAYOUT = {
  marginLeft: 24,
  marginTop: 20,
  marginRight: 24,
  marginBottom: 20,
  noteBaselineY: 26,
  noteSpacing: 20,
  noteSpacingTight: 16,
  noteSpacingNoOctave: 19,
  connectorPadding: 5,
  connectorOvalGap: 3,
  connectorMinLength: 16,
  ovalPaddingX: 8,
  ovalPaddingY: 6,
  eventMinWidth: 36,
  eventPadding: 6,
  lineSpacing: 16,
  fontSize: 15,
  fontSizeUnit: 12,
  connectorStrokeWidth: 1,
  lineMinHeight: 80,
  octaveLineWidth: 7,
  octaveLineGap: 4,
  octaveLineOffset: 2,
} as const;

// --- Layout data structures ---

export type EventLayout = {
  eventId: string;
  eventIndex: number;
  topGroup: NoteGroupLayout;
  bottomGroup: NoteGroupLayout | null;
  x: number;
  columnWidth: number;
};

export type LineLayout = {
  lineIndex: number;
  events: EventLayout[];
  y: number;
  height: number;
  width: number;
};

export type ScoreLayout = {
  lines: LineLayout[];
  viewBoxWidth: number;
  viewBoxHeight: number;
};

// --- Per-event height calculation ---

/** Total stack height using per-pair variable spacing */
function stackHeightForLabels(labels: NoteLabel[], L: { noteSpacingTight: number; noteSpacingNoOctave: number; octaveLineGap: number }): number {
  if (labels.length <= 1) return 0;
  let total = 0;
  for (let i = 1; i < labels.length; i++) {
    const below = labels[i - 1].octave <= 2 ? 2 : labels[i - 1].octave === 3 ? 1 : 0;
    const above = labels[i].octave >= 6 ? 2 : labels[i].octave === 5 ? 1 : 0;
    const extra = below * (L.octaveLineGap + 1) + above * (L.octaveLineGap + 6);
    total += extra > 0 ? L.noteSpacingTight + extra : L.noteSpacingNoOctave;
  }
  return total;
}

export function groupHeight(group: NoteGroupLayout, isMelody: boolean, L: { noteBaselineY: number; noteSpacing: number; noteSpacingTight: number; noteSpacingNoOctave: number; octaveLineGap: number; ovalPaddingY: number; fontSize: number }): number {
  const n = group.noteLabels.length;
  if (n === 0) return 0;

  const topOctave = group.noteLabels[0].octave >= 6 ? 2 : group.noteLabels[0].octave === 5 ? 1 : 0;
  const bottomOctave = (() => { const o = group.noteLabels[n - 1].octave; return o <= 2 ? 2 : o === 3 ? 1 : 0; })();
  const topOctaveReserve = (isMelody ? topOctave + 1 : topOctave) * (L.octaveLineGap + 1);

  if (n === 1) {
    const bottomReserve = Math.max(1, bottomOctave) * (L.octaveLineGap + 1);
    return L.fontSize + topOctaveReserve + bottomReserve + 4;
  }

  // Chord (2+): rect needs consistent bottom, so min 1 reserve
  const bottomOctaveReserve = Math.max(1, bottomOctave) * (L.octaveLineGap + 1);
  const stackH = stackHeightForLabels(group.noteLabels, L);
  return L.fontSize + topOctaveReserve + 1 + stackH + bottomOctaveReserve + 3;
}

/** Content extent below the last note's baseline */
function belowBaseline(labels: NoteLabel[], isChord: boolean, L: typeof LAYOUT): number {
  const lastOct = labels[labels.length - 1].octave;
  const below = lastOct <= 2 ? 2 : lastOct === 3 ? 1 : 0;
  if (isChord) {
    const reserve = Math.max(1, below) * (L.octaveLineGap + 1);
    return reserve + 3; // matches rectBottom offset from lastNoteY
  }
  return L.connectorPadding + below * (L.octaveLineGap + 2);
}

/** Content extent above the first note's baseline */
function aboveBaseline(labels: NoteLabel[], isMelody: boolean, isChord: boolean, L: typeof LAYOUT): number {
  const firstOct = labels[0].octave;
  const above = firstOct >= 6 ? 2 : firstOct === 5 ? 1 : 0;
  if (isChord) {
    const reserve = (isMelody ? above + 1 : above) * (L.octaveLineGap + 1);
    return L.fontSize + reserve + 1; // matches rectTop offset from startY
  }
  return L.fontSize + L.connectorPadding + above * (L.octaveLineGap + 2);
}

export function eventHeight(
  e: { topGroup: NoteGroupLayout; bottomGroup: NoteGroupLayout | null },
  L: typeof LAYOUT,
): number {
  const topH = groupHeight(e.topGroup, true, L);
  if (!e.bottomGroup || e.bottomGroup.noteLabels.length === 0) return topH;
  const bottomH = groupHeight(e.bottomGroup, false, L);

  const topIsChord = e.topGroup.noteLabels.length > 1;
  const bottomIsChord = e.bottomGroup.noteLabels.length > 1;

  // Renderer positions (bottom aligned by rect-bottom, not groupHeight):
  //   connectorY1 = noteBaselineY + topStackH + topBelow + gap
  //   connectorY2 = (lineH - bottomBelowBL) - bottomAboveRendering
  //   connectorY2 - connectorY1 >= connectorMinLength
  //
  // Solving for lineH:
  //   lineH >= connectorY1 + minLen + bottomAboveRendering + bottomBelowBL

  const topStackH = stackHeightForLabels(e.topGroup.noteLabels, L);
  const topBelow = belowBaseline(e.topGroup.noteLabels, topIsChord, L);
  // Always use rect-style positioning for bottom (even single notes)
  const bottomLabels = e.bottomGroup.noteLabels;
  const bottomAboveRect = aboveBaseline(bottomLabels, false, true, L); // always treat as chord
  const bottomBelowBL = stackHeightForLabels(bottomLabels, L)
    + Math.max(1, ((): number => { const o = bottomLabels[bottomLabels.length - 1].octave; return o <= 2 ? 2 : o === 3 ? 1 : 0; })()) * (L.octaveLineGap + 1) + 3;

  const fromConnector = L.noteBaselineY + topStackH + topBelow + L.connectorOvalGap
                      + L.connectorMinLength
                      + L.connectorOvalGap + bottomAboveRect + bottomBelowBL;

  return Math.max(topH, fromConnector);
}

// --- Main layout builder ---

export function buildScoreLayout(
  events: ScoreEvent[],
  eventsPerLine: number,
  labelFn: (note: ScoreNote) => NoteLabel = noteLabelFromScoreNote,
): ScoreLayout {
  const L = LAYOUT;

  // Group notes and compute column widths
  const prepared = events.map((event, idx) => {
    const grouping = groupNotesByAdjacency(event.notes, labelFn);

    const allLabels = [
      ...grouping.topGroup.noteLabels,
      ...(grouping.bottomGroup?.noteLabels ?? []),
    ];
    const maxLabelW = allLabels.length > 0
      ? Math.max(...allLabels.map((nl) => estimateLabelWidth(nl.baseName)))
      : 0;
    const contentWidth = maxLabelW * L.fontSizeUnit;
    const columnWidth = Math.max(contentWidth + L.eventPadding * 2, L.eventMinWidth);

    return {
      eventId: event.id,
      eventIndex: idx,
      topGroup: grouping.topGroup,
      bottomGroup: grouping.bottomGroup,
      columnWidth,
    };
  });

  // Split into lines
  const lineChunks: (typeof prepared[number])[][] = [];
  for (let i = 0; i < prepared.length; i += eventsPerLine) {
    lineChunks.push(prepared.slice(i, i + eventsPerLine));
  }

  // Build line layouts
  const lines: LineLayout[] = [];
  let currentY = L.marginTop;

  for (let li = 0; li < lineChunks.length; li++) {
    const chunk = lineChunks[li];

    // Per-event height calculation, take the max across the line
    const lineHeight = Math.max(
      ...chunk.map((e) => eventHeight(e, L)),
      L.lineMinHeight,
    );

    let x = L.marginLeft;
    const eventLayouts: EventLayout[] = chunk.map((e) => {
      const layout: EventLayout = {
        eventId: e.eventId,
        eventIndex: e.eventIndex,
        topGroup: e.topGroup,
        bottomGroup: e.bottomGroup,
        x,
        columnWidth: e.columnWidth,
      };
      x += e.columnWidth;
      return layout;
    });

    lines.push({
      lineIndex: li,
      events: eventLayouts,
      y: currentY,
      height: lineHeight,
      width: x + L.marginRight,
    });

    currentY += lineHeight + L.lineSpacing;
  }

  const viewBoxWidth = lines.length > 0
    ? Math.max(...lines.map((l) => l.width))
    : L.marginLeft + L.marginRight;
  const viewBoxHeight = currentY - L.lineSpacing + L.marginBottom;

  return { lines, viewBoxWidth, viewBoxHeight };
}
