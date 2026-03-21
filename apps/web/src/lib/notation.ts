import { ScoreEvent, ScoreNote, TranscriptionResult } from "@/lib/types";

const NOTE_NAME_PATTERN = /^([A-Ga-g])([#b]?)(-?\d+)$/;

export function rebuildNotationViews(events: ScoreEvent[]) {
  return {
    western: events.map((event) => event.notes.map((note) => `${note.pitchClass}${note.octave}`).join(" | ")),
    numbered: events.map((event) => event.notes.map((note) => note.labelNumber).join(" ")),
    verticalDoReMi: events.map((event) => event.notes.map((note) => note.labelDoReMi)),
  };
}

export function cloneResultWithEvents(result: TranscriptionResult, events: ScoreEvent[]): TranscriptionResult {
  return {
    ...result,
    events,
    notationViews: rebuildNotationViews(events),
  };
}

export function noteFromName(noteName: string, key: number): ScoreNote {
  const match = noteName.trim().match(NOTE_NAME_PATTERN);
  const pitchClass = `${match?.[1]?.toUpperCase() ?? "C"}${match?.[2] ?? ""}`;
  const octave = Number(match?.[3] ?? "4");
  const doremiMap: Record<string, string> = {
    C: "ド",
    "C#": "ド#",
    Db: "レb",
    D: "レ",
    "D#": "レ#",
    Eb: "ミb",
    E: "ミ",
    F: "ファ",
    "F#": "ファ#",
    Gb: "ソb",
    G: "ソ",
    "G#": "ソ#",
    Ab: "ラb",
    A: "ラ",
    "A#": "ラ#",
    Bb: "シb",
    B: "シ",
  };
  const numberMap: Record<string, string> = {
    C: "1",
    "C#": "#1",
    Db: "b2",
    D: "2",
    "D#": "#2",
    Eb: "b3",
    E: "3",
    F: "4",
    "F#": "#4",
    Gb: "b5",
    G: "5",
    "G#": "#5",
    Ab: "b6",
    A: "6",
    "A#": "#6",
    Bb: "b7",
    B: "7",
  };

  return {
    key,
    pitchClass,
    octave,
    labelDoReMi: doremiWithOctave(doremiMap[pitchClass] ?? pitchClass, octave),
    labelNumber: numberWithOctave(numberMap[pitchClass] ?? pitchClass, octave),
    frequency: 0,
  };
}

function doremiWithOctave(base: string, octave: number) {
  if (octave >= 6) {
    return `${base}..`;
  }
  if (octave === 5) {
    return `${base}.`;
  }
  if (octave === 3) {
    return `_${base}`;
  }
  if (octave <= 2) {
    return `__${base}`;
  }
  return base;
}

function numberWithOctave(base: string, octave: number) {
  if (octave >= 6) {
    return `${base}''`;
  }
  if (octave === 5) {
    return `${base}'`;
  }
  if (octave === 3) {
    return `.${base}`;
  }
  if (octave <= 2) {
    return `..${base}`;
  }
  return base;
}