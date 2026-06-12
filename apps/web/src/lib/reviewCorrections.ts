import {
  CorrectionsPayload,
  InstrumentTuning,
  ReviewOrigin,
  ScoreEvent,
  ScoreNote,
  TranscriptionResult,
} from "@/lib/types";

export type ReviewEvent = {
  id: string;
  timeSec: number;
  notes: ScoreNote[];
  origin: ReviewOrigin;
  removed: boolean;
  /** recognizer 出力由来の場合は元 event id (alternateGroupings 参照用) */
  sourceEventId: string | null;
};

export type ReviewState = {
  events: ReviewEvent[];
  nextInsertId: number;
};

const TIME_MATCH_TOLERANCE_SEC = 0.005;

const DOREMI_BY_PITCH_CLASS: Record<string, string> = {
  C: "ド",
  "C#": "ド#",
  D: "レ",
  "D#": "レ#",
  E: "ミ",
  F: "ファ",
  "F#": "ファ#",
  G: "ソ",
  "G#": "ソ#",
  A: "ラ",
  "A#": "ラ#",
  B: "シ",
};

function sortNotes(notes: ScoreNote[]): ScoreNote[] {
  return [...notes].sort((a, b) => a.frequency - b.frequency);
}

function sortEvents(events: ReviewEvent[]): ReviewEvent[] {
  return [...events].sort((a, b) => a.timeSec - b.timeSec);
}

/** result 内に登場する全 ScoreNote (events / alternates / candidateSlots) を noteName で索引化する。 */
export function buildKnownNoteIndex(result: TranscriptionResult): Map<string, ScoreNote> {
  const index = new Map<string, ScoreNote>();
  const register = (note: ScoreNote | null | undefined) => {
    if (!note) return;
    const name = `${note.pitchClass}${note.octave}`;
    if (!index.has(name)) index.set(name, note);
  };
  for (const event of result.events) {
    event.notes.forEach(register);
    for (const alt of event.alternateGroupings ?? []) {
      register(alt.alternateNote);
      alt.combinedNotes?.forEach(register);
      alt.splitInto?.forEach((group) => group.forEach(register));
    }
  }
  for (const slot of result.candidateSlots ?? []) {
    register(slot.primaryNote);
    slot.candidates.forEach(register);
  }
  return index;
}

export function noteName(note: ScoreNote): string {
  return `${note.pitchClass}${note.octave}`;
}

/**
 * noteName から ScoreNote を解決する。result 内に既出ならそれを使い、
 * 未出の音は tuning 定義から構築する (labelNumber はサーバー定義が
 * tuning スケール依存のため、未出音では空にする — review UI は固定ド表示)。
 */
export function resolveScoreNote(
  name: string,
  knownNotes: Map<string, ScoreNote>,
  tuning: InstrumentTuning,
): ScoreNote | null {
  const known = knownNotes.get(name);
  if (known) return known;
  const parsed = /^([A-G]#?)(\d)$/.exec(name);
  if (!parsed) return null;
  const tuningNote = tuning.notes.find((n) => n.noteName === name);
  if (!tuningNote) return null;
  const pitchClass = parsed[1];
  const octave = Number(parsed[2]);
  return {
    key: tuningNote.key,
    pitchClass,
    octave,
    labelDoReMi: DOREMI_BY_PITCH_CLASS[pitchClass] ?? pitchClass,
    labelNumber: "",
    frequency: tuningNote.frequency,
  };
}

export function buildInitialState(result: TranscriptionResult): ReviewState {
  return {
    events: sortEvents(
      result.events.map((event) => ({
        id: event.id,
        timeSec: event.startTimeSec,
        notes: sortNotes(event.notes),
        origin: "recognizer" as const,
        removed: false,
        sourceEventId: event.id,
      })),
    ),
    nextInsertId: 1,
  };
}

/**
 * 保存済み corrections から状態を復元する。corrections は「最終形のイベント列」
 * なので、recognizer イベントとは timeSec で突合する:
 * - 一致した correction → その recognizer イベント枠に correction の notes を採用
 * - 一致しなかった recognizer イベント → removed
 * - どの recognizer イベントとも一致しない correction → 挿入イベント
 */
export function restoreStateFromCorrections(
  result: TranscriptionResult,
  corrections: CorrectionsPayload,
): ReviewState {
  const knownNotes = buildKnownNoteIndex(result);
  const consumed = new Set<number>();
  const events: ReviewEvent[] = [];
  let nextInsertId = 1;

  const matchCorrection = (timeSec: number): number | null => {
    for (let i = 0; i < corrections.events.length; i += 1) {
      if (consumed.has(i)) continue;
      if (Math.abs(corrections.events[i].timeSec - timeSec) <= TIME_MATCH_TOLERANCE_SEC) {
        return i;
      }
    }
    return null;
  };

  for (const event of result.events) {
    const matchIndex = matchCorrection(event.startTimeSec);
    if (matchIndex === null) {
      events.push({
        id: event.id,
        timeSec: event.startTimeSec,
        notes: sortNotes(event.notes),
        origin: "recognizer",
        removed: true,
        sourceEventId: event.id,
      });
      continue;
    }
    consumed.add(matchIndex);
    const correction = corrections.events[matchIndex];
    const notes = correction.notes
      .map((name) => resolveScoreNote(name, knownNotes, result.instrumentTuning))
      .filter((note): note is ScoreNote => note !== null);
    events.push({
      id: event.id,
      timeSec: event.startTimeSec,
      notes: sortNotes(notes.length > 0 ? notes : event.notes),
      origin: correction.origin ?? "recognizer",
      removed: false,
      sourceEventId: event.id,
    });
  }

  corrections.events.forEach((correction, index) => {
    if (consumed.has(index)) return;
    const notes = correction.notes
      .map((name) => resolveScoreNote(name, knownNotes, result.instrumentTuning))
      .filter((note): note is ScoreNote => note !== null);
    if (notes.length === 0) return;
    events.push({
      id: `ins-${nextInsertId}`,
      timeSec: correction.timeSec,
      notes: sortNotes(notes),
      origin: correction.origin ?? "inserted-manual",
      removed: false,
      sourceEventId: null,
    });
    nextInsertId += 1;
  });

  return { events: sortEvents(events), nextInsertId };
}

function withEvent(
  state: ReviewState,
  eventId: string,
  update: (event: ReviewEvent) => ReviewEvent,
): ReviewState {
  return {
    ...state,
    events: state.events.map((event) => (event.id === eventId ? update(event) : event)),
  };
}

function markEdited(event: ReviewEvent): ReviewOrigin {
  return event.origin === "recognizer" ? "edited" : event.origin;
}

export function removeNote(state: ReviewState, eventId: string, name: string): ReviewState {
  return withEvent(state, eventId, (event) => {
    if (event.notes.length <= 1) return event;
    const notes = event.notes.filter((note) => noteName(note) !== name);
    if (notes.length === event.notes.length) return event;
    return { ...event, notes, origin: markEdited(event) };
  });
}

export function addNote(state: ReviewState, eventId: string, note: ScoreNote): ReviewState {
  return withEvent(state, eventId, (event) => {
    if (event.notes.some((existing) => noteName(existing) === noteName(note))) return event;
    return { ...event, notes: sortNotes([...event.notes, note]), origin: markEdited(event) };
  });
}

export function toggleRemoved(state: ReviewState, eventId: string): ReviewState {
  return withEvent(state, eventId, (event) => ({ ...event, removed: !event.removed }));
}

export function insertEvent(
  state: ReviewState,
  timeSec: number,
  notes: ScoreNote[],
  origin: ReviewOrigin,
): ReviewState {
  if (notes.length === 0) return state;
  const id = `ins-${state.nextInsertId}`;
  const event: ReviewEvent = {
    id,
    timeSec,
    notes: sortNotes(notes),
    origin,
    removed: false,
    sourceEventId: null,
  };
  return {
    events: sortEvents([...state.events, event]),
    nextInsertId: state.nextInsertId + 1,
  };
}

export function activeEvents(state: ReviewState): ReviewEvent[] {
  return state.events.filter((event) => !event.removed);
}

export function toCorrectionsPayload(state: ReviewState): CorrectionsPayload {
  return {
    version: 1,
    events: activeEvents(state).map((event) => ({
      timeSec: event.timeSec,
      notes: event.notes.map(noteName),
      origin: event.origin,
    })),
  };
}

/** DoReMiScore 表示用に ReviewEvent を ScoreEvent へ写像する。 */
export function toDisplayScoreEvents(state: ReviewState): ScoreEvent[] {
  return activeEvents(state).map((event, index) => ({
    id: event.id,
    startBeat: index,
    durationBeat: 1,
    startTimeSec: event.timeSec,
    notes: event.notes,
    isGlissLike: false,
    gesture: "ambiguous",
  }));
}
