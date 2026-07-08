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
  /** この onset に主旋律が含まれない (伴奏のみ)。GT 昇格時に role: accompaniment */
  accompanimentOnly: boolean;
  /** recognizer 出力由来の場合は元 event id (alternateGroupings 参照用) */
  sourceEventId: string | null;
};

export type ReviewState = {
  events: ReviewEvent[];
  nextInsertId: number;
};

const TIME_MATCH_TOLERANCE_SEC = 0.005;
// 再採譜 (recognizer 改善後) では event の startTimeSec が数十 ms 単位でずれうる
// (backtrack / segment 形成の変化)。保存済み corrections がそれで全滅しないよう、
// tight 突合で残ったペアはこの窓内で最近傍から束ねる (50ms は
// note_f1_benchmark の既定 onset tolerance と同値)。
const TIME_REMATCH_TOLERANCE_SEC = 0.05;

// サーバー側 PITCH_CLASS_TO_DOREMI (apps/api/app/transcription/constants.py) と同じ表記
const DOREMI_BY_PITCH_CLASS: Record<string, string> = {
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
  const parsed = /^([A-G][#b]?)(\d)$/.exec(name);
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
        accompanimentOnly: false,
        sourceEventId: event.id,
      })),
    ),
    nextInsertId: 1,
  };
}

/**
 * recognizer イベントと corrections の突合。2 段階:
 * 1. tight (±5ms): 同一 transcription 由来の完全一致
 * 2. relaxed (±50ms): 再採譜で onset がずれたペアを、全候補ペアの
 *    |Δt| 昇順 (最近傍優先) で貪欲に束ねる — 密集区間で隣のイベントに
 *    誤って束ねられるのを防ぐため、イベント順ではなく距離順で確定する
 */
function matchCorrectionsToEvents(
  events: { startTimeSec: number }[],
  corrections: CorrectionsPayload,
): Map<number, number> {
  const assignment = new Map<number, number>();
  const usedCorrections = new Set<number>();

  for (const tolerance of [TIME_MATCH_TOLERANCE_SEC, TIME_REMATCH_TOLERANCE_SEC]) {
    const pairs: { eventIndex: number; correctionIndex: number; dt: number }[] = [];
    events.forEach((event, eventIndex) => {
      if (assignment.has(eventIndex)) return;
      corrections.events.forEach((correction, correctionIndex) => {
        if (usedCorrections.has(correctionIndex)) return;
        const dt = Math.abs(correction.timeSec - event.startTimeSec);
        if (dt <= tolerance) pairs.push({ eventIndex, correctionIndex, dt });
      });
    });
    pairs.sort((a, b) => a.dt - b.dt);
    for (const pair of pairs) {
      if (assignment.has(pair.eventIndex) || usedCorrections.has(pair.correctionIndex)) continue;
      assignment.set(pair.eventIndex, pair.correctionIndex);
      usedCorrections.add(pair.correctionIndex);
    }
  }
  return assignment;
}

/**
 * 保存済み corrections から状態を復元する。corrections は「最終形のイベント列」
 * なので、recognizer イベントとは timeSec で突合する:
 * - 一致した correction → その recognizer イベント枠に correction の notes を採用
 * - 一致しなかった recognizer イベント → removed
 * - どの recognizer イベントとも一致しない correction → 挿入イベント
 *
 * 突合は tight (±5ms) → relaxed (±50ms、最近傍優先) の 2 段。再採譜で
 * recognizer の onset が少しずれても、保存済み corrections が
 * 「全 removed + 全挿入」に化けないための頑健化 (2026-07-02 監査)。
 */
export function restoreStateFromCorrections(
  result: TranscriptionResult,
  corrections: CorrectionsPayload,
): ReviewState {
  const knownNotes = buildKnownNoteIndex(result);
  const events: ReviewEvent[] = [];
  let nextInsertId = 1;

  const assignment = matchCorrectionsToEvents(result.events, corrections);
  const consumed = new Set<number>(assignment.values());

  for (const [eventIndex, event] of result.events.entries()) {
    const matchIndex = assignment.get(eventIndex) ?? null;
    if (matchIndex === null) {
      events.push({
        id: event.id,
        timeSec: event.startTimeSec,
        notes: sortNotes(event.notes),
        origin: "recognizer",
        removed: true,
        accompanimentOnly: false,
        sourceEventId: event.id,
      });
      continue;
    }
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
      accompanimentOnly: correction.accompanimentOnly ?? false,
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
      accompanimentOnly: correction.accompanimentOnly ?? false,
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
  // no-op 更新では state の参照を保つ (undo 履歴に空編集を積まないため)
  let changed = false;
  const events = state.events.map((event) => {
    if (event.id !== eventId) return event;
    const updated = update(event);
    if (updated !== event) changed = true;
    return updated;
  });
  return changed ? { ...state, events } : state;
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

/**
 * イベント内の 1 音を別の音へ置き換える (ワンタップ置換)。
 * removeNote の「最後の 1 音は消せない」ガードに縛られないので、
 * 単音イベントでも add → remove の 2 操作を経ずに置換できる。
 * 置換先が既に含まれている場合は旧音の除去に縮退する (全消しには
 * しない — イベント自体の削除は toggleRemoved の責務)。
 */
export function replaceNote(
  state: ReviewState,
  eventId: string,
  oldName: string,
  note: ScoreNote,
): ReviewState {
  return withEvent(state, eventId, (event) => {
    if (!event.notes.some((existing) => noteName(existing) === oldName)) return event;
    if (oldName === noteName(note)) return event;
    if (event.notes.some((existing) => noteName(existing) === noteName(note))) {
      const remaining = event.notes.filter((existing) => noteName(existing) !== oldName);
      if (remaining.length === 0) return event;
      return { ...event, notes: sortNotes(remaining), origin: markEdited(event) };
    }
    const notes = event.notes.map((existing) =>
      noteName(existing) === oldName ? note : existing,
    );
    return { ...event, notes: sortNotes(notes), origin: markEdited(event) };
  });
}

/** 既存イベントの timeSec を書き換える (0 秒未満はクランプ)。時刻順を保つため再ソートする。 */
export function setEventTime(state: ReviewState, eventId: string, timeSec: number): ReviewState {
  const clamped = Math.max(0, timeSec);
  const target = state.events.find((event) => event.id === eventId);
  if (!target || target.timeSec === clamped) return state;
  const events = state.events.map((event) =>
    event.id === eventId ? { ...event, timeSec: clamped, origin: markEdited(event) } : event,
  );
  return { ...state, events: sortEvents(events) };
}

export function toggleRemoved(state: ReviewState, eventId: string): ReviewState {
  return withEvent(state, eventId, (event) => ({ ...event, removed: !event.removed }));
}

/**
 * merge/split サジェスト (#16 §4.2) の適用。
 * splitEvent: イベントを notes グループ×時刻に分割する (最初のグループは
 * 元イベントを編集、残りは inserted-manual ではなく edited 由来の新規行として挿入)。
 */
export function splitEvent(
  state: ReviewState,
  eventId: string,
  groups: ScoreNote[][],
  times: number[],
): ReviewState {
  if (groups.length < 2 || groups.length !== times.length) return state;
  const target = state.events.find((e) => e.id === eventId);
  if (!target || target.removed) return state;
  let nextInsertId = state.nextInsertId;
  const newEvents: ReviewEvent[] = [];
  for (let i = 1; i < groups.length; i += 1) {
    if (groups[i].length === 0) return state;
    newEvents.push({
      id: `ins-${nextInsertId}`,
      timeSec: Math.max(0, times[i]),
      notes: sortNotes(groups[i]),
      origin: "edited",
      removed: false,
      accompanimentOnly: false,
      sourceEventId: null,
    });
    nextInsertId += 1;
  }
  const events = state.events.map((e) =>
    e.id === eventId
      ? { ...e, notes: sortNotes(groups[0]), timeSec: Math.max(0, times[0]), origin: "edited" as const }
      : e,
  );
  return { events: sortEvents([...events, ...newEvents]), nextInsertId };
}

/** merge サジェストの適用: 対象イベントの notes を統合結果に置換し、相方を削除状態にする */
export function applyMergeSuggestion(
  state: ReviewState,
  eventId: string,
  combinedNotes: ScoreNote[],
  combinesWithIds: string[],
): ReviewState {
  if (combinedNotes.length === 0) return state;
  const withIds = new Set(combinesWithIds);
  let changed = false;
  const events = state.events.map((e) => {
    if (e.id === eventId && !e.removed) {
      changed = true;
      return { ...e, notes: sortNotes(combinedNotes), origin: "edited" as const };
    }
    if (withIds.has(e.id) && !e.removed) {
      changed = true;
      return { ...e, removed: true };
    }
    return e;
  });
  return changed ? { events, nextInsertId: state.nextInsertId } : state;
}

export function toggleAccompanimentOnly(state: ReviewState, eventId: string): ReviewState {
  return withEvent(state, eventId, (event) => ({
    ...event,
    accompanimentOnly: !event.accompanimentOnly,
  }));
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
    accompanimentOnly: false,
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

/**
 * timeSec にアクティブなイベントが存在するか。candidateSlot の表示抑制に使う
 * (保存済み corrections から復元された inserted-slot イベントも含めて判定するため、
 * セッションローカルな挿入記録ではなく state 自体を見る)。
 */
export function hasActiveEventAt(
  state: ReviewState,
  timeSec: number,
  toleranceSec = TIME_MATCH_TOLERANCE_SEC,
): boolean {
  return state.events.some(
    (event) => !event.removed && Math.abs(event.timeSec - timeSec) <= toleranceSec,
  );
}

/**
 * baseRunId (#204 Phase 3) は「保存時にどの run に対する差分か」を記録するもので、
 * dirty 判定用の payload には含めたくない (内容が同じなら run だけで dirty 扱いに
 * しないため)。そのため付与は ReviewEditor の保存時スプレッド 1 箇所に集約し、この
 * 関数では扱わない (#209 review: 未使用の任意引数による二重機構を排除)。
 */
export function toCorrectionsPayload(state: ReviewState): CorrectionsPayload {
  return {
    version: 1,
    events: activeEvents(state).map((event) => ({
      timeSec: event.timeSec,
      notes: event.notes.map(noteName),
      origin: event.origin,
      ...(event.accompanimentOnly ? { accompanimentOnly: true } : {}),
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
