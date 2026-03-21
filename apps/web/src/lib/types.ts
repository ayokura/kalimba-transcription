export type TuningNote = {
  key: number;
  noteName: string;
  frequency: number;
};

export type InstrumentTuning = {
  id: string;
  name: string;
  keyCount: number;
  notes: TuningNote[];
};

export type ScoreNote = {
  key: number;
  pitchClass: string;
  octave: number;
  labelDoReMi: string;
  labelNumber: string;
  frequency: number;
};

export type ScoreEvent = {
  id: string;
  startBeat: number;
  durationBeat: number;
  notes: ScoreNote[];
  isGlissLike: boolean;
};

export type NotationViews = {
  western: string[];
  numbered: string[];
  verticalDoReMi: string[][];
};

export type TranscriptionResult = {
  instrumentTuning: InstrumentTuning;
  tempo: number;
  events: ScoreEvent[];
  notationViews: NotationViews;
  warnings: string[];
  debug?: Record<string, unknown> | null;
};

export type NotationMode = "vertical" | "numbered" | "western";
