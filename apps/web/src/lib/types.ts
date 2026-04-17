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
  startTimeSec: number;
  notes: ScoreNote[];
  isGlissLike: boolean;
  gesture: string;
};

export type NotationViews = {
  western: string[];
  numbered: string[];
  verticalDoReMi: string[][];
};

export type TranscriptionResult = {
  transactionId?: string | null;
  instrumentTuning: InstrumentTuning;
  tempo: number;
  events: ScoreEvent[];
  notationViews: NotationViews;
  warnings: string[];
  debug?: Record<string, unknown> | null;
};

export type NotationMode = "vertical" | "numbered" | "western" | "score";

export type AcquisitionMode = "live_mic" | "uploaded_file";

export type InstrumentProfileRef = {
  id: string | null;
  name: string | null;
};

export type RecordingProfileRef = {
  id: string | null;
  name: string | null;
};

export type ReviewRequestSnapshot = {
  capturedAt: string;
  scenario: string;
  expectedNote: string | null;
  expectedPerformance: unknown;
  memo: string | null;
  captureIntent: string | null;
  sourceProfile: string;
  midPerformanceStart?: boolean;
  midPerformanceEnd?: boolean;
  tuning: InstrumentTuning;
  audio: {
    sampleRate: number;
    channels: number;
    durationSec: number;
    mimeType: string;
    sizeBytes: number;
  };
};

export type ReviewEditedDraft = {
  result: TranscriptionResult;
  updatedAt: string;
} | null;

export type TranscriptionReviewSession = {
  sessionVersion: 1;
  sessionId: string;
  transactionId: string | null;
  createdAt: string;
  acquisitionMode: AcquisitionMode;
  tuning: InstrumentTuning;
  instrumentProfile: InstrumentProfileRef | null;
  recordingProfile: RecordingProfileRef | null;
  requestSnapshot: ReviewRequestSnapshot;
  responseSnapshot: TranscriptionResult;
  notationMode: NotationMode;
  activeEventId: string | null;
  editedDraft: ReviewEditedDraft;
};
