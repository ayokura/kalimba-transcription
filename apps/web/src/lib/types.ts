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
  tonic?: string | null;
};

export type ScoreNote = {
  key: number;
  pitchClass: string;
  octave: number;
  labelDoReMi: string;
  labelNumber: string;
  frequency: number;
};

export type AlternateGrouping = {
  combinesWith: string[] | null;
  combinedNotes: ScoreNote[] | null;
  splitInto: ScoreNote[][] | null;
  alternateNote: ScoreNote | null;
  reason: string;
  confidence: number;
};

export type CandidateSlot = {
  startTime: number;
  endTime: number;
  primaryNote: ScoreNote;
  candidates: ScoreNote[];
  dropReason: string;
  confidence: number;
};

export type ScoreEvent = {
  id: string;
  startBeat: number;
  durationBeat: number;
  startTimeSec: number;
  durationSec?: number | null;
  notes: ScoreNote[];
  isGlissLike: boolean;
  gesture: string;
  alternateGroupings?: AlternateGrouping[] | null;
};

export type ReviewOrigin = "recognizer" | "edited" | "inserted-slot" | "inserted-manual";

export type CorrectionEventPayload = {
  timeSec: number;
  notes: string[];
  origin: ReviewOrigin;
  /** この onset に主旋律が含まれない (伴奏のみ)。GT 昇格時に role: accompaniment */
  accompanimentOnly?: boolean;
};

export type CorrectionsPayload = {
  version: 1;
  updatedAt?: string | null;
  events: CorrectionEventPayload[];
};

export type ReviewStatusValue =
  | "recorded_only"
  | "review_started"
  | "review_completed"
  | "rerecord_needed"
  | "unusable"
  | "uncertain";

export type ReviewStatusPayload = {
  version: 1;
  status: ReviewStatusValue;
  note?: string | null;
  reviewer?: string | null;
  updatedAt?: string | null;
};

export type ReviewQueueEntry = {
  transactionId: string;
  createdAt: number;
  tuningId: string | null;
  tuningName: string | null;
  eventCount: number;
  audioSha256: string | null;
  reviewStatus: ReviewStatusValue | null;
  reviewStatusUpdatedAt: string | null;
  hasCorrections: boolean;
  hasMemo: boolean;
  warningCount: number;
  candidateSlotCount: number;
};

export type NotationViews = {
  western: string[];
  numbered: string[];
  verticalDoReMi: string[][];
};

export type TuningMismatch = {
  selectedCoverage: number;
  outsidePitchClasses: string[];
  suggestedTuningId?: string | null;
  suggestedTuningName?: string | null;
  suggestedCoverage?: number | null;
};

export type TranscriptionResult = {
  transactionId?: string | null;
  instrumentTuning: InstrumentTuning;
  tempo: number;
  events: ScoreEvent[];
  candidateSlots?: CandidateSlot[];
  notationViews: NotationViews;
  tuningMismatch?: TuningMismatch | null;
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
