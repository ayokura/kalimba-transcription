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
  // S5 agenda 2 (#141): recognizer gate が棄却の代わりに低 confidence 降格した
  // event。値は降格理由 (例 "onset-gate-no-evidence")。absent/null = 通常。
  lowConfidenceReason?: string | null;
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
  // #204 Phase 3: which recognition run these corrections were made against
  // ("legacy", or a runId from GET .../runs). Absent/null for corrections
  // saved before this field existed.
  baseRunId?: string | null;
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
  // #194 (S6): recognizer の内部 difficulty 自己評価 (0-1)。表示はせず
  // queue の優先度ソートにのみ使う。旧 payload では null/undefined。
  qualityDifficulty?: number | null;
  qualityFlag?: string | null;
  // #204 Phase 2: 表示中の response が保存された時点の recognizer fingerprint と、
  // それが現行 recognizer と異なるか (再認識対象の目印)。fingerprint 不明な
  // 旧録音では isStale は null (安全側: stale と決めつけない)。
  recognizerFingerprint?: string | null;
  isStale?: boolean | null;
};

// #204 Phase 1/2: 1 件の認識実行 (recognition run)。isLegacy=true はアップロード時
// スナップショット (response.json) を表す合成エントリで、ranAt は null。
export type RecognitionRun = {
  runId: string;
  commitSha: string | null;
  recognizerFingerprint: string | null;
  dspFingerprint: string | null;
  ranAt: string | null;
  eventCount: number;
  isLegacy: boolean;
};

export type RecognitionRunsResponse = {
  runs: RecognitionRun[];
  latestRunId: string | null;
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
