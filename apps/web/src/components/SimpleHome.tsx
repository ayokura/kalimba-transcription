"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  createTranscriptionWithCapture,
  fetchRecentTranscriptions,
  fetchTunings,
  lookupTranscriptionByHash,
} from "@/lib/api";
import {
  computeAudioLevels,
  computeBlobSha256Hex,
  MIC_AUDIO_CONSTRAINTS,
  type AudioLevels,
} from "@/lib/audio";
import { createReviewSession, saveReviewSession } from "@/lib/reviewSession";
import { saveReviewAudio } from "@/lib/reviewAudioStore";
import {
  loadRecentTranscriptions,
  pushRecentTranscription,
  removeRecentTranscription,
  type RecentTranscription,
} from "@/lib/recentTranscriptions";
import {
  clearPendingRecordings,
  deletePendingRecording,
  loadLatestPendingRecording,
  savePendingRecording,
  type PendingRecording,
} from "@/lib/pendingRecordingStore";
import { InstrumentTuning } from "@/lib/types";

type Stage = "idle" | "recording" | "ready" | "analyzing";

// -15 dB: 実データ較正 (2da2e1ac peak -17.4dB で 3 events の実質失敗を捕捉、
// ad0b0a57 peak -8.1dB の正常録音は誤警告しない)
const LOW_LEVEL_PEAK_DB = -15;
const ANALYZE_RETRY_PROMPT_MS = 20_000;

type DedupPrompt = {
  transactionId: string;
};

export function SimpleHome() {
  const router = useRouter();
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [selectedTuningId, setSelectedTuningId] = useState<string>("");
  const [recording, setRecording] = useState<Blob | null>(null);
  const [recordingSource, setRecordingSource] = useState<"mic" | "file" | null>(null);
  const [audioLevels, setAudioLevels] = useState<AudioLevels | null>(null);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentTranscription[]>([]);
  const [dedupPrompt, setDedupPrompt] = useState<DedupPrompt | null>(null);
  const [analyzeElapsed, setAnalyzeElapsed] = useState(0);
  const [pendingRestore, setPendingRestore] = useState<PendingRecording | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const analyzeRequestIdRef = useRef(0);
  const pendingIdRef = useRef<string | null>(null);

  // タブクラッシュ対策: 録音/WAV を IndexedDB にバックアップし、採譜成功で削除する。
  const backupRecording = useCallback(
    (blob: Blob, source: "mic" | "file", tuningId: string | null) => {
      const id =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : `pending-${Math.random().toString(36).slice(2)}`;
      pendingIdRef.current = id;
      savePendingRecording({ id, blob, source, tuningId, createdAt: Date.now() }).catch(() => {
        // バックアップ失敗は主導線を妨げない
        pendingIdRef.current = null;
      });
    },
    [],
  );

  const discardBackup = useCallback(() => {
    const id = pendingIdRef.current;
    pendingIdRef.current = null;
    if (id) deletePendingRecording(id).catch(() => {});
  }, []);

  useEffect(() => {
    loadLatestPendingRecording()
      .then((entry) => {
        if (entry) setPendingRestore(entry);
      })
      .catch(() => {
        // IndexedDB が使えない環境では復元プロンプトを出さない
      });
  }, []);

  useEffect(() => {
    setRecent(loadRecentTranscriptions());
    fetchRecentTranscriptions(10)
      .then((serverRecent) => {
        const existing = loadRecentTranscriptions();
        const existingIds = new Set(existing.map((e) => e.transactionId));
        let changed = false;
        for (const s of serverRecent) {
          if (existingIds.has(s.transactionId)) continue;
          pushRecentTranscription({
            transactionId: s.transactionId,
            createdAt: new Date(s.createdAt * 1000).toISOString(),
            tuningName: s.tuningName ?? "unknown",
            eventCount: s.eventCount,
          });
          changed = true;
        }
        if (changed) setRecent(loadRecentTranscriptions());
      })
      .catch(() => {
        // ignore server recent errors
      });
  }, []);

  // 録音直後に聴き直せるよう、保持中の blob を再生用 object URL にする。
  // blob が変わる / クリアされるたびに古い URL を解放してリークを防ぐ。
  useEffect(() => {
    if (!recording) {
      setPlaybackUrl(null);
      return;
    }
    const url = URL.createObjectURL(recording);
    setPlaybackUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [recording]);

  useEffect(() => {
    if (!recording) {
      setAudioLevels(null);
      return;
    }
    let cancelled = false;
    computeAudioLevels(recording)
      .then((levels) => {
        if (!cancelled) setAudioLevels(levels);
      })
      .catch(() => {
        if (!cancelled) setAudioLevels(null);
      });
    return () => {
      cancelled = true;
    };
  }, [recording]);

  useEffect(() => {
    fetchTunings()
      .then((list) => {
        setTunings(list);
        if (list[0]) setSelectedTuningId(list[0].id);
      })
      .catch(() => setError("チューニング情報の取得に失敗しました。"));
  }, []);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    if (stage !== "analyzing") {
      setAnalyzeElapsed(0);
      return;
    }
    const start = Date.now();
    const id = window.setInterval(() => {
      setAnalyzeElapsed(Date.now() - start);
    }, 500);
    return () => window.clearInterval(id);
  }, [stage]);

  const selectedTuning = tunings.find((t) => t.id === selectedTuningId) ?? null;

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia(MIC_AUDIO_CONSTRAINTS);
      streamRef.current = stream;
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setRecording(blob);
        setRecordingSource("mic");
        setStage("ready");
        backupRecording(blob, "mic", selectedTuningId || null);
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setStage("recording");
    } catch {
      setError("マイクの利用許可が必要です。ブラウザ設定を確認してください。");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
  }

  function handleFilePick(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setRecording(file);
    setRecordingSource("file");
    setStage("ready");
    setError(null);
    backupRecording(file, "file", selectedTuningId || null);
  }

  function resetRecording() {
    setRecording(null);
    setRecordingSource(null);
    setStage("idle");
    setError(null);
    setDedupPrompt(null);
    discardBackup();
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleRestorePending() {
    if (!pendingRestore) return;
    setRecording(pendingRestore.blob);
    setRecordingSource(pendingRestore.source);
    setStage("ready");
    setError(null);
    pendingIdRef.current = pendingRestore.id;
    if (pendingRestore.tuningId && tunings.some((t) => t.id === pendingRestore.tuningId)) {
      setSelectedTuningId(pendingRestore.tuningId);
    }
    setPendingRestore(null);
  }

  function handleDiscardPending() {
    setPendingRestore(null);
    clearPendingRecordings().catch(() => {});
  }

  const runTranscription = useCallback(
    async (blob: Blob, tuning: InstrumentTuning, source: "mic" | "file" | null, force: boolean) => {
      const requestId = ++analyzeRequestIdRef.current;
      setStage("analyzing");
      setError(null);
      try {
        const capture = await createTranscriptionWithCapture(blob, tuning, { force });
        if (requestId !== analyzeRequestIdRef.current) return; // superseded
        const session = createReviewSession({
          capture,
          acquisitionMode: source === "file" ? "uploaded_file" : "live_mic",
          notationMode: "score",
          activeEventId: null,
        });
        saveReviewSession(session);
        saveReviewAudio(session.sessionId, capture.audioWav);
        const transactionId = capture.responsePayload.transactionId;
        if (transactionId) {
          discardBackup();
          pushRecentTranscription({
            transactionId,
            createdAt: new Date().toISOString(),
            tuningName: tuning.name,
            eventCount: capture.responsePayload.events.length,
          });
          router.push(`/score/${transactionId}`);
        } else {
          setError("サーバー保管に失敗しました。もう一度お試しください。");
          setStage("ready");
        }
      } catch (err) {
        if (requestId !== analyzeRequestIdRef.current) return;
        setError(err instanceof Error ? err.message : "採譜に失敗しました。");
        setStage("ready");
      }
    },
    [router, discardBackup],
  );

  async function handleAnalyze() {
    if (!recording || !selectedTuning) return;
    setError(null);
    setDedupPrompt(null);
    try {
      const hash = await computeBlobSha256Hex(recording);
      const existing = await lookupTranscriptionByHash(hash, selectedTuning.id);
      if (existing) {
        setDedupPrompt({ transactionId: existing });
        return;
      }
    } catch {
      // ignore hash/lookup failures and proceed to POST
    }
    await runTranscription(recording, selectedTuning, recordingSource, false);
  }

  async function handleForceRerun() {
    if (!recording || !selectedTuning) return;
    setDedupPrompt(null);
    await runTranscription(recording, selectedTuning, recordingSource, true);
  }

  function handleOpenExisting(transactionId: string) {
    setDedupPrompt(null);
    router.push(`/score/${transactionId}`);
  }

  async function handleResend() {
    if (!recording || !selectedTuning) return;
    await runTranscription(recording, selectedTuning, recordingSource, false);
  }

  const isRecording = stage === "recording";
  const isAnalyzing = stage === "analyzing";
  const canAnalyze = Boolean(recording && selectedTuning) && !isAnalyzing;
  const showRetryPrompt = isAnalyzing && analyzeElapsed >= ANALYZE_RETRY_PROMPT_MS;

  return (
    <main className="simple-home">
      <h1 className="simple-home-title">カリンバ譜面</h1>

      {pendingRestore && !recording ? (
        <div className="simple-home-pending" role="dialog" aria-label="pending-recording-prompt">
          <p className="simple-home-pending-text">
            送信されていない録音があります ({formatRelativeTime(new Date(pendingRestore.createdAt).toISOString())})。
            復元して採譜しますか?
          </p>
          <div className="simple-home-pending-actions">
            <button type="button" className="simple-home-btn primary" onClick={handleRestorePending}>
              復元する
            </button>
            <button type="button" className="simple-home-btn ghost" onClick={handleDiscardPending}>
              破棄する
            </button>
          </div>
        </div>
      ) : null}

      <section className="simple-home-step">
        <label className="simple-home-label" htmlFor="simple-home-tuning">
          調律
        </label>
        <select
          id="simple-home-tuning"
          className="simple-home-select"
          value={selectedTuningId}
          onChange={(e) => setSelectedTuningId(e.target.value)}
          disabled={isAnalyzing}
        >
          {tunings.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </section>

      <section className="simple-home-step">
        <p className="simple-home-label">演奏を用意</p>
        {!recording ? (
          <div className="simple-home-record-row">
            {!isRecording ? (
              <>
                <button
                  type="button"
                  className="simple-home-btn primary"
                  onClick={startRecording}
                  disabled={isAnalyzing}
                >
                  録音する
                </button>
                <label className="simple-home-btn secondary" htmlFor="simple-home-file">
                  WAV をアップロード
                </label>
                <input
                  id="simple-home-file"
                  ref={fileInputRef}
                  type="file"
                  accept="audio/wav,audio/x-wav"
                  onChange={handleFilePick}
                  hidden
                />
              </>
            ) : (
              <button type="button" className="simple-home-btn primary" onClick={stopRecording}>
                録音を停止
              </button>
            )}
          </div>
        ) : (
          <div className="simple-home-recording-ready">
            <div className="simple-home-ready-head">
              <p className="simple-home-ready-text">
                {recordingSource === "file" ? "WAV を選択しました。" : "録音を保持しています。"}
              </p>
              <button
                type="button"
                className="simple-home-btn ghost"
                onClick={resetRecording}
                disabled={isAnalyzing}
              >
                やり直す
              </button>
            </div>
            {playbackUrl ? (
              <audio
                className="simple-home-audio"
                src={playbackUrl}
                controls
                preload="metadata"
                aria-label="録音の聴き直し"
              />
            ) : null}
          </div>
        )}
      </section>

      {audioLevels && recording && audioLevels.peakDb < LOW_LEVEL_PEAK_DB ? (
        <p className="simple-home-warning">
          録音の音量が小さすぎます (ピーク {audioLevels.peakDb.toFixed(1)} dB)。
          iOS では外部オーディオインターフェースが使えず、内蔵マイクで遠くから拾っている可能性があります。
          マイクを近づけるか、PC で録音した WAV を「WAV をアップロード」から読み込ませてください。
        </p>
      ) : null}

      {dedupPrompt ? (
        <div className="simple-home-dedup" role="dialog" aria-label="dedup-prompt">
          <p className="simple-home-dedup-text">
            この録音は以前採譜済みです。既存の結果を開きますか? それとも改めて採譜しますか?
          </p>
          <div className="simple-home-dedup-actions">
            <button
              type="button"
              className="simple-home-btn primary"
              onClick={() => handleOpenExisting(dedupPrompt.transactionId)}
            >
              結果を開く
            </button>
            <button
              type="button"
              className="simple-home-btn secondary"
              onClick={handleForceRerun}
            >
              改めて採譜
            </button>
          </div>
        </div>
      ) : null}

      <section className="simple-home-step">
        <button
          type="button"
          className="simple-home-btn primary large"
          onClick={handleAnalyze}
          disabled={!canAnalyze}
        >
          {isAnalyzing ? "採譜中…" : "自動採譜する"}
        </button>
      </section>

      {showRetryPrompt ? (
        <div className="simple-home-retry">
          <p className="simple-home-retry-text">
            時間がかかっています。通信トラブルで届いていない可能性があります。
          </p>
          <button type="button" className="simple-home-btn secondary" onClick={handleResend}>
            再送信する
          </button>
        </div>
      ) : null}

      {error ? <p className="simple-home-error">{error}</p> : null}

      {recent.length > 0 ? (
        <section className="simple-home-recent">
          <div className="simple-home-recent-head">
            <p className="simple-home-label">これまでの採譜</p>
            <button
              type="button"
              className="simple-home-queue-link"
              onClick={() => router.push("/review/queue")}
            >
              確認キューを開く →
            </button>
          </div>
          <ul className="simple-home-recent-list">
            {recent.map((entry) => (
              <li key={entry.transactionId} className="simple-home-recent-item">
                <button
                  type="button"
                  className="simple-home-recent-btn"
                  onClick={() => router.push(`/score/${entry.transactionId}`)}
                >
                  <span className="simple-home-recent-primary">
                    {formatRelativeTime(entry.createdAt)} · {entry.tuningName}
                  </span>
                  <span className="simple-home-recent-secondary">
                    {entry.eventCount} イベント
                  </span>
                </button>
                <button
                  type="button"
                  className="simple-home-recent-remove"
                  aria-label="履歴から削除"
                  onClick={() => {
                    removeRecentTranscription(entry.transactionId);
                    setRecent(loadRecentTranscriptions());
                  }}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </main>
  );
}

function formatRelativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const diffSec = Math.floor((Date.now() - then) / 1000);
  if (diffSec < 60) return "たった今";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} 分前`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)} 時間前`;
  if (diffSec < 86400 * 7) return `${Math.floor(diffSec / 86400)} 日前`;
  const d = new Date(iso);
  return `${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()}`;
}
