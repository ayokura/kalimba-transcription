"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { createTranscriptionWithCapture, fetchTunings } from "@/lib/api";
import { createReviewSession, saveReviewSession } from "@/lib/reviewSession";
import { saveReviewAudio } from "@/lib/reviewAudioStore";
import { InstrumentTuning } from "@/lib/types";

type Stage = "idle" | "recording" | "ready" | "analyzing";

export function SimpleHome() {
  const router = useRouter();
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [selectedTuningId, setSelectedTuningId] = useState<string>("");
  const [recording, setRecording] = useState<Blob | null>(null);
  const [recordingSource, setRecordingSource] = useState<"mic" | "file" | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  const selectedTuning = tunings.find((t) => t.id === selectedTuningId) ?? null;

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
  }

  function resetRecording() {
    setRecording(null);
    setRecordingSource(null);
    setStage("idle");
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleAnalyze() {
    if (!recording || !selectedTuning) return;
    setError(null);
    setStage("analyzing");
    try {
      const capture = await createTranscriptionWithCapture(recording, selectedTuning);
      const session = createReviewSession({
        capture,
        acquisitionMode: recordingSource === "file" ? "uploaded_file" : "live_mic",
        notationMode: "score",
        activeEventId: null,
      });
      saveReviewSession(session);
      saveReviewAudio(session.sessionId, capture.audioWav);
      const transactionId = capture.responsePayload.transactionId;
      if (transactionId) {
        router.push(`/score/${transactionId}`);
      } else {
        setError("サーバー保管に失敗しました。もう一度お試しください。");
        setStage("ready");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "採譜に失敗しました。");
      setStage("ready");
    }
  }

  const isRecording = stage === "recording";
  const isAnalyzing = stage === "analyzing";
  const canAnalyze = Boolean(recording && selectedTuning) && !isAnalyzing;

  return (
    <main className="simple-home">
      <h1 className="simple-home-title">カリンバ譜面</h1>

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
        )}
      </section>

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

      {error ? <p className="simple-home-error">{error}</p> : null}
    </main>
  );
}
