"use client";

import { useEffect, useRef, useState } from "react";

type RecorderPanelProps = {
  disabled?: boolean;
  onRecordingReady: (blob: Blob | null) => void;
};

export function RecorderPanel({ disabled, onRecordingReady }: RecorderPanelProps) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function startRecording() {
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        onRecordingReady(blob);
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setError("マイクの利用許可が必要です。ブラウザ設定を確認してください。");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Recorder</p>
          <h2>マイク録音</h2>
        </div>
        <span className={`pill ${isRecording ? "live" : ""}`}>{isRecording ? "Recording" : "Ready"}</span>
      </div>
      <p className="muted">
        録音後に解析 API へ送信します。リアルタイム採譜ではなく、録音を止めたあとに変換する方式です。
      </p>
      <div className="row">
        <button className="primary" onClick={startRecording} disabled={Boolean(disabled) || isRecording}>
          録音開始
        </button>
        <button className="secondary" onClick={stopRecording} disabled={!isRecording}>
          録音停止
        </button>
        <button className="ghost" onClick={() => onRecordingReady(null)} disabled={isRecording}>
          録音をクリア
        </button>
      </div>
      {error ? <p className="error">{error}</p> : null}
    </section>
  );
}