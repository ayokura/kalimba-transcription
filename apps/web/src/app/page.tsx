"use client";

import { useEffect, useMemo, useState } from "react";

import { EditorPanel } from "@/components/EditorPanel";
import { ExpectedKeySelector } from "@/components/ExpectedKeySelector";
import { NotationPanel } from "@/components/NotationPanel";
import { RecorderPanel } from "@/components/RecorderPanel";
import { TuningPanel } from "@/components/TuningPanel";
import {
  ManualCaptureExpectedEvent,
  ManualCaptureExpectedPerformance,
  TranscriptionCapture,
  createTranscriptionWithCapture,
  fetchTunings,
} from "@/lib/api";
import { buildCaptureArchive, downloadBlob } from "@/lib/archive";
import { InstrumentTuning, NotationMode, TranscriptionResult, TuningNote } from "@/lib/types";

function buildCustomTuning(customName: string, customNotes: string): InstrumentTuning {
  const notes = customNotes
    .split(",")
    .map((noteName, index) => ({
      key: index + 1,
      noteName: noteName.trim(),
      frequency: 0,
    }))
    .filter((note) => note.noteName.length > 0);

  return {
    id: "custom",
    name: customName,
    keyCount: notes.length,
    notes,
  };
}

function buildDefaultCaptureId(capture: TranscriptionCapture) {
  const day = capture.generatedAt.slice(0, 10);
  const scenario = capture.requestPayload.scenario || "manual-test";
  return `${day}-${scenario}-${capture.requestPayload.tuning.id}`;
}

function buildExpectedDisplay(notes: TuningNote[]): string {
  return notes.map((note) => note.noteName).join(" + ");
}

function buildExpectedSummary(events: ManualCaptureExpectedEvent[]): string {
  if (events.length === 0) {
    return "";
  }

  const orderedCounts = new Map<string, number>();
  for (const event of events) {
    orderedCounts.set(event.display, (orderedCounts.get(event.display) ?? 0) + 1);
  }

  return Array.from(orderedCounts.entries())
    .map(([display, count]) => (count > 1 ? `${display} x ${count}` : display))
    .join(" / ");
}

function buildExpectedPerformance(events: ManualCaptureExpectedEvent[]): ManualCaptureExpectedPerformance | null {
  if (events.length === 0) {
    return null;
  }

  return {
    source: "clickable-kalimba-ui",
    version: 1,
    summary: buildExpectedSummary(events),
    events,
  };
}

export default function HomePage() {
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [selectedTuningId, setSelectedTuningId] = useState<string>("");
  const [customName, setCustomName] = useState("Custom Tuning");
  const [customNotes, setCustomNotes] = useState("C4,D4,E4,G4,A4,C5,E5,G5");
  const [recording, setRecording] = useState<Blob | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [notationMode, setNotationMode] = useState<NotationMode>("vertical");
  const [activeEventId, setActiveEventId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [captureCaseId, setCaptureCaseId] = useState("");
  const [captureMemo, setCaptureMemo] = useState("");
  const [lastCapture, setLastCapture] = useState<TranscriptionCapture | null>(null);
  const [isSavingCapture, setIsSavingCapture] = useState(false);
  const [pendingExpectedKeys, setPendingExpectedKeys] = useState<number[]>([]);
  const [expectedRepeatCount, setExpectedRepeatCount] = useState("1");
  const [expectedEvents, setExpectedEvents] = useState<ManualCaptureExpectedEvent[]>([]);

  useEffect(() => {
    fetchTunings()
      .then((loadedTunings) => {
        setTunings(loadedTunings);
        setSelectedTuningId(loadedTunings[0]?.id ?? "custom");
      })
      .catch((loadError) => {
        setError(loadError instanceof Error ? loadError.message : "調律の取得に失敗しました。");
      });
  }, []);

  const selectedTuning =
    selectedTuningId === "custom"
      ? buildCustomTuning(customName, customNotes)
      : tunings.find((tuning) => tuning.id === selectedTuningId) ?? null;

  const tuningSignature = useMemo(() => {
    if (!selectedTuning) {
      return "none";
    }
    const noteSignature = selectedTuning.notes.map((note) => `${note.key}:${note.noteName}`).join("|");
    return `${selectedTuning.id}:${noteSignature}`;
  }, [selectedTuning]);

  useEffect(() => {
    setPendingExpectedKeys([]);
    setExpectedEvents([]);
    setExpectedRepeatCount("1");
  }, [tuningSignature]);

  const expectedPerformance = useMemo(() => buildExpectedPerformance(expectedEvents), [expectedEvents]);
  const expectedNote = expectedPerformance?.summary ?? "";

  function handleToggleExpectedKey(key: number) {
    setPendingExpectedKeys((current) => (current.includes(key) ? current.filter((value) => value !== key) : [...current, key]));
  }

  function handleCommitExpectedSelection() {
    if (!selectedTuning) {
      return;
    }

    const selectedNotes = selectedTuning.notes.filter((note) => pendingExpectedKeys.includes(note.key));
    if (selectedNotes.length === 0) {
      return;
    }

    const repeatCount = Math.max(1, Number.parseInt(expectedRepeatCount || "1", 10) || 1);
    const display = buildExpectedDisplay(selectedNotes);

    setExpectedEvents((current) => {
      const nextEvents = [...current];
      for (let index = 0; index < repeatCount; index += 1) {
        nextEvents.push({
          index: nextEvents.length + 1,
          keys: selectedNotes.map((note) => ({ key: note.key, noteName: note.noteName })),
          display,
        });
      }
      return nextEvents;
    });

    setPendingExpectedKeys([]);
    setExpectedRepeatCount("1");
  }

  function handleUndoExpectedEvent() {
    setExpectedEvents((current) => current.slice(0, -1).map((event, index) => ({ ...event, index: index + 1 })));
  }

  function handleClearExpectedPerformance() {
    setPendingExpectedKeys([]);
    setExpectedEvents([]);
    setExpectedRepeatCount("1");
  }

  async function handleAnalyze() {
    if (!recording || !selectedTuning) {
      setError("録音と調律設定が必要です。");
      return;
    }

    setError(null);
    setIsAnalyzing(true);

    try {
      const capture = await createTranscriptionWithCapture(recording, selectedTuning, {
        scenario: captureCaseId,
        expectedNote,
        expectedPerformance,
        memo: captureMemo,
      });
      setResult(capture.responsePayload);
      setActiveEventId(capture.responsePayload.events[0]?.id ?? null);
      setLastCapture(capture);
      if (captureCaseId.trim().length === 0) {
        setCaptureCaseId(buildDefaultCaptureId(capture));
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "解析に失敗しました。");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function handleDownloadCapture() {
    if (!lastCapture) {
      return;
    }

    setError(null);
    setIsSavingCapture(true);

    try {
      const caseId = captureCaseId.trim().length > 0 ? captureCaseId.trim() : buildDefaultCaptureId(lastCapture);
      const archive = await buildCaptureArchive({
        caseId,
        audioWav: lastCapture.audioWav,
        requestPayload: {
          ...lastCapture.requestPayload,
          scenario: caseId,
          expectedNote: expectedNote.trim() || lastCapture.requestPayload.expectedNote,
          expectedPerformance: expectedPerformance ?? lastCapture.requestPayload.expectedPerformance,
          memo: captureMemo.trim() || lastCapture.requestPayload.memo,
        },
        responsePayload: lastCapture.responsePayload,
        notes: {
          expectedNote,
          expectedPerformance,
          memo: captureMemo,
        },
      });
      downloadBlob(archive, `${caseId}.zip`);
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : "保存パックの作成に失敗しました。");
    } finally {
      setIsSavingCapture(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Kalimba Score MVP</p>
          <h1>カリンバ演奏を録音して、そのまま譜面へ。</h1>
          <p className="hero-copy">
            マイク録音からカリンバ向けの譜面へ変換します。ドレミ縦並び譜を中心に、数字譜と通常表記も同じ解析結果から切り替え表示できます。
          </p>
        </div>
        <div className="hero-card">
          <p>対応範囲</p>
          <ul>
            <li>ブラウザ録音</li>
            <li>サーバー側解析</li>
            <li>和音イベント検出</li>
            <li>複数調律とカスタム音列</li>
          </ul>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="workspace-grid">
        <div className="stack gap-xl">
          <RecorderPanel disabled={isAnalyzing || isSavingCapture} onRecordingReady={setRecording} />
          <TuningPanel
            tunings={tunings}
            selectedId={selectedTuningId}
            onSelect={setSelectedTuningId}
            customName={customName}
            customNotes={customNotes}
            onCustomNameChange={setCustomName}
            onCustomNotesChange={setCustomNotes}
          />
          <section className="panel workflow-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Workflow</p>
                <h2>解析を実行</h2>
              </div>
              <span className="pill">{recording ? "Recording Ready" : "Awaiting Recording"}</span>
            </div>
            <p className="muted workflow-lead">
              {recording
                ? "期待演奏をレールに組み立ててから解析すると、保存パックの再現性が上がります。"
                : "録音後に expected performance を組み立てると、手動テストの意図をそのまま残せます。"}
            </p>

            <div className="workflow-composer-grid">
              <div className="workflow-card workflow-card-primary stack gap-lg">
                <div className="workflow-card-header">
                  <div>
                    <span className="eyebrow">Expected Performance</span>
                    <h3>期待演奏を組み立てる</h3>
                  </div>
                  <span className="muted">選択中の和音を上、キー入力を下に配置</span>
                </div>

                <ExpectedKeySelector
                  tuning={selectedTuning}
                  selectedKeys={pendingExpectedKeys}
                  repeatCount={expectedRepeatCount}
                  onToggleKey={handleToggleExpectedKey}
                  onRepeatCountChange={setExpectedRepeatCount}
                  onCommitSelection={handleCommitExpectedSelection}
                  onClearSelection={() => setPendingExpectedKeys([])}
                />

                <div className="expected-performance-panel">
                  <div className="panel-header compact">
                    <div>
                      <strong>期待演奏シーケンス</strong>
                      <p className="muted">保存パックには event 単位で記録されます。</p>
                    </div>
                    <div className="row wrap">
                      <button type="button" className="ghost" onClick={handleUndoExpectedEvent} disabled={expectedEvents.length === 0}>
                        最後を取り消す
                      </button>
                      <button
                        type="button"
                        className="ghost"
                        onClick={handleClearExpectedPerformance}
                        disabled={expectedEvents.length === 0 && pendingExpectedKeys.length === 0}
                      >
                        すべてクリア
                      </button>
                    </div>
                  </div>
                  <div className="expected-summary-box large summary-hero">
                    <span className="eyebrow">Summary</span>
                    <strong>{expectedNote || "未設定"}</strong>
                  </div>
                  {expectedEvents.length > 0 ? (
                    <div className="expected-event-list compact-list">
                      {expectedEvents.map((event) => (
                        <div key={`${event.index}-${event.display}`} className="expected-event-card compact">
                          <span className="pill">#{event.index}</span>
                          <div className="stack event-copy">
                            <strong>{event.display}</strong>
                            <span className="muted">{event.keys.map((key) => `${key.noteName} (#${key.key})`).join(" / ")}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="muted">まだ期待演奏は追加されていません。</p>
                  )}
                </div>
              </div>

              <div className="workflow-card workflow-card-secondary stack gap-lg">
                <div className="workflow-card-header">
                  <div>
                    <span className="eyebrow">Capture Metadata</span>
                    <h3>ケース情報</h3>
                  </div>
                  <span className="muted">補助情報</span>
                </div>
                <label className="stack">
                  <span>ケースID (任意)</span>
                  <input
                    value={captureCaseId}
                    onChange={(event) => setCaptureCaseId(event.target.value)}
                    placeholder="2026-03-21-d5-single-note-01"
                  />
                </label>
                <label className="stack">
                  <span>テストメモ (任意)</span>
                  <textarea value={captureMemo} rows={6} onChange={(event) => setCaptureMemo(event.target.value)} />
                </label>
              </div>
            </div>

            <div className="workflow-action-dock stack gap-lg">
              <button className="primary wide" onClick={handleAnalyze} disabled={!recording || !selectedTuning || isAnalyzing}>
                {isAnalyzing ? "解析中..." : "自動採譜する"}
              </button>
              <button className="secondary wide" onClick={handleDownloadCapture} disabled={!lastCapture || isAnalyzing || isSavingCapture}>
                {isSavingCapture ? "保存パック作成中..." : "解析結果を保存パックでダウンロード"}
              </button>
              <p className="muted">保存内容: audio.wav / request.json / response.json / notes.md</p>
            </div>
          </section>
        </div>

        <div className="stack gap-xl">
          <NotationPanel result={result} mode={notationMode} onModeChange={setNotationMode} />
          <EditorPanel
            result={result}
            activeEventId={activeEventId}
            onActiveEventIdChange={setActiveEventId}
            tuning={selectedTuning}
            onResultChange={setResult}
          />
        </div>
      </div>
    </main>
  );
}
