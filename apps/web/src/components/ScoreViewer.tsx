"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useRouter } from "next/navigation";

import { BeatGridScore } from "@/components/BeatGridScore";
import { DoReMiScore } from "@/components/DoReMiScore";
import { ScoreExportButtons } from "@/components/ScoreExportButtons";
import { TxIdBadge } from "@/components/TxIdBadge";
import {
  boostDbForPeak,
  closeAudioBoost,
  ensureAudioBoost,
  type AudioBoostChain,
} from "@/lib/audioBoost";
import { computeAudioLevels } from "@/lib/audio";
import {
  createTranscriptionRun,
  createTranscriptionWithCapture,
  fetchMemo,
  fetchTranscription,
  fetchTranscriptionAudioBlob,
  fetchTranscriptionRun,
  fetchTranscriptionRuns,
  fetchTunings,
  fetchCorrections,
  saveMemo,
} from "@/lib/api";
import { findEventById, findEventIdAtSec } from "@/lib/eventTiming";
import { pushRecentTranscription } from "@/lib/recentTranscriptions";
import {
  isMovableNumberApplicable,
  movableDoLabelFn,
  movableNumberLabelFn,
  noteLabelFromScoreNote,
  tonicReferenceOctave,
} from "@/lib/scoreLayout";
import { restoreStateFromCorrections, toDisplayScoreEvents } from "@/lib/reviewCorrections";
import {
  InstrumentTuning,
  RecognitionRun,
  RecognitionRunsResponse,
  ScoreEvent,
  TranscriptionResult,
  TuningMismatch,
} from "@/lib/types";

const EMPTY_RUNS: RecognitionRunsResponse = { runs: [], latestRunId: null };

type LabelMode = "fixed" | "movable" | "movableNumber";
const LABEL_MODE_STORAGE_KEY = "kalimba:score-label-mode";

function isLabelMode(value: string | null): value is LabelMode {
  return value === "fixed" || value === "movable" || value === "movableNumber";
}

type LoadState =
  | { kind: "loading" }
  | {
      kind: "ready";
      result: TranscriptionResult;
      audioUrl: string;
      initialMemo: string;
      correctedEvents: ScoreEvent[] | null;
      correctionsBaseRunId: string | null;
      peakDb: number | null;
      runs: RecognitionRunsResponse;
    }
  | { kind: "error"; message: string };

const MEMO_SAVE_DEBOUNCE_MS = 800;

export function ScoreViewer({ transactionId }: { transactionId: string }) {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    async function load() {
      try {
        // runs の取得失敗は致命的ではない (#204 Phase 2 の run 切替 UI が
        // 出せないだけ) なので空扱いにフォールバックする。
        const [result, audioBlob, memo, corrections, runs] = await Promise.all([
          fetchTranscription(transactionId),
          fetchTranscriptionAudioBlob(transactionId),
          fetchMemo(transactionId).catch(() => ""),
          fetchCorrections(transactionId).catch(() => null),
          fetchTranscriptionRuns(transactionId).catch(() => EMPTY_RUNS),
        ]);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(audioBlob);
        // 静音録音の試聴ブースト用 (lib/audioBoost)。失敗しても再生には影響しない
        const levels = await computeAudioLevels(audioBlob).catch(() => null);
        if (cancelled) return;
        // #202 / #209 review: corrections が保存済みなら編集後イベント列を導出する。
        // corrections はその base run に対する diff。明示 baseRunId があればその run の
        // payload に restore する (latest に rematch すると再認識後に誤整列した corrected
        // を表示してしまう)。明示 baseRunId 無しの corrections は latest (= 表示中の
        // 解決済み result) に対するものとみなす (#202 の既定挙動を踏襲・no-regression)。
        // runs が取得できなくても (latestRunId=null) corrections は表示する — restore は
        // result に対して行えるため、runs 不在で corrected を落とさない (#209 review @105)。
        let correctedEvents: ScoreEvent[] | null = null;
        let correctionsBaseRunId: string | null = null;
        if (corrections && corrections.events.length > 0) {
          const explicitBase = corrections.baseRunId ?? null;
          let baseResult = result; // 既定は latest (表示中の解決済み result)
          if (explicitBase && explicitBase !== runs.latestRunId) {
            try {
              baseResult = await fetchTranscriptionRun(transactionId, explicitBase);
            } catch {
              baseResult = result; // base run が取れないときは latest に best-effort
            }
            if (cancelled) return;
          }
          try {
            correctedEvents = toDisplayScoreEvents(
              restoreStateFromCorrections(baseResult, corrections),
            );
            // 明示 base があればその run、無ければ latest (null = runs 不明時)。
            correctionsBaseRunId = explicitBase ?? runs.latestRunId ?? null;
          } catch {
            correctedEvents = null;
            correctionsBaseRunId = null;
          }
        }
        setState({
          kind: "ready",
          result,
          audioUrl: objectUrl,
          initialMemo: memo,
          correctedEvents,
          // #209 review: corrections がどの run に対するものかを保持し、その run を
          // 閲覧しているときだけ corrected を出す (別 run 閲覧時は note で誘導)。
          correctionsBaseRunId,
          peakDb: levels?.peakDb ?? null,
          runs,
        });
      } catch (err) {
        if (cancelled) return;
        setState({
          kind: "error",
          message: err instanceof Error ? err.message : "読み込みに失敗しました。",
        });
      }
    }

    load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [transactionId]);

  if (state.kind === "loading") {
    return (
      <main className="score-viewer-shell">
        <p className="muted">読み込み中…</p>
      </main>
    );
  }

  if (state.kind === "error") {
    return (
      <main className="score-viewer-shell">
        <p className="empty">読み込めませんでした: {state.message}</p>
      </main>
    );
  }

  return (
    // key で transactionId ごとに remount させ、run 切替用の useState が前の
    // 録音の値を持ち越さないようにする (#209 review: 同一インスタンスに別
    // transactionId が来ても useState 初期化子は再実行されないため)。
    <ScoreViewerReady
      key={transactionId}
      transactionId={transactionId}
      result={state.result}
      audioUrl={state.audioUrl}
      initialMemo={state.initialMemo}
      correctedEvents={state.correctedEvents}
      correctionsBaseRunId={state.correctionsBaseRunId}
      peakDb={state.peakDb}
      initialRuns={state.runs}
    />
  );
}

type ReadyProps = {
  transactionId: string;
  result: TranscriptionResult;
  audioUrl: string;
  initialMemo: string;
  correctedEvents: ScoreEvent[] | null;
  correctionsBaseRunId: string | null;
  peakDb: number | null;
  initialRuns: RecognitionRunsResponse;
};

function ScoreViewerReady({
  transactionId,
  result: latestResult,
  audioUrl,
  initialMemo,
  correctedEvents,
  correctionsBaseRunId,
  peakDb,
  initialRuns,
}: ReadyProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeEventId, setActiveEventId] = useState<string | null>(null);

  // #204 Phase 2: 認識履歴 (recognition runs) の切替。selectedRunId が最新 run
  // (または runs が空) のときは load() が返した最新解決結果をそのまま使う。
  // それ以外を選ぶと GET .../runs/{runId} で該当 run の全文を取得して表示する。
  const [runsState, setRunsState] = useState<RecognitionRunsResponse>(initialRuns);
  // #209 review: corrections があれば既定でその base run を選択して corrected を
  // 既定表示する (#202「修正済みを既定表示」を run-correct に踏襲)。無ければ latest。
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    correctionsBaseRunId ?? initialRuns.latestRunId,
  );
  // "最新" の内容そのもの。load() が返した latestResult prop で初期化するが、
  // 再認識 (handleRerun) が起きるとその新しい run が「最新」になるので更新する。
  // これを latestResult prop 自体で代用すると (#204 Phase 2 の実装バグとして
  // 発見): 再認識直後は selectedRunId/runsState.latestRunId が新しい runId に
  // 揃うため isViewingLatestRun が true に戻り、下の effect が「最新に戻す」
  // 分岐を実行する。その時 latestResult (初回読み込み時点の古いスナップショット)
  // へreset してしまうと、再認識で表示したはずの新しい結果が即座に古い結果へ
  // 巻き戻ってしまう。
  const [latestKnownResult, setLatestKnownResult] = useState<TranscriptionResult>(latestResult);
  const [displayResult, setDisplayResult] = useState<TranscriptionResult>(latestResult);
  const [runSwitchBusy, setRunSwitchBusy] = useState(false);
  const [runSwitchError, setRunSwitchError] = useState<string | null>(null);
  const [rerunBusy, setRerunBusy] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const isViewingLatestRun = selectedRunId === null || selectedRunId === runsState.latestRunId;

  useEffect(() => {
    if (isViewingLatestRun) {
      setDisplayResult(latestKnownResult);
      setRunSwitchError(null);
      // #209 review: 履歴 run の fetch 進行中に「最新」へ戻ると、前 effect の
      // cleanup が cancelled=true にして in-flight の finally が busy 解除を
      // skip する。この分岐でも解除しないと selector が disabled のまま固まる。
      setRunSwitchBusy(false);
      return;
    }
    let cancelled = false;
    setRunSwitchBusy(true);
    setRunSwitchError(null);
    fetchTranscriptionRun(transactionId, selectedRunId as string)
      .then((run) => {
        if (cancelled) return;
        setDisplayResult(run);
      })
      .catch((err) => {
        if (cancelled) return;
        setRunSwitchError(
          err instanceof Error ? err.message : "この認識結果の読み込みに失敗しました。",
        );
      })
      .finally(() => {
        if (!cancelled) setRunSwitchBusy(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRunId, isViewingLatestRun, transactionId, latestKnownResult]);

  const handleRerun = useCallback(async () => {
    setRerunBusy(true);
    setRerunError(null);
    try {
      const created = await createTranscriptionRun(transactionId);
      // #209 review: created は既にサーバへ append 済み。まず結果を表示に反映
      // してから履歴一覧を更新する。こうしないと後続の fetchTranscriptionRuns が
      // 失敗したとき「再認識は成功したのに失敗表示 → ユーザー再試行で重複 run」
      // に陥る (runs は append-only)。
      setLatestKnownResult(created.result);
      setDisplayResult(created.result);
      try {
        const refreshed = await fetchTranscriptionRuns(transactionId);
        setRunsState(refreshed);
        setSelectedRunId(created.runId);
      } catch {
        // 履歴一覧の更新だけ失敗: created は表示済み。selectedRunId=null で
        // 「最新」扱いにして created.result を表示し続ける (dropdown は次回
        // ロードで整合)。重複 run を生む再試行を促さない。
        setSelectedRunId(null);
        setRunSwitchError("再認識は成功しましたが履歴一覧の更新に失敗しました。リロードで反映されます。");
      }
    } catch (err) {
      setRerunError(err instanceof Error ? err.message : "再認識に失敗しました。");
    } finally {
      setRerunBusy(false);
    }
  }, [transactionId]);

  // ダウンストリームの表示ロジック (labelMode / events / warnings / export 等)
  // は全て「今表示している認識結果」に対して働けばよいので、result という名前を
  // ここでシャドウする (result.xxx への既存参照をそのまま activeResult として使う)。
  const result = displayResult;

  // 静音録音のブースト + 片チャンネル無音ステレオの両耳化 (lib/audioBoost)
  const boostChainRef = useRef<AudioBoostChain | null>(null);
  const boostDb = boostDbForPeak(peakDb);
  const ensureBoost = useCallback(() => {
    ensureAudioBoost(audioRef.current, boostChainRef, boostDb);
  }, [boostDb]);
  useEffect(() => {
    return () => closeAudioBoost(boostChainRef);
  }, []);

  const tonic = result.instrumentTuning.tonic ?? null;
  const movableAvailable = Boolean(tonic);
  const tonicRefOctave = useMemo(
    () => tonicReferenceOctave(result.instrumentTuning, tonic),
    [result.instrumentTuning, tonic],
  );

  // #202: 修正済み版があれば既定でそれを表示。認識結果へは切替可能。
  // correctedEvents は読み込み時の run (initialRuns.latestRunId) に対して
  // 算出されたものなので (#204 Phase 2)、過去 run を閲覧中はもちろん、
  // 再認識で「最新」が別の run に変わった後もそのままでは整列が保証できない
  // (再認識後は isViewingLatestRun は true に戻るが、それは新しい run に
  // 対して) — なので「読み込み時と同じ run を見ているか」で厳密にゲートする。
  // #209 review: corrections はその base run に束ねる。いま選んでいる run が
  // corrections の base run のときだけ corrected を出す (selectedRunId===null は
  // 「最新」= runsState.latestRunId を指す)。別 run を閲覧中は無言で消さず note で
  // base run へ誘導する。これで再認識後に誤整列した corrected を表示しない。
  const normalizedSelectedRunId = selectedRunId ?? runsState.latestRunId;
  // 明示 base が無い corrections (correctionsBaseRunId=null) は latest に対するものと
  // みなす (#202)。runs 不明時は latestRunId も null になり、既定選択も null なので一致する。
  const effectiveCorrectionsRun = correctionsBaseRunId ?? runsState.latestRunId;
  const isViewingCorrectionsBase =
    correctedEvents !== null && normalizedSelectedRunId === effectiveCorrectionsRun;
  const [viewSource, setViewSource] = useState<"corrected" | "recognized">(
    correctedEvents ? "corrected" : "recognized",
  );
  const showCorrectedToggle = isViewingCorrectionsBase && Boolean(correctedEvents);
  // corrections はあるが別 run (再認識後の新 run 等) を閲覧中: 保存済み修正を
  // 無言で消さず、base run を選べば見られる旨の note を出す (P3)。
  const showCorrectionsElsewhereNote = Boolean(correctedEvents) && !isViewingCorrectionsBase;
  const events =
    showCorrectedToggle && viewSource === "corrected" && correctedEvents
      ? correctedEvents
      : result.events;

  const allNotes = useMemo(
    () => events.flatMap((e) => e.notes),
    [events],
  );
  const movableNumberAvailable = useMemo(
    () => isMovableNumberApplicable(allNotes, tonic),
    [allNotes, tonic],
  );

  const [labelMode, setLabelMode] = useState<LabelMode>("fixed");
  const scoreAreaRef = useRef<HTMLElement | null>(null);
  // #202 案 A' プロトタイプの dev フラグ (?notation=beatgrid)。判定材料用で
  // UI には切替を出さない。
  const [useBeatGrid, setUseBeatGrid] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    setUseBeatGrid(new URLSearchParams(window.location.search).get("notation") === "beatgrid");
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(LABEL_MODE_STORAGE_KEY);
    if (!isLabelMode(stored)) return;
    if (stored === "movable" && movableAvailable) setLabelMode("movable");
    else if (stored === "movableNumber" && movableNumberAvailable) setLabelMode("movableNumber");
  }, [movableAvailable, movableNumberAvailable]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(LABEL_MODE_STORAGE_KEY, labelMode);
  }, [labelMode]);

  const labelFn = useMemo(() => {
    if (labelMode === "movable" && tonic) return movableDoLabelFn(tonic, tonicRefOctave);
    if (labelMode === "movableNumber" && tonic) return movableNumberLabelFn(tonic, tonicRefOctave);
    return noteLabelFromScoreNote;
  }, [labelMode, tonic, tonicRefOctave]);

  const shareUrl = useMemo(() => {
    if (typeof window === "undefined") return "";
    return window.location.href;
  }, []);

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const next = findEventIdAtSec(events, audio.currentTime);
    if (next !== activeEventId) {
      setActiveEventId(next);
    }
  }, [events, activeEventId]);

  const handleScoreEventTap = useCallback(
    (eventId: string) => {
      const audio = audioRef.current;
      const event = findEventById(events, eventId);
      if (!audio || !event) return;
      audio.currentTime = event.startTimeSec;
      setActiveEventId(eventId);
    },
    [events],
  );

  return (
    <main className="score-viewer-shell">
      <header className="score-viewer-header">
        <div className="score-viewer-header-row">
          <Link href="/" className="score-viewer-home-link">
            ← トップへ
          </Link>
          <h1 className="score-viewer-title">カリンバ譜面</h1>
          <TxIdBadge id={transactionId} />
        </div>
        <ShareUrlRow url={shareUrl} />
      </header>

      <RunHistoryPanel
        runs={runsState.runs}
        latestRunId={runsState.latestRunId}
        selectedRunId={selectedRunId}
        onSelect={setSelectedRunId}
        switchBusy={runSwitchBusy}
        switchError={runSwitchError}
        onRerun={handleRerun}
        rerunBusy={rerunBusy}
        rerunError={rerunError}
      />

      {result.tuningMismatch ? (
        <TuningMismatchBanner
          transactionId={transactionId}
          mismatch={result.tuningMismatch}
          currentTuningName={result.instrumentTuning.name}
        />
      ) : null}

      {result.warnings.length > 0 ? (
        <div className="warning-box">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}

      <MemoEditor transactionId={transactionId} initialMemo={initialMemo} />

      <section className="score-viewer-playback">
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          onTimeUpdate={handleTimeUpdate}
          onPlay={ensureBoost}
          className="score-viewer-audio"
        />
        {boostDb > 0 ? (
          <p className="muted score-viewer-boost-note">
            試聴 +{boostDb.toFixed(0)}dB ブースト中 (元 peak {peakDb?.toFixed(1)}dB の静音録音)
          </p>
        ) : null}
      </section>

      <section className="score-viewer-score" ref={scoreAreaRef}>
        {showCorrectedToggle ? (
          <div className="score-viewer-source-row">
            <span className={`score-source-badge${viewSource === "corrected" ? " corrected" : ""}`}>
              {viewSource === "corrected" ? "修正済み版を表示中" : "認識結果 (未修正) を表示中"}
            </span>
            <button
              type="button"
              className="score-export-btn"
              onClick={() => setViewSource((v) => (v === "corrected" ? "recognized" : "corrected"))}
            >
              {viewSource === "corrected" ? "認識結果を表示" : "修正済み版を表示"}
            </button>
          </div>
        ) : showCorrectionsElsewhereNote ? (
          <p className="muted score-viewer-run-note">
            この録音には別の認識結果に対して保存された修正版があります。上の履歴からその認識結果を選ぶと修正版を表示・切替できます。
          </p>
        ) : null}
        <div className="score-viewer-mode-toggle" role="group" aria-label="ドレミ表記">
          <button
            type="button"
            className={`score-viewer-mode-btn${labelMode === "fixed" ? " active" : ""}`}
            onClick={() => setLabelMode("fixed")}
          >
            固定ド
          </button>
          <button
            type="button"
            className={`score-viewer-mode-btn${labelMode === "movable" ? " active" : ""}`}
            onClick={() => movableAvailable && setLabelMode("movable")}
            disabled={!movableAvailable}
            title={movableAvailable ? undefined : "この調律には tonic が設定されていません"}
          >
            移動ド{tonic ? ` (${tonic})` : ""}
          </button>
          <button
            type="button"
            className={`score-viewer-mode-btn${labelMode === "movableNumber" ? " active" : ""}`}
            onClick={() => movableNumberAvailable && setLabelMode("movableNumber")}
            disabled={!movableNumberAvailable}
            title={
              movableNumberAvailable
                ? undefined
                : tonic
                ? "スケール外の音が含まれているため使用できません"
                : "この調律には tonic が設定されていません"
            }
          >
            数字{tonic ? ` (${tonic}=1)` : ""}
          </button>
        </div>
        {useBeatGrid ? (
          <BeatGridScore
            events={events}
            activeEventId={activeEventId}
            onActiveEventIdChange={handleScoreEventTap}
            labelFn={labelFn}
          />
        ) : (
          <DoReMiScore
            events={events}
            activeEventId={activeEventId}
            onActiveEventIdChange={handleScoreEventTap}
            labelFn={labelFn}
          />
        )}
        <ScoreExportButtons
          scoreAreaRef={scoreAreaRef}
          fileBaseName={`kalimba-score-${transactionId.slice(0, 8)}`}
        />
      </section>

      <section className="score-viewer-review-link-row">
        <Link href={`/score/${transactionId}/review`} className="score-viewer-review-link">
          結果を確認・修正する →
        </Link>
      </section>

      <RetranscribeSection
        transactionId={transactionId}
        currentTuningId={result.instrumentTuning.id}
      />

      <footer className="score-viewer-footer">
        <p className="muted">
          {result.instrumentTuning.name} · Tempo {result.tempo.toFixed(1)} BPM · {events.length} events
        </p>
      </footer>
    </main>
  );
}

function useRetranscribe(transactionId: string) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const retranscribe = useCallback(
    async (tuning: InstrumentTuning) => {
      setBusy(true);
      setError(null);
      try {
        const audioBlob = await fetchTranscriptionAudioBlob(transactionId);
        const capture = await createTranscriptionWithCapture(audioBlob, tuning, {
          force: true,
        });
        const newId = capture.responsePayload.transactionId;
        if (!newId) throw new Error("新しい transactionId が返されませんでした。");
        pushRecentTranscription({
          transactionId: newId,
          createdAt: new Date().toISOString(),
          tuningName: tuning.name,
          eventCount: capture.responsePayload.events.length,
        });
        router.push(`/score/${newId}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : "再採譜に失敗しました。");
        setBusy(false);
      }
    },
    [transactionId, router],
  );

  return { busy, error, retranscribe };
}

function TuningMismatchBanner({
  transactionId,
  mismatch,
  currentTuningName,
}: {
  transactionId: string;
  mismatch: TuningMismatch;
  currentTuningName: string;
}) {
  const { busy, error, retranscribe } = useRetranscribe(transactionId);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const outside = mismatch.outsidePitchClasses.join(", ");

  async function handleSuggested() {
    if (!mismatch.suggestedTuningId) return;
    setFetchError(null);
    try {
      const tunings = await fetchTunings();
      const suggested = tunings.find((t) => t.id === mismatch.suggestedTuningId);
      if (!suggested) throw new Error("提案された調律が見つかりませんでした。");
      await retranscribe(suggested);
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : "再採譜に失敗しました。");
    }
  }

  return (
    <section className="score-viewer-mismatch" role="alert">
      <p className="score-viewer-mismatch-text">
        この録音には選択した調律 ({currentTuningName}) にない音
        {outside ? ` (${outside})` : ""} が強く含まれています。
        {mismatch.suggestedTuningName
          ? ` ${mismatch.suggestedTuningName} の演奏かもしれません。`
          : " 調律の選択が合っているか確認してください。"}
      </p>
      {mismatch.suggestedTuningId ? (
        <button
          type="button"
          className="score-viewer-mismatch-btn"
          onClick={handleSuggested}
          disabled={busy}
        >
          {busy ? "再採譜中…" : `${mismatch.suggestedTuningName} で再採譜`}
        </button>
      ) : null}
      {error || fetchError ? (
        <p className="score-viewer-retranscribe-error">{error ?? fetchError}</p>
      ) : null}
    </section>
  );
}

function RetranscribeSection({
  transactionId,
  currentTuningId,
}: {
  transactionId: string;
  currentTuningId: string;
}) {
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [selectedTuningId, setSelectedTuningId] = useState<string>(currentTuningId);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const { busy, error, retranscribe } = useRetranscribe(transactionId);

  useEffect(() => {
    fetchTunings()
      .then((list) => setTunings(list))
      .catch(() => setFetchError("調律一覧の取得に失敗しました。"));
  }, []);

  const selectedTuning = tunings.find((t) => t.id === selectedTuningId) ?? null;

  async function handleRetranscribe() {
    if (!selectedTuning) return;
    await retranscribe(selectedTuning);
  }

  return (
    <section className="score-viewer-retranscribe">
      <p className="score-viewer-retranscribe-label">この録音を別の条件で再採譜</p>
      <div className="score-viewer-retranscribe-row">
        <select
          className="score-viewer-retranscribe-select"
          value={selectedTuningId}
          onChange={(e) => setSelectedTuningId(e.target.value)}
          disabled={busy || tunings.length === 0}
        >
          {tunings.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="score-viewer-retranscribe-btn"
          onClick={handleRetranscribe}
          disabled={busy || !selectedTuning}
        >
          {busy ? "再採譜中…" : "再採譜"}
        </button>
      </div>
      {error || fetchError ? (
        <p className="score-viewer-retranscribe-error">{error ?? fetchError}</p>
      ) : null}
    </section>
  );
}

function formatRunTimestamp(ranAt: string | null): string {
  if (!ranAt) return "";
  try {
    return new Date(ranAt).toLocaleString();
  } catch {
    return ranAt;
  }
}

function formatRunLabel(run: RecognitionRun): string {
  if (run.isLegacy) {
    return `アップロード時点 (レガシー) · ${run.eventCount} events`;
  }
  const fp = run.recognizerFingerprint ? run.recognizerFingerprint.slice(0, 8) : "fp不明";
  return `${formatRunTimestamp(run.ranAt)} · ${fp} · ${run.eventCount} events`;
}

// #204 Phase 2: 認識履歴 (runs) の切替 + 現行認識器での再認識。
// GET .../runs で返る全 run (legacy 含む) を新しい順のまま select の選択肢にする。
function RunHistoryPanel({
  runs,
  latestRunId,
  selectedRunId,
  onSelect,
  switchBusy,
  switchError,
  onRerun,
  rerunBusy,
  rerunError,
}: {
  runs: RecognitionRun[];
  latestRunId: string | null;
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  switchBusy: boolean;
  switchError: string | null;
  onRerun: () => void;
  rerunBusy: boolean;
  rerunError: string | null;
}) {
  const effectiveSelected = selectedRunId ?? latestRunId ?? "";

  return (
    <section className="score-viewer-runs">
      <div className="score-viewer-runs-row">
        {runs.length > 0 ? (
          <select
            className="score-viewer-runs-select"
            value={effectiveSelected}
            onChange={(e) => onSelect(e.target.value)}
            disabled={switchBusy}
            aria-label="認識結果の履歴"
          >
            {runs.map((run) => (
              <option key={run.runId} value={run.runId}>
                {run.runId === latestRunId ? "最新 · " : ""}
                {formatRunLabel(run)}
              </option>
            ))}
          </select>
        ) : (
          <span className="muted">認識履歴を取得できませんでした。</span>
        )}
        <button
          type="button"
          className="score-viewer-rerun-btn"
          onClick={onRerun}
          disabled={rerunBusy}
          title="保存済みの録音+調律のまま、現在の認識器で再認識して履歴に追加します (新しい transaction は作られません)"
        >
          {rerunBusy ? "再認識中…" : "現在の認識器で再認識"}
        </button>
      </div>
      {switchBusy ? <p className="muted">読み込み中…</p> : null}
      {switchError ? <p className="score-viewer-retranscribe-error">{switchError}</p> : null}
      {rerunError ? <p className="score-viewer-retranscribe-error">{rerunError}</p> : null}
    </section>
  );
}

function ShareUrlRow({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // ignore
    }
  };

  return (
    <div className="score-viewer-share-row">
      <input
        className="score-viewer-url"
        type="text"
        value={url}
        readOnly
        onFocus={(e) => e.currentTarget.select()}
      />
      <button type="button" className="score-viewer-copy-btn" onClick={handleCopy}>
        {copied ? "コピーしました" : "URL をコピー"}
      </button>
    </div>
  );
}

function MemoEditor({
  transactionId,
  initialMemo,
}: {
  transactionId: string;
  initialMemo: string;
}) {
  const [memo, setMemo] = useState(initialMemo);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savedRef = useRef(initialMemo);

  useEffect(() => {
    if (memo === savedRef.current) {
      setSaveState("idle");
      return;
    }
    setSaveState("saving");
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        await saveMemo(transactionId, memo);
        savedRef.current = memo;
        setSaveState("saved");
      } catch {
        setSaveState("error");
      }
    }, MEMO_SAVE_DEBOUNCE_MS);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [memo, transactionId]);

  return (
    <section className="score-viewer-memo">
      <label className="score-viewer-memo-label" htmlFor="score-viewer-memo-input">
        メモ
      </label>
      <textarea
        id="score-viewer-memo-input"
        className="score-viewer-memo-input"
        value={memo}
        onChange={(e) => setMemo(e.target.value)}
        placeholder="演奏の気づきやノートをここに…"
        rows={2}
      />
      <p className="score-viewer-memo-status muted">
        {saveState === "saving" && "保存中…"}
        {saveState === "saved" && "保存しました"}
        {saveState === "error" && "保存できませんでした"}
        {saveState === "idle" && "\u00a0"}
      </p>
    </section>
  );
}
