"use client";

// 開発用 GT ドラフト裁定ページ (第 2 期 S2、sprint-plan-2026-07b)。
// gt_draft.py が出力した整列表 (rows.json) を読み、行クリックで該当時刻に
// シーク再生しながらワンタップで裁定 → verdict を auto-save する。
// ユーザーの裁定完了後、エージェントが verdict.json から最終 ground_truth.json
// (method: ear_verified) を生成する。main nav からはリンクしない。
// 撤去条件: GT 化の運用が落ち着いた時点で /api/dev/gt-drafts と一緒に削除する。

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchGtDrafts,
  saveGtDraftVerdict,
  type GtDraft,
  type GtDraftRow,
  type GtDraftRowVerdict,
  type GtDraftVerdict,
} from "@/lib/api";

const FLAG_LABEL: Record<GtDraftRow["flag"], string> = {
  ok: "✅",
  pitch: "⚠PITCH",
  chord: "⚠CHORD",
  extra: "⚠EXTRA",
  ear: "耳確認",
};

const PLAY_LEAD_SEC = 0.7;
const PLAY_SNIPPET_SEC = 2.5;

function emptyVerdict(): GtDraftVerdict {
  return { rows: {}, unplaced: {}, done: false };
}

function formatNotes(notes: string[] | null | undefined): string {
  return notes && notes.length > 0 ? notes.join("+") : "—";
}

export default function DebugGtReviewPage() {
  const [drafts, setDrafts] = useState<GtDraft[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [verdicts, setVerdicts] = useState<Record<string, GtDraftVerdict>>({});
  const [saveState, setSaveState] = useState<Record<string, "saving" | "saved" | "error">>({});
  const [pickerRow, setPickerRow] = useState<number | null>(null);
  const [pickerNotes, setPickerNotes] = useState<string[]>([]);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pauseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  useEffect(() => {
    fetchGtDrafts()
      .then((sorted) => {
        // 軽い録音から裁定できるよう行数昇順 (敵対的テイク → 崩壊自由演奏)
        const data = [...sorted].sort((a, b) => a.rows.length - b.rows.length);
        setDrafts(data);
        setSelected((prev) => prev ?? data[0]?.tx8 ?? null);
        const initial: Record<string, GtDraftVerdict> = {};
        for (const draft of data) {
          initial[draft.tx8] = draft.verdict ?? emptyVerdict();
        }
        setVerdicts(initial);
        setLoadError(null);
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : "読み込み失敗"));
  }, []);

  const draft = useMemo(
    () => drafts.find((d) => d.tx8 === selected) ?? null,
    [drafts, selected],
  );
  const verdict = draft ? (verdicts[draft.tx8] ?? emptyVerdict()) : emptyVerdict();

  const scheduleSave = useCallback((tx8: string, next: GtDraftVerdict) => {
    setSaveState((s) => ({ ...s, [tx8]: "saving" }));
    if (saveTimers.current[tx8]) clearTimeout(saveTimers.current[tx8]);
    saveTimers.current[tx8] = setTimeout(() => {
      saveGtDraftVerdict(tx8, next)
        .then(() => setSaveState((s) => ({ ...s, [tx8]: "saved" })))
        .catch(() => setSaveState((s) => ({ ...s, [tx8]: "error" })));
    }, 600);
  }, []);

  const updateVerdict = useCallback(
    (tx8: string, mutate: (v: GtDraftVerdict) => GtDraftVerdict) => {
      setVerdicts((prev) => {
        const next = mutate(prev[tx8] ?? emptyVerdict());
        scheduleSave(tx8, next);
        return { ...prev, [tx8]: next };
      });
    },
    [scheduleSave],
  );

  const setRowVerdict = useCallback(
    (tx8: string, index: number, rowVerdict: GtDraftRowVerdict | null) => {
      updateVerdict(tx8, (v) => {
        const rows = { ...v.rows };
        if (rowVerdict === null) {
          delete rows[String(index)];
        } else {
          rows[String(index)] = rowVerdict;
        }
        return { ...v, rows };
      });
      setPickerRow(null);
    },
    [updateVerdict],
  );

  const playAt = useCallback((timeSec: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    if (pauseTimer.current) clearTimeout(pauseTimer.current);
    audio.currentTime = Math.max(0, timeSec - PLAY_LEAD_SEC);
    void audio.play();
    pauseTimer.current = setTimeout(
      () => audio.pause(),
      (PLAY_LEAD_SEC + PLAY_SNIPPET_SEC) * 1000,
    );
  }, []);

  const decidedCount = useCallback(
    (d: GtDraft): number => {
      const v = verdicts[d.tx8] ?? emptyVerdict();
      return d.rows.filter((row) => v.rows[String(row.index)] || row.flag === "ok").length;
    },
    [verdicts],
  );

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Dev GT Review (temporary)</p>
          <h1>GT ドラフトの試聴裁定</h1>
          <p className="hero-copy">
            行の ▶ で該当時刻を再生し、⚠ 行を裁定してください。裁定は自動保存されます。
            全録音の裁定が終わったらチャットで知らせてもらえれば、GT 化と benchmark 登録はエージェント側で行います。
          </p>
        </div>
      </section>

      {loadError ? (
        <section className="panel">
          <div className="warning-box">
            <p>{loadError}</p>
            <p className="muted">
              先に `uv run python scripts/audio-analysis/gt_draft.py &lt;tx&gt;` を実行してください。
            </p>
          </div>
        </section>
      ) : null}

      {drafts.length > 0 ? (
        <section className="panel">
          <div className="row wrap" style={{ gap: 8 }}>
            {drafts.map((d) => {
              const done = (verdicts[d.tx8] ?? emptyVerdict()).done;
              return (
                <button
                  key={d.tx8}
                  type="button"
                  className={`review-btn review-btn-small${d.tx8 === selected ? " review-btn-primary" : ""}`}
                  onClick={() => {
                    setSelected(d.tx8);
                    setPickerRow(null);
                  }}
                >
                  {done ? "✔ " : ""}
                  {d.tx8} ({decidedCount(d)}/{d.rows.length})
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      {draft ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">
                {draft.menuId ? `adversarial: ${draft.menuId}` : "自由演奏"}
              </p>
              <h2>{draft.txId.slice(0, 8)}</h2>
            </div>
            <span className="muted">
              {draft.durationSec}s / {draft.rows.length} onsets / 裁定 {decidedCount(draft)}/
              {draft.rows.length}
              {saveState[draft.tx8] === "saving" ? " / 保存中…" : ""}
              {saveState[draft.tx8] === "saved" ? " / 保存済み" : ""}
              {saveState[draft.tx8] === "error" ? " / ⚠保存失敗 (再操作で再試行)" : ""}
            </span>
          </div>

          {draft.memo ? <p className="muted">memo: {draft.memo}</p> : null}

          <div style={{ position: "sticky", top: 0, zIndex: 5, background: "var(--background, #fff)", padding: "6px 0" }}>
            <audio
              ref={audioRef}
              controls
              preload="auto"
              src={`/api/transcriptions/${draft.txId}/audio`}
              style={{ width: "100%" }}
            />
          </div>

          <div className="row wrap" style={{ gap: 8, margin: "8px 0" }}>
            <button
              type="button"
              className="review-btn review-btn-small"
              onClick={() =>
                updateVerdict(draft.tx8, (v) => {
                  const rows = { ...v.rows };
                  for (const row of draft.rows) {
                    if (!rows[String(row.index)] && row.flag !== "ok") {
                      rows[String(row.index)] = { decision: "accept" };
                    }
                  }
                  return { ...v, rows };
                })
              }
            >
              未裁定を全てドラフト通りにする
            </button>
            <label className="muted" style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={verdict.done}
                onChange={(e) =>
                  updateVerdict(draft.tx8, (v) => ({ ...v, done: e.target.checked }))
                }
              />
              この録音の裁定完了
            </label>
          </div>

          <div className="stack" style={{ gap: 6 }}>
            {draft.rows.map((row) => {
              const rv = verdict.rows[String(row.index)];
              const effective =
                rv ?? (row.flag === "ok" ? ({ decision: "accept" } as GtDraftRowVerdict) : null);
              const finalNotes =
                effective?.decision === "ignore"
                  ? null
                  : effective?.decision === "fix"
                    ? (effective.notes ?? [])
                    : effective
                      ? row.draftNotes
                      : null;
              const top1 = row.top[0];
              const showTop1Button =
                top1 && formatNotes(row.draftNotes) !== top1.note && row.flag !== "ear";
              return (
                <div
                  key={row.index}
                  className="review-card"
                  style={{
                    padding: "6px 10px",
                    opacity: effective?.decision === "ignore" ? 0.45 : 1,
                    borderLeft:
                      effective == null
                        ? "3px solid #e0a800"
                        : effective.decision === "ignore"
                          ? "3px solid #999"
                          : "3px solid #2e9e44",
                  }}
                >
                  <div className="row wrap" style={{ gap: 10, alignItems: "baseline" }}>
                    <button
                      type="button"
                      className="review-btn review-btn-small"
                      onClick={() => playAt(row.timeSec)}
                      title={`${row.timeSec.toFixed(2)}s から再生`}
                    >
                      ▶ {row.timeSec.toFixed(2)}s
                    </button>
                    <span style={{ fontFamily: "monospace" }}>
                      #{row.index} {FLAG_LABEL[row.flag]}
                    </span>
                    <span>
                      top-1: <strong>{top1 ? top1.note : "?"}</strong>
                      {top1 && !row.lowEvidence ? (
                        <span className="muted"> ({top1.share.toFixed(2)})</span>
                      ) : null}
                      {row.lowEvidence ? <span className="muted"> (score≤0)</span> : null}
                      {row.top.length > 1 ? (
                        <span className="muted">
                          {" "}
                          / {row.top.slice(1).map((c) => `${c.note} ${c.share.toFixed(2)}`).join(" / ")}
                        </span>
                      ) : null}
                    </span>
                    {row.expectedNotes ? (
                      <span className="muted">
                        期待 [{row.expectedIndex}] {formatNotes(row.expectedNotes)}
                      </span>
                    ) : null}
                    {finalNotes !== null && effective ? (
                      <span className="pill">GT: {formatNotes(finalNotes)}</span>
                    ) : null}
                    {effective?.decision === "ignore" ? <span className="pill">無視</span> : null}
                  </div>
                  <div className="row wrap" style={{ gap: 6, marginTop: 4 }}>
                    <button
                      type="button"
                      className={`review-btn review-btn-small${effective?.decision === "accept" ? " review-btn-primary" : ""}`}
                      onClick={() => setRowVerdict(draft.tx8, row.index, { decision: "accept" })}
                    >
                      ドラフト通り ({formatNotes(row.draftNotes)})
                    </button>
                    {showTop1Button ? (
                      <button
                        type="button"
                        className={`review-btn review-btn-small${
                          effective?.decision === "fix" &&
                          formatNotes(effective.notes ?? []) === top1.note
                            ? " review-btn-primary"
                            : ""
                        }`}
                        onClick={() =>
                          setRowVerdict(draft.tx8, row.index, {
                            decision: "fix",
                            notes: [top1.note],
                          })
                        }
                      >
                        top-1 ({top1.note}) にする
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className={`review-btn review-btn-small${pickerRow === row.index ? " review-btn-primary" : ""}`}
                      onClick={() => {
                        setPickerRow(pickerRow === row.index ? null : row.index);
                        setPickerNotes(
                          rv?.decision === "fix" ? (rv.notes ?? []) : row.draftNotes,
                        );
                      }}
                    >
                      修正…
                    </button>
                    <button
                      type="button"
                      className={`review-btn review-btn-small${effective?.decision === "ignore" ? " review-btn-primary" : ""}`}
                      onClick={() => setRowVerdict(draft.tx8, row.index, { decision: "ignore" })}
                    >
                      無視 (ノイズ/演奏外)
                    </button>
                  </div>
                  {pickerRow === row.index ? (
                    <div className="row wrap" style={{ gap: 4, marginTop: 6 }}>
                      {draft.tuningNotes.map((note) => (
                        <button
                          key={note}
                          type="button"
                          className={`review-btn review-btn-small${pickerNotes.includes(note) ? " review-btn-primary" : ""}`}
                          onClick={() =>
                            setPickerNotes((prev) =>
                              prev.includes(note)
                                ? prev.filter((n) => n !== note)
                                : [...prev, note],
                            )
                          }
                        >
                          {note}
                        </button>
                      ))}
                      <button
                        type="button"
                        className="review-btn review-btn-small review-btn-primary"
                        disabled={pickerNotes.length === 0}
                        onClick={() =>
                          setRowVerdict(draft.tx8, row.index, {
                            decision: "fix",
                            notes: [...pickerNotes],
                          })
                        }
                      >
                        この音で確定 ({formatNotes(pickerNotes)})
                      </button>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>

          {draft.unplacedExpected.length > 0 ? (
            <div className="warning-box" style={{ marginTop: 12 }}>
              <p>
                <strong>⚠MISS</strong> — onset が見つからなかった期待イベント。弾いた場合は再生位置を
                合わせて「ここに置く」を、弾いていなければ「破棄」を。
              </p>
              {draft.unplacedExpected.map((u) => {
                const uv = verdict.unplaced[String(u.index)];
                return (
                  <div key={u.index} className="row wrap" style={{ gap: 8, margin: "6px 0" }}>
                    <span>
                      期待 [{u.index + 1}] {formatNotes(u.notes)}
                    </span>
                    <button
                      type="button"
                      className={`review-btn review-btn-small${uv?.decision === "place" ? " review-btn-primary" : ""}`}
                      onClick={() => {
                        const t = audioRef.current?.currentTime ?? 0;
                        updateVerdict(draft.tx8, (v) => ({
                          ...v,
                          unplaced: {
                            ...v.unplaced,
                            [String(u.index)]: {
                              decision: "place",
                              timeSec: Math.round(t * 1000) / 1000,
                            },
                          },
                        }));
                      }}
                    >
                      ここに置く{uv?.decision === "place" ? ` (${uv.timeSec?.toFixed(2)}s)` : ""}
                    </button>
                    <button
                      type="button"
                      className={`review-btn review-btn-small${uv?.decision === "discard" ? " review-btn-primary" : ""}`}
                      onClick={() =>
                        updateVerdict(draft.tx8, (v) => ({
                          ...v,
                          unplaced: {
                            ...v.unplaced,
                            [String(u.index)]: { decision: "discard" },
                          },
                        }))
                      }
                    >
                      破棄 (弾いていない)
                    </button>
                  </div>
                );
              })}
            </div>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
