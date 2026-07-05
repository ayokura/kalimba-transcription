// dogfooding 計測 (用途検証, docs/usage-validation-criteria.md) 用の操作ログ。
// review UI の修正操作を localStorage に自動計測し、7 分類カウント (手動の
// 正の字カウントは非現実的) と分類別所要時間 (重み較正の実測データ) を得る。
//
// 計測はあくまで副作用。失敗しても編集操作そのものを壊してはいけないため、
// 呼び出し側 (logOp) は例外を握りつぶす。

/**
 * usage-validation-criteria.md の修正操作 7 分類 + 制御系 (undo/redo) +
 * どれにも当てはまらない操作 ("other")。
 *
 * 7 分類との対応 (詳細は ReviewEditor.tsx の呼び出し箇所コメント参照):
 *   1. 偽の候補の除去      -> "candidate-remove"
 *   2. 偽の認識の除去      -> "event-remove"
 *   3. シングル/弾き直し判定 -> "restrike-judgment"
 *   4. 和音からの一部削除   -> "chord-note-remove"
 *   5. 和音への一部追加     -> "chord-note-add"
 *   6. 候補に無い onset 追加 (単音) -> "onset-insert-single"
 *   7. 候補に無い onset 追加 (複数音) -> "onset-insert-multi" (現 UI に専用の
 *      1 アクションは無く、実際には onset-insert-single + chord-note-add の
 *      組み合わせで実現される。値としては将来の専用 UI 用に確保)
 */
export type OpClass =
  | "candidate-remove"
  | "event-remove"
  | "restrike-judgment"
  | "chord-note-remove"
  | "chord-note-add"
  | "onset-insert-single"
  | "onset-insert-multi"
  | "undo"
  | "redo"
  | "other";

export type OpLogMeta = {
  /** 対象イベントの時刻 (秒)。修正必要音率の touched-notes 近似に使う */
  timeSec?: number;
  /** 対象の音名 (例 "C4")。複数音が絡む操作は複数要素 */
  notes?: string[];
};

export type OpLogEntry = {
  ts: number;
  cls: OpClass;
  meta?: OpLogMeta;
};

const STORAGE_PREFIX = "kalimba.opLog.v1.";
const MAX_ENTRIES_PER_TX = 2000;
/** 操作間の空白 (離席等) を積算しないための active-time 頭打ち秒数 */
const ACTIVE_TIME_GAP_CAP_SEC = 120;
/** touched-notes 集計での timeSec 丸め粒度 (float 誤差吸収用) */
const TOUCHED_NOTE_TIME_BUCKET_SEC = 0.05;

const OP_CLASSES: readonly OpClass[] = [
  "candidate-remove",
  "event-remove",
  "restrike-judgment",
  "chord-note-remove",
  "chord-note-add",
  "onset-insert-single",
  "onset-insert-multi",
  "undo",
  "redo",
  "other",
];

function isOpClass(value: unknown): value is OpClass {
  return typeof value === "string" && (OP_CLASSES as readonly string[]).includes(value);
}

function isOpLogMeta(value: unknown): value is OpLogMeta {
  if (value === null || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  if (v.timeSec !== undefined && typeof v.timeSec !== "number") return false;
  if (v.notes !== undefined) {
    if (!Array.isArray(v.notes) || !v.notes.every((n) => typeof n === "string")) return false;
  }
  return true;
}

function isOpLogEntry(value: unknown): value is OpLogEntry {
  if (value === null || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  if (typeof v.ts !== "number" || !isOpClass(v.cls)) return false;
  if (v.meta !== undefined && !isOpLogMeta(v.meta)) return false;
  return true;
}

function storageKey(txId: string): string {
  return `${STORAGE_PREFIX}${txId}`;
}

function getStorage(): Storage | null {
  try {
    if (typeof window === "undefined" || !window.localStorage) return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

/** 保存済み操作ログを読み出す。壊れた/型不一致のデータは黙って除外する。 */
export function loadOpLog(txId: string): OpLogEntry[] {
  const storage = getStorage();
  if (!storage) return [];
  try {
    const raw = storage.getItem(storageKey(txId));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isOpLogEntry);
  } catch {
    return [];
  }
}

/**
 * 操作を 1 件記録する。計測失敗 (localStorage 不可 / 容量超過等) が編集操作
 * そのものを壊さないよう、例外は必ずここで握りつぶす。
 *
 * 呼び出し側は「二重計上」を避けるため、React の state updater 関数 (例:
 * setState の functional 更新) の内側では呼ばないこと — StrictMode の
 * 開発時二重呼び出しでログが二重になりうる。ReviewEditor.tsx の `apply` は
 * この関数を updater の外 (関数本体) で呼ぶことでこれを回避している。
 */
export function logOp(txId: string, cls: OpClass, meta?: OpLogMeta): void {
  try {
    const storage = getStorage();
    if (!storage) return;
    const entries = loadOpLog(txId);
    entries.push(meta ? { ts: Date.now(), cls, meta } : { ts: Date.now(), cls });
    const trimmed =
      entries.length > MAX_ENTRIES_PER_TX
        ? entries.slice(entries.length - MAX_ENTRIES_PER_TX)
        : entries;
    storage.setItem(storageKey(txId), JSON.stringify(trimmed));
  } catch {
    // 計測は副作用。失敗しても編集は継続する (silent)。
  }
}

export function clearOpLog(txId: string): void {
  try {
    getStorage()?.removeItem(storageKey(txId));
  } catch {
    // noop
  }
}

export type OpLogSummary = {
  /** 分類別件数 (登場しない分類も 0 で埋める) */
  countsByClass: Record<OpClass, number>;
  totalCount: number;
  /** 操作間ギャップを ACTIVE_TIME_GAP_CAP_SEC で頭打ちして合算した秒数 */
  activeTimeSec: number;
  /** 最初〜最後の操作の壁時計経過秒数 (頭打ちなし) */
  wallTimeSec: number;
  /** 分類別の平均経過秒数 (直前操作からの経過、頭打ちあり — 重み較正用) */
  avgGapSecByClass: Partial<Record<OpClass, number>>;
  /** distinct (丸め timeSec, note) の件数 (修正必要音率の近似分子に使う) */
  touchedNoteCount: number;
  firstTs: number | null;
  lastTs: number | null;
};

function emptyCounts(): Record<OpClass, number> {
  const counts = {} as Record<OpClass, number>;
  for (const cls of OP_CLASSES) counts[cls] = 0;
  return counts;
}

export function summarizeOpLog(entries: OpLogEntry[]): OpLogSummary {
  const sorted = [...entries].sort((a, b) => a.ts - b.ts);
  const countsByClass = emptyCounts();
  const gapSumByClass = {} as Record<OpClass, number>;
  const gapCountByClass = {} as Record<OpClass, number>;
  const touched = new Set<string>();

  let activeTimeSec = 0;
  let prevTs: number | null = null;

  for (const entry of sorted) {
    countsByClass[entry.cls] += 1;

    if (prevTs !== null) {
      const gapSec = Math.max(0, (entry.ts - prevTs) / 1000);
      const cappedGapSec = Math.min(gapSec, ACTIVE_TIME_GAP_CAP_SEC);
      activeTimeSec += cappedGapSec;
      gapSumByClass[entry.cls] = (gapSumByClass[entry.cls] ?? 0) + cappedGapSec;
      gapCountByClass[entry.cls] = (gapCountByClass[entry.cls] ?? 0) + 1;
    }
    prevTs = entry.ts;

    const timeSec = entry.meta?.timeSec;
    const notes = entry.meta?.notes;
    if (typeof timeSec === "number" && notes && notes.length > 0) {
      const bucket = Math.round(timeSec / TOUCHED_NOTE_TIME_BUCKET_SEC) * TOUCHED_NOTE_TIME_BUCKET_SEC;
      for (const note of notes) {
        touched.add(`${bucket.toFixed(2)}:${note}`);
      }
    }
  }

  const avgGapSecByClass: Partial<Record<OpClass, number>> = {};
  for (const cls of OP_CLASSES) {
    const count = gapCountByClass[cls];
    if (count) avgGapSecByClass[cls] = (gapSumByClass[cls] ?? 0) / count;
  }

  const firstTs = sorted.length > 0 ? sorted[0].ts : null;
  const lastTs = sorted.length > 0 ? sorted[sorted.length - 1].ts : null;
  const wallTimeSec = firstTs !== null && lastTs !== null ? (lastTs - firstTs) / 1000 : 0;

  return {
    countsByClass,
    totalCount: sorted.length,
    activeTimeSec,
    wallTimeSec,
    avgGapSecByClass,
    touchedNoteCount: touched.size,
    firstTs,
    lastTs,
  };
}
