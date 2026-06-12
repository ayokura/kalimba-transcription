"use client";

// タブクラッシュ対策の録音バックアップ (IndexedDB)。
// 録音/WAV 選択直後に保存し、採譜成功または「やり直す」で削除する。
// POST 前にタブが落ちた場合、次回マウント時に復元プロンプトを出すための層。

export type PendingRecording = {
  id: string;
  blob: Blob;
  source: "mic" | "file";
  tuningId: string | null;
  createdAt: number;
};

const DB_NAME = "kalimba-pending-recordings";
const DB_VERSION = 1;
const STORE_NAME = "recordings";
// これより古い録音は復元価値が低い (別端末で採譜し直している可能性が高い)
const MAX_AGE_MS = 24 * 60 * 60 * 1000;

function idbAvailable(): boolean {
  return typeof indexedDB !== "undefined";
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
  });
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDb();
  try {
    return await requestToPromise(fn(db.transaction(STORE_NAME, mode).objectStore(STORE_NAME)));
  } finally {
    db.close();
  }
}

export async function savePendingRecording(entry: PendingRecording): Promise<void> {
  if (!idbAvailable()) return;
  await withStore("readwrite", (store) => store.put(entry));
}

export async function deletePendingRecording(id: string): Promise<void> {
  if (!idbAvailable()) return;
  await withStore("readwrite", (store) => store.delete(id));
}

export async function clearPendingRecordings(): Promise<void> {
  if (!idbAvailable()) return;
  await withStore("readwrite", (store) => store.clear());
}

/** 最新の復元候補を返す。期限切れ entry はこのタイミングで掃除する。 */
export async function loadLatestPendingRecording(
  now: number = Date.now(),
): Promise<PendingRecording | null> {
  if (!idbAvailable()) return null;
  const all = await withStore("readonly", (store) => store.getAll() as IDBRequest<PendingRecording[]>);
  const expired = all.filter((e) => now - e.createdAt > MAX_AGE_MS);
  for (const e of expired) {
    await deletePendingRecording(e.id).catch(() => {});
  }
  const fresh = all.filter((e) => now - e.createdAt <= MAX_AGE_MS);
  if (fresh.length === 0) return null;
  return fresh.reduce((latest, e) => (e.createdAt > latest.createdAt ? e : latest));
}
