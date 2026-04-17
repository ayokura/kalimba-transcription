const STORAGE_KEY = "kalimba:recent-transactions";
const MAX_ENTRIES = 10;

export type RecentTranscription = {
  transactionId: string;
  createdAt: string;
  tuningName: string;
  eventCount: number;
};

function isRecent(value: unknown): value is RecentTranscription {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Partial<RecentTranscription>;
  return (
    typeof v.transactionId === "string" &&
    typeof v.createdAt === "string" &&
    typeof v.tuningName === "string" &&
    typeof v.eventCount === "number"
  );
}

export function loadRecentTranscriptions(): RecentTranscription[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecent);
  } catch {
    return [];
  }
}

export function pushRecentTranscription(entry: RecentTranscription): void {
  if (typeof window === "undefined") return;
  try {
    const existing = loadRecentTranscriptions().filter(
      (e) => e.transactionId !== entry.transactionId,
    );
    const next = [entry, ...existing].slice(0, MAX_ENTRIES);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // ignore quota errors
  }
}

export function removeRecentTranscription(transactionId: string): void {
  if (typeof window === "undefined") return;
  try {
    const next = loadRecentTranscriptions().filter(
      (e) => e.transactionId !== transactionId,
    );
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
}
