import { ScoreEvent } from "@/lib/types";

export function findEventIdAtSec(events: ScoreEvent[], currentSec: number): string | null {
  if (events.length === 0 || currentSec < events[0].startTimeSec) {
    return null;
  }

  let lo = 0;
  let hi = events.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (events[mid].startTimeSec <= currentSec) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return events[lo].id;
}

export function findEventById(events: ScoreEvent[], eventId: string): ScoreEvent | null {
  return events.find((event) => event.id === eventId) ?? null;
}
