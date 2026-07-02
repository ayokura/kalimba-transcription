import type { Page } from "@playwright/test";

// review 編集系 e2e の共有 mock。tx-e2e は「単音 C4 イベント 1 つ +
// 次候補 E4 (alternateGroupings) + dropped 候補スロット E4」という
// FP 削除 / FN 挿入 / 音高置換の 3 シナリオを全部演じられる最小構成。

export const wavBytes = Buffer.from(
  "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=",
  "base64",
);

export const tuning = {
  id: "kalimba-17-c",
  name: "17 Key C Major",
  keyCount: 2,
  tonic: "C",
  notes: [
    { key: 1, noteName: "C4", frequency: 261.626 },
    { key: 2, noteName: "E4", frequency: 329.628 },
  ],
};

export const c4 = {
  key: 1,
  pitchClass: "C",
  octave: 4,
  labelDoReMi: "ド",
  labelNumber: "1",
  frequency: 261.626,
};

export const e4 = {
  key: 2,
  pitchClass: "E",
  octave: 4,
  labelDoReMi: "ミ",
  labelNumber: "3",
  frequency: 329.628,
};

export const transcription = {
  transactionId: "tx-e2e",
  instrumentTuning: tuning,
  tempo: 120,
  events: [
    {
      id: "evt-1",
      startBeat: 0,
      durationBeat: 1,
      startTimeSec: 0,
      durationSec: 0.5,
      notes: [c4],
      isGlissLike: false,
      gesture: "ambiguous",
      alternateGroupings: [
        {
          combinesWith: null,
          combinedNotes: null,
          splitInto: null,
          alternateNote: e4,
          reason: "soft_rejected:test",
          confidence: 0.35,
        },
      ],
    },
  ],
  candidateSlots: [
    {
      startTime: 1,
      endTime: 1.2,
      primaryNote: e4,
      candidates: [],
      dropReason: "orphan-onset-no-segment",
      confidence: 0.5,
    },
  ],
  notationViews: {
    western: ["C4"],
    numbered: ["1"],
    verticalDoReMi: [["ド"]],
  },
  tuningMismatch: null,
  warnings: [],
  debug: null,
};

export async function mockTranscriptionApi(page: Page) {
  await page.route("**/api/transcriptions/tx-e2e", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(transcription),
    });
  });
  await page.route("**/api/transcriptions/tx-e2e/audio", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "audio/wav",
      body: wavBytes,
    });
  });
  await page.route("**/api/transcriptions/tx-e2e/memo", async (route) => {
    if (route.request().method() === "PUT") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ memo: "" }) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ memo: "" }) });
  });
  await page.route("**/api/transcriptions/tx-e2e/corrections", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ corrections: null }),
    });
  });
  await page.route("**/api/tunings", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([tuning]),
    });
  });
}
