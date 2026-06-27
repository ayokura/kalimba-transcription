import { expect, test, type Page } from "@playwright/test";

const wavBytes = Buffer.from(
  "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=",
  "base64",
);

const tuning = {
  id: "kalimba-17-c",
  name: "17 Key C Major",
  keyCount: 2,
  tonic: "C",
  notes: [
    { key: 1, noteName: "C4", frequency: 261.626 },
    { key: 2, noteName: "E4", frequency: 329.628 },
  ],
};

const c4 = {
  key: 1,
  pitchClass: "C",
  octave: 4,
  labelDoReMi: "ド",
  labelNumber: "1",
  frequency: 261.626,
};

const e4 = {
  key: 2,
  pitchClass: "E",
  octave: 4,
  labelDoReMi: "ミ",
  labelNumber: "3",
  frequency: 329.628,
};

const transcription = {
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

async function mockTranscriptionApi(page: Page) {
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

test("browser-only WASM demo page renders", async ({ page }) => {
  await page.goto("/wasm-demo");

  await expect(page.getByRole("heading", { name: /WASM onset/ })).toBeVisible();
  await expect(page.getByText(/zero server round-trip/)).toBeVisible();
  await expect(page.locator('input[type="file"]')).toBeVisible();
});

test("score page can open the persistent review page with mocked transaction data", async ({ page }) => {
  await mockTranscriptionApi(page);

  await page.goto("/score/tx-e2e");

  await expect(page.getByRole("heading", { name: "カリンバ譜面" })).toBeVisible();
  await expect(page.getByText("ド").first()).toBeVisible();

  await page.getByRole("link", { name: /結果を確認・修正する/ }).click();

  await expect(page).toHaveURL(/\/score\/tx-e2e\/review$/);
  await expect(page.getByRole("heading", { name: "確認と修正" })).toBeVisible();
  await expect(page.getByRole("button", { name: /保存/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /追加/ }).first()).toBeVisible();
});
