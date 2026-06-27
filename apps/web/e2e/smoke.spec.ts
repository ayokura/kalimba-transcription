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
  // Candidate-first slot card surfaces one-tap adopt chips for dropped segments.
  await expect(page.getByText("候補をそのまま採用:")).toBeVisible();
});

test("review page shows the status panel and can set recorded_only", async ({ page }) => {
  await mockTranscriptionApi(page);
  let savedStatus: string | null = null;
  await page.route("**/api/transcriptions/tx-e2e/review-status", async (route) => {
    if (route.request().method() === "PUT") {
      const body = JSON.parse(route.request().postData() ?? "{}");
      savedStatus = body.status;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ reviewStatus: { version: 1, status: body.status } }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ reviewStatus: null }),
    });
  });

  await page.goto("/score/tx-e2e/review");
  await expect(page.getByRole("heading", { name: "確認と修正" })).toBeVisible();
  await page.getByRole("button", { name: /録音だけ提出/ }).click();
  await expect(page.getByText("状態を保存しました")).toBeVisible();
  expect(savedStatus).toBe("recorded_only");
});

test("review page adopts a dropped-segment candidate with one tap", async ({ page }) => {
  await mockTranscriptionApi(page);
  await page.goto("/score/tx-e2e/review");
  await expect(page.getByRole("heading", { name: "確認と修正" })).toBeVisible();

  // The mocked transcription has a candidate slot (dropped primary E4). The
  // candidate-first UX surfaces a one-tap adopt chip.
  await expect(page.getByText("候補をそのまま採用:")).toBeVisible();
  const adoptChip = page.getByRole("button", { name: /E4/ }).first();
  await adoptChip.click();

  // After adopting, an unsaved-change indicator appears (save becomes enabled).
  await expect(page.getByText("未保存の修正があります")).toBeVisible();
});

test("review queue lists recordings and filters by status", async ({ page }) => {
  await page.route("**/api/review-queue**", async (route) => {
    const url = new URL(route.request().url());
    const status = url.searchParams.get("status");
    const all = [
      {
        transactionId: "tx-e2e",
        createdAt: 1_700_000_000,
        tuningId: "kalimba-17-c",
        tuningName: "17 Key C Major",
        eventCount: 12,
        audioSha256: "abc",
        reviewStatus: "review_started",
        reviewStatusUpdatedAt: null,
        hasCorrections: true,
        hasMemo: false,
        warningCount: 1,
        candidateSlotCount: 2,
      },
      {
        transactionId: "tx-other",
        createdAt: 1_700_000_500,
        tuningId: "kalimba-17-c",
        tuningName: "17 Key C Major",
        eventCount: 3,
        audioSha256: "def",
        reviewStatus: "rerecord_needed",
        reviewStatusUpdatedAt: null,
        hasCorrections: false,
        hasMemo: false,
        warningCount: 0,
        candidateSlotCount: 0,
      },
    ];
    const rows = status ? all.filter((r) => r.reviewStatus === status) : all;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(rows),
    });
  });

  await page.goto("/review/queue");
  await expect(page.getByRole("heading", { name: "確認キュー" })).toBeVisible();
  await expect(page.getByText("2 件")).toBeVisible();

  await page.getByRole("button", { name: "録り直しが必要" }).click();
  await expect(page.getByText("1 件")).toBeVisible();
});
