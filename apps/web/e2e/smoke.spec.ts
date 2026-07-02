import { expect, test } from "@playwright/test";

import { mockTranscriptionApi } from "./fixtures/reviewMock";

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

test("saved corrections survive re-transcription onset shifts (round-trip)", async ({ page }) => {
  await mockTranscriptionApi(page);
  // 保存時点の corrections は timeSec=0.03 (現 transcription の evt-1 は 0)。
  // 再採譜で onset が 30ms ずれた状況を模す: relaxed 突合で evt-1 に束ねられ、
  // 「削除済 + 挿入」に分解されないことを検証する。
  await page.route("**/api/transcriptions/tx-e2e/corrections", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        corrections: {
          version: 1,
          events: [{ timeSec: 0.03, notes: ["C4", "E4"], origin: "edited" }],
        },
      }),
    });
  });

  await page.goto("/score/tx-e2e/review");
  await expect(page.getByRole("heading", { name: "確認と修正" })).toBeVisible();

  // 削除済カードが存在しない = correction が recognizer イベント枠に復元された
  // (突合が壊れていると evt-1 が「削除済」+ ins-1 挿入に分解される)
  await expect(page.getByText("修正済").first()).toBeVisible();
  await expect(page.getByText("削除済")).toHaveCount(0);
  // 復元直後は保存済み状態と等価 (dirty ではない) — 保存ボタンは無効のまま
  await expect(page.getByRole("button", { name: "修正を保存" })).toBeDisabled();
});
