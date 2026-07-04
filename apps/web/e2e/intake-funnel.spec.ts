import { expect, test, type Page } from "@playwright/test";

import { transcription, tuning } from "./fixtures/reviewMock";

// SimpleHome intake funnel e2e (第 2 期 S3 最小形):
// 録音済み blob 注入 (WAV アップロード / fake-media 録音) → 採譜 → /score 遷移、
// dedup プロンプト分岐、IndexedDB pending 復元を検証する。
// マイク許可ダイアログ層はスコープ外 (chromium は fake-ui flag で常に許可)。

// toWavWithMetadata が decodeAudioData を通すため、有効な PCM WAV を生成する
function makeWavBuffer(durationSec = 0.4, sampleRate = 16000): Buffer {
  const sampleCount = Math.floor(durationSec * sampleRate);
  const buf = Buffer.alloc(44 + sampleCount * 2);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + sampleCount * 2, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16); // PCM chunk size
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(sampleCount * 2, 40);
  for (let i = 0; i < sampleCount; i += 1) {
    const value = Math.round(Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 0.5 * 32767);
    buf.writeInt16LE(value, 44 + i * 2);
  }
  return buf;
}

type FunnelMockOptions = {
  /** by-hash が既存 transaction を返す (dedup プロンプト経路) */
  dedupHit?: boolean;
};

type FunnelMockState = {
  postCount: number;
  lastPostForce: boolean;
};

async function mockIntakeApi(page: Page, options: FunnelMockOptions = {}): Promise<FunnelMockState> {
  const state: FunnelMockState = { postCount: 0, lastPostForce: false };
  await page.route("**/api/tunings", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([tuning]),
    });
  });
  await page.route("**/api/transcriptions/recent**", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
  });
  await page.route("**/api/transcriptions/by-hash/**", async (route) => {
    if (options.dedupHit) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ transactionId: "tx-e2e" }),
      });
      return;
    }
    await route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "not found" }),
    });
  });
  await page.route("**/api/transcriptions", async (route) => {
    if (route.request().method() !== "POST") {
      await route.fallback();
      return;
    }
    state.postCount += 1;
    state.lastPostForce = (route.request().postData() ?? "").includes(
      'name="force"',
    );
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(transcription),
    });
  });
  // 遷移先 /score/tx-e2e の表示分
  await page.route("**/api/transcriptions/tx-e2e", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(transcription),
    });
  });
  await page.route("**/api/transcriptions/tx-e2e/**", async (route) => {
    const url = route.request().url();
    if (url.endsWith("/audio")) {
      await route.fulfill({ status: 200, contentType: "audio/wav", body: makeWavBuffer(0.1) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
  return state;
}

async function uploadWav(page: Page) {
  await page.setInputFiles("#simple-home-file", {
    name: "recording.wav",
    mimeType: "audio/wav",
    buffer: makeWavBuffer(),
  });
  await expect(page.getByText("WAV を選択しました。")).toBeVisible();
}

test("WAV 注入 → 採譜 → /score 遷移 (主導線)", async ({ page }) => {
  const state = await mockIntakeApi(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "カリンバ譜面" })).toBeVisible();

  await uploadWav(page);
  await page.getByRole("button", { name: "自動採譜する" }).click();

  await expect(page).toHaveURL(/\/score\/tx-e2e$/);
  expect(state.postCount).toBe(1);
  expect(state.lastPostForce).toBe(false);
});

test("dedup: 既採譜の録音は プロンプト → 結果を開く で POST せず遷移", async ({ page }) => {
  const state = await mockIntakeApi(page, { dedupHit: true });
  await page.goto("/");
  await uploadWav(page);
  await page.getByRole("button", { name: "自動採譜する" }).click();

  const prompt = page.getByRole("dialog", { name: "dedup-prompt" });
  await expect(prompt).toBeVisible();
  await prompt.getByRole("button", { name: "結果を開く" }).click();

  await expect(page).toHaveURL(/\/score\/tx-e2e$/);
  expect(state.postCount).toBe(0);
});

test("dedup: 改めて採譜 は force 付きで POST して遷移", async ({ page }) => {
  const state = await mockIntakeApi(page, { dedupHit: true });
  await page.goto("/");
  await uploadWav(page);
  await page.getByRole("button", { name: "自動採譜する" }).click();

  const prompt = page.getByRole("dialog", { name: "dedup-prompt" });
  await expect(prompt).toBeVisible();
  await prompt.getByRole("button", { name: "改めて採譜" }).click();

  await expect(page).toHaveURL(/\/score\/tx-e2e$/);
  expect(state.postCount).toBe(1);
  expect(state.lastPostForce).toBe(true);
});

test("pending 復元: リロード後に復元プロンプト → 復元 → 採譜まで到達", async ({ page }) => {
  const state = await mockIntakeApi(page);
  await page.goto("/");
  // blob 注入時点で IndexedDB にバックアップされる (タブクラッシュ対策)
  await uploadWav(page);

  await page.reload();
  const prompt = page.getByRole("dialog", { name: "pending-recording-prompt" });
  await expect(prompt).toBeVisible();
  await prompt.getByRole("button", { name: "復元する" }).click();

  await expect(page.getByText("WAV を選択しました。")).toBeVisible();
  await page.getByRole("button", { name: "自動採譜する" }).click();
  await expect(page).toHaveURL(/\/score\/tx-e2e$/);
  expect(state.postCount).toBe(1);
});

test("pending 復元: 破棄すると次回リロードでプロンプトが出ない", async ({ page }) => {
  await mockIntakeApi(page);
  await page.goto("/");
  await uploadWav(page);

  await page.reload();
  const prompt = page.getByRole("dialog", { name: "pending-recording-prompt" });
  await expect(prompt).toBeVisible();
  await prompt.getByRole("button", { name: "破棄する" }).click();
  await expect(prompt).not.toBeVisible();

  await page.reload();
  await expect(page.getByRole("heading", { name: "カリンバ譜面" })).toBeVisible();
  await expect(
    page.getByRole("dialog", { name: "pending-recording-prompt" }),
  ).not.toBeVisible();
});

test("fake-media 録音 → 停止 → 採譜 → /score 遷移", async ({ page, browserName }) => {
  test.skip(browserName !== "chromium", "fake media device は chromium flag 前提");
  const state = await mockIntakeApi(page);
  await page.goto("/");

  await page.getByRole("button", { name: "録音する" }).click();
  await expect(page.getByRole("button", { name: "録音を停止" })).toBeVisible();
  await page.waitForTimeout(600);
  await page.getByRole("button", { name: "録音を停止" }).click();

  await expect(page.getByText("録音を保持しています。")).toBeVisible();
  await page.getByRole("button", { name: "自動採譜する" }).click();
  await expect(page).toHaveURL(/\/score\/tx-e2e$/);
  expect(state.postCount).toBe(1);
});
