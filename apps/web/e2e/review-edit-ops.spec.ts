import { expect, test } from "./fixtures/ops";
import { mockTranscriptionApi } from "./fixtures/reviewMock";

// 代表修正シナリオ 3 本の操作数計測 (sprint-plan-2026-07 S4)。
// 各テストの expect(ops.count) が現行 UI での操作数の権威。UX 改修で
// フローが変わったらここの期待値を更新し、削減をコミット履歴で追う。

test("ops: FP 削除 (誤検出イベントの除去)", async ({ page, ops }) => {
  await mockTranscriptionApi(page);
  await page.goto("/score/tx-e2e/review");
  await expect(page.getByRole("heading", { name: "確認と修正" })).toBeVisible();

  // (1) イベント選択 → (2) 削除
  await ops.click(page.getByRole("button", { name: /0\.00s/ }));
  await ops.click(page.getByRole("button", { name: "このイベントを削除" }));

  await expect(page.getByText("削除済").first()).toBeVisible();
  expect(ops.count).toBe(2);
});

test("ops: FN 挿入 (候補スロットのワンタップ採用)", async ({ page, ops }) => {
  await mockTranscriptionApi(page);
  await page.goto("/score/tx-e2e/review");
  await expect(page.getByText("候補をそのまま採用:")).toBeVisible();

  // (1) 採用チップをワンタップ
  await ops.click(page.getByRole("button", { name: /＋ミ E4/ }));

  await expect(page.getByText("未保存の修正があります")).toBeVisible();
  expect(ops.count).toBe(1);
});

test("ops: 音高置換 (単音イベント C4 → E4)", async ({ page, ops }) => {
  await mockTranscriptionApi(page);
  await page.goto("/score/tx-e2e/review");
  await expect(page.getByRole("heading", { name: "確認と修正" })).toBeVisible();

  // 単音イベントは鍵盤 picker の置換モードが既定。
  // (1) イベント選択 → (2) 鍵盤で E4 をタップ = 2 操作で置換完了。
  // 旧 UI (select で追加 → × で除去) は 3 操作だった (2aaa13c 時点の計測)。
  await ops.click(page.getByRole("button", { name: /0\.00s/ }));
  await ops.click(
    page.getByRole("group", { name: "音を選択" }).getByRole("button", { name: /E4/ }),
  );

  await expect(page.getByText("未保存の修正があります")).toBeVisible();
  await expect(page.getByRole("button", { name: /0\.00s/ })).toContainText("ミ");
  expect(ops.count).toBe(2);
});
