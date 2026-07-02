import { test as base, type Locator } from "@playwright/test";

// 修正シナリオの「操作数」を数える fixture。Playwright の action を
// ops.click / ops.select 経由で発行した回数 = ユーザーのタップ/選択操作数
// として数える (audio スクラブ等の付随操作は数えない)。
// 計測値はテスト内の expect で固定し、annotation (type: "ops") にも残す。

export type Ops = {
  readonly count: number;
  click(locator: Locator): Promise<void>;
  select(locator: Locator, value: string): Promise<void>;
};

export const test = base.extend<{ ops: Ops }>({
  ops: async ({}, use, testInfo) => {
    let count = 0;
    const ops: Ops = {
      get count() {
        return count;
      },
      async click(locator) {
        count += 1;
        await locator.click();
      },
      async select(locator, value) {
        count += 1;
        await locator.selectOption(value);
      },
    };
    await use(ops);
    testInfo.annotations.push({ type: "ops", description: String(count) });
  },
});

export { expect } from "@playwright/test";
