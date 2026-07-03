// 敵対的セルフ録音の「破壊メニュー票」(第 2 期 S1、sprint-plan-2026-07b)。
// 認識器を意図的に壊す演奏の設計図。各項目は既知の弱点機構から逆算している。
// events (音名列) は expectedPerformance として capture request に添付され、
// score_alignment 系の自動整列 → GT 化の差分確認だけで済む形にするための
// 機械可読期待シーケンス。null は自由演奏 (期待シーケンス非適用)。
// 人間可読版: docs/adversarial-recording-menu.md (この TS が正)。

export type AdversarialMenuItem = {
  id: string;
  title: string;
  /** 狙う機構 (何を壊しに行くか) */
  target: string;
  /** 予想される失敗モード */
  expectedFailure: string;
  /** 演奏指示 (ユーザー向け) */
  instructions: string;
  /** 期待イベント列 (音名の配列 = 和音)。null = 自由演奏 */
  events: string[][] | null;
  intent: "strict_chord" | "slide_chord" | "arpeggio" | "separated_notes" | "unknown";
};

export const ADVERSARIAL_MENU: AdversarialMenuItem[] = [
  {
    id: "carryover-mask",
    title: "残響マスキング (Mech2 carryover)",
    target: "carryover vs re-attack 判別 — 直前音の残響に弱い新規打鍵が飲まれる",
    expectedFailure: "2 音目 (弱い C5) の見逃し。17ea7626 の C5@11.55s 型",
    instructions:
      "B4 を強めに弾き、減衰しきる前 (約 0.5〜1 秒後) に C5 をごく弱く弾く。これを 4 回繰り返す。",
    events: [["B4"], ["C5"], ["B4"], ["C5"], ["B4"], ["C5"], ["B4"], ["C5"]],
    intent: "separated_notes",
  },
  {
    id: "dense-cluster",
    title: "密集連打 (Mech3 密集誤選択)",
    target: "密集域のスペクトル誤選択 — 近接時刻の複数音で選択を誤る",
    expectedFailure: "D5/F5 の見逃し + 弾いていない E6 の捏造 (13.323s 型)",
    instructions: "D5 と F5 をできるだけ速く交互に 4 回ずつ弾き、最後に E6 を 1 回。全体を 2 回。",
    events: [
      ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["E6"],
      ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["D5"], ["F5"], ["E6"],
    ],
    intent: "separated_notes",
  },
  {
    id: "weak-attack",
    title: "消え入る弱打 (weak attack)",
    target: "broadband spectral flux が閾値に届かない弱 attack",
    expectedFailure: "弱打の見逃し (onset 自体が立たない)",
    instructions: "C4 → E4 → G4 → C5 を、聞こえるか聞こえないかの pp で 1 音ずつ。2 巡する。",
    events: [["C4"], ["E4"], ["G4"], ["C5"], ["C4"], ["E4"], ["G4"], ["C5"]],
    intent: "separated_notes",
  },
  {
    id: "adjacent-tine",
    title: "物理隣接 tine の同時打 (#138 leakage)",
    target: "隣接 tine への振動 leak と和音構成音の分離",
    expectedFailure: "同時打の片方欠落、または leak 音の捏造",
    instructions: "物理的に隣り合う C4+E4 を同時に 4 回。次に高音側 C6+E6 を同時に 4 回。",
    events: [
      ["C4", "E4"], ["C4", "E4"], ["C4", "E4"], ["C4", "E4"],
      ["C6", "E6"], ["C6", "E6"], ["C6", "E6"], ["C6", "E6"],
    ],
    intent: "strict_chord",
  },
  {
    id: "gliss-sweep",
    title: "フルレンジ・グリッサンド",
    target: "gliss の segment 分割と方向推定 (イベント化の境界)",
    expectedFailure: "イベントの過分割/欠落、gliss 内の音の混同",
    instructions:
      "最低音から最高音まで一気に上行グリッサンド、間を置いて下行。各 2 回。速さは自由。",
    events: null,
    intent: "slide_chord",
  },
  {
    id: "mute-reattack",
    title: "即ミュート → 再打鍵 (mute-dip)",
    target: "mute-dip 再打鍵 rescue の限界 (ミュート接触音と再打鍵の判別)",
    expectedFailure: "再打鍵の見逃し、またはミュート接触音の捏造",
    instructions: "C4 を弾いて 1 秒以内に指で止め、すぐ C4 を弾き直す。これを 4 回。",
    events: [["C4"], ["C4"], ["C4"], ["C4"], ["C4"], ["C4"], ["C4"], ["C4"]],
    intent: "separated_notes",
  },
  {
    id: "dynamics-contrast",
    title: "極端なダイナミクス差",
    target: "per-recording noise floor 較正と gain 系閾値の絶対量依存",
    expectedFailure: "pp 側の全滅、または ff 直後の残響 FP",
    instructions: "C5 → E5 → G5 を pp で 1 巡、直後に同じ並びを ff で 1 巡。",
    events: [["C5"], ["E5"], ["G5"], ["C5"], ["E5"], ["G5"]],
    intent: "separated_notes",
  },
  {
    id: "tremolo",
    title: "最速トレモロ",
    target: "onset 検出の時間分解能 (wait/pre_max 窓) と反復音の統合",
    expectedFailure: "連打の数え落とし (8 回 → 数回に統合)",
    instructions: "C5 をできるだけ速く、正確に 8 回連打する (回数を意識して)。",
    events: [["C5"], ["C5"], ["C5"], ["C5"], ["C5"], ["C5"], ["C5"], ["C5"]],
    intent: "separated_notes",
  },
];
