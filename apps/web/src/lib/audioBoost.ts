// 試聴用 WebAudio チェーン (2026-07-05)。
// - ブースト: audio 要素の volume 上限 1.0 を GainNode で超える (静音録音対策)
// - モノラル化: iPhone 録音は片チャンネル無音のステレオで、素の再生だと
//   左耳だけから鳴る。GainNode を explicit mono にしてダウンミックス →
//   destination で両耳へアップミックスする。無音 R との平均で振幅は半分に
//   なるが、補償は掛けない (真のモノラル入力に一律 ×2 するとクリップし得る
//   ため安全側)。半減分はブースト対象の静音録音では体感上問題にならない。
// createMediaElementSource は element ごとに一度しか作れないため、
// チェーンは element と一緒に保持し identity 変化で作り直す。

export type AudioBoostChain = {
  el: HTMLAudioElement;
  ctx: AudioContext;
  gain: GainNode;
};

export const BOOST_TARGET_PEAK_DB = -6;
export const BOOST_MAX_DB = 30;

export function boostDbForPeak(peakDb: number | null | undefined): number {
  if (typeof peakDb !== "number" || peakDb >= BOOST_TARGET_PEAK_DB) return 0;
  return Math.min(BOOST_TARGET_PEAK_DB - peakDb, BOOST_MAX_DB);
}

/** 再生開始時に呼ぶ (onPlay)。チェーンを (再) 構築し gain を更新する。 */
export function ensureAudioBoost(
  el: HTMLAudioElement | null,
  chainRef: { current: AudioBoostChain | null },
  boostDb: number,
): void {
  if (!el) return;
  try {
    let chain = chainRef.current;
    if (!chain || chain.el !== el) {
      if (chain) void chain.ctx.close().catch(() => {});
      const ctx = new AudioContext();
      const source = ctx.createMediaElementSource(el);
      const gain = ctx.createGain();
      // 片チャンネル無音ステレオを両耳にする: explicit mono でダウンミックス
      gain.channelCount = 1;
      gain.channelCountMode = "explicit";
      gain.channelInterpretation = "speakers";
      source.connect(gain);
      gain.connect(ctx.destination);
      chain = { el, ctx, gain };
      chainRef.current = chain;
    }
    chain.gain.gain.value = Math.pow(10, boostDb / 20);
    void chain.ctx.resume();
  } catch {
    // WebAudio 不可の環境では素の再生にフォールバック
  }
}

export function closeAudioBoost(chainRef: { current: AudioBoostChain | null }): void {
  const chain = chainRef.current;
  chainRef.current = null;
  if (chain) void chain.ctx.close().catch(() => {});
}
