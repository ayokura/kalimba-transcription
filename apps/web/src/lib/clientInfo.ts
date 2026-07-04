// 録音デバイス推定用の client メタデータ (2026-07-05 テスターFB)。
// UA だけでは iPadOS Safari が Mac として名乗るため判別できない —
// platform + maxTouchPoints を併記して server 側 request.json に残す
// (解釈はしない。recording-profile 較正 #173 の材料として生データを保存)。
// micLabel は録音時の MediaStreamTrack.label で、オーディオインターフェース
// 利用時は "USB Audio CODEC" 等のデバイス名が入る (内蔵マイクなら
// "iPhone マイク" 等)。ファイルアップロードでは取得不能なので null。

export type ClientDeviceInfo = {
  micLabel: string | null;
  platform: string | null;
  maxTouchPoints: number | null;
  uaPlatform: string | null;
  hardwareConcurrency: number | null;
};

export function collectClientDeviceInfo(micLabel: string | null): ClientDeviceInfo {
  if (typeof navigator === "undefined") {
    return {
      micLabel,
      platform: null,
      maxTouchPoints: null,
      uaPlatform: null,
      hardwareConcurrency: null,
    };
  }
  const uaData = (navigator as { userAgentData?: { platform?: string } }).userAgentData;
  return {
    micLabel,
    platform: navigator.platform ?? null,
    maxTouchPoints:
      typeof navigator.maxTouchPoints === "number" ? navigator.maxTouchPoints : null,
    uaPlatform: uaData?.platform ?? null,
    hardwareConcurrency:
      typeof navigator.hardwareConcurrency === "number"
        ? navigator.hardwareConcurrency
        : null,
  };
}
