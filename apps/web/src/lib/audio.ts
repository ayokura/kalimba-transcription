export type WavMetadata = {
  sampleRate: number;
  channels: number;
  durationSec: number;
};

export type AudioLevels = {
  peakDb: number;
  rmsDb: number;
};

// 楽器録音用のマイク constraints。ブラウザ既定の音声通話処理
// (エコーキャンセル / ノイズ抑制 / 自動ゲイン) を無効化して生音を取得する。
// これらは通話向けで、特に Android Chrome では攻撃的に効き、カリンバの持続音や
// attack をダッキング (= 「変なミュート」) して潰すため、楽器採譜では必ず切る。
// boolean 指定は exact ではない hint なので、非対応端末でも getUserMedia は失敗しない。
export const MIC_AUDIO_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    channelCount: 1,
  },
};

export async function computeBlobSha256Hex(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  const bytes = new Uint8Array(digest);
  let hex = "";
  for (const b of bytes) {
    hex += b.toString(16).padStart(2, "0");
  }
  return hex;
}

export async function computeAudioLevels(blob: Blob): Promise<AudioLevels> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const channels = audioBuffer.numberOfChannels;
    const len = audioBuffer.length;
    let peak = 0;
    let sumSq = 0;
    for (let c = 0; c < channels; c += 1) {
      const data = audioBuffer.getChannelData(c);
      for (let i = 0; i < len; i += 1) {
        const v = Math.abs(data[i]);
        if (v > peak) peak = v;
        sumSq += data[i] * data[i];
      }
    }
    const samples = channels * len;
    const rms = samples > 0 ? Math.sqrt(sumSq / samples) : 0;
    const toDb = (v: number) => (v > 0 ? 20 * Math.log10(v) : Number.NEGATIVE_INFINITY);
    return { peakDb: toDb(peak), rmsDb: toDb(rms) };
  } finally {
    await audioContext.close();
  }
}

export async function blobToWav(blob: Blob): Promise<Blob> {
  const { wavBlob } = await toWavWithMetadata(blob);
  return wavBlob;
}

export async function toWavWithMetadata(blob: Blob): Promise<{ wavBlob: Blob; metadata: WavMetadata }> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const wavBuffer = encodeWav(audioBuffer);
    return {
      wavBlob: new Blob([wavBuffer], { type: "audio/wav" }),
      metadata: {
        sampleRate: audioBuffer.sampleRate,
        channels: audioBuffer.numberOfChannels,
        durationSec: Number(audioBuffer.duration.toFixed(3)),
      },
    };
  } finally {
    await audioContext.close();
  }
}

function encodeWav(audioBuffer: AudioBuffer): ArrayBuffer {
  const channels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const channelData = Array.from({ length: channels }, (_, index) => audioBuffer.getChannelData(index));
  const sampleCount = audioBuffer.length;
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + sampleCount * blockAlign);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + sampleCount * blockAlign, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, sampleCount * blockAlign, true);

  let offset = 44;
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channelIndex][sampleIndex] ?? 0));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += bytesPerSample;
    }
  }

  return buffer;
}

function writeString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
