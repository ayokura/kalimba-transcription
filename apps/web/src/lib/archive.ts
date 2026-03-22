import type { CaptureAssessmentStatus, ManualCaptureExpectedPerformance, ManualCaptureRequestPayload } from "@/lib/api";
import type { TranscriptionResult } from "@/lib/types";

const textEncoder = new TextEncoder();

export type CaptureArchiveInput = {
  caseId: string;
  audioWav: Blob;
  requestPayload: ManualCaptureRequestPayload;
  responsePayload: TranscriptionResult;
  notes: {
    expectedNote: string;
    expectedPerformance?: ManualCaptureExpectedPerformance | null;
    memo: string;
    tester?: string;
    verdict?: CaptureAssessmentStatus;
    reviewSummary?: string;
    reviewReason?: string;
    recaptureGuidance?: string[];
  };
};

type ZipEntry = {
  name: string;
  data: Uint8Array;
};

export async function buildCaptureArchive(input: CaptureArchiveInput): Promise<Blob> {
  const folder = sanitizePathPart(input.caseId);
  const notesMd = buildNotesMarkdown(input);
  const entries: ZipEntry[] = [
    {
      name: `${folder}/audio.wav`,
      data: new Uint8Array(await input.audioWav.arrayBuffer()),
    },
    {
      name: `${folder}/request.json`,
      data: textEncoder.encode(`${JSON.stringify(input.requestPayload, null, 2)}\n`),
    },
    {
      name: `${folder}/response.json`,
      data: textEncoder.encode(`${JSON.stringify(input.responsePayload, null, 2)}\n`),
    },
    {
      name: `${folder}/notes.md`,
      data: textEncoder.encode(notesMd),
    },
  ];

  const zipBytes = createZipArchive(entries);
  return new Blob([zipBytes as unknown as BlobPart], { type: "application/zip" });
}

export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function buildNotesMarkdown(input: CaptureArchiveInput): string {
  const tester = input.notes.tester?.trim() || "manual";
  const verdict = input.notes.verdict?.trim() || "pending";
  const expectedNote = input.notes.expectedNote.trim() || "(not specified)";
  const expectedPerformance = input.notes.expectedPerformance ?? null;
  const captureIntent = input.requestPayload.captureIntent ?? "unknown";
  const memo = input.notes.memo.trim() || "(empty)";
  const reviewSummary = input.notes.reviewSummary?.trim() || "(not specified)";
  const reviewReason = input.notes.reviewReason?.trim() || "(not specified)";
  const recaptureGuidance = input.notes.recaptureGuidance?.filter((item) => item.trim().length > 0) ?? [];
  const performanceSection = buildExpectedPerformanceMarkdown(expectedPerformance);

  return [
    "# Manual Notes",
    "",
    `- tester: ${tester}`,
    `- verdict: ${verdict}`,
    `- scenario: ${input.requestPayload.scenario}`,
    `- expected note: ${expectedNote}`,
    `- capture intent: ${captureIntent}`,
    `- captured at: ${input.requestPayload.capturedAt}`,
    "",
    "## Expected Performance",
    "",
    performanceSection,
    "",
    "## Review",
    "",
    `- summary: ${reviewSummary}`,
    `- reason: ${reviewReason}`,
    "",
    "## Recapture Guidance",
    "",
    ...(recaptureGuidance.length > 0 ? recaptureGuidance.map((item) => `- ${item}`) : ["(not specified)"]),
    "",
    "## Memo",
    "",
    memo,
    "",
  ].join("\n");
}

function buildExpectedPerformanceMarkdown(expectedPerformance: ManualCaptureExpectedPerformance | null): string {
  if (!expectedPerformance || expectedPerformance.events.length === 0) {
    return "(not specified)";
  }

  const lines = [`- summary: ${expectedPerformance.summary}`, "", "### Events", ""];
  for (const event of expectedPerformance.events) {
    const keys = event.keys.map((key) => `${key.noteName} (#${key.key})`).join(", ");
    lines.push(`- ${event.index}. ${event.display} :: ${keys}`);
  }
  return lines.join("\n");
}

function sanitizePathPart(raw: string): string {
  const normalized = raw.trim().replace(/[^a-zA-Z0-9._-]/g, "-");
  return normalized.length > 0 ? normalized : "manual-capture";
}

function createZipArchive(entries: ZipEntry[]): Uint8Array {
  const localChunks: Uint8Array[] = [];
  const centralChunks: Uint8Array[] = [];
  let localOffset = 0;

  for (const entry of entries) {
    const nameBytes = textEncoder.encode(entry.name);
    const crc = crc32(entry.data);
    const localHeader = new Uint8Array(30 + nameBytes.length);
    const localView = new DataView(localHeader.buffer);

    localView.setUint32(0, 0x04034b50, true);
    localView.setUint16(4, 20, true);
    localView.setUint16(6, 0, true);
    localView.setUint16(8, 0, true);
    localView.setUint16(10, 0, true);
    localView.setUint16(12, 0, true);
    localView.setUint32(14, crc, true);
    localView.setUint32(18, entry.data.length, true);
    localView.setUint32(22, entry.data.length, true);
    localView.setUint16(26, nameBytes.length, true);
    localView.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);

    localChunks.push(localHeader, entry.data);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const centralView = new DataView(centralHeader.buffer);
    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, 0, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, 0, true);
    centralView.setUint16(14, 0, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, entry.data.length, true);
    centralView.setUint32(24, entry.data.length, true);
    centralView.setUint16(28, nameBytes.length, true);
    centralView.setUint16(30, 0, true);
    centralView.setUint16(32, 0, true);
    centralView.setUint16(34, 0, true);
    centralView.setUint16(36, 0, true);
    centralView.setUint32(38, 0, true);
    centralView.setUint32(42, localOffset, true);
    centralHeader.set(nameBytes, 46);
    centralChunks.push(centralHeader);

    localOffset += localHeader.length + entry.data.length;
  }

  const centralSize = sumLength(centralChunks);
  const endHeader = new Uint8Array(22);
  const endView = new DataView(endHeader.buffer);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, entries.length, true);
  endView.setUint16(10, entries.length, true);
  endView.setUint32(12, centralSize, true);
  endView.setUint32(16, localOffset, true);
  endView.setUint16(20, 0, true);

  const totalSize = sumLength(localChunks) + centralSize + endHeader.length;
  const archive = new Uint8Array(totalSize);
  let writeOffset = 0;

  for (const chunk of localChunks) {
    archive.set(chunk, writeOffset);
    writeOffset += chunk.length;
  }
  for (const chunk of centralChunks) {
    archive.set(chunk, writeOffset);
    writeOffset += chunk.length;
  }
  archive.set(endHeader, writeOffset);

  return archive;
}

function sumLength(chunks: Uint8Array[]): number {
  return chunks.reduce((sum, chunk) => sum + chunk.length, 0);
}

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const value of data) {
    const index = (crc ^ value) & 0xff;
    crc = (crc >>> 8) ^ CRC32_TABLE[index];
  }
  return (crc ^ 0xffffffff) >>> 0;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let c = i;
    for (let j = 0; j < 8; j += 1) {
      c = (c & 1) === 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();
