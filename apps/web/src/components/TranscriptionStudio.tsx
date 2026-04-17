"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { ExpectedKeySelector } from "@/components/ExpectedKeySelector";
import { InitialResultSummary } from "@/components/InitialResultSummary";
import { NotationPanel } from "@/components/NotationPanel";
import { RecorderPanel } from "@/components/RecorderPanel";
import { TuningPanel } from "@/components/TuningPanel";
import { removeReviewAudio, saveReviewAudio } from "@/lib/reviewAudioStore";
import { createReviewSession, removeReviewSession, saveReviewSession } from "@/lib/reviewSession";
import {
  CaptureAssessmentDetails,
  CaptureIntent,
  ManualCaptureExpectedEvent,
  ManualCaptureExpectedPart,
  ManualCaptureExpectedPerformance,
  TranscriptionCapture,
  createTranscriptionWithCapture,
  fetchTunings,
} from "@/lib/api";
import { buildCaptureArchive, downloadBlob } from "@/lib/archive";
import { InstrumentTuning, NotationMode, TranscriptionResult, TuningNote } from "@/lib/types";

type TranscriptionStudioProps = {
  mode: "user" | "debug";
};

const INTENT_OPTIONS: Array<{ value: CaptureIntent; label: string; description: string }> = [
  { value: "strict_chord", label: "同時和音", description: "同じタイミングで鳴らす前提です。" },
  { value: "slide_chord", label: "スライド和音", description: "少しずらしてなぞる和音をまとめて扱います。" },
  { value: "arpeggio", label: "アルペジオ", description: "和音を順に分散して鳴らす意図です。" },
  { value: "separated_notes", label: "単音列", description: "明確に区切った単音列です。" },
  { value: "unknown", label: "未指定", description: "意図をまだ固定しません。" },
];

function normalizeCaptureIntentFamily(intent: string | null | undefined): CaptureIntent | "ambiguous" {
  if (intent === "strict_chord" || intent === "slide_chord" || intent === "arpeggio" || intent === "separated_notes" || intent === "unknown" || intent === "ambiguous") {
    return intent;
  }
  return "unknown";
}

function buildCustomTuning(customName: string, customNotes: string): InstrumentTuning {
  const notes = customNotes
    .split(",")
    .map((noteName, index) => ({
      key: index + 1,
      noteName: noteName.trim(),
      frequency: 0,
    }))
    .filter((note) => note.noteName.length > 0);

  return {
    id: "custom",
    name: customName,
    keyCount: notes.length,
    notes,
  };
}

function slugifyCasePart(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/\+/g, " plus ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");
}

function buildEventSlug(event: ManualCaptureExpectedEvent) {
  return slugifyCasePart(event.display || event.keys.map((key) => key.noteName).join("-"));
}

function areContiguousEventKeys(event: ManualCaptureExpectedEvent) {
  const sortedKeys = event.keys.map((key) => key.key).sort((left, right) => left - right);
  if (sortedKeys.length < 2) {
    return false;
  }
  return sortedKeys[sortedKeys.length - 1] - sortedKeys[0] + 1 === sortedKeys.length;
}

function inferGlissDirection(events: ManualCaptureExpectedEvent[]) {
  const firstAverage = events[0].keys.reduce((sum, key) => sum + key.key, 0) / events[0].keys.length;
  const lastAverage = events[events.length - 1].keys.reduce((sum, key) => sum + key.key, 0) / events[events.length - 1].keys.length;
  if (lastAverage > firstAverage + 0.5) {
    return "ascending";
  }
  if (lastAverage < firstAverage - 0.5) {
    return "descending";
  }
  return "mixed";
}

function buildScenarioSlug(expectedPerformance: ManualCaptureExpectedPerformance | null, captureIntent: CaptureIntent) {
  const events = expectedPerformance?.events ?? [];
  const captureIntentFamily = normalizeCaptureIntentFamily(captureIntent);
  const intentSlug = captureIntentFamily !== "unknown" ? `-${captureIntentFamily.replace(/_/g, "-")}` : "";
  if (events.length === 0) {
    return `manual-test${intentSlug}`;
  }

  const uniqueDisplays = new Set(events.map((event) => event.display));
  const firstEvent = events[0];
  const lastEvent = events[events.length - 1];

  if (uniqueDisplays.size === 1) {
    return `${buildEventSlug(firstEvent)}-repeat-${String(events.length).padStart(2, "0")}${intentSlug}`;
  }

  const sameWidth = events.every((event) => event.keys.length === firstEvent.keys.length);
  const contiguousSweep = sameWidth && events.every(areContiguousEventKeys);
  if (contiguousSweep) {
    return `${firstEvent.keys.length}-note-${inferGlissDirection(events)}-slide-${buildEventSlug(firstEvent)}-to-${buildEventSlug(lastEvent)}${intentSlug}`;
  }

  return `${buildEventSlug(firstEvent)}-to-${buildEventSlug(lastEvent)}-sequence-${String(events.length).padStart(2, "0")}${intentSlug}`;
}

function buildDefaultCaptureIdForValues(
  generatedAt: string,
  tuning: InstrumentTuning | null,
  expectedPerformance: ManualCaptureExpectedPerformance | null,
  captureIntent: CaptureIntent,
) {
  const day = generatedAt.slice(0, 10);
  const scenario = buildScenarioSlug(expectedPerformance, captureIntent);
  const tuningId = tuning?.id ?? "capture";
  return `${day}-${scenario}-${tuningId}`;
}

function buildExpectedDisplay(notes: TuningNote[]): string {
  return notes.map((note) => note.noteName).join(" + ");
}

function buildExpectedDisplayFromKeys(notes: Array<{ noteName: string }>): string {
  return notes.map((note) => note.noteName).join(" + ");
}

function parseCaptureIntentValue(value: unknown): CaptureIntent | null | "invalid" {
  if (value == null) {
    return null;
  }
  if (typeof value !== "string") {
    return "invalid";
  }
  const normalized = normalizeCaptureIntentFamily(value.trim());
  return normalized === "ambiguous" ? "invalid" : normalized;
}

function buildPartBreakdown(event: ManualCaptureExpectedEvent): string | null {
  if (!event.parts || event.parts.length === 0) {
    return null;
  }
  return event.parts.map((part) => {
    const label = part.display?.trim() || buildExpectedDisplayFromKeys(part.keys);
    return part.intent ? `${buildIntentLabel(part.intent)}(${label})` : label;
  }).join(" / ");
}

function resolveExpectedNotes(
  rawValues: unknown,
  tuning: InstrumentTuning,
  noteByName: Map<string, TuningNote>,
): { notes: TuningNote[]; error: string | null } {
  if (!Array.isArray(rawValues) || rawValues.length === 0) {
    return { notes: [], error: "音配列が空です。" };
  }

  const noteByKey = new Map(tuning.notes.map((note) => [note.key, note]));
  const resolvedNotes: TuningNote[] = [];
  for (const rawValue of rawValues) {
    let resolved: TuningNote | undefined;
    if (typeof rawValue === "string") {
      resolved = noteByName.get(rawValue.trim().toUpperCase());
      if (!resolved) {
        return { notes: [], error: `調律に存在しない音名です: ${rawValue}` };
      }
    } else if (typeof rawValue === "number" && Number.isInteger(rawValue)) {
      resolved = noteByKey.get(rawValue);
      if (!resolved) {
        return { notes: [], error: `調律に存在しないキー番号です: ${rawValue}` };
      }
    } else {
      return { notes: [], error: `音配列の値を解釈できませんでした: ${String(rawValue)}` };
    }

    if (!resolvedNotes.some((note) => note.key === resolved.key)) {
      resolvedNotes.push(resolved);
    }
  }

  resolvedNotes.sort((left, right) => left.key - right.key);
  return { notes: resolvedNotes, error: null };
}

function parseJsonExpectedPerformanceText(
  value: string,
  tuning: InstrumentTuning,
  fallbackIntent: CaptureIntent,
): { events: ManualCaptureExpectedEvent[]; error: string | null; suggestedCaptureIntent?: CaptureIntent } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    return { events: [], error: "JSON として解釈できませんでした。" };
  }

  let rawEvents: unknown;
  let suggestedCaptureIntent: CaptureIntent | undefined;
  if (Array.isArray(parsed)) {
    rawEvents = parsed;
  } else if (parsed && typeof parsed === "object" && Array.isArray((parsed as { events?: unknown }).events)) {
    rawEvents = (parsed as { events: unknown[] }).events;
    const defaultCaptureIntent = parseCaptureIntentValue((parsed as { defaultCaptureIntent?: unknown }).defaultCaptureIntent);
    if (defaultCaptureIntent === "invalid") {
      return { events: [], error: "defaultCaptureIntent が不正です。" };
    }
    if (defaultCaptureIntent) {
      suggestedCaptureIntent = defaultCaptureIntent;
    }
  } else {
    return { events: [], error: "JSON は expectedPerformance オブジェクトか event 配列で指定してください。" };
  }

  const noteByName = new Map(
    tuning.notes.map((note) => [note.noteName.trim().toUpperCase(), note]),
  );
  const events: ManualCaptureExpectedEvent[] = [];

  for (const [eventIndex, rawEvent] of (rawEvents as unknown[]).entries()) {
    if (!rawEvent || typeof rawEvent !== "object") {
      return { events: [], error: `events[${eventIndex + 1}] を解釈できませんでした。` };
    }

    const rawEventRecord = rawEvent as {
      notes?: unknown;
      keys?: unknown;
      display?: unknown;
      intent?: unknown;
      repeat?: unknown;
      parts?: unknown;
    };

    const eventIntent = parseCaptureIntentValue(rawEventRecord.intent);
    if (eventIntent === "invalid") {
      return { events: [], error: `events[${eventIndex + 1}].intent が不正です。` };
    }

    const repeatCount = Math.max(1, Number.parseInt(String(rawEventRecord.repeat ?? "1"), 10) || 1);
    const directNoteSource = rawEventRecord.notes ?? rawEventRecord.keys ?? null;
    if (directNoteSource && rawEventRecord.parts) {
      return { events: [], error: `events[${eventIndex + 1}] は notes/keys と parts を同時に指定できません。` };
    }

    let eventNotes: TuningNote[] = [];
    let eventParts: ManualCaptureExpectedPart[] | null = null;

    if (rawEventRecord.parts != null) {
      if (!Array.isArray(rawEventRecord.parts) || rawEventRecord.parts.length === 0) {
        return { events: [], error: `events[${eventIndex + 1}].parts が空です。` };
      }

      const mergedNotes = new Map<number, TuningNote>();
      eventParts = [];
      for (const [partIndex, rawPart] of rawEventRecord.parts.entries()) {
        if (!rawPart || typeof rawPart !== "object") {
          return { events: [], error: `events[${eventIndex + 1}].parts[${partIndex + 1}] を解釈できませんでした。` };
        }
        const rawPartRecord = rawPart as { notes?: unknown; keys?: unknown; display?: unknown; intent?: unknown };
        const resolvedPart = resolveExpectedNotes(rawPartRecord.notes ?? rawPartRecord.keys, tuning, noteByName);
        if (resolvedPart.error) {
          return { events: [], error: `events[${eventIndex + 1}].parts[${partIndex + 1}]: ${resolvedPart.error}` };
        }
        const partIntent = parseCaptureIntentValue(rawPartRecord.intent);
        if (partIntent === "invalid") {
          return { events: [], error: `events[${eventIndex + 1}].parts[${partIndex + 1}].intent が不正です。` };
        }

        for (const note of resolvedPart.notes) {
          mergedNotes.set(note.key, note);
        }
        eventParts.push({
          keys: resolvedPart.notes.map((note) => ({ key: note.key, noteName: note.noteName })),
          display: typeof rawPartRecord.display === "string" && rawPartRecord.display.trim().length > 0
            ? rawPartRecord.display.trim()
            : buildExpectedDisplay(resolvedPart.notes),
          intent: partIntent,
        });
      }

      eventNotes = Array.from(mergedNotes.values()).sort((left, right) => left.key - right.key);
    } else {
      const resolvedEvent = resolveExpectedNotes(directNoteSource, tuning, noteByName);
      if (resolvedEvent.error) {
        return { events: [], error: `events[${eventIndex + 1}]: ${resolvedEvent.error}` };
      }
      eventNotes = resolvedEvent.notes;
    }

    if (eventNotes.length === 0) {
      return { events: [], error: `events[${eventIndex + 1}] の音が空です。` };
    }

    const partIntents = [...new Set((eventParts ?? []).map((part) => part.intent).filter((intent): intent is CaptureIntent => Boolean(intent)))];
    const resolvedEventIntent = eventIntent ?? (partIntents.length === 1 ? partIntents[0] : null);
    const display = typeof rawEventRecord.display === "string" && rawEventRecord.display.trim().length > 0
      ? rawEventRecord.display.trim()
      : buildExpectedDisplay(eventNotes);

    for (let repeatIndex = 0; repeatIndex < repeatCount; repeatIndex += 1) {
      events.push({
        index: events.length + 1,
        keys: eventNotes.map((note) => ({ key: note.key, noteName: note.noteName })),
        display,
        intent: resolvedEventIntent ?? (fallbackIntent !== "unknown" ? fallbackIntent : null),
        parts: eventParts,
      });
    }
  }

  if (!suggestedCaptureIntent) {
    const explicitIntents = [...new Set(events.map((event) => event.intent).filter((intent): intent is CaptureIntent => Boolean(intent)))];
    if (explicitIntents.length === 1) {
      suggestedCaptureIntent = explicitIntents[0];
    } else if (explicitIntents.length > 1) {
      suggestedCaptureIntent = "unknown";
    }
  }

  return { events, error: null, suggestedCaptureIntent };
}

function buildExpectedSummary(events: ManualCaptureExpectedEvent[]): string {
  if (events.length === 0) {
    return "";
  }

  const orderedCounts = new Map<string, number>();
  for (const event of events) {
    orderedCounts.set(event.display, (orderedCounts.get(event.display) ?? 0) + 1);
  }

  return Array.from(orderedCounts.entries())
    .map(([display, count]) => (count > 1 ? `${display} x ${count}` : display))
    .join(" / ");
}

function parseExpectedPerformanceText(
  value: string,
  tuning: InstrumentTuning,
  fallbackIntent: CaptureIntent,
): { events: ManualCaptureExpectedEvent[]; error: string | null; suggestedCaptureIntent?: CaptureIntent } {
  const trimmed = value.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    return parseJsonExpectedPerformanceText(trimmed, tuning, fallbackIntent);
  }

  const normalized = value
    .replace(/\r\n/g, "\n")
    .replace(/→/g, "/")
    .replace(/->/g, "/")
    .replace(/／/g, "/");

  const rawTokens = normalized
    .split(/(?:\n|\/)+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0);

  if (rawTokens.length === 0) {
    return { events: [], error: "インポートする expected performance が空です。" };
  }

  const noteByName = new Map(
    tuning.notes.map((note) => [note.noteName.trim().toUpperCase(), note]),
  );

  const events: ManualCaptureExpectedEvent[] = [];

  for (const rawToken of rawTokens) {
    const repeatMatch = rawToken.match(/^(.*?)(?:\s*[x×]\s*(\d+))$/i);
    const tokenWithoutRepeat = repeatMatch?.[1]?.trim() || rawToken;
    const repeatCount = Math.max(1, Number.parseInt(repeatMatch?.[2] ?? "1", 10) || 1);
    const intentMatch = tokenWithoutRepeat.match(/^(strict_chord|slide_chord|arpeggio|separated_notes|unknown)\s*:\s*(.+)$/i);
    const eventIntent = intentMatch ? normalizeCaptureIntentFamily(intentMatch[1]) : null;
    const eventBody = intentMatch?.[2]?.trim() || tokenWithoutRepeat;

    const noteTokens = eventBody
      .split("+")
      .map((token) => token.trim())
      .filter((token) => token.length > 0);

    if (noteTokens.length === 0) {
      return { events: [], error: `event を解釈できませんでした: ${rawToken}` };
    }

    const resolvedNotes: TuningNote[] = [];
    for (const noteToken of noteTokens) {
      const resolved = noteByName.get(noteToken.toUpperCase());
      if (!resolved) {
        return { events: [], error: `調律に存在しない音名です: ${noteToken}` };
      }
      if (!resolvedNotes.some((note) => note.key === resolved.key)) {
        resolvedNotes.push(resolved);
      }
    }

    resolvedNotes.sort((left, right) => left.key - right.key);
    const display = buildExpectedDisplay(resolvedNotes);
    for (let index = 0; index < repeatCount; index += 1) {
      events.push({
        index: events.length + 1,
        keys: resolvedNotes.map((note) => ({ key: note.key, noteName: note.noteName })),
        display,
        intent: eventIntent && eventIntent !== "ambiguous"
          ? eventIntent
          : fallbackIntent !== "unknown"
            ? fallbackIntent
            : null,
      });
    }
  }

  return { events, error: null, suggestedCaptureIntent: fallbackIntent !== "unknown" ? fallbackIntent : undefined };
}

function buildExpectedPerformance(events: ManualCaptureExpectedEvent[], fallbackIntent: CaptureIntent): ManualCaptureExpectedPerformance | null {
  if (events.length === 0) {
    return null;
  }

  const explicitIntents = [...new Set(events.map((event) => event.intent).filter((intent): intent is CaptureIntent => Boolean(intent)))];
  const defaultCaptureIntent = explicitIntents.length === 1
    ? explicitIntents[0]
    : fallbackIntent !== "unknown"
      ? fallbackIntent
      : null;

  return {
    source: "clickable-kalimba-ui",
    version: 1,
    summary: buildExpectedSummary(events),
    defaultCaptureIntent,
    events,
  };
}

function buildExpectedEventDisplay(event: ManualCaptureExpectedEvent) {
  return [...event.keys]
    .sort((left, right) => left.key - right.key)
    .map((key) => key.noteName)
    .join(" + ");
}

function buildDetectedEventDisplay(event: TranscriptionResult["events"][number], noteNamesByKey: Map<number, string>) {
  return [...event.notes]
    .sort((left, right) => left.key - right.key)
    .map((note) => noteNamesByKey.get(note.key) ?? `${note.pitchClass}${note.octave}`)
    .join(" + ");
}

function buildIntentLabel(intent: string | null | undefined) {
  const normalizedIntent = normalizeCaptureIntentFamily(intent);
  if (normalizedIntent === "ambiguous") {
    return "要確認";
  }
  return INTENT_OPTIONS.find((option) => option.value === normalizedIntent)?.label ?? "未指定";
}

function buildCaptureAssessment(
  expectedPerformance: ManualCaptureExpectedPerformance | null,
  result: TranscriptionResult | null,
  noteNamesByKey: Map<number, string>,
  captureIntent: CaptureIntent | null,
): CaptureAssessmentDetails | null {
  if (!expectedPerformance || expectedPerformance.events.length === 0 || !result) {
    return null;
  }

  const expectedEvents = expectedPerformance.events.map(buildExpectedEventDisplay);
  const detectedEvents = result.events.map((event) => buildDetectedEventDisplay(event, noteNamesByKey));
  const gestureCounts = result.events.reduce<Record<string, number>>((counts, event) => {
    counts[event.gesture] = (counts[event.gesture] ?? 0) + 1;
    return counts;
  }, {});
  const dominantDetectedGesture = normalizeCaptureIntentFamily(
    Object.entries(gestureCounts).sort((left, right) => right[1] - left[1])[0]?.[0] ?? "ambiguous",
  );
  const total = Math.max(expectedEvents.length, detectedEvents.length);
  const events = Array.from({ length: total }, (_, index) => ({
    index: index + 1,
    expected: expectedEvents[index] ?? null,
    detected: detectedEvents[index] ?? null,
    matches: (expectedEvents[index] ?? null) === (detectedEvents[index] ?? null),
  }));

  const normalizedCaptureIntent = normalizeCaptureIntentFamily(captureIntent);
  const mismatchCount = events.filter((event) => !event.matches).length;
  const extraEventCount = Math.max(0, detectedEvents.length - expectedEvents.length);
  const missingEventCount = Math.max(0, expectedEvents.length - detectedEvents.length);

  if (mismatchCount === 0) {
    return {
      status: "completed",
      label: "一致",
      summary: "期待演奏と検出結果が一致しています。",
      reason: "event 数と各 note-set がすべて一致しています。fixture 化して regression に昇格できます。",
      mismatchCount,
      expectedEventCount: expectedEvents.length,
      detectedEventCount: detectedEvents.length,
      extraEventCount,
      missingEventCount,
      events,
    };
  }

  const severeFragmentation =
    detectedEvents.length >= expectedEvents.length + 2 ||
    (expectedEvents.length > 0 && detectedEvents.length >= Math.ceil(expectedEvents.length * 1.5));

  if (severeFragmentation) {
    if (normalizedCaptureIntent === "strict_chord" && dominantDetectedGesture === "slide_chord") {
      return {
        status: "rerecord",
        label: "再録音",
        summary: "strict chord としては取り直し推奨です。",
        reason: `${buildIntentLabel(normalizedCaptureIntent)} を期待しましたが、検出側は ${buildIntentLabel(dominantDetectedGesture)} 優勢でした。同時打鍵の厳密さを上げた再録音を推奨します。`,
        mismatchCount,
        expectedEventCount: expectedEvents.length,
        detectedEventCount: detectedEvents.length,
        extraEventCount,
        missingEventCount,
        events,
      };
    }

    if (normalizedCaptureIntent === "slide_chord" || normalizedCaptureIntent === "arpeggio") {
      return {
        status: "pending",
        label: "改善対象",
        summary: "録音意図に対して segmentation がまだ粗いです。",
        reason: `${buildIntentLabel(normalizedCaptureIntent)} としては現実的な崩れ方です。検出側は ${buildIntentLabel(dominantDetectedGesture)} 優勢でした。認識改善ターゲットとして保持してください。`,
        mismatchCount,
        expectedEventCount: expectedEvents.length,
        detectedEventCount: detectedEvents.length,
        extraEventCount,
        missingEventCount,
        events,
      };
    }

    return {
      status: "review_needed",
      label: "要確認",
      summary: "event の分割または束ね方に大きな差があります。",
      reason: `${buildIntentLabel(normalizedCaptureIntent)} の想定に対して差が大きく、検出側は ${buildIntentLabel(dominantDetectedGesture)} 優勢でした。録音意図と diff を確認してから fixture status を決めてください。`,
      mismatchCount,
      expectedEventCount: expectedEvents.length,
      detectedEventCount: detectedEvents.length,
      extraEventCount,
      missingEventCount,
      events,
    };
  }

  return {
    status: "pending",
    label: "改善対象",
    summary: "一部の note-set または event 数がずれています。",
    reason: `録音意図: ${buildIntentLabel(normalizedCaptureIntent)} / 検出傾向: ${buildIntentLabel(dominantDetectedGesture)}。認識改善の対象です。必要なら expected と detected の差分を見て、演奏意図の再確認も行ってください。`,
    mismatchCount,
    expectedEventCount: expectedEvents.length,
    detectedEventCount: detectedEvents.length,
    extraEventCount,
    missingEventCount,
    events,
  };
}

function buildRecaptureGuidance(
  review: CaptureAssessmentDetails | null,
  captureIntent: CaptureIntent,
): string[] {
  if (!review || review.status === "completed") {
    return [];
  }

  const normalizedCaptureIntent = normalizeCaptureIntentFamily(captureIntent);
  const guidance: string[] = [];
  if (normalizedCaptureIntent === "strict_chord") {
    guidance.push(
      "各反復で対象キーを同時に弾き、指をずらして順に入れない。",
      "各反復の間に明確な無音を入れる。",
      "和音の開始を揃え、スライド和音にならないようにする。",
    );
  } else if (normalizedCaptureIntent === "slide_chord") {
    guidance.push(
      "毎回ほぼ同じ方向と速さで順に鳴らす。",
      "各反復の間に明確な無音を入れる。",
      "対象キー以外に触れて出る接触音を減らす。",
    );
  } else if (normalizedCaptureIntent === "arpeggio") {
    guidance.push(
      "和音を一方向に分散して鳴らし、各音の順序を毎回そろえる。",
      "各反復の間に明確な無音を入れる。",
      "スライドのように連続接触させず、順次打鍵として分ける。",
    );
  } else if (normalizedCaptureIntent === "separated_notes") {
    guidance.push(
      "各音をはっきり区切り、次の音まで十分に待つ。",
      "低音残響が長い場合でも、新しい打鍵の間隔を広めに取る。",
    );
  }

  if (review.status === "rerecord") {
    guidance.push("今回の録音は意図と検出結果の差が大きいため、同じ expected performance のまま録り直して比較する。");
  } else if (review.status === "review_needed") {
    guidance.push("再録音前に expected と detected の差分を見て、演奏意図自体が正しいか確認する。");
  }

  return guidance;
}

function buildResultOnlyAssessment(result: TranscriptionResult | null): CaptureAssessmentDetails | null {
  if (!result || result.warnings.length === 0) {
    return null;
  }

  return {
    status: "review_needed",
    label: "要確認",
    summary: "自動採譜結果の確認が必要です。",
    reason: result.warnings.join(" / "),
    mismatchCount: 0,
    expectedEventCount: 0,
    detectedEventCount: result.events.length,
    extraEventCount: 0,
    missingEventCount: 0,
    events: [],
  };
}

function buildDraftCaptureScenario(captureCaseId: string, suggestedCaptureId: string) {
  return captureCaseId.trim().length > 0 ? captureCaseId.trim() : suggestedCaptureId;
}

function buildTuningSignature(tuning: InstrumentTuning | null | undefined) {
  if (!tuning) {
    return "none";
  }
  const noteSignature = tuning.notes.map((note) => `${note.key}:${note.noteName}`).join("|");
  return `${tuning.id}:${noteSignature}`;
}

function normalizeOptionalText(value: string | null | undefined) {
  const trimmed = value?.trim() ?? "";
  return trimmed.length > 0 ? trimmed : null;
}

function serializeExpectedPerformance(expectedPerformance: ManualCaptureExpectedPerformance | null) {
  return JSON.stringify(expectedPerformance ?? null);
}

function hasProtectedExpectedDraft(
  pendingExpectedKeys: number[],
  expectedEvents: ManualCaptureExpectedEvent[],
  expectedImportText: string,
) {
  return pendingExpectedKeys.length > 0 || expectedEvents.length > 0 || expectedImportText.trim().length > 0;
}

export function TranscriptionStudio({ mode }: TranscriptionStudioProps) {
  const isDebug = mode === "debug";
  const router = useRouter();
  const [tunings, setTunings] = useState<InstrumentTuning[]>([]);
  const [selectedTuningId, setSelectedTuningId] = useState<string>("");
  const [customName, setCustomName] = useState("Custom Tuning");
  const [customNotes, setCustomNotes] = useState("C4,D4,E4,G4,A4,C5,E5,G5");
  const [recording, setRecording] = useState<Blob | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [notationMode, setNotationMode] = useState<NotationMode>(isDebug ? "western" : "score");
  const [activeEventId, setActiveEventId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [captureCaseId, setCaptureCaseId] = useState("");
  const [captureMemo, setCaptureMemo] = useState("");
  const [captureIntent, setCaptureIntent] = useState<CaptureIntent>("unknown");
  const [lastCapture, setLastCapture] = useState<TranscriptionCapture | null>(null);
  const [latestReviewSessionId, setLatestReviewSessionId] = useState<string | null>(null);
  const [isSavingCapture, setIsSavingCapture] = useState(false);
  const [pendingExpectedKeys, setPendingExpectedKeys] = useState<number[]>([]);
  const [expectedRepeatCount, setExpectedRepeatCount] = useState("1");
  const [expectedEvents, setExpectedEvents] = useState<ManualCaptureExpectedEvent[]>([]);
  const [expectedImportText, setExpectedImportText] = useState("");
  const [expectedImportError, setExpectedImportError] = useState<string | null>(null);

  useEffect(() => {
    fetchTunings()
      .then((loadedTunings) => {
        setTunings(loadedTunings);
        setSelectedTuningId(loadedTunings[0]?.id ?? "custom");
      })
      .catch((loadError) => {
        setError(loadError instanceof Error ? loadError.message : "調律の取得に失敗しました。");
      });
  }, []);

  const selectedTuning =
    selectedTuningId === "custom"
      ? buildCustomTuning(customName, customNotes)
      : tunings.find((tuning) => tuning.id === selectedTuningId) ?? null;

  const tuningSignature = useMemo(() => buildTuningSignature(selectedTuning), [selectedTuning]);

  useEffect(() => {
    setPendingExpectedKeys([]);
    setExpectedEvents([]);
    setExpectedRepeatCount("1");
    setExpectedImportText("");
    setExpectedImportError(null);
    clearAnalysisState();
  }, [tuningSignature]);

  const expectedPerformance = useMemo(() => buildExpectedPerformance(expectedEvents, captureIntent), [captureIntent, expectedEvents]);
  const hasProtectedDraft = useMemo(
    () => hasProtectedExpectedDraft(pendingExpectedKeys, expectedEvents, expectedImportText),
    [expectedEvents, expectedImportText, pendingExpectedKeys],
  );
  const analyzedNoteNamesByKey = useMemo(() => new Map((lastCapture?.requestPayload.tuning.notes ?? []).map((note) => [note.key, note.noteName])), [lastCapture]);
  const expectedNote = expectedPerformance?.summary ?? "";
  const suggestedCaptureId = useMemo(
    () => buildDefaultCaptureIdForValues(new Date().toISOString(), selectedTuning, expectedPerformance, captureIntent),
    [captureIntent, selectedTuning, expectedPerformance],
  );
  const currentScenario = useMemo(() => buildDraftCaptureScenario(captureCaseId, suggestedCaptureId), [captureCaseId, suggestedCaptureId]);
  const draftChangedSinceAnalysis = useMemo(() => {
    if (!lastCapture) {
      return false;
    }
    return (
      lastCapture.requestPayload.scenario !== currentScenario
      || lastCapture.requestPayload.captureIntent !== captureIntent
      || lastCapture.requestPayload.memo !== normalizeOptionalText(captureMemo)
      || lastCapture.requestPayload.expectedNote !== normalizeOptionalText(expectedNote)
      || buildTuningSignature(lastCapture.requestPayload.tuning) !== tuningSignature
      || serializeExpectedPerformance(lastCapture.requestPayload.expectedPerformance) !== serializeExpectedPerformance(expectedPerformance)
    );
  }, [captureIntent, captureMemo, currentScenario, expectedNote, expectedPerformance, lastCapture, tuningSignature]);
  const captureReview = useMemo(
    () => buildCaptureAssessment(
      lastCapture?.requestPayload.expectedPerformance ?? null,
      lastCapture?.responsePayload ?? null,
      analyzedNoteNamesByKey,
      lastCapture?.requestPayload.captureIntent ?? null,
    ),
    [analyzedNoteNamesByKey, lastCapture],
  );
  const recaptureGuidance = useMemo(() => buildRecaptureGuidance(captureReview, lastCapture?.requestPayload.captureIntent ?? "unknown"), [captureReview, lastCapture]);
  const userReview = useMemo(() => buildResultOnlyAssessment(result), [result]);
  const activeReview = isDebug ? captureReview : userReview;
  const analysisStatus = useMemo(() => {
    if (!recording) {
      return {
        label: "Awaiting Recording",
        detail: "録音を用意すると解析できます。",
      };
    }
    if (!lastCapture) {
      return {
        label: "Recording Ready",
        detail: "録音は準備できています。解析を実行してください。",
      };
    }
    if (draftChangedSinceAnalysis) {
      return {
        label: "Analysis Stale",
        detail: "期待演奏やメモなどが解析後に変わりました。表示中の review と保存パックは前回解析の snapshot を参照しています。",
      };
    }
    return {
      label: "Analysis Ready",
      detail: "現在の録音に対する解析結果が利用可能です。",
    };
  }, [draftChangedSinceAnalysis, lastCapture, recording]);

  useEffect(() => {
    setLastCapture((current) => {
      if (!current) {
        return current;
      }
      return result === current.responsePayload ? current : null;
    });
  }, [result]);

  function clearAnalysisState() {
    setResult(null);
    setLastCapture(null);
    setActiveEventId(null);
    setLatestReviewSessionId((current) => {
      if (current) {
        removeReviewSession(current);
        removeReviewAudio(current);
      }
      return null;
    });
  }

  function confirmTuningDraftDiscard() {
    if (!hasProtectedDraft) {
      return true;
    }
    return window.confirm("調律を変更すると expected performance の下書きと現在の解析 snapshot が破棄されます。続行しますか？");
  }

  function handleTuningSelect(nextId: string) {
    if (nextId === selectedTuningId) {
      return;
    }
    if (!confirmTuningDraftDiscard()) {
      return;
    }
    setSelectedTuningId(nextId);
  }

  function handleCustomNameChange(nextValue: string) {
    if (nextValue === customName) {
      return;
    }
    if (!confirmTuningDraftDiscard()) {
      return;
    }
    setCustomName(nextValue);
  }

  function handleCustomNotesChange(nextValue: string) {
    if (nextValue === customNotes) {
      return;
    }
    if (!confirmTuningDraftDiscard()) {
      return;
    }
    setCustomNotes(nextValue);
  }

  function handleRecordingReady(blob: Blob | null) {
    setRecording(blob);
    clearAnalysisState();
    setError(null);
  }

  function handleToggleExpectedKey(key: number) {
    setPendingExpectedKeys((current) => (current.includes(key) ? current.filter((value) => value !== key) : [...current, key]));
  }

  function handleCommitExpectedSelection() {
    if (!selectedTuning) {
      return;
    }

    const selectedNotes = selectedTuning.notes.filter((note) => pendingExpectedKeys.includes(note.key));
    if (selectedNotes.length === 0) {
      return;
    }

    const repeatCount = Math.max(1, Number.parseInt(expectedRepeatCount || "1", 10) || 1);
    const display = buildExpectedDisplay(selectedNotes);

    setExpectedEvents((current) => {
      const nextEvents = [...current];
      for (let index = 0; index < repeatCount; index += 1) {
        nextEvents.push({
          index: nextEvents.length + 1,
          keys: selectedNotes.map((note) => ({ key: note.key, noteName: note.noteName })),
          display,
          intent: captureIntent !== "unknown" ? captureIntent : null,
        });
      }
      return nextEvents;
    });

    setPendingExpectedKeys([]);
    setExpectedRepeatCount("1");
  }

  function handleUndoExpectedEvent() {
    setExpectedEvents((current) => current.slice(0, -1).map((event, index) => ({ ...event, index: index + 1 })));
  }

  function handleClearExpectedPerformance() {
    setPendingExpectedKeys([]);
    setExpectedEvents([]);
    setExpectedRepeatCount("1");
    setExpectedImportError(null);
  }

  function handleImportExpectedPerformance() {
    if (!selectedTuning) {
      return;
    }

    const parsed = parseExpectedPerformanceText(expectedImportText, selectedTuning, captureIntent);
    if (parsed.error) {
      setExpectedImportError(parsed.error);
      return;
    }

    setExpectedEvents(parsed.events);
    if (parsed.suggestedCaptureIntent) {
      setCaptureIntent(parsed.suggestedCaptureIntent);
    }
    setPendingExpectedKeys([]);
    setExpectedRepeatCount("1");
    setExpectedImportError(null);
  }

  async function handleAnalyze() {
    if (!recording || !selectedTuning) {
      setError("録音と調律設定が必要です。");
      return;
    }

    setError(null);
    setIsAnalyzing(true);

    try {
      const resolvedCaseId = captureCaseId.trim().length > 0 ? captureCaseId.trim() : suggestedCaptureId;
      const capture = await createTranscriptionWithCapture(recording, selectedTuning, {
        scenario: resolvedCaseId,
        expectedNote,
        expectedPerformance,
        memo: captureMemo,
        captureIntent,
      });
      setCaptureCaseId(resolvedCaseId);
      setResult(capture.responsePayload);
      const nextActiveEventId = capture.responsePayload.events[0]?.id ?? null;
      setActiveEventId(nextActiveEventId);
      setLastCapture(capture);
      if (!isDebug) {
        const reviewSession = createReviewSession({
          capture,
          acquisitionMode: "live_mic",
          notationMode,
          activeEventId: nextActiveEventId,
        });
        saveReviewSession(reviewSession);
        saveReviewAudio(reviewSession.sessionId, capture.audioWav);
        setLatestReviewSessionId((current) => {
          if (current && current !== reviewSession.sessionId) {
            removeReviewSession(current);
            removeReviewAudio(current);
          }
          return reviewSession.sessionId;
        });
        const transactionId = capture.responsePayload.transactionId;
        if (transactionId) {
          router.push(`/score/${transactionId}`);
        }
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "解析に失敗しました。");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function handleDownloadCapture() {
    if (!lastCapture) {
      return;
    }

    setError(null);
    setIsSavingCapture(true);

    try {
      const caseId = lastCapture.requestPayload.scenario;
      const archive = await buildCaptureArchive({
        caseId,
        audioWav: lastCapture.audioWav,
        requestPayload: lastCapture.requestPayload,
        responsePayload: lastCapture.responsePayload,
        notes: {
          expectedNote: lastCapture.requestPayload.expectedNote ?? "",
          expectedPerformance: lastCapture.requestPayload.expectedPerformance,
          memo: lastCapture.requestPayload.memo ?? "",
          verdict: captureReview?.status,
          reviewSummary: captureReview?.summary,
          reviewReason: captureReview?.reason,
          recaptureGuidance,
        },
      });
      downloadBlob(archive, `${caseId}.zip`);
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : "保存パックの作成に失敗しました。");
    } finally {
      setIsSavingCapture(false);
    }
  }

  const debugMain = (
    <div className="stack gap-xl debug-main-stack">
      <section className="panel workflow-panel debug-workflow-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Debug Capture</p>
            <h2>再現データを収集</h2>
          </div>
          <span className="pill">{recording ? "Recording Ready" : "Awaiting Recording"}</span>
        </div>
        <p className="muted workflow-lead">
          expected performance、解析実行、保存パック作成を主導線にして、認識評価を最短で回せる形にします。
        </p>

        <div className="workflow-card workflow-card-primary stack gap-lg debug-primary-card">
          <div className="workflow-card-header">
            <div>
              <span className="eyebrow">Capture Intent</span>
              <h3>録音の意図を指定する</h3>
            </div>
            <span className="muted">{analysisStatus.label}</span>
          </div>
          <p className="muted">{analysisStatus.detail}</p>
          <div className="intent-selector">
            {INTENT_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`intent-option ${captureIntent === option.value ? "active" : ""}`}
                onClick={() => setCaptureIntent(option.value)}
              >
                <strong>{option.label}</strong>
                <span>{option.description}</span>
              </button>
            ))}
          </div>

          <div className="workflow-card-header">
            <div>
              <span className="eyebrow">Expected Performance</span>
              <h3>期待演奏を組み立てる</h3>
            </div>
            <span className="muted">event 単位で保存</span>
          </div>

          <ExpectedKeySelector
            tuning={selectedTuning}
            selectedKeys={pendingExpectedKeys}
            repeatCount={expectedRepeatCount}
            onToggleKey={handleToggleExpectedKey}
            onRepeatCountChange={setExpectedRepeatCount}
            onCommitSelection={handleCommitExpectedSelection}
            onClearSelection={() => setPendingExpectedKeys([])}
          />

          <div className="stack gap-sm">
            <div className="panel-header compact">
              <div>
                <strong>文字列からインポート</strong>
                <p className="muted">例: `C4 / slide_chord:E4 + G4 + B4 + D5 x 4`。複合イベントは JSON でも指定できます。</p>
              </div>
              <button type="button" className="ghost" onClick={handleImportExpectedPerformance} disabled={!selectedTuning || expectedImportText.trim().length === 0}>
                文字列を反映
              </button>
            </div>
            <textarea
              rows={3}
              value={expectedImportText}
              onChange={(event) => {
                setExpectedImportText(event.target.value);
                if (expectedImportError) {
                  setExpectedImportError(null);
                }
              }}
              placeholder={"C4 / D4 / E4\nslide_chord:E4 + G4 + B4 + D5 x 4\n\n{\"events\":[{\"parts\":[{\"intent\":\"slide_chord\",\"notes\":[\"C5\",\"D5\"]},{\"intent\":\"strict_chord\",\"notes\":[\"F4\"]}]}]}"}
            />
            {expectedImportError ? <p className="error-text">{expectedImportError}</p> : null}
          </div>

          <div className="expected-performance-panel">
            <div className="panel-header compact">
              <div>
                <strong>期待演奏シーケンス</strong>
                <p className="muted">保存パックには event 単位で記録されます。</p>
              </div>
              <div className="row wrap">
                <button type="button" className="ghost" onClick={handleUndoExpectedEvent} disabled={expectedEvents.length === 0}>
                  最後を取り消す
                </button>
                <button type="button" className="ghost" onClick={handleClearExpectedPerformance} disabled={expectedEvents.length === 0 && pendingExpectedKeys.length === 0}>
                  すべてクリア
                </button>
              </div>
            </div>
            <div className="expected-summary-box large summary-hero">
              <span className="eyebrow">Summary</span>
              <strong>{expectedNote || "未設定"}</strong>
              <span className="muted">録音意図: {buildIntentLabel(captureIntent)}</span>
            </div>
            {expectedEvents.length > 0 ? (
              <div className="expected-event-list compact-list">
                {expectedEvents.map((event) => (
                  <div key={`${event.index}-${event.display}`} className="expected-event-card compact">
                    <span className="pill">#{event.index}</span>
                    <div className="stack event-copy">
                      <strong>{event.display}</strong>
                      <span className="muted">{event.keys.map((key) => `${key.noteName} (#${key.key})`).join(" / ")}</span>
                      {event.intent ? <span className="muted">intent: {buildIntentLabel(event.intent)}</span> : null}
                      {buildPartBreakdown(event) ? <span className="muted">parts: {buildPartBreakdown(event)}</span> : null}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">まだ期待演奏は追加されていません。</p>
            )}
          </div>
          <div className="workflow-action-dock stack gap-lg debug-action-dock">
            <button className="primary wide" onClick={handleAnalyze} disabled={!recording || !selectedTuning || isAnalyzing}>
              {isAnalyzing ? "解析中..." : "自動採譜する"}
            </button>
            <button className="secondary wide" onClick={handleDownloadCapture} disabled={!lastCapture || isAnalyzing || isSavingCapture}>
              {isSavingCapture ? "保存パック作成中..." : "解析 snapshot を保存パックでダウンロード"}
            </button>
            <p className="muted">保存内容: audio.wav / request.json / response.json / notes.md。保存パックは直近の解析 snapshot をそのまま出力します。</p>
          </div>
        </div>
      </section>

      <NotationPanel result={result} mode={notationMode} onModeChange={setNotationMode} review={captureReview} />

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Debug Metadata</p>
            <h2>収集情報</h2>
          </div>
        </div>
        <div className="stack gap-lg">
          <div className="summary-strip">
            <span>{lastCapture?.requestPayload.scenario ?? "manual-test"}</span>
            <span>{lastCapture?.requestPayload.audio.sampleRate ?? "-"} Hz</span>
            <span>{lastCapture?.requestPayload.audio.durationSec ?? "-"} sec</span>
            <span>intent {buildIntentLabel(lastCapture?.requestPayload.captureIntent ?? captureIntent)}</span>
          </div>
          <p className="muted">通常表記を優先して認識結果を確認し、必要なら保存パックを fixture に取り込みます。</p>
          {draftChangedSinceAnalysis ? (
            <div className="warning-box">
              <p>現在の入力ドラフトは解析 snapshot と一致していません。保存パックと review は前回解析時の request/response を基準にしています。</p>
            </div>
          ) : null}
          {captureReview ? (
            <div className={`review-box status-${captureReview.status}`}>
              <div className="panel-header compact">
                <div>
                  <p className="eyebrow">Capture Review</p>
                  <strong>{captureReview.summary}</strong>
                </div>
                <span className={`pill review-pill status-${captureReview.status}`}>{captureReview.label}</span>
              </div>
              <p>{captureReview.reason}</p>
              <div className="summary-strip">
                <span>expected {captureReview.expectedEventCount}</span>
                <span>detected {captureReview.detectedEventCount}</span>
                <span>mismatch {captureReview.mismatchCount}</span>
              </div>
              {recaptureGuidance.length > 0 ? (
                <div className="recapture-guidance">
                  <span className="eyebrow">Recapture Guidance</span>
                  <ul>
                    {recaptureGuidance.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : lastCapture?.requestPayload.expectedPerformance ? (
            <div className="warning-box">
              <p>Expected: {lastCapture.requestPayload.expectedPerformance.summary}</p>
            </div>
          ) : null}
          {result?.debug ? <pre className="debug-pre">{JSON.stringify(result.debug, null, 2)}</pre> : <p className="empty">解析後に debug 情報をここへ表示します。</p>}
        </div>
      </section>
    </div>
  );

  const debugSide = (
    <div className="stack gap-xl debug-side-stack">
      <RecorderPanel disabled={isAnalyzing || isSavingCapture} hasRecording={Boolean(recording)} onRecordingReady={handleRecordingReady} />
      <div className="file-upload-row">
        <label className="file-upload-label">
          WAVファイルをアップロード
          <input
            type="file"
            accept="audio/wav,audio/wave,.wav"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleRecordingReady(file);
            }}
            disabled={isAnalyzing || isSavingCapture}
          />
        </label>
      </div>
      <TuningPanel
        tunings={tunings}
        selectedId={selectedTuningId}
        onSelect={handleTuningSelect}
        customName={customName}
        customNotes={customNotes}
        hasProtectedDraft={hasProtectedDraft}
        onCustomNameChange={handleCustomNameChange}
        onCustomNotesChange={handleCustomNotesChange}
      />
      <section className="panel workflow-card workflow-card-secondary debug-secondary-card">
        <div className="workflow-card-header">
          <div>
            <p className="eyebrow">Capture Metadata</p>
            <h2>ケース情報</h2>
          </div>
          <span className="muted">補助情報</span>
        </div>
        <div className="stack gap-lg">
          <label className="stack">
            <span>ケースID (任意)</span>
            <input value={captureCaseId} onChange={(event) => setCaptureCaseId(event.target.value)} placeholder={suggestedCaptureId} />
            <span className="muted">空欄なら `{suggestedCaptureId}` を使います。</span>
          </label>
          <label className="stack">
            <span>テストメモ (任意)</span>
            <textarea value={captureMemo} rows={3} onChange={(event) => setCaptureMemo(event.target.value)} />
          </label>
        </div>
      </section>
    </div>
  );

  const userPrimary = (
    <div className="stack gap-xl">
      <RecorderPanel disabled={isAnalyzing || isSavingCapture} hasRecording={Boolean(recording)} onRecordingReady={handleRecordingReady} />
      <div className="file-upload-row">
        <label className="file-upload-label">
          WAVファイルをアップロード
          <input
            type="file"
            accept="audio/wav,audio/wave,.wav"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleRecordingReady(file);
            }}
            disabled={isAnalyzing || isSavingCapture}
          />
        </label>
      </div>
      <TuningPanel
        tunings={tunings}
        selectedId={selectedTuningId}
        onSelect={handleTuningSelect}
        customName={customName}
        customNotes={customNotes}
        hasProtectedDraft={hasProtectedDraft}
        onCustomNameChange={handleCustomNameChange}
        onCustomNotesChange={handleCustomNotesChange}
      />

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Workflow</p>
            <h2>解析を実行</h2>
          </div>
        </div>
        <p className="muted">録音と調律を用意したら解析します。詳細な再現データ収集は debug capture 画面で行います。</p>
        <div className="summary-strip">
          <span>{analysisStatus.label}</span>
          <span>{analysisStatus.detail}</span>
        </div>
        <div className="debug-capture-grid">
          <button className="primary wide" onClick={handleAnalyze} disabled={!recording || !selectedTuning || isAnalyzing}>
            {isAnalyzing ? "解析中..." : "自動採譜する"}
          </button>
          <a href="/debug/capture" className="debug-link-card">
            <strong>Debug capture 画面へ</strong>
            <span>expected performance、保存パック、認識評価を扱います。</span>
          </a>
        </div>
      </section>
    </div>
  );

  const userSecondary = (
    <div className="stack gap-xl">
      <InitialResultSummary result={result} review={activeReview} reviewSessionId={latestReviewSessionId} />
    </div>
  );

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">{isDebug ? "Kalimba Debug Capture" : "Kalimba Score MVP"}</p>
          <h1>{isDebug ? "認識評価用のデータ収集と比較。" : "カリンバ演奏を録音して、そのまま譜面へ。"}</h1>
          <p className="hero-copy">
            {isDebug
              ? "手動テスト用の expected performance、保存パック、認識結果比較を 1 画面で扱います。通常表記を優先して認識精度を確認できます。"
              : "マイク録音からカリンバ向けの譜面へ変換します。ドレミ縦並び譜を中心に、数字譜と通常表記も同じ解析結果から切り替え表示できます。"}
          </p>
        </div>
        <div className="hero-card">
          <p>{isDebug ? "Debug Shortcuts" : "対応範囲"}</p>
          <ul>
            {isDebug ? (
              <>
                <li><a href="/">利用者向け画面へ戻る</a></li>
                <li>録音意図の指定</li>
                <li>Expected performance の入力</li>
                <li>Capture pack の保存</li>
              </>
            ) : (
              <>
                <li>ブラウザ録音</li>
                <li>サーバー側解析</li>
                <li>和音イベント検出</li>
                <li><a href="/debug/capture">Debug capture 画面</a></li>
              </>
            )}
          </ul>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className={isDebug ? "workspace-grid debug-workspace-grid" : result ? "workspace-grid" : "workspace-grid workspace-grid-preanalysis"}>
        {isDebug ? (
          <>
            {debugMain}
            {debugSide}
          </>
        ) : (
          <>
            {userPrimary}
            {result ? userSecondary : null}
          </>
        )}
      </div>
    </main>
  );
}
