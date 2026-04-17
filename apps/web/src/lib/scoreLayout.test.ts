import { describe, expect, it } from "vitest";

import {
  isMovableNumberApplicable,
  movableDoLabelFn,
  movableNumberLabelFn,
  noteLabelFromScoreNote,
  tonicReferenceOctave,
} from "@/lib/scoreLayout";
import { InstrumentTuning, ScoreNote } from "@/lib/types";

function note(pitchClass: string, octave: number): ScoreNote {
  return {
    key: 1,
    pitchClass,
    octave,
    labelDoReMi: "",
    labelNumber: "",
    frequency: 0,
  };
}

describe("movableDoLabelFn", () => {
  it("returns fixed-do labels when tonic is null or unknown", () => {
    const fallback = movableDoLabelFn(null);
    expect(fallback(note("C", 4))).toEqual({ baseName: "ド", octave: 4 });
    const fallback2 = movableDoLabelFn("ZZ");
    expect(fallback2(note("C", 4))).toEqual({ baseName: "ド", octave: 4 });
  });

  it("labels the tonic as ド with octave-4 anchored to tonic's octave-4", () => {
    const fn = movableDoLabelFn("D");
    expect(fn(note("D", 4))).toEqual({ baseName: "ド", octave: 4 });
    expect(fn(note("D", 5))).toEqual({ baseName: "ド", octave: 5 });
    expect(fn(note("D", 3))).toEqual({ baseName: "ド", octave: 3 });
  });

  it("labels scale degrees in D major correctly", () => {
    const fn = movableDoLabelFn("D");
    expect(fn(note("E", 4)).baseName).toBe("レ");
    expect(fn(note("F#", 4)).baseName).toBe("ミ");
    expect(fn(note("G", 4)).baseName).toBe("ファ");
    expect(fn(note("A", 4)).baseName).toBe("ソ");
    expect(fn(note("B", 4)).baseName).toBe("ラ");
    expect(fn(note("C#", 5)).baseName).toBe("シ");
  });

  it("labels scale degrees in Bb major with accidentals", () => {
    const fn = movableDoLabelFn("Bb");
    // Bb major: Bb C D Eb F G A
    expect(fn(note("Bb", 4)).baseName).toBe("ド");
    expect(fn(note("C", 5)).baseName).toBe("レ");
    expect(fn(note("D", 5)).baseName).toBe("ミ");
    expect(fn(note("Eb", 5)).baseName).toBe("ファ");
    expect(fn(note("F", 5)).baseName).toBe("ソ");
    expect(fn(note("G", 5)).baseName).toBe("ラ");
    expect(fn(note("A", 5)).baseName).toBe("シ");
  });

  it("anchors octave dots to tonic's octave-4 regardless of pitch class ordering", () => {
    // Bb tonic: notes below Bb4 but in same octave visually go to a lower "scale octave"
    const fn = movableDoLabelFn("Bb");
    // F4 is below Bb4 in pitch, so it belongs to the lower scale octave
    expect(fn(note("F", 4))).toEqual({ baseName: "ソ", octave: 3 });
    // F5 is above Bb4, within the octave-4 scale range
    expect(fn(note("F", 5))).toEqual({ baseName: "ソ", octave: 4 });
    // Bb5 is one octave above tonic → octave 5
    expect(fn(note("Bb", 5))).toEqual({ baseName: "ド", octave: 5 });
  });

  it("labels non-diatonic chromatic notes with sharp form", () => {
    const fn = movableDoLabelFn("C");
    // In C major tonic: C# is +1 semitone = ド#
    expect(fn(note("C#", 4)).baseName).toBe("ド#");
    expect(fn(note("D#", 4)).baseName).toBe("レ#");
    expect(fn(note("F#", 4)).baseName).toBe("ファ#");
  });
});

describe("movableNumberLabelFn", () => {
  it("labels C major tonic scale as 1-7", () => {
    const fn = movableNumberLabelFn("C");
    expect(fn(note("C", 4)).baseName).toBe("1");
    expect(fn(note("D", 4)).baseName).toBe("2");
    expect(fn(note("E", 4)).baseName).toBe("3");
    expect(fn(note("F", 4)).baseName).toBe("4");
    expect(fn(note("G", 4)).baseName).toBe("5");
    expect(fn(note("A", 4)).baseName).toBe("6");
    expect(fn(note("B", 4)).baseName).toBe("7");
  });

  it("labels G major scale as 1-7 regardless of actual pitch", () => {
    const fn = movableNumberLabelFn("G");
    expect(fn(note("G", 4)).baseName).toBe("1");
    expect(fn(note("A", 4)).baseName).toBe("2");
    expect(fn(note("B", 4)).baseName).toBe("3");
    expect(fn(note("C", 5)).baseName).toBe("4");
    expect(fn(note("D", 5)).baseName).toBe("5");
    expect(fn(note("E", 5)).baseName).toBe("6");
    expect(fn(note("F#", 5)).baseName).toBe("7");
  });

  it("anchors octave to tonic (Bb4 tonic → F4 is lower-octave 5)", () => {
    const fn = movableNumberLabelFn("Bb");
    expect(fn(note("Bb", 4))).toEqual({ baseName: "1", octave: 4 });
    expect(fn(note("F", 4))).toEqual({ baseName: "5", octave: 3 });
    expect(fn(note("Bb", 5))).toEqual({ baseName: "1", octave: 5 });
  });

  it("falls back to fixed-do when tonic is missing", () => {
    const fn = movableNumberLabelFn(null);
    expect(fn(note("C", 4))).toEqual({ baseName: "ド", octave: 4 });
  });
});

describe("isMovableNumberApplicable", () => {
  it("returns true for all diatonic notes in C major", () => {
    const notes = [note("C", 4), note("E", 4), note("G", 4), note("F", 5)];
    expect(isMovableNumberApplicable(notes, "C")).toBe(true);
  });

  it("returns false when any note is chromatic to the tonic", () => {
    const notes = [note("C", 4), note("C#", 4)];
    expect(isMovableNumberApplicable(notes, "C")).toBe(false);
  });

  it("returns true for G major scale notes (including F#)", () => {
    const notes = [note("G", 4), note("A", 4), note("F#", 5)];
    expect(isMovableNumberApplicable(notes, "G")).toBe(true);
  });

  it("returns false when F is used in G major (non-diatonic)", () => {
    const notes = [note("G", 4), note("F", 5)];
    expect(isMovableNumberApplicable(notes, "G")).toBe(false);
  });

  it("returns false when tonic is missing", () => {
    expect(isMovableNumberApplicable([note("C", 4)], null)).toBe(false);
  });
});

function buildTuning(names: string[], tonic: string | null): InstrumentTuning {
  return {
    id: "test",
    name: "Test",
    keyCount: names.length,
    tonic,
    notes: names.map((noteName, i) => ({ key: i + 1, noteName, frequency: 0 })),
  };
}

describe("tonicReferenceOctave", () => {
  it("returns the lowest octave of tonic tines", () => {
    const tuning = buildTuning(["A5", "G5", "E5", "C5", "A4", "G4", "C4", "E4", "G3"], "G");
    expect(tonicReferenceOctave(tuning, "G")).toBe(3);
  });

  it("ignores tines of other pitch classes sharing a letter prefix (B vs Bb)", () => {
    // Tuning has B4 but tonic is Bb; should NOT pick up B4 as a match.
    const tuning = buildTuning(["B4", "Bb5"], "Bb");
    expect(tonicReferenceOctave(tuning, "Bb")).toBe(5);
  });

  it("defaults to 4 when tonic or tuning is missing", () => {
    const tuning = buildTuning(["C4"], "C");
    expect(tonicReferenceOctave(tuning, null)).toBe(4);
    expect(tonicReferenceOctave(null, "C")).toBe(4);
  });

  it("defaults to 4 when tonic is not found in the tuning", () => {
    const tuning = buildTuning(["C4", "D4"], "A");
    expect(tonicReferenceOctave(tuning, "A")).toBe(4);
  });
});

describe("movableDoLabelFn with custom tonicRefOctave", () => {
  it("renders G3 as the no-dot reference when refOctave=3 (G-low kalimba)", () => {
    const fn = movableDoLabelFn("G", 3);
    expect(fn(note("G", 3))).toEqual({ baseName: "ド", octave: 4 });
    expect(fn(note("G", 4))).toEqual({ baseName: "ド", octave: 5 });
    expect(fn(note("B", 3))).toEqual({ baseName: "ミ", octave: 4 });
  });

  it("renders G4 as no-dot and G5 as .-dot when refOctave=4 (standard G)", () => {
    const fn = movableDoLabelFn("G", 4);
    expect(fn(note("G", 4))).toEqual({ baseName: "ド", octave: 4 });
    expect(fn(note("G", 5))).toEqual({ baseName: "ド", octave: 5 });
  });
});

describe("movableNumberLabelFn with custom tonicRefOctave", () => {
  it("aligns number-label octave to G-low's G3", () => {
    const fn = movableNumberLabelFn("G", 3);
    expect(fn(note("G", 3))).toEqual({ baseName: "1", octave: 4 });
    expect(fn(note("D", 4))).toEqual({ baseName: "5", octave: 4 });
    expect(fn(note("G", 4))).toEqual({ baseName: "1", octave: 5 });
  });
});

describe("noteLabelFromScoreNote (fixed-do)", () => {
  it("returns pitch-class syllable with actual octave", () => {
    expect(noteLabelFromScoreNote(note("C", 4))).toEqual({ baseName: "ド", octave: 4 });
    expect(noteLabelFromScoreNote(note("F#", 5))).toEqual({ baseName: "ファ#", octave: 5 });
    expect(noteLabelFromScoreNote(note("Bb", 4))).toEqual({ baseName: "シb", octave: 4 });
  });
});
