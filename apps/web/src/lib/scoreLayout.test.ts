import { describe, expect, it } from "vitest";

import { movableDoLabelFn, noteLabelFromScoreNote } from "@/lib/scoreLayout";
import { ScoreNote } from "@/lib/types";

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

describe("noteLabelFromScoreNote (fixed-do)", () => {
  it("returns pitch-class syllable with actual octave", () => {
    expect(noteLabelFromScoreNote(note("C", 4))).toEqual({ baseName: "ド", octave: 4 });
    expect(noteLabelFromScoreNote(note("F#", 5))).toEqual({ baseName: "ファ#", octave: 5 });
    expect(noteLabelFromScoreNote(note("Bb", 4))).toEqual({ baseName: "シb", octave: 4 });
  });
});
