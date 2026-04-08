from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from math import pow
from typing import Callable

_NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
_NOTE_NAME_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


@dataclass(frozen=True)
class Note:
    """Immutable representation of a musical note with all derived properties."""

    name: str           # "G4"
    pitch_class: str    # "G"
    octave: int         # 4
    frequency: float    # 391.995...
    midi: int           # 67

    @staticmethod
    def from_name(name: str) -> Note:
        match = _NOTE_NAME_RE.match(name.strip())
        if not match:
            raise ValueError(f"Invalid note name: {name}")
        pitch = f"{match.group(1).upper()}{match.group(2)}"
        octave = int(match.group(3))
        midi = _NOTE_TO_SEMITONE[pitch] + (octave + 1) * 12
        frequency = 440.0 * pow(2.0, (midi - 69) / 12.0)
        return Note(
            name=f"{pitch}{octave}",
            pitch_class=pitch,
            octave=octave,
            frequency=frequency,
            midi=midi,
        )

    def semitone_distance(self, other: Note) -> int:
        return abs(self.midi - other.midi)

    def is_octave_of(self, other: Note) -> bool:
        return self.pitch_class == other.pitch_class and self.octave != other.octave

    def octave_above(self) -> Note:
        return Note.from_name(f"{self.pitch_class}{self.octave + 1}")

    def octave_below(self) -> Note:
        return Note.from_name(f"{self.pitch_class}{self.octave - 1}")


@dataclass
class NoteCandidate:
    key: int
    note: Note
    score: float = 0.0
    onset_gain: float | None = None

    @property
    def note_name(self) -> str:
        return self.note.name

    @property
    def frequency(self) -> float:
        return self.note.frequency

    @property
    def pitch_class(self) -> str:
        return self.note.pitch_class

    @property
    def octave(self) -> int:
        return self.note.octave


@dataclass
class RawEvent:
    start_time: float
    end_time: float
    notes: list[NoteCandidate]
    is_gliss_like: bool
    primary_note_name: str = ""
    primary_score: float = 0.0
    # True when this event came from a segment whose duration was below
    # SHORT_SEGMENT_SECONDARY_GUARD_DURATION and only the primary survived.
    # Downstream suppress / merge / future per-sub-onset logic should treat
    # this primary as a tentative single-note attack from a window too narrow
    # for full FFT analysis (typically a gap-mute-dip rescue).
    from_short_segment_guard: bool = False
    # Sub-onset times within this segment, propagated from segment_peaks so
    # downstream pipeline passes (e.g., #153 Phase A.4 narrow-FFT primary
    # recovery) can re-run pick_matching_sub_onset for new candidate notes
    # without recomputing onset detection.  Empty when the segment came from
    # a path that did not produce sub-onset times.
    sub_onsets: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class RepeatedPatternPass:
    name: str
    fn: Callable[[list["RawEvent"]], list["RawEvent"]]
    merge_after: bool = True


@dataclass(slots=True)
class NoteHypothesis:
    candidate: NoteCandidate
    score: float
    fundamental_energy: float
    overtone_energy: float
    fundamental_ratio: float
    subharmonic_alias_energy: float
    octave_alias_energy: float
    octave_alias_ratio: float
    octave_alias_penalty: float
    second_harmonic_energy: float = 0.0
    harmonics: list[dict[str, float]] | None = None
    subharmonics: list[dict[str, float]] | None = None


@dataclass(frozen=True)
class Segment:
    """A time segment with collector provenance tracking."""

    start_time: float
    end_time: float
    sources: frozenset[str] = dataclass_field(default_factory=frozenset)
    merged_from: tuple["Segment", ...] = ()
    merge_reason: str = ""
    end_estimated: bool = False
    trimmed_from: "Segment | None" = None
    confirmed_primary: Note | None = None
    hint_primary: Note | None = None

    def __iter__(self):
        yield self.start_time
        yield self.end_time

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.start_time
        if index == 1:
            return self.end_time
        raise IndexError(index)


@dataclass(frozen=True)
class GapAttackCandidates:
    inter_ranges: list[list[float]]
    leading: list[float]
    trailing: list[float]


@dataclass(frozen=True)
class OnsetWaveformStats:
    kurtosis: float
    crest: float
    post_autocorr_20ms: float
    diff_centroid: float
    post_sustain_ratio: float


@dataclass
class OnsetAttackProfile:
    onset_time: float
    broadband_onset_gain: float
    high_band_spectral_flux: float
    broadband_spectral_flux: float
    is_valid_attack: bool
