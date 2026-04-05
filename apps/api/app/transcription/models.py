from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Callable


@dataclass
class NoteCandidate:
    key: int
    note_name: str
    frequency: float
    pitch_class: str
    octave: int
    score: float = 0.0
    onset_gain: float | None = None


@dataclass
class RawEvent:
    start_time: float
    end_time: float
    notes: list[NoteCandidate]
    is_gliss_like: bool
    primary_note_name: str = ""
    primary_score: float = 0.0


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
