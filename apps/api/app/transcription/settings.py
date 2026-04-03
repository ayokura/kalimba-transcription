"""Shared runtime settings for the recognizer pipeline.

Values here are intended to be overridden in tests, diagnostic scripts,
and future side-by-side experiments.  Structural constants that should
never change at runtime stay in ``constants.py``.

Usage in production code::

    from . import settings
    cfg = settings.get()
    if cfg.use_attack_validated_gap_collector:
        ...

Usage in tests::

    from app.transcription.settings import override

    with override(ablate_multi_onset_gap=True):
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class RecognizerSettings:
    """Recognizer feature flags and ablation switches.

    All fields must have defaults matching production behaviour.
    """

    # Feature flags
    use_attack_validated_gap_collector: bool = True
    filter_gap_onsets_by_attack_profile: bool = True
    use_iterative_harmonic_suppression: bool = True

    # Ablation switches (True = disable the feature)
    ablate_sparse_gap_tail: bool = False
    ablate_multi_onset_gap: bool = False
    ablate_collapse_active_range_head: bool = False
    ablate_snap_range_start_to_onset: bool = False

    # Gate-level ablation: set of gate reason strings to skip.
    # See GATE_CATEGORIES in peaks.py for the full list.
    # Example: frozenset({"recent-carryover-candidate", "weak-upper-secondary"})
    disabled_gates: frozenset[str] = frozenset()


_DEFAULTS = RecognizerSettings()
_current: RecognizerSettings = _DEFAULTS


def get() -> RecognizerSettings:
    """Return the active settings snapshot."""
    return _current


@contextmanager
def override(**kwargs: Any) -> Iterator[RecognizerSettings]:
    """Temporarily replace settings for the duration of a ``with`` block.

    Only fields defined on :class:`RecognizerSettings` are accepted;
    unknown keys raise ``TypeError``.

    ::

        with override(ablate_multi_onset_gap=True):
            # ablation is active inside this block
            ...
        # original settings restored here
    """
    global _current
    previous = _current
    _current = replace(_current, **kwargs)
    try:
        yield _current
    finally:
        _current = previous


def reset() -> None:
    """Restore production defaults.  Useful in test teardown."""
    global _current
    _current = _DEFAULTS
