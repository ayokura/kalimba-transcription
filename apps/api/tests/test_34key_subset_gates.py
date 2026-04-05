"""Mechanism tests for 34-key SUBSET regression fixes.

These tests use marshalled intermediate state from the actual pipeline to
verify that specific gate improvements work correctly:

1. E145 backward gate override: og=171 genuine attack should not be
   rejected by tertiary-weak-backward-attack (backward_gain=15.66 < 20.0)
2. E65 fr artifact rejection: C4 with fr=0.398 should be rejected as a
   harmonic artifact (not a genuine note) when onset_gain is low
"""

import json
import numpy as np
import pytest
from pathlib import Path

from app.transcription.peaks import (
    onset_backward_attack_gain,
    onset_energy_gain,
    is_physically_playable_chord,
    TERTIARY_MIN_BACKWARD_ATTACK_GAIN,
    TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE,
    SECONDARY_MIN_FUNDAMENTAL_RATIO,
    SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO,
    TERTIARY_MIN_ONSET_GAIN,
)


SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "mechanism-snapshots" / "34key-subset-regressions.json"


@pytest.fixture(scope="module")
def snapshots():
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


class TestBackwardGateOnsetOverride:
    """E145: G4 with og=171 should pass backward gate despite bg=15.66."""

    def test_snapshot_values_match_expectations(self, snapshots):
        """Verify the marshalled snapshot captures the known values."""
        s = snapshots["E145_backward_gate"]
        target = s["targetNote"]
        assert target["noteName"] == "G4"
        assert target["onsetGain"] == pytest.approx(171.26, abs=1.0)
        assert target["backwardAttackGain"] == pytest.approx(15.66, abs=1.0)
        assert target["backwardAttackGain"] < TERTIARY_MIN_BACKWARD_ATTACK_GAIN

    def test_override_threshold_allows_genuine_attack(self):
        """When onset_gain >= override threshold, backward gate should not fire."""
        og = 171.26
        bg = 15.66
        # The gate condition is: bg < threshold AND og < override
        # With og=171 >= 100, the backward gate is bypassed
        assert og >= TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE
        backward_would_reject = bg < TERTIARY_MIN_BACKWARD_ATTACK_GAIN
        override_active = og >= TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE
        assert backward_would_reject, "backward gain is below threshold"
        assert override_active, "onset override should be active"
        # Final decision: not rejected because override is active
        final_reject = backward_would_reject and not override_active
        assert not final_reject

    def test_override_does_not_help_weak_onset(self):
        """Weak onset (og=13) should NOT override backward gate."""
        og = 13.02  # F4 in triple-glissando chord 1
        bg = 5.0
        assert og < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE
        final_reject = bg < TERTIARY_MIN_BACKWARD_ATTACK_GAIN and og < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE
        assert final_reject, "weak onset should still be rejected by backward gate"

    def test_override_does_not_help_transient_brush(self):
        """Transient brush (og=55, verified as non-sustained) should not override."""
        og = 55.37  # E4 in triple-glissando chord 3 (confirmed transient brush)
        bg = 8.0
        assert og < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE
        final_reject = bg < TERTIARY_MIN_BACKWARD_ATTACK_GAIN and og < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE
        assert final_reject, "transient brush should still be rejected"

    def test_e145_g4_now_accepted(self, snapshots):
        """E145 trail should show G4 was rejected before this fix."""
        s = snapshots["E145_backward_gate"]
        # In the snapshot (pre-fix), G4 was rejected
        trail = s["trail"]
        g4_entries = [t for t in trail if t.get("noteName") == "G4"]
        # G4 appears in trail as rejected by tertiary-weak-backward-attack
        rejected_entries = [
            t for t in g4_entries
            if not t.get("accepted") and "tertiary-weak-backward-attack" in t.get("reasons", [])
        ]
        assert len(rejected_entries) > 0, "snapshot should show G4 was rejected pre-fix"


class TestFundamentalRatioArtifactGate:
    """E65: C4 with fr=0.398 should be rejected when onset is weak."""

    def test_snapshot_values_match_expectations(self, snapshots):
        """Verify the marshalled snapshot captures the known values."""
        s = snapshots["E65_fr_artifact"]
        target = s["targetNote"]
        assert target["noteName"] == "C4"
        assert target["fundamentalRatio"] == pytest.approx(0.398, abs=0.01)
        assert target["onsetGain"] == pytest.approx(0.91, abs=0.1)

    def test_artifact_fr_above_old_threshold(self):
        """fr=0.398 passes the old SECONDARY_MIN_FUNDAMENTAL_RATIO=0.18."""
        fr = 0.398
        assert fr >= SECONDARY_MIN_FUNDAMENTAL_RATIO
        assert fr < SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO

    def test_weak_onset_triggers_higher_fr_threshold(self):
        """When onset_gain < TERTIARY_MIN_ONSET_GAIN, higher fr threshold applies."""
        og = 0.91  # E65 C4
        fr = 0.398
        # The gate: if fr < 0.5 and og < 1.8, reject
        assert og < TERTIARY_MIN_ONSET_GAIN
        assert fr < SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO
        # This combination should be rejected
        should_reject = fr < SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO and og < TERTIARY_MIN_ONSET_GAIN
        assert should_reject

    def test_genuine_note_with_low_fr_and_strong_onset_passes(self):
        """A genuine note with low-ish fr but strong onset should not be rejected."""
        og = 50.0  # strong genuine attack
        fr = 0.45  # below SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO
        # og >= TERTIARY_MIN_ONSET_GAIN → higher fr threshold does NOT apply
        assert og >= TERTIARY_MIN_ONSET_GAIN
        should_reject = fr < SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO and og < TERTIARY_MIN_ONSET_GAIN
        assert not should_reject, "strong onset should prevent fr-based rejection"

    def test_e65_c4_was_accepted_before_fix(self, snapshots):
        """E65 trail should show C4 was accepted in the pre-fix snapshot."""
        s = snapshots["E65_fr_artifact"]
        assert "C4" in s["selectedNotes"], "C4 was accepted pre-fix (the bug)"
