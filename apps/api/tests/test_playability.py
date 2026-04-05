"""Mechanism tests for is_physically_playable_chord with layer awareness.

Validates that multi-layer kalimbas correctly reject cross-layer consecutive
key pairs (e.g. key 17 on bottom layer + key 18 on top layer).
"""

import pytest

from app.transcription.peaks import is_physically_playable_chord


# ---------------------------------------------------------------------------
# 17-key (single layer) — no key_layers, behavior unchanged
# ---------------------------------------------------------------------------

class TestSingleLayer:
    def test_single_note(self):
        assert is_physically_playable_chord([5]) is True

    def test_two_adjacent(self):
        assert is_physically_playable_chord([3, 4]) is True

    def test_two_non_adjacent(self):
        assert is_physically_playable_chord([1, 10]) is True

    def test_three_consecutive(self):
        assert is_physically_playable_chord([5, 6, 7]) is True

    def test_four_consecutive(self):
        assert is_physically_playable_chord([5, 6, 7, 8]) is True

    def test_slide_plus_strict(self):
        # slide group [5,6,7] + strict group [10]
        assert is_physically_playable_chord([5, 6, 7, 10]) is True

    def test_five_notes_rejected(self):
        assert is_physically_playable_chord([1, 2, 3, 4, 5]) is False

    def test_non_consecutive_triple(self):
        # [1,3,5] — no split produces a valid slide+strict
        assert is_physically_playable_chord([1, 3, 5]) is False


# ---------------------------------------------------------------------------
# 34-key (two layers) — key_layers provided
# ---------------------------------------------------------------------------

# Bottom layer: keys 1-17, top layer: keys 18-34
LAYER_34 = {k: (0 if k <= 17 else 1) for k in range(1, 35)}


class TestMultiLayer:
    def test_same_layer_adjacent(self):
        """Keys 5,6 both on layer 0 — consecutive."""
        assert is_physically_playable_chord([5, 6], key_layers=LAYER_34) is True

    def test_cross_layer_boundary_rejected(self):
        """Keys 17 (layer 0) and 18 (layer 1) are NOT physically adjacent."""
        assert is_physically_playable_chord([17, 18], key_layers=LAYER_34) is False

    def test_top_layer_adjacent(self):
        """Keys 18,19 both on layer 1 — consecutive."""
        assert is_physically_playable_chord([18, 19], key_layers=LAYER_34) is True

    def test_cross_layer_non_adjacent_ok(self):
        """Keys 5 (layer 0) and 20 (layer 1) — far apart, 2 notes always playable
        unless they claim to be adjacent (which they don't since diff > 1)."""
        assert is_physically_playable_chord([5, 20], key_layers=LAYER_34) is True

    def test_three_notes_same_layer(self):
        """Keys 8,9,10 all on layer 0 — consecutive slide."""
        assert is_physically_playable_chord([8, 9, 10], key_layers=LAYER_34) is True

    def test_three_notes_crossing_layer_splittable(self):
        """Keys 16,17,18 — 17→18 crosses layer, but slide=[16,17] + strict=[18] is valid."""
        assert is_physically_playable_chord([16, 17, 18], key_layers=LAYER_34) is True

    def test_three_consecutive_crossing_layer_unsplittable(self):
        """Keys 17,18,19 — all three must be consecutive for a slide, but 17→18
        crosses layers. No valid slide+strict split exists because [18,19]
        is a valid slide but [17] strict is fine, so this IS playable."""
        # slide=[18,19] + strict=[17] → valid
        assert is_physically_playable_chord([17, 18, 19], key_layers=LAYER_34) is True

    def test_four_notes_spanning_layer_boundary(self):
        """Keys 16,17,18,19 — no single slide can span the boundary."""
        # Possible splits: slide=[16,17]+strict=[18,19] — strict needs adjacent same-layer,
        # 18,19 are same layer → valid!
        assert is_physically_playable_chord([16, 17, 18, 19], key_layers=LAYER_34) is True

    def test_cross_layer_strict_pair_rejected(self):
        """Keys 5,6,17,18 — slide=[5,6] + strict=[17,18] — strict pair crosses layers."""
        assert is_physically_playable_chord([5, 6, 17, 18], key_layers=LAYER_34) is False

    def test_four_notes_same_layer(self):
        """Keys 20,21,22,23 all on layer 1."""
        assert is_physically_playable_chord([20, 21, 22, 23], key_layers=LAYER_34) is True

    def test_slide_plus_strict_cross_layer(self):
        """Slide [5,6,7] on layer 0, strict [20] on layer 1 — different layers but
        they're not claimed to be consecutive, so this is valid (each group is
        internally consistent)."""
        assert is_physically_playable_chord([5, 6, 7, 20], key_layers=LAYER_34) is True

    def test_no_key_layers_backward_compatible(self):
        """Without key_layers, cross-boundary keys 17,18 appear consecutive."""
        assert is_physically_playable_chord([17, 18]) is True


class TestMultiLayerEdgeCases:
    def test_single_note_with_layers(self):
        assert is_physically_playable_chord([17], key_layers=LAYER_34) is True

    def test_all_layer_zero(self):
        """When all notes are layer 0, key_layers has no effect."""
        all_zero = {k: 0 for k in range(1, 18)}
        assert is_physically_playable_chord([15, 16, 17], key_layers=all_zero) is True
