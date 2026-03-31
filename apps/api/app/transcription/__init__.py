from __future__ import annotations

import importlib

_legacy = importlib.import_module(f"{__name__}._legacy")

from . import audio, constants, events, models, patterns, peaks, profiles, segments
from ._legacy import (
    _find_note_attack_time,
    _has_mute_dip_reattack,
    _is_residual_decay,
    _note_band_energy,
    parse_disabled_repeated_pattern_passes,
    transcribe_audio,
)
from .audio import *  # noqa: F401,F403
from .constants import *  # noqa: F401,F403
from .events import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .patterns import *  # noqa: F401,F403
from .peaks import *  # noqa: F401,F403
from .profiles import *  # noqa: F401,F403
from .segments import *  # noqa: F401,F403
