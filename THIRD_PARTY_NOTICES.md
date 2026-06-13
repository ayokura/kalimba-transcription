# Third-Party Notices

This project includes code derived from third-party open-source software.
The required copyright and license notices are reproduced below. They must be
retained in all copies and in any distributed build (including a future
browser-side WebAssembly bundle that ships the ported code to end users).

---

## librosa

Portions of `apps/api/app/transcription/segments.py` are ported from or closely
derived from **librosa** (https://librosa.org), which is distributed under the
ISC License. librosa is also used as a runtime dependency.

Functions derived from librosa source (each annotated in-code):

- `_peak_pick_numpy` — port of librosa's `__peak_pick` guvectorize kernel
  (`librosa/util/utils.py`)
- `_onset_detect_numpy` — replacement for `librosa.onset.onset_detect`
- `_onset_backtrack_numpy` — port of `librosa.onset.onset_backtrack`
- `_mel_filterbank` — Slaney-normalised mel filterbank derived from
  `librosa.filters.mel`
- `_onset_strength_numpy` — derived from `librosa.onset.onset_strength`
  (mel spectral flux)

(Other numpy helpers in that module — `_rms_numpy`, `_frames_to_time_numpy`,
`_audio_duration_sec`, and the autocorrelation tempo estimator replacing
`librosa.beat.beat_track` — are independent reimplementations of public,
well-known DSP procedures and are not derived from librosa's source.)

### ISC License

```
ISC License

Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```
