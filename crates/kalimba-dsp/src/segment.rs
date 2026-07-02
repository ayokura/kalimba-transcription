//! Segment-stage DSP shared between the server (pyo3) and browser (wasm):
//! active-range extraction from the RMS envelope.
//!
//! Port of the active-range head of `detect_segments`
//! (apps/api/app/transcription/segments.py) — the first B1 slice
//! (sprint-plan 2026-07 S5). The heuristic constants mirror the Python
//! originals (threshold formula 0.18 / 2.2 / 0.45 / 0.01 floor, -0.02s /
//! +0.08s margins, 0.06s merge gap). Python remains the production
//! implementation; the fixture parity harness (tools/check_wasm_parity.cjs)
//! pins agreement on real recordings, so a retune on either side fails CI
//! instead of drifting silently.

/// `np.median` on a float32 vector: mean of the middle pair for even
/// lengths (numpy keeps float32 accumulation there), middle element
/// otherwise. Empty input yields 0.0 (detect_segments never passes one).
fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Active-range RMS threshold from `detect_segments`:
/// `max(max_rms * 0.18, min(median_rms * 2.2, max_rms * 0.45), 0.01)`
/// evaluated in f64 on f32-precision max/median, matching the Python
/// `float(np.max(rms))` / `float(np.median(rms))` promotions.
pub fn rms_threshold(rms: &[f32]) -> f64 {
    if rms.is_empty() {
        return 0.01;
    }
    let max_rms = rms.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let median_rms = median_f32(rms) as f64;
    (max_rms * 0.18)
        .max((median_rms * 2.2).min(max_rms * 0.45))
        .max(0.01)
}

/// Raw active ranges: contiguous runs of `rms >= threshold`, opened with a
/// -0.02s margin (clamped at 0) and closed at the first inactive frame time
/// +0.08s. A run still open at the end extends to `duration_sec` with no end
/// margin. Returns flat `[start0, end0, start1, end1, ...]` seconds for
/// simple Float64Array marshalling.
pub fn raw_active_ranges(
    rms: &[f32],
    sample_rate: i64,
    hop_length: usize,
    duration_sec: f64,
) -> Vec<f64> {
    let threshold = rms_threshold(rms);
    let sr = sample_rate as f64;
    let hop = hop_length as f64;
    let frame_time = |i: usize| (i as f64) * hop / sr;

    let mut ranges: Vec<f64> = Vec::new();
    let mut active_start: Option<usize> = None;
    for (index, &value) in rms.iter().enumerate() {
        let is_active = (value as f64) >= threshold;
        if is_active && active_start.is_none() {
            active_start = Some(index);
        } else if !is_active {
            if let Some(start) = active_start {
                ranges.push((frame_time(start) - 0.02).max(0.0));
                ranges.push(frame_time(index.min(rms.len() - 1)) + 0.08);
                active_start = None;
            }
        }
    }
    if let Some(start) = active_start {
        ranges.push((frame_time(start) - 0.02).max(0.0));
        ranges.push(duration_sec);
    }
    ranges
}

/// Merge time-ordered `[start, end]` pairs whose gap is within
/// `gap_tolerance` seconds. Flat-pair layout in and out.
pub fn merge_time_ranges(flat_ranges: &[f64], gap_tolerance: f64) -> Vec<f64> {
    let mut merged: Vec<f64> = Vec::new();
    for pair in flat_ranges.chunks_exact(2) {
        let (start, end) = (pair[0], pair[1]);
        if let [.., prev_start, prev_end] = merged.as_mut_slice() {
            if start <= *prev_end + gap_tolerance {
                let _ = prev_start;
                *prev_end = prev_end.max(end);
                continue;
            }
        }
        merged.push(start);
        merged.push(end);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_matches_numpy_semantics() {
        assert_eq!(median_f32(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_f32(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_f32(&[]), 0.0);
    }

    #[test]
    fn trailing_open_range_extends_to_duration() {
        // rms: quiet, loud, loud (run still open at the end)
        let rms = [0.0, 1.0, 1.0];
        let flat = raw_active_ranges(&rms, 100, 10, 0.5);
        // threshold = max(0.18, min(2.2, 0.45), 0.01) = 0.45; frames 1,2 active
        assert_eq!(flat, vec![(0.1f64 - 0.02), 0.5]);
    }

    #[test]
    fn closed_range_gets_end_margin() {
        let rms = [1.0, 1.0, 0.0, 0.0];
        let flat = raw_active_ranges(&rms, 100, 10, 1.0);
        assert_eq!(flat, vec![0.0, 0.2 + 0.08]);
    }

    #[test]
    fn merge_respects_gap_tolerance() {
        let flat = [0.0, 1.0, 1.05, 2.0, 3.0, 4.0];
        assert_eq!(merge_time_ranges(&flat, 0.06), vec![0.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            merge_time_ranges(&flat, 0.01),
            vec![0.0, 1.0, 1.05, 2.0, 3.0, 4.0]
        );
    }
}
