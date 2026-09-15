//! CTC forced alignment over an arbitrary unit sequence.
//!
//! Given a `(T, n_labels)` emission matrix and a known label sequence,
//! the Viterbi-style dynamic program finds the most probable CTC path
//! through the frames, yielding a frame range per unit. This is the
//! algorithmic core of `Aligner`'s neural mode; it is model-independent
//! and unit-type-agnostic (phonemes, word pieces, characters).

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::VoiceError;

/// One aligned unit: `label` belongs to word `word_index`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentUnit {
    /// Acoustic label (e.g. phoneme id) for this unit.
    pub label: usize,
    /// Index of the word this unit belongs to.
    pub word_index: usize,
}

impl AlignmentUnit {
    pub fn new(label: usize, word_index: usize) -> Self {
        AlignmentUnit { label, word_index }
    }
}

/// Frame range of one aligned unit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnitTiming {
    /// Index into the unit sequence.
    pub unit_index: usize,
    /// Acoustic label of the unit.
    pub label: usize,
    /// Word this unit belongs to.
    pub word_index: usize,
    /// First frame emitting this unit (inclusive).
    pub start_frame: usize,
    /// Last frame emitting this unit (inclusive).
    pub end_frame: usize,
}

/// Complete forced alignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ForcedAlignment {
    pub units: Vec<UnitTiming>,
    /// Viterbi path log-probability (natural log), for diagnostics.
    pub score: f32,
}

/// CTC state machine for `L` units: `2L` states
/// (`u0, blank, u1, blank, …, u_{L-1}, trailing blank`).
fn state_label(state: usize, units: &[AlignmentUnit]) -> Option<usize> {
    if state % 2 == 1 {
        None // blank state
    } else {
        units.get(state / 2).map(|u| u.label)
    }
}

/// Forced alignment via CTC Viterbi with blank transitions.
///
/// `emissions` are frame log-probabilities or logits `(T, n_labels)`;
/// they are converted with log-softmax per frame internally.
pub fn ctc_align(
    emissions: &Array2<f32>,
    units: &[AlignmentUnit],
    blank: usize,
) -> Result<ForcedAlignment, VoiceError> {
    if units.is_empty() {
        return Ok(ForcedAlignment::default());
    }
    let t_len = emissions.nrows();
    let n_labels = emissions.ncols();
    let l = units.len();
    let states = 2 * l;
    if t_len < states {
        return Err(VoiceError::decode(format!(
            "audio has {t_len} frames but {l} units need at least {states}"
        )));
    }
    // Precompute per-frame log-softmax over the labels we touch.
    let frame_log_prob = |t: usize, label: Option<usize>| -> f32 {
        let row = emissions.row(t);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let target = match label {
            Some(l) => row[l.min(n_labels - 1)],
            None => row[blank.min(n_labels - 1)],
        };
        for v in row.iter() {
            sum += (v - max).exp();
        }
        (target - max) - sum.ln()
    };

    // dp[t][s]; use rolling rows. NEG_INFINITY = unreachable.
    let neg_inf = f32::NEG_INFINITY;
    let mut dp = vec![vec![neg_inf; states]; 2];
    let mut back = vec![vec![0u8; states]; t_len];

    dp[0][0] = frame_log_prob(0, state_label(0, units));
    if states > 1 {
        dp[0][1] = frame_log_prob(0, None);
    }
    for v in dp[0].iter_mut().skip(2) {
        *v = neg_inf;
    }

    for t in 1..t_len {
        let prev = std::mem::take(&mut dp[(t - 1) % 2]);
        let mut cur = std::mem::take(&mut dp[t % 2]);
        for s in 0..states {
            // Allowed predecessors: stay, skip one state; and skip two
            // only when the target and the state two back are both
            // non-blank AND distinct labels (standard CTC constraint).
            let mut best = prev[s];
            let mut best_prev = s as u8;
            if s >= 1 && prev[s - 1] > best {
                best = prev[s - 1];
                best_prev = (s - 1) as u8;
            }
            if s >= 2 {
                let repeat_allowed = match (state_label(s, units), state_label(s - 2, units)) {
                    (Some(a), Some(b)) => a != b,
                    _ => true, // through a blank, always allowed
                };
                if repeat_allowed && prev[s - 2] > best {
                    best = prev[s - 2];
                    best_prev = (s - 2) as u8;
                }
            }
            if best == neg_inf {
                cur[s] = neg_inf;
            } else {
                cur[s] = best + frame_log_prob(t, state_label(s, units));
            }
            back[t][s] = best_prev;
        }
        dp[(t - 1) % 2] = prev;
        dp[t % 2] = cur;
    }

    // Backtrack from the last two states (final blank or final unit).
    let final_row = &dp[(t_len - 1) % 2];
    let mut s = if final_row[states - 1] >= final_row[states - 2] {
        states - 1
    } else {
        states - 2
    };
    let score = final_row[s];
    // Frames in which each state was active.
    let mut state_frames: Vec<Vec<usize>> = vec![Vec::new(); states];
    state_frames[s].push(t_len - 1);
    for t in (1..t_len).rev() {
        s = back[t][s] as usize;
        state_frames[s].push(t - 1);
    }
    for frames in state_frames.iter_mut() {
        frames.reverse();
    }

    // Unit boundaries: union of its label-state frames.
    let mut units_out = Vec::with_capacity(l);
    for (u, unit) in units.iter().enumerate() {
        let state = 2 * u;
        let frames = &state_frames[state];
        if frames.is_empty() {
            return Err(VoiceError::decode(format!("unit {u} received no frames")));
        }
        units_out.push(UnitTiming {
            unit_index: u,
            label: unit.label,
            word_index: unit.word_index,
            start_frame: frames[0],
            end_frame: *frames.last().unwrap(),
        });
    }
    Ok(ForcedAlignment {
        units: units_out,
        score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Two units (labels 1 and 2) with clean time blocks and blanks
    /// between: frames 0-4 → 1, 5-9 → blank, 10-14 → 2.
    fn clean_emissions() -> Array2<f32> {
        let mut e = Array2::from_elem((15, 3), -5.0);
        for t in 0..5 {
            e[[t, 1]] = 5.0;
        }
        for t in 5..10 {
            e[[t, 0]] = 5.0;
        }
        for t in 10..15 {
            e[[t, 2]] = 5.0;
        }
        e
    }

    #[test]
    fn aligns_two_units_to_their_blocks() {
        let units = vec![AlignmentUnit::new(1, 0), AlignmentUnit::new(2, 1)];
        let alignment = ctc_align(&clean_emissions(), &units, 0).unwrap();
        assert_eq!(alignment.units.len(), 2);
        let u0 = alignment.units[0];
        let u1 = alignment.units[1];
        assert!(u0.start_frame <= 1 && u0.end_frame >= 3, "u0: {u0:?}");
        assert!(u1.start_frame >= 9 && u1.end_frame >= 13, "u1: {u1:?}");
        assert!(u1.start_frame > u0.end_frame, "monotonic");
    }

    #[test]
    fn repeated_labels_stay_separated() {
        // Same label twice: frames 0-4 → label 1, 5-9 blank, 10-14 → label 1.
        let mut e = Array2::from_elem((15, 2), -5.0);
        for t in 0..5 {
            e[[t, 1]] = 5.0;
        }
        for t in 5..10 {
            e[[t, 0]] = 5.0;
        }
        for t in 10..15 {
            e[[t, 1]] = 5.0;
        }
        let units = vec![AlignmentUnit::new(1, 0), AlignmentUnit::new(1, 1)];
        let alignment = ctc_align(&e, &units, 0).unwrap();
        let u0 = alignment.units[0];
        let u1 = alignment.units[1];
        assert!(
            u1.start_frame >= 9,
            "second unit must come after the blank gap"
        );
        assert!(u0.end_frame < u1.start_frame);
    }

    #[test]
    fn too_short_audio_is_rejected() {
        let e = Array2::from_elem((3, 3), 0.0);
        let units = vec![AlignmentUnit::new(1, 0), AlignmentUnit::new(2, 1)];
        assert!(ctc_align(&e, &units, 0).is_err());
    }

    #[test]
    fn empty_sequence_aligns_to_nothing() {
        let e = Array2::from_elem((10, 3), 0.0);
        let alignment = ctc_align(&e, &[], 0).unwrap();
        assert!(alignment.units.is_empty());
    }

    #[test]
    fn score_is_finite_and_boundable() {
        let units = vec![AlignmentUnit::new(1, 0), AlignmentUnit::new(2, 1)];
        let alignment = ctc_align(&clean_emissions(), &units, 0).unwrap();
        assert!(alignment.score.is_finite());
        // 15 frames × ln(softmax≈1) ≈ small negative.
        assert!(alignment.score > -1e-2, "score {}", alignment.score);
    }
}
