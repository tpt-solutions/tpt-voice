//! Synthetic test signals: reproducible audio fixtures for pipelines,
//! examples and benchmarks.

use crate::nn::Rng;

/// Pure sine tone.
pub fn sine(hz: f32, secs: f64, sample_rate: u32, amp: f32) -> Vec<f32> {
    let n = (secs * f64::from(sample_rate)) as usize;
    (0..n)
        .map(|i| amp * (std::f32::consts::TAU * hz * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Seeded white noise (deterministic across platforms and runs).
pub fn white_noise(secs: f64, sample_rate: u32, amp: f32, seed: u64) -> Vec<f32> {
    let n = (secs * f64::from(sample_rate)) as usize;
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.uniform_range(-amp, amp)).collect()
}

/// Harmonic stack with `1/k` amplitudes — a buzzy, voice-like source.
pub fn harmonic_stack(
    f0: f32,
    n_harmonics: usize,
    secs: f64,
    sample_rate: u32,
    amp: f32,
) -> Vec<f32> {
    let n = (secs * f64::from(sample_rate)) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let mut v = 0.0f32;
            for k in 1..=n_harmonics {
                v += (std::f32::consts::TAU * f0 * k as f32 * t).sin() / k as f32;
            }
            amp * v
        })
        .collect()
}

/// Harmonic stack with slow vibrato — closer to a real voice.
pub fn speech_like(f0: f32, secs: f64, sample_rate: u32, amp: f32) -> Vec<f32> {
    let n = (secs * f64::from(sample_rate)) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let vibrato = 1.0 + 0.02 * (std::f32::consts::TAU * 5.0 * t).sin();
            let mut v = 0.0f32;
            for k in 1..=6 {
                v += (std::f32::consts::TAU * f0 * k as f32 * vibrato * t).sin() / k as f32;
            }
            amp * v
        })
        .collect()
}

/// Builds an alternating pattern: `voices` is a list of (f0, secs)
/// segments separated by `gap_secs` of silence — a synthetic "meeting".
pub fn alternating_speakers(
    voices: &[(f32, f64)],
    gap_secs: f64,
    sample_rate: u32,
    amp: f32,
) -> Vec<f32> {
    let mut out = Vec::new();
    let gap = (gap_secs * f64::from(sample_rate)) as usize;
    for &(f0, secs) in voices {
        out.extend(harmonic_stack(f0, 8, secs, sample_rate, amp));
        out.extend(std::iter::repeat(0.0).take(gap));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::rms;

    #[test]
    fn lengths_match_duration() {
        assert_eq!(sine(440.0, 1.0, 16_000, 0.5).len(), 16_000);
        assert_eq!(white_noise(0.5, 16_000, 0.1, 1).len(), 8_000);
        assert_eq!(speech_like(200.0, 2.0, 16_000, 0.4).len(), 32_000);
    }

    #[test]
    fn deterministic() {
        assert_eq!(
            white_noise(0.1, 16_000, 0.1, 7),
            white_noise(0.1, 16_000, 0.1, 7)
        );
        assert_ne!(
            white_noise(0.1, 16_000, 0.1, 7),
            white_noise(0.1, 16_000, 0.1, 8)
        );
    }

    #[test]
    fn amplitudes_are_sane() {
        let s = sine(440.0, 0.1, 16_000, 0.5);
        assert!((rms(&s) - 0.353).abs() < 0.01, "sine rms ≈ amp/√2");
        let n = white_noise(0.1, 16_000, 0.2, 3);
        assert!(n.iter().all(|v| v.abs() <= 0.2));
        let quiet = alternating_speakers(&[(150.0, 0.5), (400.0, 0.5)], 0.5, 16_000, 0.4);
        // Each voice contributes speech + a trailing gap.
        assert_eq!(quiet.len(), 32_000);
    }
}
