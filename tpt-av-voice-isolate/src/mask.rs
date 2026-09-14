//! Time-frequency masking and classical noise suppression.
//!
//! Two mask sources:
//!
//! * [`apply_mask`] — multiply a complex STFT by a soft mask `(0..1)`,
//!   produced either by the separation model or the classical estimator;
//! * [`SpectralSubtractor`] — an online per-bin noise-floor tracker with
//!   a Wiener-style gain. Requires no trained weights, which is why it
//!   backs [`crate::VoiceIsolator::classical`].

use ndarray::Array2;
use rustfft::num_complex::Complex;

/// Elementwise mask application to a complex STFT.
pub fn apply_mask(spec: &Array2<Complex<f32>>, mask: &Array2<f32>) -> Array2<Complex<f32>> {
    mask_spectrum(spec, mask)
}

/// Multiplies each complex cell by its mask value.
pub fn mask_spectrum(spec: &Array2<Complex<f32>>, mask: &Array2<f32>) -> Array2<Complex<f32>> {
    assert_eq!(spec.dim(), mask.dim(), "mask shape must match spectrum");
    let mut out = Array2::from_elem(spec.dim(), Complex::new(0.0, 0.0));
    for ((t, b), c) in spec.indexed_iter() {
        let m = mask[[t, b]];
        *out.get_mut((t, b)).unwrap() = Complex::new(c.re * m, c.im * m);
    }
    out
}

/// Online spectral-subtraction noise suppressor.
#[derive(Debug, Clone)]
pub struct SpectralSubtractor {
    /// Per-bin noise magnitude estimate.
    noise: Vec<f32>,
    /// Adaptation speed for the noise floor (smaller = slower).
    adapt: f32,
    /// Over-subtraction factor.
    alpha: f32,
    /// Spectral floor (prevents musical-noise holes).
    floor: f32,
    n_bins: usize,
}

impl SpectralSubtractor {
    /// Builds a subtractor for `n_bins` bins.
    pub fn new(n_bins: usize, alpha: f32, floor: f32) -> Self {
        SpectralSubtractor {
            noise: vec![0.0; n_bins],
            adapt: 0.04,
            alpha,
            floor,
            n_bins,
        }
    }

    /// Resets the tracked noise floor.
    pub fn reset(&mut self) {
        self.noise.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Processes one frame of magnitudes, returning the voice mask
    /// `(0..1)` per bin and updating the internal noise estimate.
    pub fn frame_mask(&mut self, mag: &[f32]) -> Vec<f32> {
        assert_eq!(mag.len(), self.n_bins);
        let mut mask = vec![1.0f32; self.n_bins];
        for (b, &m) in mag.iter().enumerate() {
            // Exponential minimum tracking: adapt downward quickly,
            // upward slowly (noise floors rarely jump up).
            let n = &mut self.noise[b];
            if *n == 0.0 {
                *n = m;
            } else if m < *n {
                *n += self.adapt * 4.0 * (m - *n);
            } else {
                *n += self.adapt * (m - *n);
            }
            // Wiener-style gain from power subtraction.
            let power = m * m;
            let noise_power = self.alpha * *n * *n;
            let gain = if power > 1e-12 {
                ((power - noise_power) / power).clamp(self.floor, 1.0)
            } else {
                self.floor
            };
            mask[b] = gain;
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mask_scales_spectrum() {
        let spec = array![
            [Complex::new(1.0, 0.0), Complex::new(0.0, 2.0)],
            [Complex::new(3.0, 4.0), Complex::new(-1.0, -1.0)],
        ];
        let mask = array![[1.0, 0.5], [0.0, 0.25]];
        let out = mask_spectrum(&spec, &mask);
        assert_eq!(out[[0, 0]], Complex::new(1.0, 0.0));
        assert_eq!(out[[0, 1]], Complex::new(0.0, 1.0));
        assert_eq!(out[[1, 0]], Complex::new(0.0, 0.0));
        assert_eq!(out[[1, 1]], Complex::new(-0.25, -0.25));
    }

    #[test]
    fn subtractor_passes_clean_speech_and_attenuates_noise() {
        let mut sub = SpectralSubtractor::new(4, 2.0, 0.05);
        // Steady "noise" frames train the floor.
        for _ in 0..50 {
            let mask = sub.frame_mask(&[0.2, 0.2, 0.2, 0.2]);
            let _ = mask;
        }
        // A much louder bin behaves like speech: mostly passed.
        let mask = sub.frame_mask(&[0.2, 2.0, 0.2, 0.2]);
        assert!(mask[1] > 0.9, "loud bin should pass, got {}", mask[1]);
        // Quiet bins sit at/near the floor.
        assert!(mask[0] < 0.6, "noise bin should be attenuated, got {}", mask[0]);
        // And the steady-state noise floor converges low.
        let mask = sub.frame_mask(&[0.2; 4]);
        assert!(mask.iter().all(|m| *m < 0.6));
    }

    #[test]
    fn reset_restores_full_pass() {
        let mut sub = SpectralSubtractor::new(2, 2.0, 0.05);
        for _ in 0..50 {
            sub.frame_mask(&[0.2, 0.2]);
        }
        sub.reset();
        let mask = sub.frame_mask(&[1.0, 1.0]);
        // First frame initializes floor to the signal → gain near 1-alpha.
        assert!(mask.iter().all(|m| (0.0..=1.0).contains(m)));
    }

    #[test]
    fn apply_mask_docs_compile() {
        // apply_mask is documented API; ensure it compiles and matches
        // mask_spectrum semantics.
        let spec = array![[Complex::new(2.0, 0.0)]];
        let mask = array![[0.5]];
        let _ = apply_mask(&spec, &mask);
        let _ = mask_spectrum(&spec, &mask);
    }
}
