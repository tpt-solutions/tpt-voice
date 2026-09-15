//! Inverse STFT with weighted overlap-add.
//!
//! Hann analysis + Hann synthesis windows with overlap-add of the
//! squared window normalizes to a constant in the interior for hop
//! sizes that divide the window (COLA), so `istft(stft(x)) == x` up to
//! edge effects.

use ndarray::Array2;
use rustfft::num_complex::Complex;

use tpt_av_voice_utils::dsp::{hann_window, RealFft};
use tpt_av_voice_utils::VoiceError;

use crate::stft::StftConfig;

/// Cached inverse STFT.
pub struct Istft {
    config: StftConfig,
    fft: RealFft,
    window: Vec<f32>,
}

impl Istft {
    pub fn new(config: StftConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        Ok(Istft {
            fft: RealFft::new(config.n_fft),
            window: hann_window(config.n_fft),
            config,
        })
    }

    /// Reconstructs `out_len` samples from a `(n_frames, n_bins)`
    /// half-spectrum.
    pub fn reconstruct(&self, spec: &Array2<Complex<f32>>, out_len: usize) -> Vec<f32> {
        let n = self.config.n_fft;
        let hop = self.config.hop_size;
        let mut acc = vec![0.0f32; out_len + n];
        let mut wsum = vec![0.0f32; out_len + n];
        for t in 0..spec.nrows() {
            let half: Vec<Complex<f32>> = spec.slice(ndarray::s![t, ..]).to_vec();
            let frame = self.fft.inverse_half(&half);
            let start = t * hop;
            for (i, s) in frame.iter().enumerate() {
                let w = self.window[i];
                acc[start + i] += s * w;
                wsum[start + i] += w * w;
            }
        }
        acc.truncate(out_len);
        wsum.truncate(out_len);
        // Where the squared-window sum is ~0 (the very edges) pass 0.
        acc.iter()
            .zip(wsum)
            .map(|(s, w)| if w > 1e-8 { s / w } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stft::Stft;

    #[test]
    fn round_trips() {
        let cfg = StftConfig::default();
        let stft = Stft::new(cfg.clone()).unwrap();
        let istft = Istft::new(cfg).unwrap();

        let len = 16_000;
        let x: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.05).sin() * 0.7 + (i as f32 * 0.0013).cos() * 0.3)
            .collect();
        let spec = stft.forward(&x);
        let y = istft.reconstruct(&spec, len);
        // The very first/last samples have zero window weight (periodic
        // Hann vanishes at k=0), so compare the covered interior.
        let (lo, hi) = (512, len - 512);
        for (i, (a, b)) in x[lo..hi].iter().zip(&y[lo..hi]).enumerate() {
            assert!((a - b).abs() < 1e-3, "at {}: {a} vs {b}", i + lo);
        }
        // Edges decay toward silence instead of blowing up.
        assert!(y[0].abs() < 1.0 && y[len - 1].abs() < 1.0);
    }

    #[test]
    fn silence_stays_silent() {
        let cfg = StftConfig::default();
        let istft = Istft::new(cfg).unwrap();
        let spec = Array2::from_elem((4, 257), Complex::new(0.0, 0.0));
        let y = istft.reconstruct(&spec, 1_024);
        assert!(y.iter().all(|v| v.abs() < 1e-6));
    }
}
