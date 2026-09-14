//! Short-Time Fourier Transform.

use ndarray::Array2;
use rustfft::num_complex::Complex;

use tpt_av_voice_utils::dsp::{hann_window, RealFft};
use tpt_av_voice_utils::VoiceError;

/// STFT geometry. Defaults follow the voice-isolation conventions:
/// 32 ms window @ 16 kHz, 50 % overlap.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StftConfig {
    /// FFT size in samples (power of two recommended).
    pub n_fft: usize,
    /// Hop between frames in samples.
    pub hop_size: usize,
}

impl Default for StftConfig {
    fn default() -> Self {
        StftConfig { n_fft: 512, hop_size: 256 }
    }
}

impl StftConfig {
    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.n_fft == 0 || !self.n_fft.is_power_of_two() {
            return Err(VoiceError::config("n_fft must be a power of two"));
        }
        if self.hop_size == 0 || self.hop_size > self.n_fft {
            return Err(VoiceError::config("hop_size must be in (0, n_fft]"));
        }
        Ok(())
    }

    /// Number of non-negative frequency bins.
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Cached forward STFT.
pub struct Stft {
    config: StftConfig,
    fft: RealFft,
    window: Vec<f32>,
}

impl Stft {
    pub fn new(config: StftConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        Ok(Stft { fft: RealFft::new(config.n_fft), window: hann_window(config.n_fft), config })
    }

    pub fn config(&self) -> &StftConfig {
        &self.config
    }

    /// Transforms `samples` into a `(n_frames, n_bins)` complex matrix.
    /// The tail shorter than one frame is dropped (the isolator pads).
    pub fn forward(&self, samples: &[f32]) -> Array2<Complex<f32>> {
        let n_frames = samples.len().saturating_sub(self.config.n_fft) / self.config.hop_size + 1;
        let bins = self.config.n_bins();
        let mut out = Array2::from_elem((n_frames, bins), Complex::new(0.0, 0.0));
        for t in 0..n_frames {
            let start = t * self.config.hop_size;
            let frame = &samples[start..(start + self.config.n_fft).min(samples.len())];
            let windowed: Vec<f32> =
                frame.iter().zip(&self.window).map(|(s, w)| s * w).collect();
            let spectrum = self.fft.forward(&windowed);
            for (b, c) in spectrum.iter().enumerate().take(bins) {
                out[[t, b]] = *c;
            }
        }
        out
    }

    /// Magnitudes `(n_frames, n_bins)`.
    pub fn magnitudes(&self, samples: &[f32]) -> Array2<f32> {
        self.forward(samples).mapv(|c| c.norm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(StftConfig::default().validate().is_ok());
        assert!(StftConfig { n_fft: 500, hop_size: 128 }.validate().is_err());
        assert!(StftConfig { n_fft: 512, hop_size: 0 }.validate().is_err());
        assert!(StftConfig { n_fft: 512, hop_size: 600 }.validate().is_err());
    }

    #[test]
    fn shapes_match_config() {
        let stft = Stft::new(StftConfig::default()).unwrap();
        let spec = stft.forward(&vec![0.0; 16_000]);
        // (16000 - 512) / 256 + 1 = 61 frames
        assert_eq!(spec.dim(), (61, 257));
    }

    #[test]
    fn tone_lands_in_expected_bin() {
        let stft = Stft::new(StftConfig::default()).unwrap();
        let tone: Vec<f32> = (0..8_192)
            .map(|i| (std::f32::consts::TAU * 1_000.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mag = stft.magnitudes(&tone);
        // 1 kHz @ 16 kHz over 512 samples → bin 32.
        let row = mag.row(0);
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 32);
    }
}
