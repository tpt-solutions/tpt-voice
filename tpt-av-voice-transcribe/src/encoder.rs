//! Audio feature extraction: Whisper-style log-mel spectrograms.
//!
//! 16 kHz audio is framed with 25 ms windows / 10 ms hops, transformed
//! with an FFT, folded through a mel filter bank and compressed with a
//! Whisper-style log normalization. Output is `(n_mels, T)`.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::dsp::{hann_window, mel_filterbank, RealFft};
use tpt_av_voice_utils::VoiceError;

/// Mel front-end configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MelConfig {
    /// Expected input sample rate in Hz.
    pub sample_rate: u32,
    /// FFT window length in samples (400 = 25 ms @ 16 kHz).
    pub window_size: usize,
    /// Hop between frames in samples (160 = 10 ms @ 16 kHz).
    pub hop_size: usize,
    /// Number of mel channels (Whisper uses 80).
    pub n_mels: usize,
    /// Lower edge of the mel bank in Hz.
    pub f_min: f32,
    /// Upper edge of the mel bank in Hz (0 = Nyquist).
    pub f_max: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        MelConfig {
            sample_rate: 16_000,
            window_size: 400,
            hop_size: 160,
            n_mels: 80,
            f_min: 0.0,
            f_max: 8000.0,
        }
    }
}

impl MelConfig {
    /// Validates frame/hop/mel parameters.
    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.sample_rate == 0 {
            return Err(VoiceError::config("sample_rate must be > 0"));
        }
        if self.window_size == 0 || self.hop_size == 0 {
            return Err(VoiceError::config("window_size and hop_size must be > 0"));
        }
        if self.n_mels == 0 {
            return Err(VoiceError::config("n_mels must be > 0"));
        }
        Ok(())
    }
}

/// Cached mel front-end (FFT plans, window, filter bank).
pub struct MelFrontend {
    config: MelConfig,
    fft: RealFft,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
}

impl MelFrontend {
    /// Builds a front-end from a validated [`MelConfig`].
    pub fn new(config: MelConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        let fft = RealFft::new(config.window_size);
        let window = hann_window(config.window_size);
        let f_max = if config.f_max <= 0.0 {
            config.sample_rate as f32 / 2.0
        } else {
            config.f_max
        };
        let filterbank = mel_filterbank(
            config.n_mels,
            config.window_size,
            config.sample_rate,
            config.f_min,
            f_max,
        );
        Ok(MelFrontend {
            config,
            fft,
            window,
            filterbank,
        })
    }

    /// Borrows the configuration.
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Computes the raw (linear) mel energy matrix `(n_mels, T)`.
    /// Apply a log transform (see [`MelFrontend::log_mel`]) before use.
    pub fn mel_energy(&self, samples: &[f32]) -> Array2<f32> {
        let n_frames = if samples.len() > self.config.window_size {
            (samples.len() - self.config.window_size) / self.config.hop_size + 1
        } else {
            0
        };
        let mut out = Array2::<f32>::zeros((self.config.n_mels, n_frames.max(1)));
        if n_frames == 0 {
            return out;
        }
        let n_bins = self.fft.size() / 2 + 1;
        for t in 0..n_frames {
            let start = t * self.config.hop_size;
            let frame = &samples[start..start + self.config.window_size];
            let windowed: Vec<f32> = frame.iter().zip(&self.window).map(|(s, w)| s * w).collect();
            let spectrum = self.fft.forward(&windowed);
            let mut power = Vec::with_capacity(n_bins);
            for c in &spectrum[..n_bins] {
                power.push(c.norm_sqr() / self.fft.size() as f32);
            }
            for (m, filter) in self.filterbank.iter().enumerate() {
                out[[m, t]] = filter.iter().zip(&power).map(|(c, p)| c * p).sum::<f32>();
            }
        }
        out
    }

    /// Whisper-style normalized log-mel `(n_mels, T)`:
    /// `log10` then shift/scale so the loudest bin sits near 1.0 and the
    /// floor is clamped 8 dB below the peak.
    pub fn log_mel(&self, samples: &[f32]) -> Array2<f32> {
        let mut mel = self.mel_energy(samples);
        for v in mel.iter_mut() {
            *v = v.max(1e-10).log10();
        }
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in mel.iter_mut() {
            *v = ((*v - max) + 8.0).max(0.0) / 8.0 * 2.0 - 1.0;
        }
        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontend() -> MelFrontend {
        MelFrontend::new(MelConfig::default()).unwrap()
    }

    #[test]
    fn config_validation() {
        assert!(MelConfig::default().validate().is_ok());
        let bad = MelConfig {
            hop_size: 0,
            ..MelConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn output_shape() {
        let fe = frontend();
        let samples = vec![0.0f32; 16_000];
        let mel = fe.mel_energy(&samples);
        assert_eq!(mel.dim(), (80, 98));
        let mel = fe.log_mel(&samples);
        assert_eq!(mel.dim(), (80, 98));
    }

    #[test]
    fn short_input_padded() {
        let fe = frontend();
        let mel = fe.log_mel(&[0.5, -0.5]);
        assert_eq!(mel.dim(), (80, 1));
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tone_has_energy_in_high_mels() {
        // A 4 kHz tone must light up the upper part of the mel bank,
        // while a 100 Hz tone stays in the bottom.
        let fe = frontend();
        let mk = |hz: f32| {
            let v: Vec<f32> = (0..16_000)
                .map(|i| (std::f32::consts::TAU * hz * i as f32 / 16_000.0).sin())
                .collect();
            fe.log_mel(&v)
        };
        let high = mk(4_000.0);
        let low = mk(100.0);
        let high_top: f32 = high.slice(ndarray::s![60.., ..]).mean().unwrap();
        let low_top: f32 = low.slice(ndarray::s![60.., ..]).mean().unwrap();
        assert!(high_top > low_top + 0.1, "{high_top} vs {low_top}");
        let low_bottom: f32 = low.slice(ndarray::s![..10, ..]).mean().unwrap();
        let high_bottom: f32 = high.slice(ndarray::s![..10, ..]).mean().unwrap();
        assert!(
            low_bottom > high_bottom + 0.3,
            "{low_bottom} vs {high_bottom}"
        );
    }

    #[test]
    fn log_mel_is_normalized() {
        let fe = frontend();
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mel = fe.log_mel(&tone);
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((max - 1.0).abs() < 1e-4, "max {max}");
        assert!(mel.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
