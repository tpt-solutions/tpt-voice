//! Audio feature extraction: Whisper-style log-mel spectrograms.
//!
//! 16 kHz audio is framed with 25 ms windows / 10 ms hops, transformed
//! with an FFT, folded through a mel filter bank and compressed with a
//! Whisper-style log normalization. Output is `(n_mels, T)`.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::dsp::{hann_window, RealFft};
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

/// Whisper pads or truncates every input to 30 s before the front end.
pub const WHISPER_N_SAMPLES: usize = 30 * 16_000;

/// Slaney-scaled triangular mel filters, area-normalised, as used by the
/// Whisper feature extractor (librosa `norm="slaney"`, `mel_scale="slaney"`).
/// Returns one row per mel band over `n_fft / 2 + 1` FFT bins.
pub fn slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;
    let hz_to_mel = |hz: f64| {
        if hz >= MIN_LOG_HZ {
            min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
        } else {
            hz / F_SP
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel >= min_log_mel {
            MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
        } else {
            mel * F_SP
        }
    };

    let n_bins = n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * f64::from(sample_rate) / n_fft as f64)
        .collect();
    let mel_lo = hz_to_mel(f64::from(f_min));
    let mel_hi = hz_to_mel(f64::from(f_max));
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f64 / (n_mels + 1) as f64))
        .collect();

    (0..n_mels)
        .map(|m| {
            let (lo, mid, hi) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
            let enorm = 2.0 / (hi - lo);
            fft_freqs
                .iter()
                .map(|&f| {
                    let rising = (f - lo) / (mid - lo);
                    let falling = (hi - f) / (hi - mid);
                    (rising.min(falling).max(0.0) * enorm) as f32
                })
                .collect()
        })
        .collect()
}

/// Reflect-pads `x` by `pad` samples on both ends (`torch.stft` centring).
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((0..pad).map(|i| x[n - 2 - i]));
    out
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
        let filterbank = slaney_mel_filterbank(
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

    /// Whisper-exact log-mel `(n_mels, 3000)` for a clip of any length.
    ///
    /// Matches `WhisperFeatureExtractor`: the clip is padded or truncated to
    /// 30 s, frames are centred with reflect padding, power is unnormalised,
    /// the last frame is dropped, and the result is `log10` clamped to 8 units
    /// below the global peak and scaled to `(x + 4) / 4`.
    pub fn log_mel(&self, samples: &[f32]) -> Array2<f32> {
        let n_fft = self.config.window_size;
        let hop = self.config.hop_size;
        let n_bins = n_fft / 2 + 1;

        let mut audio = vec![0.0_f32; WHISPER_N_SAMPLES];
        let keep = samples.len().min(WHISPER_N_SAMPLES);
        audio[..keep].copy_from_slice(&samples[..keep]);
        let padded = reflect_pad(&audio, n_fft / 2);
        let n_frames = (padded.len() - n_fft) / hop; // drops the final frame

        let n_mels = self.config.n_mels;
        let mut mel = Array2::<f32>::zeros((n_mels, n_frames));
        for t in 0..n_frames {
            let start = t * hop;
            let windowed: Vec<f32> = padded[start..start + n_fft]
                .iter()
                .zip(&self.window)
                .map(|(s, w)| s * w)
                .collect();
            let spectrum = self.fft.forward(&windowed);
            let power: Vec<f32> = spectrum[..n_bins].iter().map(|c| c.norm_sqr()).collect();
            for (m, filter) in self.filterbank.iter().enumerate() {
                mel[[m, t]] = filter.iter().zip(&power).map(|(c, p)| c * p).sum::<f32>();
            }
        }

        mel.mapv_inplace(|v| v.max(1e-10).log10());
        let peak = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        mel.mapv_inplace(|v| (v.max(peak - 8.0) + 4.0) / 4.0);
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
        // Raw energy keeps the framing without centring: 98 frames per second.
        let mel = fe.mel_energy(&samples);
        assert_eq!(mel.dim(), (80, 98));
        // Whisper's front end always yields 30 s of frames (3000).
        let mel = fe.log_mel(&samples);
        assert_eq!(mel.dim(), (80, 3000));
    }

    #[test]
    fn short_input_padded() {
        let fe = frontend();
        let mel = fe.log_mel(&[0.5, -0.5]);
        assert_eq!(mel.dim(), (80, 3000));
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
            fe.mel_energy(&v).mapv(|e| e.max(1e-10).log10())
        };
        let high = mk(4_000.0);
        let low = mk(100.0);
        let high_top: f32 = high.slice(ndarray::s![60.., ..]).mean().unwrap();
        let low_top: f32 = low.slice(ndarray::s![60.., ..]).mean().unwrap();
        assert!(high_top > low_top + 1.0, "{high_top} vs {low_top}");
        let low_bottom: f32 = low.slice(ndarray::s![..10, ..]).mean().unwrap();
        let high_bottom: f32 = high.slice(ndarray::s![..10, ..]).mean().unwrap();
        assert!(low_bottom > high_bottom + 1.0, "{low_bottom} vs {high_bottom}");
    }

    #[test]
    fn log_mel_is_clamped_to_eight_log_units_below_peak() {
        // Whisper's normalisation: (max(x, peak - 8) + 4) / 4, so the output
        // spans exactly 2.0 from the floor to the peak.
        let fe = frontend();
        let tone: Vec<f32> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let mel = fe.log_mel(&tone);
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = mel.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max - min <= 2.0 + 1e-3, "span {}", max - min);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn slaney_filterbank_has_expected_shape_and_coverage() {
        let bank = slaney_mel_filterbank(80, 400, 16_000, 0.0, 8_000.0);
        assert_eq!(bank.len(), 80);
        assert!(bank.iter().all(|row| row.len() == 201));
        // Every FFT bin between the first and last band centre is covered.
        let covered: Vec<bool> = (0..201)
            .map(|k| bank.iter().any(|row| row[k] > 0.0))
            .collect();
        assert!(covered[2..190].iter().all(|&c| c), "gaps in the mel coverage");
        // Each triangle is area-normalised, so its peak falls as the band widens.
        let peak = |row: &Vec<f32>| row.iter().copied().fold(0.0_f32, f32::max);
        assert!(peak(&bank[0]) > peak(&bank[79]));
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let out = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(out, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }
}
