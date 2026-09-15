//! Shared DSP primitives: windows, FFT glue, mel filter banks, MFCC.
//!
//! Used by `tpt-av-voice-transcribe` (log-mel features), `tpt-av-voice-isolate`
//! (STFT) and `tpt-av-voice-diarize` (MFCC embeddings).

use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Periodic Hann window (`w[k] = 0.5 - 0.5*cos(2πk/n)`), the COLA-friendly
/// variant used for STFT overlap-add.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 - 0.5 * (std::f32::consts::TAU * k as f32 / n as f32).cos())
        .collect()
}

/// Symmetric Hann window (`w[k] = 0.5 - 0.5*cos(2πk/(n-1))`) as used by
/// classical MFCC front-ends.
pub fn hann_window_symmetric(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|k| 0.5 - 0.5 * (std::f32::consts::TAU * k as f32 / (n - 1) as f32).cos())
        .collect()
}

/// Root-mean-square of a sample slice (0.0 for empty input).
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Slices `signal` into overlapping frames of `frame_len` samples advanced
/// by `hop` samples. The tail shorter than `frame_len` is dropped.
pub fn frames(signal: &[f32], frame_len: usize, hop: usize) -> impl Iterator<Item = &[f32]> {
    debug_assert!(frame_len > 0 && hop > 0);
    signal.windows(frame_len).step_by(hop)
}

/// Real-valued FFT front-end with cached rustfft plans.
pub struct RealFft {
    n: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl RealFft {
    /// Plans FFTs of size `n` (any factorization; rustfft handles primes).
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        RealFft {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    /// FFT size.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Forward transform of a real signal of length `n` (shorter input is
    /// zero-padded; longer input is truncated).
    pub fn forward(&self, input: &[f32]) -> Vec<rustfft::num_complex::Complex<f32>> {
        let mut buf: Vec<_> = input
            .iter()
            .take(self.n)
            .map(|&s| rustfft::num_complex::Complex::new(s, 0.0))
            .collect();
        buf.resize(self.n, rustfft::num_complex::Complex::new(0.0, 0.0));
        self.forward.process(&mut buf);
        buf
    }

    /// Inverse transform of a Hermitian half-spectrum (`n/2 + 1` bins),
    /// returning the real output scaled by `1/n` (i.e. properly
    /// normalized so `inverse(forward(x)) == x`).
    pub fn inverse_half(&self, half: &[rustfft::num_complex::Complex<f32>]) -> Vec<f32> {
        assert!(
            half.len() == self.n / 2 + 1,
            "expected {} bins, got {}",
            self.n / 2 + 1,
            half.len()
        );
        let mut buf = vec![rustfft::num_complex::Complex::new(0.0, 0.0); self.n];
        buf[..half.len()].copy_from_slice(half);
        // Hermitian mirror: X[n-k] = conj(X[k]).
        for k in 1..self.n / 2 {
            let c = half[k];
            buf[self.n - k] = rustfft::num_complex::Complex::new(c.re, -c.im);
        }
        self.inverse.process(&mut buf);
        let scale = 1.0 / self.n as f32;
        buf.into_iter().map(|c| c.re * scale).collect()
    }
}

/// Converts Hz to mel (HTK scale, `2595·log10(1 + f/700)`).
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts mel to Hz (inverse of [`hz_to_mel`]).
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Builds a mel filter bank: `n_mels` triangular filters over
/// `[f_min, f_max]` Hz, each a dense vector of length `n_fft / 2 + 1`.
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let f_max = if f_max <= f_min {
        sample_rate as f32 / 2.0
    } else {
        f_max
    };
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    // n_mels + 2 boundary points, evenly spaced in mel.
    let points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let bin_hz = sample_rate as f32 / n_fft as f32;

    let mut bank = vec![vec![0.0f32; n_bins]; n_mels];
    for (m, filter) in bank.iter_mut().enumerate() {
        let (left, center, right) = (points[m], points[m + 1], points[m + 2]);
        for (bin, coeff) in filter.iter_mut().enumerate() {
            let f = bin as f32 * bin_hz;
            if f > left && f < right {
                *coeff = if f <= center {
                    (f - left) / (center - left)
                } else {
                    (right - f) / (right - center)
                };
            }
        }
    }
    bank
}

/// Applies a mel filter bank to a power spectrum.
pub fn mel_energies(filterbank: &[Vec<f32>], power_spectrum: &[f32]) -> Vec<f32> {
    filterbank
        .iter()
        .map(|filter| {
            filter
                .iter()
                .zip(power_spectrum)
                .map(|(c, p)| c * p)
                .sum::<f32>()
                .max(1e-10)
                .ln()
        })
        .collect()
}

/// DCT-II along a vector (used as the MFCC decorrelation step).
pub fn dct_ii(input: &[f32]) -> Vec<f32> {
    let m = input.len();
    (0..m)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(n, &x)| {
                    x * (std::f32::consts::PI * k as f32 * (n as f32 + 0.5) / m as f32).cos()
                })
                .sum::<f32>()
        })
        .collect()
}

/// Configuration for [`mfcc`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MfccConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Analysis window length in samples (default 400 = 25 ms @ 16 kHz).
    pub frame_len: usize,
    /// Hop between frames in samples (default 160 = 10 ms @ 16 kHz).
    pub hop: usize,
    /// Number of mel filters.
    pub n_mels: usize,
    /// Number of MFCC coefficients kept (including c0).
    pub n_mfcc: usize,
    /// Lower filter bank edge in Hz.
    pub f_min: f32,
    /// Upper filter bank edge in Hz (0 = Nyquist).
    pub f_max: f32,
    /// Pre-emphasis coefficient (0 disables).
    pub pre_emphasis: f32,
}

impl Default for MfccConfig {
    fn default() -> Self {
        MfccConfig {
            sample_rate: 16_000,
            frame_len: 400,
            hop: 160,
            n_mels: 40,
            n_mfcc: 20,
            f_min: 20.0,
            f_max: 0.0,
            pre_emphasis: 0.97,
        }
    }
}

impl MfccConfig {
    /// Validates frame/hop/mel counts.
    pub fn validate(&self) -> Result<(), crate::VoiceError> {
        if self.frame_len == 0 || self.hop == 0 {
            return Err(crate::VoiceError::config("frame_len and hop must be > 0"));
        }
        if self.n_mels == 0 || self.n_mfcc == 0 || self.n_mfcc > self.n_mels {
            return Err(crate::VoiceError::config("require 0 < n_mfcc <= n_mels"));
        }
        Ok(())
    }
}

/// Standard MFCC front-end: pre-emphasis → framing → Hann → power
/// spectrum → mel filter bank → ln → DCT-II.
pub fn mfcc(signal: &[f32], config: &MfccConfig) -> Result<Vec<Vec<f32>>, crate::VoiceError> {
    config.validate()?;
    let n_fft = config.frame_len.next_power_of_two();
    let fft = RealFft::new(n_fft);
    let window = hann_window(config.frame_len);
    let filterbank = mel_filterbank(
        config.n_mels,
        n_fft,
        config.sample_rate,
        config.f_min,
        config.f_max,
    );

    // Pre-emphasized signal (y[0] = x[0]).
    let mut emph = Vec::with_capacity(signal.len());
    if let Some(first) = signal.first() {
        emph.push(*first);
    }
    for w in signal.windows(2) {
        emph.push(w[1] - config.pre_emphasis * w[0]);
    }

    let mut out = Vec::new();
    for frame in frames(&emph, config.frame_len, config.hop) {
        let windowed: Vec<f32> = frame.iter().zip(&window).map(|(s, w)| s * w).collect();
        let spectrum = fft.forward(&windowed);
        let power: Vec<f32> = spectrum[..n_fft / 2 + 1]
            .iter()
            .map(|c| c.norm_sqr() / n_fft as f32)
            .collect();
        out.push(dct_ii(&mel_energies(&filterbank, &power))[..config.n_mfcc].to_vec());
    }
    Ok(out)
}

/// Resamples `samples` from `from_hz` to `to_hz` by linear interpolation.
/// Quality is adequate for pipeline plumbing; production decoding should
/// receive correctly-rate-matched audio (e.g. from `tpt-cadence`).
pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(to_hz) / f64::from(from_hz);
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(out_len);
    let step = 1.0 / ratio;
    for i in 0..out_len {
        let pos = i as f64 * step;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let s0 = samples[i0.min(samples.len() - 1)];
        let s1 = samples[(i0 + 1).min(samples.len() - 1)];
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_shapes() {
        let w = hann_window(4);
        assert!((w[0]).abs() < 1e-6);
        assert!((w[1] - 0.5).abs() < 1e-6);
        assert!((w[2] - 1.0).abs() < 1e-6);
        let ws = hann_window_symmetric(5);
        assert!((ws[0]).abs() < 1e-6);
        assert!((ws[2] - 1.0).abs() < 1e-6);
        assert!((ws[1] - ws[3]).abs() < 1e-6);
    }

    #[test]
    fn fft_round_trips() {
        let n = 64;
        let fft = RealFft::new(n);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin() + 0.2).collect();
        let spectrum = fft.forward(&x);
        let back = fft.inverse_half(&spectrum[..n / 2 + 1]);
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn fft_detects_tone() {
        // 1 kHz sine @ 8 kHz over 256 samples → peak bin at 32.
        let n = 256;
        let fft = RealFft::new(n);
        let x: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 1000.0 * i as f32 / 8000.0).sin())
            .collect();
        let spectrum = fft.forward(&x);
        // Only the non-negative frequency half is meaningful for a real signal.
        let mags: Vec<f32> = spectrum[..n / 2 + 1].iter().map(|c| c.norm()).collect();
        let peak = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 32);
    }

    #[test]
    fn mel_round_trip() {
        assert!((mel_to_hz(hz_to_mel(1000.0)) - 1000.0).abs() < 1e-3);
        let bank = mel_filterbank(26, 512, 16_000, 20.0, 8000.0);
        assert_eq!(bank.len(), 26);
        for filter in &bank {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0, "empty mel filter");
        }
        // First filter starts near f_min, last ends near f_max.
        assert!(bank[0].iter().take(4).any(|&c| c > 0.0));
    }

    #[test]
    fn mfcc_shape_and_energy() {
        let cfg = MfccConfig::default();
        // 440 Hz tone: comfortably inside the mel filter bank.
        let signal: Vec<f32> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let m = mfcc(&signal, &cfg).unwrap();
        // (16000 - 400) / 160 + 1 = 98 frames
        assert_eq!(m.len(), 98);
        assert_eq!(m[0].len(), cfg.n_mfcc);
        // Silence → strongly negative log energies.
        let silent = mfcc(&vec![0.0; 16_000], &cfg).unwrap();
        assert!(silent[0][0] < 0.0);
        // A loud tone has more energy than silence (c0 = log-energy term).
        assert!(m[0][0] > silent[0][0]);
    }

    #[test]
    fn rms_and_frames() {
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert_eq!(rms(&[]), 0.0);
        let signal: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let f: Vec<&[f32]> = frames(&signal, 4, 2).collect();
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(f[3], &[6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn dct_ii_known_values() {
        // DCT-II of a constant vector: only c0 non-zero.
        let d = dct_ii(&[1.0; 4]);
        assert!((d[0] - 4.0).abs() < 1e-5);
        assert!(d[1].abs() < 1e-5 && d[2].abs() < 1e-5 && d[3].abs() < 1e-5);
    }

    #[test]
    fn resample_basics() {
        // Same rate → unchanged.
        let x = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&x, 16_000, 16_000), x);
        // Upsample 2× keeps endpoints, doubles length.
        let y = resample_linear(&x, 8_000, 16_000);
        assert_eq!(y.len(), 6);
        assert!((y[0] - 0.1).abs() < 1e-6);
        assert!((y[4] - 0.3).abs() < 1e-6);
        // Downsample halves the length.
        assert_eq!(resample_linear(&x, 16_000, 8_000).len(), 2);
        // Mono tone keeps its frequency.
        let tone: Vec<f32> = (0..8_000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 8_000.0).sin())
            .collect();
        let up = resample_linear(&tone, 8_000, 16_000);
        let fft = RealFft::new(1024);
        let spec = fft.forward(&up[..1024]);
        let mags: Vec<f32> = spec[..513].iter().map(|c| c.norm()).collect();
        let peak = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 28, "440 Hz @ 16 kHz over 1024 samples → bin 28");
    }
}
