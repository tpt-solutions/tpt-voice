//! Voice activity detection and speech segmentation.
//!
//! Energy-based VAD with an adaptive noise floor: frame RMS is compared
//! (in dBFS) against a rolling noise estimate with hysteresis, then
//! speech runs shorter than `min_speech_secs` are dropped and gaps
//! shorter than `min_silence_secs` are bridged.

use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::dsp::{hann_window, rms, RealFft};
use tpt_av_voice_utils::{time::samples_to_secs, TimeRange, VoiceError};

/// VAD tuning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadConfig {
    /// Analysis window in samples (25 ms @ 16 kHz).
    pub frame_len: usize,
    /// Hop between frames in samples (10 ms @ 16 kHz).
    pub hop: usize,
    /// Speech must exceed the noise floor by this many dB.
    pub snr_threshold_db: f32,
    /// Drop speech bursts shorter than this.
    pub min_speech_secs: f64,
    /// Bridge silence gaps shorter than this.
    pub min_silence_secs: f64,
    /// Pad each speech region by this much on both sides.
    pub padding_secs: f64,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            frame_len: 400,
            hop: 160,
            snr_threshold_db: 8.0,
            min_speech_secs: 0.25,
            min_silence_secs: 0.4,
            padding_secs: 0.1,
        }
    }
}

/// Detected speech regions, in time order.
pub type SpeechRegions = Vec<TimeRange>;

/// Energy VAD over mono PCM.
pub struct VoiceActivityDetector {
    config: VadConfig,
}

impl VoiceActivityDetector {
    pub fn new(config: VadConfig) -> Result<Self, VoiceError> {
        if config.frame_len == 0 || config.hop == 0 {
            return Err(VoiceError::config("VAD frame_len and hop must be > 0"));
        }
        Ok(VoiceActivityDetector { config })
    }

    /// Detects speech regions in mono PCM.
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> SpeechRegions {
        if samples.len() < self.config.frame_len {
            return Vec::new();
        }
        let window = hann_window(self.config.frame_len);
        let fft = RealFft::new(self.config.frame_len.next_power_of_two());

        // Frame energies in dBFS, weighted toward perceptual loudness
        // (band-limited: DC and ultra-high bins contribute little).
        let mut frame_db: Vec<f32> = Vec::new();
        for frame in samples
            .windows(self.config.frame_len)
            .step_by(self.config.hop)
        {
            let windowed: Vec<f32> = frame.iter().zip(&window).map(|(s, w)| s * w).collect();
            let spectrum = fft.forward(&windowed);
            let bins = spectrum.len() / 2;
            // Weighted band energy: 80 Hz .. 4 kHz dominates speech.
            let bin_hz = sample_rate as f32 / fft.size() as f32;
            let lo = (80.0 / bin_hz) as usize;
            let hi = ((4_000.0 / bin_hz) as usize).min(bins).max(lo + 1);
            let energy: f32 = spectrum[lo..hi].iter().map(|c| c.norm_sqr()).sum();
            let db = 10.0 * (energy.max(1e-12)).log10();
            frame_db.push(db);
        }

        // Noise floor: 10th percentile of frame energies (robust when
        // the recording is mostly speech with brief pauses).
        let mut sorted = frame_db.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = sorted[sorted.len() / 10];
        let threshold = noise_floor + self.config.snr_threshold_db;

        // Above/below hysteresis into boolean frames.
        let active: Vec<bool> = frame_db.iter().map(|&db| db > threshold).collect();

        // Run-length into regions.
        let frame_secs = samples_to_secs(self.config.hop, sample_rate);
        let mut regions: Vec<(usize, usize)> = Vec::new(); // frame indices
        let mut start: Option<usize> = None;
        for (i, &a) in active.iter().enumerate() {
            if a && start.is_none() {
                start = Some(i);
            } else if !a {
                if let Some(s) = start {
                    regions.push((s, i));
                    start = None;
                }
            }
        }
        if let Some(s) = start {
            regions.push((s, active.len()));
        }

        // Bridge short gaps.
        let min_gap_frames = (self.config.min_silence_secs / frame_secs).round() as usize;
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (s, e) in regions {
            match merged.last_mut() {
                Some(last) if s - last.1 < min_gap_frames => last.1 = e,
                _ => merged.push((s, e)),
            }
        }

        // Drop short bursts, convert to time, apply padding.
        let min_len_frames = (self.config.min_speech_secs / frame_secs).round() as usize;
        let pad_frames = (self.config.padding_secs / frame_secs).round() as usize;
        let total_frames = active.len();
        merged
            .into_iter()
            .filter(|(s, e)| e - s >= min_len_frames)
            .map(|(s, e)| {
                let s = s.saturating_sub(pad_frames);
                let e = (e + pad_frames).min(total_frames);
                TimeRange::new(
                    samples_to_secs(s * self.config.hop, sample_rate),
                    samples_to_secs(e * self.config.hop, sample_rate),
                )
            })
            .collect()
    }

    /// Borrows the configuration.
    pub fn config(&self) -> &VadConfig {
        &self.config
    }
}

/// RMS of a slice (re-exported convenience for streaming VAD checks).
pub fn frame_rms(samples: &[f32]) -> f32 {
    rms(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> VoiceActivityDetector {
        VoiceActivityDetector::new(VadConfig::default()).unwrap()
    }

    /// Voiced-like buzz with a given fundamental.
    fn buzz(hz: f32, secs: f64, amp: f32) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let mut v = 0.0f32;
                for k in 1..=5 {
                    v += (std::f32::consts::TAU * hz * k as f32 * t).sin() / k as f32;
                }
                v * amp
            })
            .collect()
    }

    #[test]
    fn silence_produces_no_regions() {
        let d = detector();
        assert!(d.detect(&vec![0.0; 16_000], 16_000).is_empty());
    }

    #[test]
    fn alternating_speech_is_segmented() {
        let d = detector();
        let mut pcm = Vec::new();
        pcm.extend(buzz(180.0, 1.0, 0.4)); // speech
        pcm.extend(vec![0.0; 16_000]); // 1 s silence
        pcm.extend(buzz(180.0, 1.0, 0.4)); // speech
        let regions = d.detect(&pcm, 16_000);
        assert_eq!(regions.len(), 2, "got {regions:?}");
        assert!(regions[0].end <= regions[1].start);
        assert!(regions[0].duration() > 0.8);
    }

    #[test]
    fn short_bursts_are_dropped() {
        let d = detector();
        let mut pcm = vec![0.0; 16_000];
        pcm.extend(buzz(200.0, 0.1, 0.4)); // 100 ms — too short
        pcm.extend(vec![0.0; 16_000]);
        assert!(d.detect(&pcm, 16_000).is_empty());
    }

    #[test]
    fn gaps_are_bridged() {
        let d = VoiceActivityDetector::new(VadConfig {
            min_silence_secs: 0.5,
            ..VadConfig::default()
        })
        .unwrap();
        let mut pcm = Vec::new();
        pcm.extend(buzz(180.0, 1.0, 0.4));
        pcm.extend(vec![0.0; 4_000]); // 250 ms gap — bridged
        pcm.extend(buzz(180.0, 1.0, 0.4));
        let regions = d.detect(&pcm, 16_000);
        assert_eq!(regions.len(), 1, "gap should bridge, got {regions:?}");
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(VoiceActivityDetector::new(VadConfig {
            hop: 0,
            ..Default::default()
        })
        .is_err());
    }
}
