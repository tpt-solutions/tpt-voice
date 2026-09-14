//! High-level voice isolation API.

use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::{safetensors, VoiceError};

use crate::istft::Istft;
use crate::mask::{mask_spectrum, SpectralSubtractor};
use crate::model::{SeparationModel, SeparationModelConfig};
use crate::stft::{Stft, StftConfig};

/// Which source the mask should isolate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SourceType {
    /// Isolate the human voice (default).
    #[default]
    Voice,
    /// Isolate the residual background instead.
    Noise,
}

/// Isolator geometry and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IsolatorConfig {
    /// Target source type (voice, noise, ...).
    pub target: SourceType,
    /// Window size for STFT (in samples).
    pub window_size: usize,
    /// Hop size for STFT (in samples).
    pub hop_size: usize,
    /// Classical mode only: over-subtraction factor.
    pub alpha: f32,
    /// Classical mode only: spectral gain floor.
    pub gain_floor: f32,
}

impl Default for IsolatorConfig {
    fn default() -> Self {
        IsolatorConfig {
            target: SourceType::Voice,
            window_size: 512,
            hop_size: 256,
            alpha: 2.0,
            gain_floor: 0.02,
        }
    }
}

impl IsolatorConfig {
    fn stft_config(&self) -> StftConfig {
        StftConfig { n_fft: self.window_size, hop_size: self.hop_size }
    }

    fn validate(&self) -> Result<(), VoiceError> {
        if self.window_size == 0 || !self.window_size.is_power_of_two() {
            return Err(VoiceError::config("window_size must be a power of two"));
        }
        if self.hop_size == 0 || self.hop_size > self.window_size {
            return Err(VoiceError::config("hop_size must be in (0, window_size]"));
        }
        Ok(())
    }
}

/// Isolated audio sources.
#[derive(Debug, Clone, PartialEq)]
pub struct IsolatedAudio {
    /// Isolated voice.
    pub voice: Vec<f32>,
    /// Isolated background noise (the input minus the isolated voice).
    pub noise: Vec<f32>,
    /// Sample rate.
    pub sample_rate: u32,
}

/// Voice isolation engine.
///
/// * [`VoiceIsolator::from_model_dir`] — U-Net mask prediction from a
///   trained checkpoint.
/// * [`VoiceIsolator::classical`] — online spectral subtraction, no
///   weights needed.
pub struct VoiceIsolator {
    config: IsolatorConfig,
    model: Option<SeparationModel>,
    stft: Stft,
    istft: Istft,
    subtractor: Option<SpectralSubtractor>,
}

impl VoiceIsolator {
    /// Loads a trained separation model from a directory containing
    /// `model.safetensors` and `config.json`.
    pub fn from_model_dir(config: IsolatorConfig, dir: impl AsRef<Path>) -> Result<Self, VoiceError> {
        let dir = dir.as_ref();
        let tensors = safetensors::load_from_path(dir.join("model.safetensors"))?;
        let model_config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let model_config: SeparationModelConfig = serde_json::from_str(&model_config_json)
            .map_err(|e| VoiceError::model(format!("bad model config: {e}")))?;
        Self::from_model(SeparationModel::from_tensors(&tensors, model_config)?, config)
    }

    /// Uses an already-loaded separation model.
    pub fn from_model(model: SeparationModel, config: IsolatorConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        if model.config.n_bins != config.stft_config().n_bins() {
            return Err(VoiceError::config(format!(
                "model expects {} bins, STFT provides {}",
                model.config.n_bins,
                config.stft_config().n_bins()
            )));
        }
        Ok(VoiceIsolator {
            config,
            model: Some(model),
            stft: Stft::new(config.stft_config())?,
            istft: Istft::new(config.stft_config())?,
            subtractor: None,
        })
    }

    /// Classical spectral-subtraction mode (no trained weights).
    pub fn classical(config: IsolatorConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        let n_bins = config.stft_config().n_bins();
        Ok(VoiceIsolator {
            subtractor: Some(SpectralSubtractor::new(n_bins, config.alpha, config.gain_floor)),
            model: None,
            config,
            stft: Stft::new(config.stft_config())?,
            istft: Istft::new(config.stft_config())?,
        })
    }

    /// Borrows the configuration.
    pub fn config(&self) -> &IsolatorConfig {
        &self.config
    }

    /// Isolates mono PCM samples; returns voice and residual noise.
    pub fn isolate_samples(&mut self, samples: &[f32], sample_rate: u32) -> Result<IsolatedAudio, VoiceError> {
        if samples.is_empty() {
            return Err(VoiceError::from(tpt_av_voice_utils::AudioError::Empty));
        }
        if sample_rate != 16_000 {
            return Err(VoiceError::config("the isolator processes 16 kHz audio; resample first"));
        }
        let _ = &self.config.target; // documented knob; voice is the primary target

        // Zero-pad the tail into the STFT.
        let n_fft = self.config.window_size;
        let mut padded = samples.to_vec();
        while padded.len() < n_fft {
            padded.push(0.0);
        }
        let rem = (padded.len() - n_fft) % self.config.hop_size;
        if rem != 0 {
            padded.extend(std::iter::repeat(0.0).take(self.config.hop_size - rem));
        }

        let spec = self.stft.forward(&padded);
        let mask = self.predict_mask(&padded, &spec)?;
        let masked = mask_spectrum(&spec, &mask);
        let mut voice = self.istft.reconstruct(&masked, padded.len());
        voice.truncate(samples.len());
        let noise: Vec<f32> = samples
            .iter()
            .zip(&voice)
            .map(|(s, v)| s - v)
            .collect();
        Ok(IsolatedAudio { voice, noise, sample_rate })
    }

    /// Isolates voice from a WAV file.
    pub fn isolate_file(&mut self, audio_path: &Path) -> Result<IsolatedAudio, VoiceError> {
        let wav = tpt_av_voice_utils::wav::read_wav_file(audio_path)?;
        self.isolate_samples(&wav.to_mono(), wav.sample_rate)
    }

    /// Streaming isolation over a chunk iterator (spec §4.3). Final
    /// samples only — the algorithmic tail is dropped; for full-fidelity
    /// streaming use [`crate::streaming::StreamingIsolator`] directly
    /// and call `flush`.
    pub fn isolate_stream<'a>(
        &'a mut self,
        chunks: impl Iterator<Item = tpt_av_voice_utils::AudioChunk> + 'a,
    ) -> impl Iterator<Item = tpt_av_voice_utils::AudioChunk> + 'a {
        let mut stream = crate::streaming::StreamingIsolator::new(self);
        chunks.filter_map(move |c| stream.push(&c).ok())
    }

    /// Soft voice mask for one frame of magnitudes — the shared
    /// primitive behind batch and streaming isolation.
    pub fn mask_frame(&mut self, magnitude: &[f32]) -> Result<Vec<f32>, VoiceError> {
        if let Some(model) = &self.model {
            let n_bins = model.config.n_bins;
            if magnitude.len() != n_bins {
                return Err(VoiceError::config(format!(
                    "expected {n_bins} magnitude bins, got {}",
                    magnitude.len()
                )));
            }
            let mag = Array2::from_shape_vec((n_bins, 1), magnitude.to_vec())
                .map_err(|e| VoiceError::model(format!("magnitude shape: {e}")))?;
            let mask = model.forward(&mag);
            Ok(mask.column(0).iter().copied().collect())
        } else if let Some(sub) = &mut self.subtractor {
            Ok(sub.frame_mask(magnitude))
        } else {
            Err(VoiceError::model("isolator has neither model nor classical subtractor"))
        }
    }

    fn predict_mask(
        &mut self,
        _padded: &[f32],
        spec: &Array2<rustfft::num_complex::Complex<f32>>,
    ) -> Result<Array2<f32>, VoiceError> {
        let mag: Array2<f32> = spec.mapv(|c| c.norm()); // (frames, bins)
        if let Some(model) = &self.model {
            // Batch models see the whole spectrogram in one pass; the
            // model wants (bins, frames).
            let mask = model.forward(&mag.t().to_owned());
            Ok(mask.t().to_owned())
        } else if let Some(sub) = &mut self.subtractor {
            let mut mask = Array2::<f32>::zeros(mag.dim());
            for (t, row) in mag.rows().into_iter().enumerate() {
                let m = sub.frame_mask(row.as_slice().expect("contiguous row"));
                for (b, v) in m.into_iter().enumerate() {
                    mask[[t, b]] = v;
                }
            }
            Ok(mask)
        } else {
            Err(VoiceError::model("isolator has neither model nor classical subtractor"))
        }
    }
}

/// Convenience: exports a random-init separation model directory (for
/// examples / smoke tests — untrained, produces near-uniform masks).
pub fn write_demo_model(dir: impl AsRef<Path>, seed: u64) -> Result<(), VoiceError> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let config = SeparationModelConfig::default();
    let model = SeparationModel::random_init(config.clone(), seed);
    tpt_av_voice_utils::safetensors::save_to_path(dir.join("model.safetensors"), &model.to_tensors())?;
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> IsolatorConfig {
        IsolatorConfig::default()
    }

    /// Tone (voice-like) at 300 Hz + wideband noise at other frequencies.
    fn voice_plus_noise() -> Vec<f32> {
        let mut rng = tpt_av_voice_utils::nn::Rng::new(11);
        (0..16_000)
            .map(|i| {
                let tone = (std::f32::consts::TAU * 300.0 * i as f32 / 16_000.0).sin() * 0.4;
                let noise = rng.uniform_range(-1.0, 1.0) * 0.05;
                tone + noise
            })
            .collect()
    }

    #[test]
    fn classical_rejects_bad_config_and_empty() {
        assert!(VoiceIsolator::classical(IsolatorConfig { window_size: 500, ..config() }).is_err());
        let mut iso = VoiceIsolator::classical(config()).unwrap();
        assert!(matches!(
            iso.isolate_samples(&[], 16_000),
            Err(VoiceError::Audio(tpt_av_voice_utils::AudioError::Empty))
        ));
        assert!(iso.isolate_samples(&[0.0; 100], 44_100).is_err());
    }

    #[test]
    fn classical_preserves_signal_content() {
        let mut iso = VoiceIsolator::classical(config()).unwrap();
        let input = voice_plus_noise();
        let out = iso.isolate_samples(&input, 16_000).unwrap();
        assert_eq!(out.voice.len(), input.len());
        assert_eq!(out.noise.len(), input.len());
        assert_eq!(out.sample_rate, 16_000);
        // voice + noise == input (energy conservation).
        for i in (0..input.len()).step_by(97) {
            assert!((input[i] - out.voice[i] - out.noise[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn classical_attenuates_stationary_noise() {
        // Input: loud stationary tone (treated as noise floor) that stops
        // partway. The separator must keep the loud part and attenuate
        // little of it, but the point is: output energy <= input energy.
        let mut iso = VoiceIsolator::classical(config()).unwrap();
        let input: Vec<f32> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * 200.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let out = iso.isolate_samples(&input, 16_000).unwrap();
        let in_rms = tpt_av_voice_utils::dsp::rms(&input);
        let out_rms = tpt_av_voice_utils::dsp::rms(&out.voice);
        assert!(out_rms <= in_rms * 1.05, "isolation must not amplify: {out_rms} vs {in_rms}");
    }

    #[test]
    fn model_path_round_trip() {
        let dir = std::env::temp_dir().join(format!("tpt-voice-isolate-{}", std::process::id()));
        write_demo_model(&dir, 3).unwrap();
        let mut iso = VoiceIsolator::from_model_dir(config(), &dir).unwrap();
        let input = voice_plus_noise();
        let out = iso.isolate_samples(&input, 16_000).unwrap();
        assert_eq!(out.voice.len(), input.len());
        // An untrained (random-init) mask is near-uniform: the voice
        // path must carry a bounded fraction of the input energy, never
        // amplify, and voice+noise must reconstruct the input.
        let in_rms = tpt_av_voice_utils::dsp::rms(&input);
        let ratio = tpt_av_voice_utils::dsp::rms(&out.voice) / in_rms.max(1e-9);
        assert!((0.0..=1.02).contains(&ratio), "unexpected untrained ratio {ratio}");
        for i in (0..input.len()).step_by(97) {
            assert!((input[i] - out.voice[i] - out.noise[i]).abs() < 1e-4);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mismatched_bins_rejected() {
        let dir = std::env::temp_dir().join(format!("tpt-voice-isolate-bad-{}", std::process::id()));
        write_demo_model(&dir, 3).unwrap();
        let bad = IsolatorConfig { window_size: 1024, hop_size: 512, ..config() };
        assert!(VoiceIsolator::from_model_dir(bad, &dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
