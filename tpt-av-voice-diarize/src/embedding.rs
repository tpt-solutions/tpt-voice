//! Speaker embedding extraction.
//!
//! Two embedders are provided:
//!
//! * [`EmbeddingModel`] — a TDNN-style network over MFCC frames
//!   (conv layers → global mean/std pooling → projection), loaded from
//!   safetensors weights for trained x-vector-style models;
//! * [`ClassicalEmbedding`] — deterministic MFCC mean+std statistics
//!   with no weights at all, so diarization works out of the box.

use tpt_av_voice_utils::dsp::{mfcc, MfccConfig};
use tpt_av_voice_utils::nn::{Conv1d, LayerNorm, Linear, Rng};
use tpt_av_voice_utils::{safetensors::Tensor, VoiceError};

/// Anything that turns an utterance into a fixed-size speaker vector.
pub trait SpeakerEmbedder: Send {
    /// Embeds one utterance (mono PCM).
    fn embed(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, VoiceError>;
    /// Embedding dimensionality.
    fn dim(&self) -> usize;
}

/// L2-normalizes in place; zero vectors stay zero.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine distance `1 − cos(a, b)` in `[0, 2]`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

/// MFCC front-end shared by both embedders.
pub(crate) fn mfcc_frames(samples: &[f32], sample_rate: u32) -> Result<Vec<Vec<f32>>, VoiceError> {
    let config = MfccConfig {
        sample_rate,
        ..MfccConfig::default()
    };
    mfcc(samples, &config)
}

/// TDNN-style embedding network: MFCC → (conv1d → tanh)³ → global
/// mean+std pool → linear → L2 normalize.
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    convs: Vec<Conv1d>,
    norms: Vec<LayerNorm>,
    proj: Linear,
    n_mfcc: usize,
}

impl EmbeddingModel {
    /// Output embedding dimension.
    pub const OUTPUT_DIM: usize = 256;

    /// Loads a trained model from decoded tensors (names from
    /// [`EmbeddingModel::to_tensors`]).
    pub fn from_tensors(
        tensors: &std::collections::BTreeMap<String, Tensor>,
    ) -> Result<Self, VoiceError> {
        let get_conv = |name: &str| -> Result<Conv1d, VoiceError> {
            let w = tensors
                .get(&format!("{name}.weight"))
                .ok_or_else(|| VoiceError::model(format!("missing {name}.weight")))?;
            let b = tensors
                .get(&format!("{name}.bias"))
                .ok_or_else(|| VoiceError::model(format!("missing {name}.bias")))?;
            if w.shape.len() != 3 {
                return Err(VoiceError::model(format!("{name}.weight must be 3-D")));
            }
            Ok(Conv1d::new(
                ndarray::Array3::from_shape_vec(
                    [w.shape[0], w.shape[1], w.shape[2]],
                    w.data.clone(),
                )
                .map_err(|e| VoiceError::model(format!("{name}.weight: {e}")))?,
                ndarray::Array1::from(b.data.clone()),
            ))
        };
        let mut convs = Vec::new();
        let mut norms = Vec::new();
        for i in 0..3 {
            convs.push(get_conv(&format!("tdnn{i}"))?);
            let s = tensors
                .get(&format!("tdnn{i}.ln.weight"))
                .ok_or_else(|| VoiceError::model(format!("missing tdnn{i}.ln.weight")))?;
            let b = tensors
                .get(&format!("tdnn{i}.ln.bias"))
                .ok_or_else(|| VoiceError::model(format!("missing tdnn{i}.ln.bias")))?;
            norms.push(LayerNorm::new(
                ndarray::Array1::from(s.data.clone()),
                ndarray::Array1::from(b.data.clone()),
            ));
        }
        let w = tensors
            .get("proj.weight")
            .ok_or_else(|| VoiceError::model("missing proj.weight"))?;
        let b = tensors
            .get("proj.bias")
            .ok_or_else(|| VoiceError::model("missing proj.bias"))?;
        let proj = Linear::new(
            ndarray::Array2::from_shape_vec([w.shape[0], w.shape[1]], w.data.clone())
                .map_err(|e| VoiceError::model(format!("proj.weight: {e}")))?,
            ndarray::Array1::from(b.data.clone()),
        );
        // Cross-validate chained dimensions so a corrupt checkpoint
        // surfaces as an error, not a panic at inference time.
        let n_mfcc = convs[0].weight.dim().1;
        for (i, conv) in convs.iter().enumerate() {
            if norms[i].scale.len() != conv.weight.dim().0 {
                return Err(VoiceError::model(format!(
                    "tdnn{i} norm/channel mismatch ({} vs {})",
                    norms[i].scale.len(),
                    conv.weight.dim().0
                )));
            }
            if i + 1 < convs.len() && conv.weight.dim().0 != convs[i + 1].weight.dim().1 {
                return Err(VoiceError::model(format!(
                    "tdnn{i} → tdnn{} channel mismatch",
                    i + 1
                )));
            }
        }
        let last_ch = convs.last().expect("three convs").weight.dim().0;
        if proj.weight.dim() != (EmbeddingModel::OUTPUT_DIM, 2 * last_ch) {
            return Err(VoiceError::model(format!(
                "proj.weight must be ({}, {}), got {:?}",
                EmbeddingModel::OUTPUT_DIM,
                2 * last_ch,
                proj.weight.dim()
            )));
        }
        Ok(EmbeddingModel {
            convs,
            norms,
            proj,
            n_mfcc,
        })
    }

    /// Deterministic random init (for tests/demo export).
    pub fn random_init(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let channels = [20, 32, 32];
        let mut convs = Vec::new();
        let mut norms = Vec::new();
        let mut ch_in = 20usize; // MfccConfig::default().n_mfcc
        for ch_out in channels {
            let mut c = Conv1d::with_init(&mut rng, ch_in, ch_out, 5);
            c.stride = 1;
            convs.push(c);
            norms.push(LayerNorm::unit(ch_out));
            ch_in = ch_out;
        }
        // proj: (2*32 pooled stats) → 256
        let proj = Linear::with_init(&mut rng, 2 * 32, Self::OUTPUT_DIM);
        EmbeddingModel {
            convs,
            norms,
            proj,
            n_mfcc: 20,
        }
    }

    /// Serializes parameters for checkpoint export.
    pub fn to_tensors(&self) -> std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let mut out = std::collections::BTreeMap::new();
        for (i, c) in self.convs.iter().enumerate() {
            out.insert(
                format!("tdnn{i}.weight"),
                (
                    c.weight.shape().to_vec(),
                    c.weight.iter().copied().collect(),
                ),
            );
            out.insert(
                format!("tdnn{i}.bias"),
                (vec![c.bias.len()], c.bias.iter().copied().collect()),
            );
            out.insert(
                format!("tdnn{i}.ln.weight"),
                (
                    vec![self.norms[i].scale.len()],
                    self.norms[i].scale.iter().copied().collect(),
                ),
            );
            out.insert(
                format!("tdnn{i}.ln.bias"),
                (
                    vec![self.norms[i].bias.len()],
                    self.norms[i].bias.iter().copied().collect(),
                ),
            );
        }
        out.insert(
            "proj.weight".into(),
            (
                self.proj.weight.shape().to_vec(),
                self.proj.weight.iter().copied().collect(),
            ),
        );
        out.insert(
            "proj.bias".into(),
            (
                vec![self.proj.bias.len()],
                self.proj.bias.iter().copied().collect(),
            ),
        );
        out
    }
}

impl SpeakerEmbedder for EmbeddingModel {
    fn embed(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, VoiceError> {
        let frames = mfcc_frames(samples, sample_rate)?;
        if frames.is_empty() {
            return Err(VoiceError::from(tpt_av_voice_utils::AudioError::Empty));
        }
        let t = frames.len();
        let n_mfcc = frames[0].len().min(self.n_mfcc);
        // (n_mfcc, T)
        let mut x = ndarray::Array2::<f32>::zeros((n_mfcc, t));
        for (ti, frame) in frames.iter().enumerate() {
            for (m, v) in frame.iter().take(n_mfcc).enumerate() {
                x[[m, ti]] = *v;
            }
        }
        // Normalize input features per coefficient.
        for mut row in x.rows_mut() {
            let mean = row.sum() / row.len() as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
            row.mapv_inplace(|v| (v - mean) / (var + 1e-6).sqrt());
        }

        let mut h = x;
        for (conv, ln) in self.convs.iter().zip(&self.norms) {
            let c = conv.forward(&h);
            let mut activated = ln.forward(&c.t().to_owned()).t().to_owned();
            activated.mapv_inplace(|v| v.tanh());
            h = activated;
        }

        // Global mean+std pool over time → (2C,) then project.
        let (c_count, _t) = h.dim();
        let mut stats = ndarray::Array2::<f32>::zeros((1, 2 * c_count));
        for c in 0..c_count {
            let row = h.row(c);
            let mean = row.sum() / row.len() as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
            stats[[0, c]] = mean;
            stats[[0, c_count + c]] = var.sqrt();
        }
        let mut out = self.proj.forward(&stats).row(0).to_vec();
        l2_normalize(&mut out);
        Ok(out)
    }

    fn dim(&self) -> usize {
        Self::OUTPUT_DIM
    }
}

/// Weight-free embedding: cepstral mean+variance statistics plus an
/// autocorrelation pitch (F0) feature — pitch is a strong speaker cue
/// and costs nothing to compute.
#[derive(Debug, Clone, Default)]
pub struct ClassicalEmbedding;

impl ClassicalEmbedding {
    /// Mean + std over 13 MFCC coefficients, plus one F0 feature.
    pub const OUTPUT_DIM: usize = 27;

    /// Autocorrelation pitch estimate in Hz (0.0 when unvoiced or
    /// unreliable). Searches 60–400 Hz over the middle of the clip.
    pub fn estimate_f0(samples: &[f32], sample_rate: u32) -> f32 {
        let window = (0.04 * sample_rate as f32) as usize;
        if samples.len() < window * 2 {
            return 0.0;
        }
        let start = (samples.len() - window) / 2;
        let frame = &samples[start..start + window];
        let min_lag = (sample_rate as f32 / 400.0) as usize;
        let max_lag = (sample_rate as f32 / 60.0) as usize;
        let energy: f32 = frame.iter().map(|s| s * s).sum();
        if energy < 1e-8 {
            return 0.0;
        }
        let mut best_lag = 0usize;
        let mut best_corr = 0.0f32;
        for lag in min_lag..max_lag.min(window) {
            let corr: f32 = frame[..window - lag]
                .iter()
                .zip(&frame[lag..])
                .map(|(a, b)| a * b)
                .sum();
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
        if best_lag == 0 || best_corr / energy < 0.3 {
            0.0
        } else {
            sample_rate as f32 / best_lag as f32
        }
    }
}

impl SpeakerEmbedder for ClassicalEmbedding {
    fn embed(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, VoiceError> {
        let frames = mfcc_frames(samples, sample_rate)?;
        if frames.is_empty() {
            return Err(VoiceError::from(tpt_av_voice_utils::AudioError::Empty));
        }
        let n_mfcc = 13;
        let mut out = vec![0.0f32; 2 * n_mfcc];
        let t = frames.len() as f32;
        for frame in &frames {
            for (m, &v) in frame.iter().take(n_mfcc).enumerate() {
                out[m] += v / t;
            }
        }
        for frame in &frames {
            for (m, &v) in frame.iter().take(n_mfcc).enumerate() {
                out[n_mfcc + m] += (v - out[m]) * (v - out[m]) / t;
            }
        }
        for v in out.iter_mut().skip(n_mfcc) {
            *v = v.sqrt();
        }
        // Pitch feature: log2(F0/100 Hz); 0 for unvoiced.
        let f0 = Self::estimate_f0(samples, sample_rate);
        out.push(if f0 > 0.0 { (f0 / 100.0).log2() } else { 0.0 });
        l2_normalize(&mut out);
        Ok(out)
    }

    fn dim(&self) -> usize {
        Self::OUTPUT_DIM
    }
}

// (matmul_t not needed here; pooling is O(C·T).)

#[cfg(test)]
mod tests {
    use super::*;

    fn buzz(hz: f32, secs: f64) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let mut v = 0.0f32;
                for k in 1..=8 {
                    v += (std::f32::consts::TAU * hz * k as f32 * t).sin() / k as f32;
                }
                v * 0.3
            })
            .collect()
    }

    #[test]
    fn classical_separates_distinct_voices() {
        let mut e = ClassicalEmbedding;
        let a1 = e.embed(&buzz(150.0, 1.0), 16_000).unwrap();
        let a2 = e.embed(&buzz(155.0, 1.2), 16_000).unwrap();
        let b1 = e.embed(&buzz(400.0, 1.0), 16_000).unwrap();
        assert_eq!(a1.len(), ClassicalEmbedding::OUTPUT_DIM);
        assert!(
            cosine_distance(&a1, &a2) < 0.1,
            "same voice similar: {}",
            cosine_distance(&a1, &a2)
        );
        assert!(
            cosine_distance(&a1, &b1) > 0.1,
            "different voices distinct: {}",
            cosine_distance(&a1, &b1)
        );
        assert!(cosine_distance(&a1, &b1) > cosine_distance(&a1, &a2));
    }

    #[test]
    fn embedding_model_separates_and_is_deterministic() {
        let mut m1 = EmbeddingModel::random_init(3);
        let mut m2 = EmbeddingModel::random_init(3);
        let a = m1.embed(&buzz(150.0, 1.0), 16_000).unwrap();
        let a2 = m2.embed(&buzz(150.0, 0.8), 16_000).unwrap();
        let b = m1.embed(&buzz(400.0, 1.0), 16_000).unwrap();
        assert_eq!(a.len(), EmbeddingModel::OUTPUT_DIM);
        assert!(cosine_distance(&a, &b) > cosine_distance(&a, &a2));
        // Unit norm.
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn tensors_round_trip() {
        let m = EmbeddingModel::random_init(9);
        let tensors: std::collections::BTreeMap<String, Tensor> = m
            .to_tensors()
            .into_iter()
            .map(|(k, (shape, data))| (k, Tensor { shape, data }))
            .collect();
        let loaded = EmbeddingModel::from_tensors(&tensors).unwrap();
        assert_eq!(loaded.proj.weight, m.proj.weight);
        assert_eq!(loaded.convs[0].weight, m.convs[0].weight);
    }

    #[test]
    fn empty_input_rejected() {
        let mut e = ClassicalEmbedding;
        assert!(e.embed(&[], 16_000).is_err());
        let mut m = EmbeddingModel::random_init(1);
        assert!(m.embed(&[0.0; 5], 16_000).is_err());
    }
}
