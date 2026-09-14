//! Separation model: a compact U-Net over log-magnitude STFT frames.
//!
//! Input: `(1, F_padded, T)` normalized log magnitude.
//! Output: `(F, T)` soft mask in `(0, 1)` via sigmoid.
//!
//! The encoder path (conv + pool) captures wide frequency context; the
//! decoder path (upsample + conv, with skip connections) restores
//! resolution. Randomly initialized it produces a near-uniform mask;
//! trained weights (loaded via [`SeparationModel::from_tensors`]) make
//! it selective.

use ndarray::{Array2, Array3, Array4, s};
use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::nn::{max_pool2d, upsample_nearest2d, Conv2d, LayerNorm, Rng};
use tpt_av_voice_utils::{safetensors::Tensor, VoiceError};

/// Architecture hyper-parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparationModelConfig {
    /// STFT bins the model sees before padding (n_fft / 2 + 1).
    pub n_bins: usize,
    /// Frequency axis is padded up to this multiple of 2^depth.
    pub freq_multiple: usize,
    /// Downsampling stages (each halves frequency and time).
    pub depth: usize,
    /// Channels of the first stage (doubled per stage).
    pub base_channels: usize,
}

impl Default for SeparationModelConfig {
    fn default() -> Self {
        SeparationModelConfig { n_bins: 257, freq_multiple: 8, depth: 3, base_channels: 8 }
    }
}

impl SeparationModelConfig {
    /// Padded frequency axis length.
    pub fn padded_bins(&self) -> usize {
        self.n_bins.div_ceil(self.freq_multiple) * self.freq_multiple
    }

    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.n_bins == 0 || self.base_channels == 0 || self.depth == 0 {
            return Err(VoiceError::config("n_bins, base_channels and depth must be > 0"));
        }
        if self.freq_multiple < (1usize << self.depth) {
            return Err(VoiceError::config("freq_multiple must be at least 2^depth"));
        }
        Ok(())
    }
}

/// One down- or up-sampling stage: conv → activation (+ optional norm).
#[derive(Debug, Clone)]
struct Stage {
    conv: Conv2d,
    ln: LayerNorm,
}

impl Stage {
    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let c = self.conv.forward(x);
        // Channel-wise LayerNorm over space: normalize each feature map.
        let mut out = c;
        for ch_i in 0..out.dim().0 {
            let mut map = out.slice_mut(s![ch_i, .., ..]);
            let n = map.len() as f32;
            let mean = map.sum() / n;
            let var = map.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let scale = self.ln.scale[ch_i];
            let bias = self.ln.bias[ch_i];
            map.mapv_inplace(|v| (v - mean) / (var + self.ln.eps).sqrt() * scale + bias);
        }
        out
    }
}

/// The separation network.
#[derive(Debug, Clone)]
pub struct SeparationModel {
    pub config: SeparationModelConfig,
    encoders: Vec<Stage>,
    bottleneck: Stage,
    decoders: Vec<Stage>,
    final_conv: Conv2d,
}

impl SeparationModel {
    /// Builds the model from decoded tensors (names produced by
    /// [`SeparationModel::to_tensors`]).
    pub fn from_tensors(tensors: &std::collections::BTreeMap<String, Tensor>, config: SeparationModelConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        let load_conv = |name: &str| -> Result<Conv2d, VoiceError> {
            let w = tensors
                .get(&format!("{name}.weight"))
                .ok_or_else(|| VoiceError::model(format!("missing {name}.weight")))?;
            let b = tensors
                .get(&format!("{name}.bias"))
                .ok_or_else(|| VoiceError::model(format!("missing {name}.bias")))?;
            if w.shape.len() != 4 {
                return Err(VoiceError::model(format!("{name}.weight must be 4-D")));
            }
            Ok(Conv2d::new(
                Array4::from_shape_vec(
                    [w.shape[0], w.shape[1], w.shape[2], w.shape[3]],
                    w.data.clone(),
                )
                .map_err(|e| VoiceError::model(format!("{name}.weight: {e}")))?,
                ndarray::Array1::from(b.data.clone()),
            ))
        };
        let load_ln = |name: &str, ch: usize| -> Result<LayerNorm, VoiceError> {
            let s = tensors
                .get(&format!("{name}.weight"))
                .ok_or_else(|| VoiceError::model(format!("missing {name}.weight")))?;
            let b = tensors
                .get(&format!("{name}.bias"))
                .ok_or_else(|| VoiceError::model(format!("missing {name}.bias")))?;
            if s.data.len() != ch || b.data.len() != ch {
                return Err(VoiceError::model(format!("{name} channel mismatch")));
            }
            Ok(LayerNorm::new(ndarray::Array1::from(s.data.clone()), ndarray::Array1::from(b.data.clone())))
        };

        let mut encoders = Vec::new();
        for i in 0..config.depth {
            let ch_in = if i == 0 { 1 } else { config.base_channels << (i - 1) };
            let ch_out = config.base_channels << i;
            encoders.push(Stage {
                conv: load_conv(&format!("enc{i}.conv"))?,
                ln: load_ln(&format!("enc{i}.ln"), ch_out)?,
            });
            let _ = ch_in;
        }
        let bottleneck_ch = config.base_channels << config.depth;
        let bottleneck = Stage {
            conv: load_conv("bottleneck.conv")?,
            ln: load_ln("bottleneck.ln", bottleneck_ch)?,
        };
        let mut decoders = Vec::new();
        for i in (0..config.depth).rev() {
            let skip_ch = config.base_channels << i;
            let ch_out = config.base_channels << i;
            decoders.push(Stage {
                conv: load_conv(&format!("dec{i}.conv"))?,
                ln: load_ln(&format!("dec{i}.ln"), ch_out)?,
            });
            let _ = skip_ch;
        }
        let final_conv = load_conv("final.conv")?;
        Ok(SeparationModel { config, encoders, bottleneck, decoders, final_conv })
    }

    /// Deterministic random init for tests/demo export.
    pub fn random_init(config: SeparationModelConfig, seed: u64) -> Self {
        config.validate().expect("valid config");
        let mut rng = Rng::new(seed);
        let mut mk_stage = |ch_in: usize, ch_out: usize| Stage {
            conv: Conv2d::with_init(&mut rng, ch_in, ch_out, 3, 3),
            ln: LayerNorm::unit(ch_out),
        };
        let mut encoders = Vec::new();
        for i in 0..config.depth {
            let ch_in = if i == 0 { 1 } else { config.base_channels << (i - 1) };
            encoders.push(mk_stage(ch_in, config.base_channels << i));
        }
        let bottleneck = mk_stage(config.base_channels << (config.depth - 1), config.base_channels << config.depth);
        let mut decoders = Vec::new();
        for i in (0..config.depth).rev() {
            // Upsampled features (2·base<<i) concatenated with the skip
            // (base<<i).
            let ch_in = (config.base_channels << (i + 1)) + (config.base_channels << i);
            decoders.push(mk_stage(ch_in, config.base_channels << i));
        }
        let final_conv = Conv2d::with_init(&mut rng, config.base_channels, 1, 3, 3);
        SeparationModel { config, encoders, bottleneck, decoders, final_conv }
    }

    /// Serializes every parameter for checkpoint export.
    pub fn to_tensors(&self) -> std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        fn put_conv(
            out: &mut std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>,
            name: String,
            c: &Conv2d,
        ) {
            out.insert(
                format!("{name}.weight"),
                (c.weight.shape().to_vec(), c.weight.iter().copied().collect()),
            );
            out.insert(
                format!("{name}.bias"),
                (vec![c.bias.len()], c.bias.iter().copied().collect()),
            );
        }
        fn put_stage(
            out: &mut std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>,
            name: String,
            s: &Stage,
        ) {
            put_conv(out, format!("{name}.conv"), &s.conv);
            out.insert(
                format!("{name}.ln.weight"),
                (vec![s.ln.scale.len()], s.ln.scale.iter().copied().collect()),
            );
            out.insert(
                format!("{name}.ln.bias"),
                (vec![s.ln.bias.len()], s.ln.bias.iter().copied().collect()),
            );
        }
        let mut out = std::collections::BTreeMap::new();
        for (i, e) in self.encoders.iter().enumerate() {
            put_stage(&mut out, format!("enc{i}"), e);
        }
        put_stage(&mut out, "bottleneck".into(), &self.bottleneck);
        for (i, d) in self.decoders.iter().enumerate() {
            put_stage(&mut out, format!("dec{}", self.config.depth - 1 - i), d);
        }
        put_conv(&mut out, "final.conv".into(), &self.final_conv);
        out
    }

    /// `(n_bins, T)` magnitudes → `(n_bins, T)` soft mask.
    pub fn forward(&self, magnitude: &Array2<f32>) -> Array2<f32> {
        let f_pad = self.config.padded_bins();
        let t = magnitude.ncols();
        // Pooling/upsampling only round-trips when both axes divide by
        // 2^depth, so pad time as well and crop at the end.
        let factor = 1usize << self.config.depth;
        let t_pad = t.div_ceil(factor) * factor;
        // Normalize log magnitude into roughly [-1, 1].
        let input = magnitude.mapv(|v| (v.max(1e-10)).ln() / 10.0);
        let mut x = Array3::<f32>::zeros((1, f_pad, t_pad));
        for b in 0..self.config.n_bins.min(f_pad) {
            for ti in 0..t {
                x[[0, b, ti]] = input[[b, ti]];
            }
        }

        // Encoder with skips.
        let mut skips: Vec<Array3<f32>> = Vec::new();
        for enc in &self.encoders {
            let h = gelu3(&enc.forward(&x));
            skips.push(h.clone());
            x = max_pool2d(&h, 2);
        }
        x = gelu3(&self.bottleneck.forward(&x));

        // Decoder with skip concatenation.
        for (i, dec) in self.decoders.iter().enumerate() {
            x = upsample_nearest2d(&x, 2);
            let skip = &skips[skips.len() - 1 - i];
            x = concat_channels(&x, skip);
            x = gelu3(&dec.forward(&x));
        }

        let mut mask = self.final_conv.forward(&x);
        mask.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));

        // Crop back to the real bin count and frame count.
        let mut out = Array2::<f32>::zeros((self.config.n_bins, t));
        for b in 0..self.config.n_bins.min(f_pad) {
            for ti in 0..t {
                out[[b, ti]] = mask[[0, b, ti]];
            }
        }
        out
    }
}

fn gelu3(x: &Array3<f32>) -> Array3<f32> {
    x.mapv(|v| 0.5 * v * (1.0 + (0.797_884_560_8 * (v + 0.044715 * v * v * v)).tanh()))
}

fn concat_channels(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    let (ca, h, w) = a.dim();
    let cb = b.dim().0;
    assert_eq!((h, w), (b.dim().1, b.dim().2), "skip spatial dims must match");
    let mut out = Array3::<f32>::zeros((ca + cb, h, w));
    out.slice_mut(s![..ca, .., ..]).assign(a);
    out.slice_mut(s![ca.., .., ..]).assign(b);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_shape_and_range() {
        let model = SeparationModel::random_init(SeparationModelConfig::default(), 5);
        let mag = Array2::from_shape_fn((257, 40), |(b, t)| ((b + t) as f32 * 0.1).abs() + 0.1);
        let mask = model.forward(&mag);
        assert_eq!(mask.dim(), (257, 40));
        assert!(mask.iter().all(|v| (0.0..=1.0).contains(v)), "sigmoid range");
    }

    #[test]
    fn deterministic_init() {
        let a = SeparationModel::random_init(SeparationModelConfig::default(), 1);
        let b = SeparationModel::random_init(SeparationModelConfig::default(), 1);
        assert_eq!(a.final_conv.weight, b.final_conv.weight);
    }

    #[test]
    fn tensors_round_trip() {
        let model = SeparationModel::random_init(SeparationModelConfig::default(), 9);
        let tensors = model.to_tensors();
        let loaded = SeparationModel::from_tensors(
            &tensors
                .into_iter()
                .map(|(k, (shape, data))| (k, Tensor { shape, data }))
                .collect(),
            SeparationModelConfig::default(),
        )
        .unwrap();
        assert_eq!(model.final_conv.weight, loaded.final_conv.weight);
        assert_eq!(model.encoders[0].conv.weight, loaded.encoders[0].conv.weight);
    }

    #[test]
    fn config_validation() {
        assert!(SeparationModelConfig::default().validate().is_ok());
        assert!(SeparationModelConfig { depth: 5, ..Default::default() }.validate().is_err());
    }
}
