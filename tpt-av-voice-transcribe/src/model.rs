//! Transformer encoder-decoder ASR architecture (Whisper-style), in pure
//! Rust on top of `tpt-av-voice-utils::nn`.
//!
//! * Encoder: two strided conv1d front-ends → learned positional
//!   embedding → residual attention blocks → final layer norm.
//! * Decoder: token embedding + positional embedding → causal
//!   self-attention blocks with cross-attention to the encoder output →
//!   final layer norm → vocabulary projection (tied to no weights; the
//!   final projection is a separate linear of the last layer norm).
//!
//! Weight tensor names follow the `encoder.*` / `decoder.*` convention
//! so checkpoints can be converted from Whisper-style exports.

use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::nn::{gelu, matmul, softmax_rows, Conv1d, LayerNorm, Linear};
use tpt_av_voice_utils::VoiceError;

/// Named model size presets (spec Phase 5: "larger, more accurate models"
/// are a config choice, not a code change).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl ModelSize {
    /// Canonical Whisper dimension table per size class.
    pub fn config(&self, n_vocab: usize) -> AsrModelConfig {
        let (state, layers, heads) = match self {
            ModelSize::Tiny => (384, 4, 4),
            ModelSize::Base => (512, 6, 8),
            ModelSize::Small => (768, 12, 12),
            ModelSize::Medium => (1024, 16, 16),
            ModelSize::Large => (1280, 20, 20),
        };
        AsrModelConfig {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: state,
            n_audio_head: heads,
            n_audio_layer: layers,
            n_vocab,
            n_text_ctx: 448,
            n_text_state: state,
            n_text_head: heads,
            n_text_layer: layers,
        }
    }
}

/// Full architecture hyper-parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrModelConfig {
    /// Mel channels into the encoder.
    pub n_mels: usize,
    /// Encoder context length (frames).
    pub n_audio_ctx: usize,
    /// Encoder model width.
    pub n_audio_state: usize,
    /// Encoder attention heads.
    pub n_audio_head: usize,
    /// Encoder block count.
    pub n_audio_layer: usize,
    /// Vocabulary size (token count).
    pub n_vocab: usize,
    /// Decoder context length (tokens).
    pub n_text_ctx: usize,
    /// Decoder model width.
    pub n_text_state: usize,
    /// Decoder attention heads.
    pub n_text_head: usize,
    /// Decoder block count.
    pub n_text_layer: usize,
}

impl Default for AsrModelConfig {
    fn default() -> Self {
        ModelSize::Tiny.config(500)
    }
}

impl AsrModelConfig {
    /// Sanity-checks head/divisibility invariants.
    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.n_audio_state == 0
            || self.n_audio_state % self.n_audio_head != 0
            || self.n_text_state == 0
            || self.n_text_state % self.n_text_head != 0
        {
            return Err(VoiceError::config(
                "model width must be a non-zero multiple of heads",
            ));
        }
        if self.n_vocab == 0 {
            return Err(VoiceError::config("n_vocab must be > 0"));
        }
        Ok(())
    }
}

/// Multi-head attention over `(T, d)` matrices.
#[derive(Debug, Clone)]
pub(crate) struct MultiHeadAttention {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) out: Linear,
    n_heads: usize,
}

impl MultiHeadAttention {
    fn from_tensors(
        tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
        prefix: &str,
        _d: usize,
        n_heads: usize,
    ) -> Result<Self, VoiceError> {
        let get = |name: &str| -> Result<Array2<f32>, VoiceError> {
            let t = tensors
                .get(&format!("{prefix}.{name}.weight"))
                .ok_or_else(|| {
                    VoiceError::model(format!("missing tensor {prefix}.{name}.weight"))
                })?;
            expect2(t)
        };
        let get_bias = |name: &str| -> Result<Array1<f32>, VoiceError> {
            let t = tensors
                .get(&format!("{prefix}.{name}.bias"))
                .ok_or_else(|| VoiceError::model(format!("missing tensor {prefix}.{name}.bias")))?;
            if t.data.len() != _d {
                return Err(VoiceError::model(format!(
                    "{prefix}.{name}.bias has {} elements, expected {_d}",
                    t.data.len()
                )));
            }
            Ok(Array1::from(t.data.clone()))
        };
        let get_d = |name: &str| -> Result<Array2<f32>, VoiceError> {
            let w = get(name)?;
            if w.dim() != (_d, _d) {
                return Err(VoiceError::model(format!(
                    "{prefix}.{name}.weight has shape {:?}, expected ({_d}, {_d})",
                    w.dim()
                )));
            }
            Ok(w)
        };
        Ok(MultiHeadAttention {
            query: Linear::new(get_d("query")?, get_bias("query")?),
            key: Linear::new(get_d("key")?, get_bias("key")?),
            value: Linear::new(get_d("value")?, get_bias("value")?),
            out: Linear::new(get_d("out")?, get_bias("out")?),
            n_heads,
        })
    }

    /// `x` is the query source `(Tq, d)`; `kv` the key/value source
    /// `(Tk, d)` (self-attention when `None`). `mask` holds additive
    /// scores `(Tq, Tk)` (use `f32::NEG_INFINITY` for blocked cells).
    fn forward(
        &self,
        x: &Array2<f32>,
        kv: Option<&Array2<f32>>,
        mask: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        let kv = kv.unwrap_or(x);
        let d = x.ncols();
        let head_dim = d / self.n_heads;
        let q = self.query.forward(x);
        let k = self.key.forward(kv);
        let v = self.value.forward(kv);

        let mut context = Array2::<f32>::zeros((x.nrows(), d));
        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..self.n_heads {
            let q_h = q.slice(s![.., h * head_dim..(h + 1) * head_dim]);
            let k_h = k.slice(s![.., h * head_dim..(h + 1) * head_dim]);
            let v_h = v.slice(s![.., h * head_dim..(h + 1) * head_dim]);
            // scores (Tq, Tk)
            let mut scores = matmul(q_h, k_h.t().view());
            scores.mapv_inplace(|v| v * scale);
            if let Some(m) = mask {
                let m = m.slice(s![.., ..scores.ncols()]);
                scores = scores + m;
            }
            let probs = softmax_rows(&scores);
            let ctx = matmul(probs.view(), v_h);
            context
                .slice_mut(s![.., h * head_dim..(h + 1) * head_dim])
                .assign(&ctx);
        }
        self.out.forward(&context)
    }
}

/// One pre-norm transformer block (encoder flavor: self-attention + MLP).
#[derive(Debug, Clone)]
pub(crate) struct EncoderBlock {
    pub(crate) attn_ln: LayerNorm,
    pub(crate) attn: MultiHeadAttention,
    pub(crate) mlp_ln: LayerNorm,
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
}

impl EncoderBlock {
    fn forward(&self, x: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let h = x.to_owned() + self.attn.forward(&self.attn_ln.forward(x), None, mask);
        let m = self.mlp_ln.forward(&h);
        let m = self.fc2.forward(&gelu(&self.fc1.forward(&m)));
        h + m
    }
}

/// Transformer encoder: conv front-end + self-attention stack.
#[derive(Debug, Clone)]
pub struct Encoder {
    pub config: AsrModelConfig,
    pub(crate) conv1: Conv1d,
    pub(crate) conv2: Conv1d,
    pub(crate) positional: Array2<f32>,
    pub(crate) blocks: Vec<EncoderBlock>,
    pub(crate) ln_post: LayerNorm,
}

impl Encoder {
    fn from_tensors(
        tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
        config: &AsrModelConfig,
    ) -> Result<Self, VoiceError> {
        let conv1 = load_conv1d(tensors, "encoder.conv1")?;
        let conv2 = load_conv1d(tensors, "encoder.conv2")?;
        let d = config.n_audio_state;
        if conv1.weight.dim() != (d, config.n_mels, 3) {
            return Err(VoiceError::model(format!(
                "encoder.conv1.weight must be ({d}, {}, 3)",
                config.n_mels
            )));
        }
        if conv2.weight.dim() != (d, d, 3) {
            return Err(VoiceError::model(format!(
                "encoder.conv2.weight must be ({d}, {d}, 3)"
            )));
        }
        let pos = tensors
            .get("encoder.positional_embedding")
            .ok_or_else(|| VoiceError::model("missing encoder.positional_embedding"))?;
        let positional = expect2(pos)?;
        if positional.dim() != (config.n_audio_ctx, d) {
            return Err(VoiceError::model(format!(
                "encoder.positional_embedding must be ({}, {d}), got {:?}",
                config.n_audio_ctx,
                positional.dim()
            )));
        }
        let mut blocks = Vec::with_capacity(config.n_audio_layer);
        for i in 0..config.n_audio_layer {
            let prefix = format!("encoder.blocks.{i}");
            blocks.push(EncoderBlock {
                attn_ln: load_ln(tensors, &format!("{prefix}.attn_ln"), config.n_audio_state)?,
                attn: MultiHeadAttention::from_tensors(
                    tensors,
                    &format!("{prefix}.attn"),
                    config.n_audio_state,
                    config.n_audio_head,
                )?,
                mlp_ln: load_ln(tensors, &format!("{prefix}.mlp_ln"), config.n_audio_state)?,
                fc1: load_linear(
                    tensors,
                    &format!("{prefix}.mlp.0"),
                    config.n_audio_state * 4,
                )?,
                fc2: load_linear(tensors, &format!("{prefix}.mlp.2"), config.n_audio_state)?,
            });
        }
        Ok(Encoder {
            config: config.clone(),
            conv1,
            conv2,
            positional,
            blocks,
            ln_post: load_ln(tensors, "encoder.ln_post", config.n_audio_state)?,
        })
    }

    /// `mel (n_mels, T) → audio embeddings (T/2, d)`.
    pub fn forward(&self, mel: &Array2<f32>) -> Array2<f32> {
        let mut x = self.conv2.forward(&gelu(&self.conv1.forward(mel)));
        let t = x.ncols().min(self.positional.nrows());
        x = x.slice(s![.., ..t]).to_owned() + self.positional.slice(s![..t, ..]).t();
        // x is (d, T) here; transpose to token-major.
        let x = x.t().to_owned();
        let mut h = x;
        for block in &self.blocks {
            // The encoder is bidirectional: no attention mask.
            h = block.forward(&h, None);
        }
        self.ln_post.forward(&h)
    }
}

/// One decoder block: causal self-attention + cross-attention + MLP.
#[derive(Debug, Clone)]
pub(crate) struct DecoderBlock {
    pub(crate) attn_ln: LayerNorm,
    pub(crate) attn: MultiHeadAttention,
    pub(crate) cross_ln: LayerNorm,
    pub(crate) cross_attn: MultiHeadAttention,
    pub(crate) mlp_ln: LayerNorm,
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
}

impl DecoderBlock {
    fn forward(&self, x: &Array2<f32>, audio: &Array2<f32>) -> Array2<f32> {
        let causal = causal_mask(x.nrows());
        let h = x.to_owned()
            + self
                .attn
                .forward(&self.attn_ln.forward(x), None, Some(&causal));
        let c = self.cross_ln.forward(&h);
        let h = h + self.cross_attn.forward(&c, Some(audio), None);
        let m = self.mlp_ln.forward(&h);
        let m = self.fc2.forward(&gelu(&self.fc1.forward(&m)));
        h + m
    }
}

/// Transformer decoder: token embeddings + causal blocks + LM head.
#[derive(Debug, Clone)]
pub struct Decoder {
    pub config: AsrModelConfig,
    pub(crate) token_embedding: Array2<f32>,
    pub(crate) positional: Array2<f32>,
    pub(crate) blocks: Vec<DecoderBlock>,
    pub(crate) ln: LayerNorm,
    pub(crate) head: Linear,
}

impl Decoder {
    fn from_tensors(
        tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
        config: &AsrModelConfig,
    ) -> Result<Self, VoiceError> {
        let td = config.n_text_state;
        let token_embedding = expect2(
            tensors
                .get("decoder.token_embedding.weight")
                .ok_or_else(|| VoiceError::model("missing decoder.token_embedding.weight"))?,
        )?;
        if token_embedding.dim() != (config.n_vocab, td) {
            return Err(VoiceError::model(format!(
                "decoder.token_embedding.weight must be ({}, {td}), got {:?}",
                config.n_vocab,
                token_embedding.dim()
            )));
        }
        let positional = expect2(
            tensors
                .get("decoder.positional_embedding")
                .ok_or_else(|| VoiceError::model("missing decoder.positional_embedding"))?,
        )?;
        if positional.dim() != (config.n_text_ctx, td) {
            return Err(VoiceError::model(format!(
                "decoder.positional_embedding must be ({}, {td}), got {:?}",
                config.n_text_ctx,
                positional.dim()
            )));
        }
        let mut blocks = Vec::with_capacity(config.n_text_layer);
        for i in 0..config.n_text_layer {
            let prefix = format!("decoder.blocks.{i}");
            blocks.push(DecoderBlock {
                attn_ln: load_ln(tensors, &format!("{prefix}.attn_ln"), config.n_text_state)?,
                attn: MultiHeadAttention::from_tensors(
                    tensors,
                    &format!("{prefix}.attn"),
                    config.n_text_state,
                    config.n_text_head,
                )?,
                cross_ln: load_ln(
                    tensors,
                    &format!("{prefix}.cross_attn_ln"),
                    config.n_text_state,
                )?,
                cross_attn: MultiHeadAttention::from_tensors(
                    tensors,
                    &format!("{prefix}.cross_attn"),
                    config.n_text_state,
                    config.n_text_head,
                )?,
                mlp_ln: load_ln(tensors, &format!("{prefix}.mlp_ln"), config.n_text_state)?,
                fc1: load_linear(tensors, &format!("{prefix}.mlp.0"), config.n_text_state * 4)?,
                fc2: load_linear(tensors, &format!("{prefix}.mlp.2"), config.n_text_state)?,
            });
        }
        Ok(Decoder {
            config: config.clone(),
            token_embedding,
            positional,
            blocks,
            ln: load_ln(tensors, "decoder.ln", config.n_text_state)?,
            head: load_linear(tensors, "decoder.lm_head", config.n_vocab)?,
        })
    }

    /// Embeds tokens: `(L,) → (L, d)`.
    fn embed(&self, tokens: &[usize]) -> Array2<f32> {
        let d = self.token_embedding.ncols();
        let l = tokens.len();
        let mut x = Array2::<f32>::zeros((l, d));
        for (i, &tok) in tokens.iter().enumerate() {
            let row = self
                .token_embedding
                .row(tok.min(self.token_embedding.nrows() - 1));
            x.row_mut(i).assign(&row);
        }
        let p = self.positional.nrows().min(l);
        // Add positional encodings (row-wise).
        for i in 0..p {
            let mut row = x.row_mut(i);
            let pos_row = self.positional.row(i);
            for j in 0..d {
                row[j] += pos_row[j];
            }
        }
        x
    }

    /// Logits for every prompt position: `(L, n_vocab)`.
    pub fn forward(&self, tokens: &[usize], audio: &Array2<f32>) -> Array2<f32> {
        let l = tokens.len().min(self.positional.nrows());
        let tokens = &tokens[..l];
        let mut h = self.embed(tokens);
        for block in &self.blocks {
            h = block.forward(&h, audio);
        }
        self.head.forward(&self.ln.forward(&h))
    }
}

/// The complete ASR model.
#[derive(Debug, Clone)]
pub struct AsrModel {
    pub config: AsrModelConfig,
    pub encoder: Encoder,
    pub decoder: Decoder,
}

impl AsrModel {
    /// Loads a model from decoded safetensors tensors.
    pub fn from_tensors(
        tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
        config: AsrModelConfig,
    ) -> Result<Self, VoiceError> {
        config.validate()?;
        let encoder = Encoder::from_tensors(tensors, &config)?;
        let decoder = Decoder::from_tensors(tensors, &config)?;
        Ok(AsrModel {
            config,
            encoder,
            decoder,
        })
    }

    /// Builds a model with deterministic (untrained) weights — used for
    /// pipeline tests and demo-model export. Never produces meaningful
    /// transcripts.
    pub fn random_init(config: AsrModelConfig, seed: u64) -> Self {
        use tpt_av_voice_utils::nn::Rng;

        fn rand2(rng: &mut Rng, rows: usize, cols: usize) -> Array2<f32> {
            Array2::from_shape_fn((rows, cols), |_| rng.normal() * 0.02)
        }
        fn conv(rng: &mut Rng, in_c: usize, out_c: usize, stride: usize) -> Conv1d {
            let mut c = Conv1d::with_init(rng, in_c, out_c, 3);
            c.stride = stride;
            c
        }
        fn mha(rng: &mut Rng, d: usize, heads: usize) -> MultiHeadAttention {
            MultiHeadAttention {
                query: Linear::new(rand2(rng, d, d), Array1::zeros(d)),
                key: Linear::new(rand2(rng, d, d), Array1::zeros(d)),
                value: Linear::new(rand2(rng, d, d), Array1::zeros(d)),
                out: Linear::new(rand2(rng, d, d), Array1::zeros(d)),
                n_heads: heads,
            }
        }
        fn ln(d: usize) -> LayerNorm {
            LayerNorm {
                scale: Array1::ones(d),
                bias: Array1::zeros(d),
                eps: 1e-5,
            }
        }
        fn lin(rng: &mut Rng, rows: usize, cols: usize) -> Linear {
            Linear::new(rand2(rng, rows, cols), Array1::zeros(rows))
        }

        let mut rng = Rng::new(seed);
        let d = config.n_audio_state;
        let td = config.n_text_state;

        let encoder = Encoder {
            config: config.clone(),
            conv1: conv(&mut rng, config.n_mels, d, 1),
            conv2: conv(&mut rng, d, d, 2),
            positional: rand2(&mut rng, config.n_audio_ctx, d),
            blocks: (0..config.n_audio_layer)
                .map(|_| EncoderBlock {
                    attn_ln: ln(d),
                    attn: mha(&mut rng, d, config.n_audio_head),
                    mlp_ln: ln(d),
                    fc1: lin(&mut rng, d * 4, d),
                    fc2: lin(&mut rng, d, d * 4),
                })
                .collect(),
            ln_post: ln(d),
        };
        let decoder = Decoder {
            config: config.clone(),
            token_embedding: rand2(&mut rng, config.n_vocab, td),
            positional: rand2(&mut rng, config.n_text_ctx, td),
            blocks: (0..config.n_text_layer)
                .map(|_| DecoderBlock {
                    attn_ln: ln(td),
                    attn: mha(&mut rng, td, config.n_text_head),
                    cross_ln: ln(td),
                    cross_attn: mha(&mut rng, td, config.n_text_head),
                    mlp_ln: ln(td),
                    fc1: lin(&mut rng, td * 4, td),
                    fc2: lin(&mut rng, td, td * 4),
                })
                .collect(),
            ln: ln(td),
            head: lin(&mut rng, config.n_vocab, td),
        };
        AsrModel {
            config,
            encoder,
            decoder,
        }
    }

    /// Audio `(n_mels, T)` → encoder embeddings `(T/2, d)`.
    pub fn encode(&self, mel: &Array2<f32>) -> Array2<f32> {
        self.encoder.forward(mel)
    }

    /// Token logits `(L, n_vocab)` for a prompt against encoded audio.
    pub fn decode_logits(&self, tokens: &[usize], audio: &Array2<f32>) -> Array2<f32> {
        self.decoder.forward(tokens, audio)
    }

    /// Parameter count (for logging/size reporting).
    pub fn num_parameters(&self) -> usize {
        let count_mha = |m: &MultiHeadAttention| {
            m.query.weight.len() + m.key.weight.len() + m.value.weight.len() + m.out.weight.len()
        };
        let mut n = 0;
        n += self.encoder.conv1.weight.len() + self.encoder.conv2.weight.len();
        n += self.encoder.positional.len();
        for b in &self.encoder.blocks {
            n += count_mha(&b.attn) + b.fc1.weight.len() + b.fc2.weight.len();
        }
        n += self.decoder.token_embedding.len() + self.decoder.positional.len();
        for b in &self.decoder.blocks {
            n += count_mha(&b.attn)
                + count_mha(&b.cross_attn)
                + b.fc1.weight.len()
                + b.fc2.weight.len();
        }
        n += self.decoder.head.weight.len();
        n
    }
}

// ---------------------------------------------------------------------------
// tensor loading helpers
// ---------------------------------------------------------------------------

fn expect2(t: &tpt_av_voice_utils::safetensors::Tensor) -> Result<Array2<f32>, VoiceError> {
    if t.shape.len() != 2 {
        return Err(VoiceError::model(format!(
            "expected 2-D tensor, got shape {:?}",
            t.shape
        )));
    }
    Array2::from_shape_vec([t.shape[0], t.shape[1]], t.data.clone())
        .map_err(|e| VoiceError::model(format!("tensor shape mismatch: {e}")))
}

fn load_linear(
    tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
    prefix: &str,
    out_features: usize,
) -> Result<Linear, VoiceError> {
    let t = tensors
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| VoiceError::model(format!("missing tensor {prefix}.weight")))?;
    let weight = expect2(t)?;
    if weight.nrows() != out_features {
        return Err(VoiceError::model(format!(
            "{prefix}.weight has {} rows, expected {out_features}",
            weight.nrows()
        )));
    }
    let bias = match tensors.get(&format!("{prefix}.bias")) {
        Some(t) => {
            if t.data.len() != out_features {
                return Err(VoiceError::model(format!(
                    "{prefix}.bias has {} elements, expected {out_features}",
                    t.data.len()
                )));
            }
            Array1::from(t.data.clone())
        }
        None => Array1::zeros(out_features),
    };
    Ok(Linear::new(weight, bias))
}

fn load_ln(
    tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
    prefix: &str,
    dim: usize,
) -> Result<LayerNorm, VoiceError> {
    let get = |suffix: &str| -> Result<Array1<f32>, VoiceError> {
        let t = tensors
            .get(&format!("{prefix}.{suffix}"))
            .ok_or_else(|| VoiceError::model(format!("missing tensor {prefix}.{suffix}")))?;
        if t.data.len() != dim {
            return Err(VoiceError::model(format!(
                "{prefix}.{suffix} has {} elements, expected {dim}",
                t.data.len()
            )));
        }
        Ok(Array1::from(t.data.clone()))
    };
    Ok(LayerNorm::new(get("weight")?, get("bias")?))
}

fn load_conv1d(
    tensors: &std::collections::BTreeMap<String, tpt_av_voice_utils::safetensors::Tensor>,
    prefix: &str,
) -> Result<Conv1d, VoiceError> {
    let t = tensors
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| VoiceError::model(format!("missing tensor {prefix}.weight")))?;
    if t.shape.len() != 3 {
        return Err(VoiceError::model(format!(
            "expected 3-D tensor, got {:?}",
            t.shape
        )));
    }
    let weight =
        ndarray::Array3::from_shape_vec([t.shape[0], t.shape[1], t.shape[2]], t.data.clone())
            .map_err(|e| VoiceError::model(format!("tensor shape mismatch: {e}")))?;
    let bias = tensors
        .get(&format!("{prefix}.bias"))
        .map(|t| Array1::from(t.data.clone()))
        .unwrap_or_else(|| Array1::zeros(weight.dim().0));
    let mut conv = Conv1d::new(weight, bias);
    // conv2 downsamples by 2 (Whisper convention).
    conv.stride = if prefix.ends_with("conv2") { 2 } else { 1 };
    Ok(conv)
}

/// Additive causal mask `(T, T)`: position i attends to ≤ i.
fn causal_mask(t: usize) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((t, t));
    for i in 0..t {
        for j in (i + 1)..t {
            m[[i, j]] = f32::NEG_INFINITY;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> AsrModel {
        AsrModel::random_init(ModelSize::Tiny.config(64), 0xA5)
    }

    #[test]
    fn encoder_shapes() {
        let model = tiny();
        let mel = Array2::<f32>::zeros((80, 200));
        let audio = model.encode(&mel);
        assert_eq!(audio.dim(), (100, 384));
    }

    #[test]
    fn decoder_shapes_and_determinism() {
        let model = tiny();
        let mel = Array2::<f32>::zeros((80, 40));
        let audio = model.encode(&mel);
        let logits = model.decode_logits(&[1, 2, 3], &audio);
        assert_eq!(logits.dim(), (3, 64));
        let again = model.decode_logits(&[1, 2, 3], &audio);
        assert_eq!(logits, again, "deterministic forward");
    }

    #[test]
    fn causal_mask_blocks_future() {
        let m = causal_mask(3);
        assert_eq!(m[[0, 0]], 0.0);
        assert!(m[[0, 1]] == f32::NEG_INFINITY);
        assert_eq!(m[[2, 1]], 0.0);
    }

    #[test]
    fn config_validation() {
        let mut c = ModelSize::Tiny.config(64);
        assert!(c.validate().is_ok());
        c.n_audio_head = 5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn random_init_is_reproducible() {
        let a = tiny();
        let b = AsrModel::random_init(ModelSize::Tiny.config(64), 0xA5);
        assert_eq!(a.encoder.conv1.weight, b.encoder.conv1.weight);
        assert_ne!(
            a.encoder.conv1.weight,
            AsrModel::random_init(ModelSize::Tiny.config(64), 7)
                .encoder
                .conv1
                .weight
        );
    }
}
