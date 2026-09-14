//! Model weight loading and management.
//!
//! Layout of a model directory:
//!
//! ```text
//! models/en-small/
//! ├── config.json          # AsrModelConfig
//! ├── model.safetensors    # weights (safetensors format)
//! └── tokenizer.json       # BpeTokenizer serialization
//! ```
//!
//! [`write_demo_model`] exports a tiny, randomly initialized model that
//! exercises the full pipeline (with untrained weights the transcripts
//! are gibberish — it exists for smoke tests, examples, and CI).

use std::collections::BTreeMap;
use std::path::Path;

use ndarray::{Array1, Array2, Array3};

use tpt_av_voice_utils::safetensors::{load_from_path, save_to_path, Tensor};
use tpt_av_voice_utils::VoiceError;

type TensorMap = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// Reads `model.safetensors` from a model directory.
pub fn load_model_weights(dir: impl AsRef<Path>) -> Result<BTreeMap<String, Tensor>, VoiceError> {
    load_from_path(dir.as_ref().join("model.safetensors"))
}

/// Writes weights as `model.safetensors` into a model directory.
pub fn write_model_weights(dir: impl AsRef<Path>, tensors: &TensorMap) -> Result<(), VoiceError> {
    save_to_path(dir.as_ref().join("model.safetensors"), tensors)
}

fn put2(tensors: &mut TensorMap, name: String, arr: &Array2<f32>) {
    tensors.insert(name, (arr.shape().to_vec(), arr.iter().copied().collect()));
}

fn put3(tensors: &mut TensorMap, name: String, arr: &Array3<f32>) {
    tensors.insert(name, (arr.shape().to_vec(), arr.iter().copied().collect()));
}

fn put1(tensors: &mut TensorMap, name: String, arr: &Array1<f32>) {
    tensors.insert(name, (vec![arr.len()], arr.iter().copied().collect()));
}

/// Exports a complete demo model (random weights + trained tokenizer)
/// usable by examples and smoke tests. `corpus` trains the tokenizer so
/// the vocabulary is realistic; the acoustic weights themselves are
/// untrained by construction.
pub fn write_demo_model(
    dir: impl AsRef<Path>,
    corpus: &str,
    seed: u64,
) -> Result<crate::model::AsrModelConfig, VoiceError> {
    use crate::model::{AsrModel, ModelSize};
    use crate::tokenizer::{BpeTokenizer, Tokenizer};

    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;

    let tokenizer = BpeTokenizer::train(corpus, 300)?;
    let config = ModelSize::Tiny.config(tokenizer.vocab_size());
    let model = AsrModel::random_init(config.clone(), seed);
    let mut tensors: TensorMap = BTreeMap::new();

    // --- encoder ---
    put3(&mut tensors, "encoder.conv1.weight".into(), &model.encoder.conv1.weight);
    put1(&mut tensors, "encoder.conv1.bias".into(), &model.encoder.conv1.bias);
    put3(&mut tensors, "encoder.conv2.weight".into(), &model.encoder.conv2.weight);
    put1(&mut tensors, "encoder.conv2.bias".into(), &model.encoder.conv2.bias);
    put2(&mut tensors, "encoder.positional_embedding".into(), &model.encoder.positional);
    for (i, b) in model.encoder.blocks.iter().enumerate() {
        let layer = |suffix: &str| format!("encoder.blocks.{i}.{suffix}");
        put2(&mut tensors, layer("attn.query.weight"), &b.attn.query.weight);
        put2(&mut tensors, layer("attn.key.weight"), &b.attn.key.weight);
        put2(&mut tensors, layer("attn.value.weight"), &b.attn.value.weight);
        put2(&mut tensors, layer("attn.out.weight"), &b.attn.out.weight);
        put2(&mut tensors, layer("mlp.0.weight"), &b.fc1.weight);
        put2(&mut tensors, layer("mlp.2.weight"), &b.fc2.weight);
        put1(&mut tensors, layer("attn.query.bias"), &b.attn.query.bias);
        put1(&mut tensors, layer("attn.key.bias"), &b.attn.key.bias);
        put1(&mut tensors, layer("attn.value.bias"), &b.attn.value.bias);
        put1(&mut tensors, layer("attn.out.bias"), &b.attn.out.bias);
        put1(&mut tensors, layer("mlp.0.bias"), &b.fc1.bias);
        put1(&mut tensors, layer("mlp.2.bias"), &b.fc2.bias);
        put1(&mut tensors, layer("attn_ln.weight"), &b.attn_ln.scale);
        put1(&mut tensors, layer("attn_ln.bias"), &b.attn_ln.bias);
        put1(&mut tensors, layer("mlp_ln.weight"), &b.mlp_ln.scale);
        put1(&mut tensors, layer("mlp_ln.bias"), &b.mlp_ln.bias);
    }
    put1(&mut tensors, "encoder.ln_post.weight".into(), &model.encoder.ln_post.scale);
    put1(&mut tensors, "encoder.ln_post.bias".into(), &model.encoder.ln_post.bias);

    // --- decoder ---
    put2(&mut tensors, "decoder.token_embedding.weight".into(), &model.decoder.token_embedding);
    put2(&mut tensors, "decoder.positional_embedding".into(), &model.decoder.positional);
    for (i, b) in model.decoder.blocks.iter().enumerate() {
        let layer = |suffix: &str| format!("decoder.blocks.{i}.{suffix}");
        put2(&mut tensors, layer("attn.query.weight"), &b.attn.query.weight);
        put2(&mut tensors, layer("attn.key.weight"), &b.attn.key.weight);
        put2(&mut tensors, layer("attn.value.weight"), &b.attn.value.weight);
        put2(&mut tensors, layer("attn.out.weight"), &b.attn.out.weight);
        put2(&mut tensors, layer("cross_attn.query.weight"), &b.cross_attn.query.weight);
        put2(&mut tensors, layer("cross_attn.key.weight"), &b.cross_attn.key.weight);
        put2(&mut tensors, layer("cross_attn.value.weight"), &b.cross_attn.value.weight);
        put2(&mut tensors, layer("cross_attn.out.weight"), &b.cross_attn.out.weight);
        put2(&mut tensors, layer("mlp.0.weight"), &b.fc1.weight);
        put2(&mut tensors, layer("mlp.2.weight"), &b.fc2.weight);
        put1(&mut tensors, layer("attn.query.bias"), &b.attn.query.bias);
        put1(&mut tensors, layer("attn.key.bias"), &b.attn.key.bias);
        put1(&mut tensors, layer("attn.value.bias"), &b.attn.value.bias);
        put1(&mut tensors, layer("attn.out.bias"), &b.attn.out.bias);
        put1(&mut tensors, layer("cross_attn.query.bias"), &b.cross_attn.query.bias);
        put1(&mut tensors, layer("cross_attn.key.bias"), &b.cross_attn.key.bias);
        put1(&mut tensors, layer("cross_attn.value.bias"), &b.cross_attn.value.bias);
        put1(&mut tensors, layer("cross_attn.out.bias"), &b.cross_attn.out.bias);
        put1(&mut tensors, layer("mlp.0.bias"), &b.fc1.bias);
        put1(&mut tensors, layer("mlp.2.bias"), &b.fc2.bias);
        put1(&mut tensors, layer("attn_ln.weight"), &b.attn_ln.scale);
        put1(&mut tensors, layer("attn_ln.bias"), &b.attn_ln.bias);
        put1(&mut tensors, layer("cross_attn_ln.weight"), &b.cross_ln.scale);
        put1(&mut tensors, layer("cross_attn_ln.bias"), &b.cross_ln.bias);
        put1(&mut tensors, layer("mlp_ln.weight"), &b.mlp_ln.scale);
        put1(&mut tensors, layer("mlp_ln.bias"), &b.mlp_ln.bias);
    }
    put1(&mut tensors, "decoder.ln.weight".into(), &model.decoder.ln.scale);
    put1(&mut tensors, "decoder.ln.bias".into(), &model.decoder.ln.bias);
    put2(&mut tensors, "decoder.lm_head.weight".into(), &model.decoder.head.weight);
    put1(&mut tensors, "decoder.lm_head.bias".into(), &model.decoder.head.bias);

    write_model_weights(dir, &tensors)?;
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap())?;
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_json())?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AsrModelConfig;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn demo_model_round_trips() {
        let dir = std::env::temp_dir().join(format!("tpt-voice-demo-{}", std::process::id()));
        let config = write_demo_model(&dir, "hello world hello world", 3).unwrap();
        assert!(config.n_vocab >= 258 && config.n_vocab <= 300);

        let tensors = load_model_weights(&dir).unwrap();
        let expected = crate::model::AsrModel::random_init(config.clone(), 3);
        let model = crate::model::AsrModel::from_tensors(&tensors, config.clone()).unwrap();
        assert_eq!(model.encoder.conv1.weight, expected.encoder.conv1.weight);
        assert_eq!(model.decoder.head.weight, expected.decoder.head.weight);

        let tok_json = std::fs::read_to_string(dir.join("tokenizer.json")).unwrap();
        let tok = crate::tokenizer::BpeTokenizer::from_json(&tok_json).unwrap();
        assert_eq!(tok.decode(&tok.encode("world")), "world");

        let cfg_json = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let parsed: AsrModelConfig = serde_json::from_str(&cfg_json).unwrap();
        assert_eq!(parsed, config);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_weights_error() {
        assert!(load_model_weights("/nonexistent/model/dir").is_err());
    }
}
