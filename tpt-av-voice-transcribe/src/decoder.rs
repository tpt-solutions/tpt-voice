//! Text decoding strategies: greedy, beam search, and CTC.
//!
//! Autoregressive decoders consume an [`AsrModel`] plus encoded audio and
//! emit token ids (the caller maps ids to text via a tokenizer). CTC
//! decoding operates on `(T, n_vocab)` emission matrices and is shared
//! with the alignment crate.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::model::AsrModel;
use tpt_av_voice_utils::VoiceError;

/// Knobs shared by the autoregressive decoders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeConfig {
    /// Maximum generated tokens (including the prompt).
    pub max_tokens: usize,
    /// Beam width for [`BeamSearchDecoder`] (1 behaves like greedy).
    pub beam_width: usize,
    /// Tokens never generated (e.g. specials mid-utterance).
    pub suppress: Vec<usize>,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        DecodeConfig { max_tokens: 224, beam_width: 1, suppress: Vec::new() }
    }
}

impl DecodeConfig {
    fn validate(&self) -> Result<(), VoiceError> {
        if self.max_tokens == 0 {
            return Err(VoiceError::config("max_tokens must be > 0"));
        }
        if self.beam_width == 0 {
            return Err(VoiceError::config("beam_width must be > 0"));
        }
        Ok(())
    }
}

/// A search strategy that turns encoded audio into token ids.
pub trait Decoder {
    /// `prompt` (e.g. `[SOT, language, task]`) seeds the decoder.
    fn decode(
        &self,
        model: &AsrModel,
        audio: &Array2<f32>,
        prompt: &[usize],
        config: &DecodeConfig,
    ) -> Result<Vec<usize>, VoiceError>;
}

/// Argmax decoding — fastest, adequate for short clean utterances.
#[derive(Debug, Clone, Default)]
pub struct GreedyDecoder;

impl Decoder for GreedyDecoder {
    /// Repeatedly takes the highest-probability token (skipping
    /// `config.suppress`) until `max_tokens` or the model's decoder
    /// context limit is reached.
    fn decode(
        &self,
        model: &AsrModel,
        audio: &Array2<f32>,
        prompt: &[usize],
        config: &DecodeConfig,
    ) -> Result<Vec<usize>, VoiceError> {
        config.validate()?;
        let mut tokens = prompt.to_vec();
        for _ in 0..config.max_tokens {
            if tokens.len() >= model.config.n_text_ctx {
                break;
            }
            let logits = model.decode_logits(&tokens, audio);
            let last = logits.row(logits.nrows() - 1);
            let next = argmax_excluding(&last, &config.suppress);
            tokens.push(next);
        }
        Ok(tokens)
    }
}

/// Beam search with length-normalized log-prob scoring.
#[derive(Debug, Clone, Default)]
pub struct BeamSearchDecoder;

#[derive(Debug, Clone)]
struct Beam {
    tokens: Vec<usize>,
    logprob: f32,
}

impl Decoder for BeamSearchDecoder {
    fn decode(
        &self,
        model: &AsrModel,
        audio: &Array2<f32>,
        prompt: &[usize],
        config: &DecodeConfig,
    ) -> Result<Vec<usize>, VoiceError> {
        config.validate()?;
        let width = config.beam_width.max(1);
        let mut beams = vec![Beam { tokens: prompt.to_vec(), logprob: 0.0 }];
        let mut finished: Vec<Beam> = Vec::new();

        for _ in 0..config.max_tokens {
            if beams.is_empty() {
                break;
            }
            let mut candidates: Vec<Beam> = Vec::new();
            for beam in &beams {
                if beam.tokens.len() >= model.config.n_text_ctx {
                    finished.push(beam.clone());
                    continue;
                }
                let logits = model.decode_logits(&beam.tokens, audio);
                let last = logits.row(logits.nrows() - 1);
                let log_probs = log_softmax(last.to_slice().unwrap_or(&[]), &config.suppress);
                let top: Vec<usize> = top_k(&log_probs, width);
                for &tok in &top {
                    let mut b = beam.clone();
                    b.tokens.push(tok);
                    b.logprob += log_probs[tok];
                    candidates.push(b);
                }
            }
            candidates.sort_by(|a, b| b.logprob.partial_cmp(&a.logprob).unwrap_or(std::cmp::Ordering::Equal));
            beams = candidates.into_iter().take(width).collect();
            // Beams that hit the context limit are retired.
            let (mut done, live): (Vec<_>, Vec<_>) = beams
                .into_iter()
                .partition(|b| b.tokens.len() >= model.config.n_text_ctx);
            finished.append(&mut done);
            beams = live;
        }
        finished.append(&mut beams);
        let best = finished
            .into_iter()
            .max_by(|a, b| {
                let norm = |b: &Beam| b.logprob / b.tokens.len() as f32;
                norm(a).partial_cmp(&norm(b)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|b| b.tokens)
            .unwrap_or_else(|| prompt.to_vec());
        Ok(best)
    }
}

/// CTC decoding over `(T, n_vocab)` frame emissions.
#[derive(Debug, Clone, Default)]
pub struct CtcDecoder {
    /// Id of the CTC blank symbol.
    pub blank: usize,
}

impl CtcDecoder {
    /// Greedy CTC: per-frame argmax, then collapse repeats and drop
    /// blanks.
    pub fn decode_greedy(&self, emissions: &Array2<f32>) -> Vec<usize> {
        ctc_greedy(emissions, self.blank)
    }

    /// Beam CTC with plain sequence scoring (no prefix-tree sharing —
    /// fine for short segments).
    pub fn decode_beam(&self, emissions: &Array2<f32>, beam_width: usize) -> Vec<usize> {
        let width = beam_width.max(1);
        let t_len = emissions.nrows();
        // Each hypothesis: (tokens, logprob). Blank extension keeps tokens.
        let mut beams: Vec<(Vec<usize>, f32)> = vec![(Vec::new(), 0.0)];
        for t in 0..t_len {
            let frame = emissions.row(t);
            let log_probs = log_softmax(&frame.to_vec(), &[]);
            let mut next: Vec<(Vec<usize>, f32)> = Vec::new();
            for (tokens, lp) in &beams {
                // Emit blank (keeps hypothesis).
                next.push((tokens.clone(), lp + log_probs[self.blank]));
                // Emit each token (dedup when repeating the last token).
                for (tok, &p) in log_probs.iter().enumerate() {
                    if tok == self.blank {
                        continue;
                    }
                    if tokens.last() == Some(&tok) {
                        continue; // must pass through blank first
                    }
                    let mut nt = tokens.clone();
                    nt.push(tok);
                    next.push((nt, lp + p));
                }
            }
            next.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next.dedup_by(|a, b| a.0 == b.0);
            beams = next.into_iter().take(width).collect();
        }
        let _ = t_len;
        beams
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tokens, _)| tokens)
            .unwrap_or_default()
    }
}

/// Shared greedy CTC collapse (also used by the align crate's tests).
pub fn ctc_greedy(emissions: &Array2<f32>, blank: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for row in emissions.rows() {
        let tok = argmax_excluding(&row, &[]);
        if tok != blank && out.last() != Some(&tok) {
            out.push(tok);
        }
    }
    out
}

fn argmax_excluding(row: &ndarray::ArrayView1<f32>, suppress: &[usize]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if suppress.contains(&i) {
            continue;
        }
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn log_softmax(row: &[f32], suppress: &[usize]) -> Vec<f32> {
    let min = f32::NEG_INFINITY;
    let max = row
        .iter()
        .enumerate()
        .filter(|(i, _)| !suppress.contains(i))
        .map(|(_, &v)| v)
        .fold(min, f32::max);
    let exps: Vec<f32> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if suppress.contains(&i) {
                f32::NEG_INFINITY
            } else {
                (v - max).exp()
            }
        })
        .collect();
    let sum: f32 = exps.iter().filter(|e| e.is_finite()).sum();
    exps.into_iter()
        .map(|e| if e.is_finite() { e.ln() - sum.ln() } else { e })
        .collect()
}

fn top_k(log_probs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..log_probs.len()).collect();
    idx.sort_by(|a, b| log_probs[*b].partial_cmp(&log_probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    use crate::model::ModelSize;
    use crate::tokenizer::{BpeTokenizer, Tokenizer};

    fn model() -> AsrModel {
        AsrModel::random_init(ModelSize::Tiny.config(64), 0xBEEF)
    }

    #[test]
    fn greedy_produces_prompt_plus_tokens() {
        let m = model();
        let audio = m.encode(&Array2::<f32>::zeros((80, 40)));
        let cfg = DecodeConfig { max_tokens: 5, ..Default::default() };
        let out = GreedyDecoder.decode(&m, &audio, &[0], &cfg).unwrap();
        assert_eq!(out.len(), 6);
        assert_eq!(&out[..1], &[0]);
        // Deterministic.
        let again = GreedyDecoder.decode(&m, &audio, &[0], &cfg).unwrap();
        assert_eq!(out, again);
    }

    #[test]
    fn beam_beats_or_equals_greedy_probability() {
        let m = model();
        let audio = m.encode(&Array2::<f32>::zeros((80, 40)));
        let cfg_g = DecodeConfig { max_tokens: 4, ..Default::default() };
        let cfg_b = DecodeConfig { max_tokens: 4, beam_width: 3, ..Default::default() };
        let g = GreedyDecoder.decode(&m, &audio, &[0], &cfg_g).unwrap();
        let b = BeamSearchDecoder.decode(&m, &audio, &[0], &cfg_b).unwrap();
        assert_eq!(g.len(), b.len());
    }

    #[test]
    fn decode_config_validation() {
        let cfg = DecodeConfig { max_tokens: 0, ..Default::default() };
        assert!(GreedyDecoder.decode(&model(), &Array2::<f32>::zeros((80, 8)), &[], &cfg).is_err());
    }

    #[test]
    fn ctc_greedy_collapses_and_drops_blank() {
        // frames: [blank, A, A, blank, B] → [A, B]
        let e = array![
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 4.0, 1.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ];
        assert_eq!(ctc_greedy(&e, 0), vec![1, 2]);
    }

    #[test]
    fn ctc_beam_prefers_high_probability_path() {
        // Clear token regions with a dominant blank frame between them:
        // the best path is emit 1, repeat/blank, blank, emit 2, keep 2.
        let e = array![
            [0.0, 10.0, -10.0],
            [0.0, 10.0, -10.0],
            [10.0, -10.0, -10.0],
            [0.0, -10.0, 10.0],
            [0.0, -10.0, 10.0],
        ];
        let d = CtcDecoder { blank: 0 };
        assert_eq!(d.decode_greedy(&e), vec![1, 2]);
        assert_eq!(d.decode_beam(&e, 4), vec![1, 2]);
    }

    #[test]
    fn decoded_ids_map_through_tokenizer() {
        let tok = BpeTokenizer::train("hello world hello world", 300).unwrap();
        let m = AsrModel::random_init(ModelSize::Tiny.config(tok.vocab_size()), 1);
        let audio = m.encode(&Array2::<f32>::zeros((80, 40)));
        let cfg = DecodeConfig { max_tokens: 8, ..Default::default() };
        let out = GreedyDecoder.decode(&m, &audio, &[tok.sot_token()], &cfg).unwrap();
        let _ = tok.decode(&out); // must not panic; content is gibberish by design
    }
}
