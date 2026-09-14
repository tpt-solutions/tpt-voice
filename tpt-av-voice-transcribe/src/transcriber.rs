//! The high-level transcription API.

use std::path::Path;

use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::dsp::resample_linear;
use tpt_av_voice_utils::{Language, Transcript, TranscriptSegment, VoiceError};

use crate::decoder::{DecodeConfig, Decoder, GreedyDecoder};
use crate::encoder::{MelConfig, MelFrontend};
use crate::model::{AsrModel, AsrModelConfig};
use crate::tokenizer::{BpeTokenizer, SOT, Tokenizer};

/// User-tunable transcription settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriberConfig {
    /// Language hint embedded in the decode prompt.
    pub language: Language,
    /// Maximum tokens generated per 30 s window.
    pub max_tokens_per_window: usize,
    /// Minimum segment confidence to keep a segment (0 keeps everything).
    pub min_confidence: f32,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        TranscriberConfig {
            language: Language::english(),
            max_tokens_per_window: 200,
            min_confidence: 0.0,
        }
    }
}

/// End-to-end speech-to-text engine.
///
/// ```no_run
/// use tpt_av_voice_transcribe::Transcriber;
///
/// let transcriber = Transcriber::from_model_dir("models/en-small")?;
/// let transcript = transcriber.transcribe_file("meeting.wav")?;
/// println!("{}", transcript.full_text);
/// # Ok::<(), tpt_av_voice_utils::VoiceError>(())
/// ```
pub struct Transcriber {
    model: AsrModel,
    tokenizer: BpeTokenizer,
    frontend: MelFrontend,
    config: TranscriberConfig,
    decoder: Box<dyn Decoder + Send + Sync>,
}

/// Whisper-style window: 30 s of audio per decode pass.
const WINDOW_SAMPLES: usize = 30 * 16_000;

impl Transcriber {
    /// Loads a model directory (see [`crate::weights`] for the layout).
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, VoiceError> {
        let dir = dir.as_ref();
        let tensors = crate::weights::load_model_weights(dir)?;
        let config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let model_config: AsrModelConfig = serde_json::from_str(&config_json)
            .map_err(|e| VoiceError::model(format!("bad config.json: {e}")))?;
        let model = AsrModel::from_tensors(&tensors, model_config)?;
        let tokenizer_json = std::fs::read_to_string(dir.join("tokenizer.json"))?;
        let tokenizer = BpeTokenizer::from_json(&tokenizer_json)?;
        Self::from_parts(model, tokenizer, TranscriberConfig::default())
    }

    /// Assembles a transcriber from already-loaded parts.
    pub fn from_parts(
        model: AsrModel,
        tokenizer: BpeTokenizer,
        config: TranscriberConfig,
    ) -> Result<Self, VoiceError> {
        if model.config.n_mels != 80 {
            return Err(VoiceError::config("only 80-mel models are supported"));
        }
        let frontend = MelFrontend::new(MelConfig::default())?;
        Ok(Transcriber {
            model,
            tokenizer,
            frontend,
            config,
            decoder: Box::new(GreedyDecoder),
        })
    }

    /// Replaces the default greedy decoder (e.g. with beam search).
    pub fn with_decoder(mut self, decoder: Box<dyn Decoder + Send + Sync>) -> Self {
        self.decoder = decoder;
        self
    }

    /// Replaces the settings.
    pub fn with_config(mut self, config: TranscriberConfig) -> Self {
        self.config = config;
        self
    }

    /// Transcribes a WAV file (mono downmix; resampled to 16 kHz if needed).
    pub fn transcribe_file(&self, path: impl AsRef<Path>) -> Result<Transcript, VoiceError> {
        let wav = tpt_av_voice_utils::wav::read_wav_file(path)?;
        let mono = wav.to_mono();
        self.transcribe_samples(&mono, wav.sample_rate)
    }

    /// Transcribes mono PCM samples at any sample rate.
    pub fn transcribe_samples(&self, samples: &[f32], sample_rate: u32) -> Result<Transcript, VoiceError> {
        let pcm = resample_linear(samples, sample_rate, 16_000);
        let mut transcript = Transcript::new(self.config.language.code());
        let mut start_sample = 0usize;
        let mut seg_index = 0u64;

        while start_sample < pcm.len() {
            let end = (start_sample + WINDOW_SAMPLES).min(pcm.len());
            let window = &pcm[start_sample..end];
            let window_start_secs = tpt_av_voice_utils::time::samples_to_secs(start_sample, 16_000);
            let window_end_secs = tpt_av_voice_utils::time::samples_to_secs(end, 16_000);

            let tokens = self.decode_window(window)?;
            let text = self.tokenizer.decode(&tokens);
            let text = text.trim();
            if !text.is_empty() {
                // Confidence: mean softmax probability of generated tokens,
                // recomputed cheaply from the final logits pass.
                let confidence = self.window_confidence(window, &tokens);
                if confidence >= self.config.min_confidence {
                    transcript.segments.push(TranscriptSegment {
                        id: tpt_av_voice_utils::SegmentId(seg_index),
                        start_time: window_start_secs,
                        end_time: window_end_secs,
                        text: text.to_string(),
                        speaker: None,
                        confidence,
                        words: None,
                    });
                    seg_index += 1;
                }
            }
            start_sample = end;
        }
        transcript.refresh_full_text();
        transcript.recompute_confidence();
        Ok(transcript)
    }

    /// Decodes one ≤30 s window into text token ids (generated only).
    fn decode_window(&self, window: &[f32]) -> Result<Vec<usize>, VoiceError> {
        let mel = self.frontend.log_mel(window);
        let audio = self.model.encode(&mel);
        // The bare SOT prompt is used here; trained checkpoints extend it
        // with language/task tokens as a prompt convention.
        let prompt = [self.tokenizer.sot_token()];
        let cfg = DecodeConfig {
            max_tokens: self.config.max_tokens_per_window,
            beam_width: 1,
            suppress: vec![],
        };
        let out = self.decoder.decode(&self.model, &audio, &prompt, &cfg)?;
        // Strip prompt tokens, stop at EOT if present.
        let eot = self.tokenizer.eot_token();
        let mut generated = out[prompt.len().min(out.len())..].to_vec();
        if let Some(eot) = eot {
            if let Some(pos) = generated.iter().position(|&t| t == eot) {
                generated.truncate(pos);
            }
        }
        Ok(generated)
    }

    /// Mean token probability over the generated span (single extra
    /// logits pass when the generation was short enough to reuse).
    fn window_confidence(&self, window: &[f32], generated: &[usize]) -> f32 {
        if generated.is_empty() {
            return 0.0;
        }
        // Prompt + generated log-prob pass.
        let mel = self.frontend.log_mel(window);
        let audio = self.model.encode(&mel);
        let mut tokens = vec![self.tokenizer.sot_token()];
        tokens.extend_from_slice(generated);
        let logits = self.model.decode_logits(&tokens, &audio);
        // logits[i] predicts tokens[i + 1].
        let mut total = 0.0f32;
        let mut count = 0usize;
        for (i, &tok) in generated.iter().enumerate() {
            let row = logits.row(i.min(logits.nrows() - 1));
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            total += exps[tok.min(exps.len() - 1)] / sum;
            count += 1;
        }
        if count == 0 { 0.0 } else { total / count as f32 }
    }

    /// Borrows the tokenizer (e.g. for prompt construction).
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// The prompt start token constant (`<|startoftranscript|>`).
    pub const PROMPT_TOKEN: &'static str = SOT;
}
