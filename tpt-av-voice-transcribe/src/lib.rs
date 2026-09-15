//! Pure-Rust speech-to-text.
//!
//! `tpt-av-voice-transcribe` implements a transformer encoder-decoder ASR
//! architecture (Whisper-style) entirely in Rust:
//!
//! * mel spectrogram feature extraction ([`encoder`])
//! * transformer encoder-decoder model ([`model`])
//! * greedy / beam-search / CTC text decoding ([`decoder`])
//! * byte-level BPE tokenization ([`tokenizer`])
//! * safetensors-compatible weight loading ([`weights`])
//! * high-level [`Transcriber`] API and streaming transcription ([`streaming`])
//!
//! # Example
//!
//! ```no_run
//! use tpt_av_voice_transcribe::Transcriber;
//!
//! # let pcm: Vec<f32> = Vec::new();
//! let transcriber = Transcriber::from_model_dir("models/en-small").unwrap();
//! let transcript = transcriber.transcribe_samples(&pcm, 16_000).unwrap();
//! println!("{}", transcript.full_text);
//! ```

pub mod decoder;
pub mod encoder;
pub mod model;
pub mod streaming;
pub mod tokenizer;
pub mod transcriber;
pub mod weights;

pub use decoder::{BeamSearchDecoder, CtcDecoder, DecodeConfig, GreedyDecoder};
pub use encoder::MelConfig;
pub use model::{AsrModelConfig, ModelSize};
pub use streaming::StreamingTranscriber;
pub use tokenizer::{BpeTokenizer, Tokenizer};
pub use transcriber::{Transcriber, TranscriberConfig};
pub use weights::{load_model_weights, write_model_weights};

// Convenience re-exports: the core vocabulary types users need alongside
// this crate's API, without adding tpt-av-voice-utils directly.
pub use tpt_av_voice_utils::{
    AudioChunk, Language, Transcript, TranscriptSegment, VoiceError, WordTimestamp,
};
