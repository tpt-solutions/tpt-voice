//! Shared foundation for the `tpt-voice` speech processing engine.
//!
//! This crate provides the common vocabulary types ([`Transcript`],
//! [`SpeakerLabel`], [`AudioChunk`]), error handling ([`VoiceError`]),
//! time utilities, minimal WAV I/O, shared DSP primitives (windows, FFT
//! glue, mel filter banks, MFCC), small pure-Rust neural-network building
//! blocks, safetensors-compatible weight I/O, and SIMD kernels.
//!
//! All code in the `tpt-voice` workspace is pure Rust: no C/C++ bindings,
//! no ML runtime dependencies.

pub mod audio_chunk;
pub mod dsp;
pub mod error;
pub mod language;
pub mod manifest;
pub mod nn;
pub mod safetensors;
pub mod signals;
pub mod simd;
pub mod speaker;
pub mod time;
pub mod transcript;
pub mod wav;

pub use audio_chunk::AudioChunk;
pub use error::{AudioError, VoiceError};
pub use language::Language;
pub use speaker::SpeakerLabel;
pub use time::{TimeRange, Timestamp};
pub use transcript::{SegmentId, Transcript, TranscriptSegment, WordTimestamp};
