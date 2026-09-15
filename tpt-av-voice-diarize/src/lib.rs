//! Pure-Rust speaker diarization: who spoke when.
//!
//! `tpt-av-voice-diarize` answers "which speaker is talking at each moment":
//!
//! * voice activity detection & segmentation ([`segmentation`])
//! * speaker embedding extraction (TDNN-style, with a classical MFCC
//!   statistics fallback) ([`embedding`])
//! * speaker clustering — agglomerative hierarchical or spectral
//!   ([`clustering`])
//! * high-level [`Diarizer`] API ([`diarizer`]) and streaming support
//!   ([`streaming`])
//!
//! [`annotate_transcript`] links diarization output to a
//! [`Transcript`](tpt_av_voice_utils::Transcript) so segments carry
//! `speaker` labels.

pub mod clustering;
pub mod diarizer;
pub mod embedding;
pub mod integration;
pub mod segmentation;
pub mod streaming;

pub use clustering::{cluster, ClusteringAlgorithm, ClusteringConfig};
pub use diarizer::{DiarizationResult, Diarizer, DiarizerConfig, SpeakerSegment};
pub use integration::{annotate_transcript, diarize_and_annotate};
pub use segmentation::{VadConfig, VoiceActivityDetector};
pub use streaming::StreamingDiarizer;

// Convenience re-exports for one-dependency usage.
pub use tpt_av_voice_utils::{AudioChunk, SpeakerLabel, Transcript, VoiceError};
