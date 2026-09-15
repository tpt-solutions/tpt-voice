//! Pure-Rust real-time voice isolation.
//!
//! `tpt-av-voice-isolate` separates human speech from background noise:
//!
//! * STFT / inverse STFT ([`stft`], [`istft`])
//! * time-frequency masking ([`mask`])
//! * U-Net separation model ([`model`]) or classical spectral-subtraction
//!   masking when no trained weights are available
//! * high-level [`VoiceIsolator`] API and streaming isolation ([`streaming`])
//!
//! Pipeline: `audio → STFT → mask prediction → masked STFT → ISTFT → voice`.

pub mod isolator;
pub mod istft;
pub mod mask;
pub mod model;
pub mod stft;
pub mod streaming;

pub use isolator::{IsolatedAudio, IsolatorConfig, SourceType, VoiceIsolator};
pub use istft::Istft;
pub use stft::{Stft, StftConfig};
pub use streaming::StreamingIsolator;

// Convenience re-exports for one-dependency usage.
pub use tpt_av_voice_utils::{wav, AudioChunk, VoiceError};
