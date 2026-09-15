//! Pure-Rust text-to-speech (early stage).
//!
//! `tpt-av-voice-tts` currently provides a source-filter formant
//! synthesizer driven by rule-based letter-to-sound rules. It produces
//! intelligible (robotic) speech with no external dependencies or model
//! downloads, and defines the API surface that neural voices will
//! implement later.
//!
//! ```no_run
//! use tpt_av_voice_tts::{FormantSynth, SynthConfig};
//!
//! let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
//! let pcm = synth.synthesize("Hello world").unwrap();
//! // pcm.samples, pcm.sample_rate ready for playback / WAV export.
//! ```

pub mod config;
pub mod g2p;
pub mod synth;
pub mod voice;

pub use config::SynthConfig;
pub use synth::{FormantSynth, SynthesizedAudio};
pub use voice::VoiceProfile;

// Convenience re-exports for one-dependency usage.
pub use tpt_av_voice_utils::VoiceError;
