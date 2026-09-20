//! # tpt-voice
//!
//! The one-crate facade over the `tpt-av-voice-*` speech engine: every
//! capability, one dependency.
//!
//! ```toml
//! [dependencies]
//! tpt-voice = "0.1"
//! ```
//!
//! Sub-crates are still published and usable individually; this crate
//! only re-exports them and adds cross-cutting glue:
//!
//! * [`pipeline::SpeechPipeline`] — transcribe → diarize → align in one
//!   call, returning a speaker-labelled transcript with word timestamps.
//! * [`pipeline::apply_text_edit_with_speech`] — text-based editing that
//!   synthesizes audio for inserted words via the built-in TTS, ready to
//!   splice on the `tpt-audio` timeline.
//!
//! ## Quickstart
//!
//! ```no_run
//! use tpt_voice::prelude::*;
//!
//! # let pcm: Vec<f32> = Vec::new();
//! let mut pipeline = SpeechPipeline::from_model_dir(
//!     "models/en-small",
//!     DiarizerConfig::default(),
//!     AlignerConfig::default(),
//! )
//! .unwrap();
//! let out = pipeline.run(&pcm, 16_000).unwrap();
//! for seg in &out.transcript.segments {
//!     let speaker = seg.speaker.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "?".into());
//!     println!("[{:.2}-{:.2}] {speaker}: {}", seg.start_time, seg.end_time, seg.text);
//! }
//! ```

pub use tpt_av_voice_align as align;
pub use tpt_av_voice_diarize as diarize;
pub use tpt_av_voice_isolate as isolate;
pub use tpt_av_voice_transcribe as transcribe;
pub use tpt_av_voice_tts as tts;
pub use tpt_av_voice_utils as utils;

pub mod pipeline;
pub mod prelude;
pub mod streaming_pipeline;

pub use pipeline::{apply_text_edit_with_speech, fill_insertions, PipelineOutput, SpeechPipeline};
pub use streaming_pipeline::StreamingPipeline;
