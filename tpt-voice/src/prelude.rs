//! Everything a typical `tpt-voice` user needs, one `use` away.
//!
//! ```no_run
//! use tpt_voice::prelude::*;
//!
//! let mut isolator = VoiceIsolator::classical(IsolatorConfig::default()).unwrap();
//! let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
//! ```

pub use crate::pipeline::{PipelineOutput, SpeechPipeline};
pub use tpt_av_voice_align::{
    g2p_word, AlignedTranscript, Aligner, AlignerConfig, AlignmentMode, AudioEdit, EditType,
    Phoneme, TextBasedEditor,
};
pub use tpt_av_voice_diarize::{
    annotate_transcript, DiarizationResult, Diarizer, DiarizerConfig, SpeakerSegment,
};
pub use tpt_av_voice_isolate::{IsolatedAudio, IsolatorConfig, SourceType, VoiceIsolator};
pub use tpt_av_voice_tts::{FormantSynth, SynthConfig, VoiceProfile};
pub use tpt_av_voice_utils::{
    wav, AudioChunk, Language, SpeakerLabel, TimeRange, Transcript, TranscriptSegment, VoiceError,
    WordTimestamp,
};
