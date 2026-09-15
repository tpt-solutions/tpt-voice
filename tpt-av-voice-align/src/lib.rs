//! Pure-Rust text-audio alignment and text-based editing.
//!
//! `tpt-av-voice-align` maps transcript text onto audio time ranges:
//!
//! * CTC forced alignment over arbitrary unit sequences ([`forced_align`])
//! * rule-based English grapheme-to-phoneme conversion ([`phoneme`])
//! * word-level alignment ([`word_align`])
//! * high-level [`Aligner`] API ([`aligner`])
//! * [`TextBasedEditor`] which diffs an edited transcript against the
//!   original and emits [`AudioEdit`] operations consumable by the
//!   `tpt-audio` timeline ([`editor`])

pub mod aligner;
pub mod editor;
pub mod forced_align;
pub mod phoneme;
pub mod word_align;

pub use aligner::{AlignedTranscript, Aligner, AlignerConfig, AlignmentMode};
pub use editor::{AudioEdit, EditType, TextBasedEditor};
pub use forced_align::{ctc_align, AlignmentUnit, ForcedAlignment};
pub use phoneme::{g2p_word, Phoneme};
pub use word_align::words_from_phoneme_alignment;

// Convenience re-exports for one-dependency usage.
pub use tpt_av_voice_utils::{Transcript, VoiceError, WordTimestamp};
