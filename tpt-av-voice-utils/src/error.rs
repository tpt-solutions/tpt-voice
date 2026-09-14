//! Error handling for the `tpt-voice` engine.

use std::fmt;
use std::io;

/// Errors produced anywhere in the `tpt-voice` engine.
#[derive(Debug)]
pub enum VoiceError {
    /// Filesystem or stream I/O failure.
    Io(io::Error),
    /// Audio decoding/format problems.
    Audio(AudioError),
    /// Model architecture/config problems (missing tensor, bad shape, ...).
    Model(String),
    /// Weight file problems (bad magic, truncated data, unsupported dtype).
    Weights(String),
    /// Tokenizer problems (missing vocab, bad merges, unknown encoding).
    Tokenizer(String),
    /// Decoding/search failures (empty hypotheses, invalid token stream).
    Decode(String),
    /// Invalid user configuration (zero hop size, conflicting bounds, ...).
    Config(String),
    /// Feature that is planned but not implemented yet.
    NotImplemented(String),
}

impl VoiceError {
    /// Convenience constructor for [`VoiceError::Config`].
    pub fn config(msg: impl Into<String>) -> Self {
        VoiceError::Config(msg.into())
    }

    /// Convenience constructor for [`VoiceError::Model`].
    pub fn model(msg: impl Into<String>) -> Self {
        VoiceError::Model(msg.into())
    }

    /// Convenience constructor for [`VoiceError::Weights`].
    pub fn weights(msg: impl Into<String>) -> Self {
        VoiceError::Weights(msg.into())
    }

    /// Convenience constructor for [`VoiceError::Tokenizer`].
    pub fn tokenizer(msg: impl Into<String>) -> Self {
        VoiceError::Tokenizer(msg.into())
    }

    /// Convenience constructor for [`VoiceError::Decode`].
    pub fn decode(msg: impl Into<String>) -> Self {
        VoiceError::Decode(msg.into())
    }
}

impl fmt::Display for VoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoiceError::Io(e) => write!(f, "I/O error: {e}"),
            VoiceError::Audio(e) => write!(f, "audio error: {e}"),
            VoiceError::Model(msg) => write!(f, "model error: {msg}"),
            VoiceError::Weights(msg) => write!(f, "weights error: {msg}"),
            VoiceError::Tokenizer(msg) => write!(f, "tokenizer error: {msg}"),
            VoiceError::Decode(msg) => write!(f, "decode error: {msg}"),
            VoiceError::Config(msg) => write!(f, "configuration error: {msg}"),
            VoiceError::NotImplemented(what) => write!(f, "not implemented: {what}"),
        }
    }
}

impl std::error::Error for VoiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VoiceError::Io(e) => Some(e),
            VoiceError::Audio(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for VoiceError {
    fn from(e: io::Error) -> Self {
        VoiceError::Io(e)
    }
}

impl From<AudioError> for VoiceError {
    fn from(e: AudioError) -> Self {
        VoiceError::Audio(e)
    }
}

/// Audio decoding and format errors.
#[derive(Debug)]
pub enum AudioError {
    /// Data exists but not enough of it (truncated header, empty buffer).
    Truncated(String),
    /// A recognized container holding an encoding we do not support.
    UnsupportedEncoding(String),
    /// Not a recognizable audio format at all.
    InvalidFormat(String),
    /// Operation requires a specific sample rate.
    SampleRateMismatch { expected: u32, actual: u32 },
    /// Input buffer contains no samples where some were required.
    Empty,
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::Truncated(msg) => write!(f, "audio data truncated: {msg}"),
            AudioError::UnsupportedEncoding(msg) => {
                write!(f, "unsupported audio encoding: {msg}")
            }
            AudioError::InvalidFormat(msg) => write!(f, "invalid audio format: {msg}"),
            AudioError::SampleRateMismatch { expected, actual } => {
                write!(f, "sample rate mismatch: expected {expected} Hz, got {actual} Hz")
            }
            AudioError::Empty => write!(f, "audio input is empty"),
        }
    }
}

impl std::error::Error for AudioError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_informative() {
        let e = VoiceError::config("hop_size must be > 0");
        assert_eq!(e.to_string(), "configuration error: hop_size must be > 0");

        let e = VoiceError::from(AudioError::Empty);
        assert_eq!(e.to_string(), "audio error: audio input is empty");
    }

    #[test]
    fn io_error_converts() {
        let e: VoiceError = io::Error::new(io::ErrorKind::NotFound, "nope").into();
        assert!(matches!(e, VoiceError::Io(_)));
        assert!(std::error::Error::source(&e).is_some());
    }
}
