//! Streaming transcription.
//!
//! [`StreamingTranscriber`] accepts [`AudioChunk`]s as they arrive,
//! buffers them, and finalizes a segment whenever the buffer holds a
//! complete utterance: either a silence gap is detected at the tail of
//! the buffer, or the hard window limit (30 s) is reached. The final
//! segments for the finished utterance are returned from `push`.
//!
//! ```no_run
//! use tpt_av_voice_utils::AudioChunk;
//! use tpt_av_voice_transcribe::{StreamingTranscriber, Transcriber};
//!
//! # let audio_chunks: Vec<AudioChunk> = Vec::new();
//! # let transcriber = Transcriber::from_model_dir("models/en-small").unwrap();
//! let mut stream = StreamingTranscriber::new(transcriber);
//! for chunk in audio_chunks {
//!     if let Some(transcript) = stream.push(&chunk).unwrap() {
//!         println!("final: {}", transcript.full_text);
//!     }
//! }
//! ```

use tpt_av_voice_utils::{AudioChunk, Transcript, VoiceError};

use crate::transcriber::Transcriber;

/// Silence/segmentation tuning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamConfig {
    /// RMS below which audio counts as silence (linear amplitude).
    pub silence_rms: f32,
    /// Minimum silence duration that ends an utterance, in seconds.
    pub min_silence_secs: f64,
    /// Hard cap on buffered audio, in seconds (a full buffer forces a
    /// segment even mid-speech).
    pub max_buffer_secs: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            silence_rms: 0.004,
            min_silence_secs: 0.5,
            max_buffer_secs: 30.0,
        }
    }
}

/// Incremental transcriber over chunk streams.
pub struct StreamingTranscriber {
    transcriber: Transcriber,
    config: StreamConfig,
    buffer: Vec<f32>,
    buffer_start: f64,
    silence_run_secs: f64,
    last_end: f64,
    next_sequence: u64,
}

impl StreamingTranscriber {
    /// Wraps a transcriber with default streaming settings.
    pub fn new(transcriber: Transcriber) -> Self {
        StreamingTranscriber::with_config(transcriber, StreamConfig::default())
    }

    /// Wraps a transcriber with custom streaming settings.
    pub fn with_config(transcriber: Transcriber, config: StreamConfig) -> Self {
        StreamingTranscriber {
            transcriber,
            config,
            buffer: Vec::new(),
            buffer_start: 0.0,
            silence_run_secs: 0.0,
            last_end: 0.0,
            next_sequence: 0,
        }
    }

    /// Borrows the underlying transcriber.
    pub fn transcriber(&self) -> &Transcriber {
        &self.transcriber
    }

    /// Buffers a chunk; when the buffered audio forms a complete
    /// utterance (trailing silence or window cap), the utterance is
    /// transcribed and returned as a one-segment [`Transcript`].
    pub fn push(&mut self, chunk: &AudioChunk) -> Result<Option<Transcript>, VoiceError> {
        if chunk.sample_rate != 16_000 {
            return Err(VoiceError::config(format!(
                "StreamingTranscriber expects 16 kHz chunks, got {}",
                chunk.sample_rate
            )));
        }
        if self.buffer.is_empty() {
            self.buffer_start = chunk.start_time;
        }
        self.buffer.extend_from_slice(&chunk.samples);

        // Track trailing silence.
        let tail = chunk.samples.as_slice();
        let rms = tpt_av_voice_utils::dsp::rms(tail);
        if rms < self.config.silence_rms {
            self.silence_run_secs += chunk.duration();
        } else {
            self.silence_run_secs = 0.0;
        }
        self.last_end = chunk.end_time();

        let buffered_secs = self.buffer.len() as f64 / 16_000.0;
        let should_flush = self.silence_run_secs >= self.config.min_silence_secs
            || buffered_secs >= self.config.max_buffer_secs;

        if should_flush {
            self.flush()
        } else {
            Ok(None)
        }
    }

    /// Transcribes and clears whatever is buffered right now.
    pub fn flush(&mut self) -> Result<Option<Transcript>, VoiceError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        let samples = std::mem::take(&mut self.buffer);
        self.silence_run_secs = 0.0;
        let mut transcript = self.transcriber.transcribe_samples(&samples, 16_000)?;
        // Rebase segment times onto the streaming timeline.
        for seg in &mut transcript.segments {
            seg.start_time += self.buffer_start;
            seg.end_time += self.buffer_start;
        }
        transcript.refresh_full_text();
        self.buffer_start = self.last_end;
        if transcript.segments.is_empty() {
            Ok(None)
        } else {
            Ok(Some(transcript))
        }
    }

    /// Number of samples currently buffered.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Consumes an iterator of chunks, emitting one [`Transcript`] per
    /// finalized utterance. Call [`StreamingTranscriber::flush`] after
    /// the last chunk for the tail utterance.
    pub fn process<'a>(
        &'a mut self,
        chunks: impl Iterator<Item = AudioChunk> + 'a,
    ) -> impl Iterator<Item = Result<Transcript, VoiceError>> + 'a {
        chunks.filter_map(move |chunk| {
            let seq = chunk.sequence;
            self.next_sequence = seq + 1;
            self.push(&chunk).transpose()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AsrModel, ModelSize};
    use crate::tokenizer::{BpeTokenizer, Tokenizer};
    use crate::transcriber::Transcriber;
    use tpt_av_voice_utils::audio_chunk::chunk_pcm;

    fn transcriber() -> Transcriber {
        let tok = BpeTokenizer::train("hello world silence test", 300).unwrap();
        let model = AsrModel::random_init(ModelSize::Tiny.config(tok.vocab_size()), 7);
        // Tiny decode budget keeps the debug-build test fast.
        Transcriber::from_parts(
            model,
            tok,
            crate::transcriber::TranscriberConfig {
                max_tokens_per_window: 4,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn silence_gaps_produce_segments() {
        let mut stream = StreamingTranscriber::new(transcriber());
        // 1 s loud tone, 0.6 s silence, 1 s loud tone, 0.6 s silence.
        let mut pcm = Vec::new();
        for _ in 0..2 {
            pcm.extend((0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.5));
            pcm.extend(std::iter::repeat(0.0).take(10_000)); // 0.625 s silence
        }
        let segments: Vec<Transcript> = stream
            .process(chunk_pcm(pcm, 16_000, 4_000))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        stream.flush().unwrap();
        assert!(
            !segments.is_empty(),
            "expected at least one flushed utterance"
        );
        for seg in &segments {
            assert_eq!(seg.segments.len(), 1);
            assert!(seg.segments[0].end_time > seg.segments[0].start_time);
        }
    }

    #[test]
    fn flush_drains_buffer() {
        let mut stream = StreamingTranscriber::new(transcriber());
        let chunk = AudioChunk::new(vec![0.1; 8_000], 16_000, 0.0, 0);
        stream.push(&chunk).unwrap();
        assert_eq!(stream.buffered_samples(), 8_000);
        // Unbroken loud audio: flush returns whatever is there.
        let out = stream.flush().unwrap();
        assert!(out.is_some());
        assert_eq!(stream.buffered_samples(), 0);
        assert!(stream.flush().unwrap().is_none());
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut stream = StreamingTranscriber::new(transcriber());
        let chunk = AudioChunk::new(vec![0.0; 100], 44_100, 0.0, 0);
        assert!(stream.push(&chunk).is_err());
    }
}
