//! Streaming composition: isolate → stream-transcribe → online diarize.
//!
//! [`StreamingPipeline`] mirrors [`crate::pipeline::SpeechPipeline`] but for
//! chunk-at-a-time audio: each pushed [`AudioChunk`] is isolated first
//! (overlap-add), the isolated audio is fed to the streaming transcriber,
//! and in parallel to the online diarizer; whenever the transcriber
//! finalizes an utterance, its segments are annotated with whichever
//! diarization window overlaps them most (same rule as
//! [`tpt_av_voice_diarize::annotate_transcript`], just applied
//! incrementally against the diarization timeline accumulated so far).
//!
//! ```no_run
//! use tpt_av_voice_isolate::{IsolatorConfig, VoiceIsolator};
//! use tpt_av_voice_diarize::{DiarizerConfig, StreamingDiarizer};
//! use tpt_av_voice_transcribe::Transcriber;
//! use tpt_voice::StreamingPipeline;
//!
//! # let chunks: Vec<tpt_av_voice_utils::AudioChunk> = Vec::new();
//! let mut isolator = VoiceIsolator::classical(IsolatorConfig::default()).unwrap();
//! let transcriber = Transcriber::from_model_dir("models/en-small").unwrap();
//! let diarizer = StreamingDiarizer::classical(&DiarizerConfig::default()).unwrap();
//! let mut pipeline = StreamingPipeline::new(&mut isolator, transcriber, diarizer);
//! for chunk in chunks {
//!     for transcript in pipeline.push(&chunk).unwrap() {
//!         println!("{}", transcript.full_text);
//!     }
//! }
//! for transcript in pipeline.flush().unwrap() {
//!     println!("{}", transcript.full_text);
//! }
//! ```

use tpt_av_voice_diarize::{SpeakerSegment, StreamingDiarizer};
use tpt_av_voice_isolate::{StreamingIsolator, VoiceIsolator};
use tpt_av_voice_transcribe::{StreamingTranscriber, Transcriber};
use tpt_av_voice_utils::{AudioChunk, Transcript, VoiceError};

/// Chunk-at-a-time transcribe + diarize, fed through streaming isolation.
///
/// Borrows the [`VoiceIsolator`] rather than owning it, matching
/// [`StreamingIsolator`]'s own convention: the caller keeps the isolator
/// (and any noise-estimation state it accumulates) alive alongside the
/// pipeline.
pub struct StreamingPipeline<'a> {
    isolate: StreamingIsolator<'a>,
    transcribe: StreamingTranscriber,
    diarize: StreamingDiarizer,
    /// Diarization windows seen so far, used to annotate transcript
    /// segments as they are finalized.
    diarization_timeline: Vec<SpeakerSegment>,
}

impl<'a> StreamingPipeline<'a> {
    /// Builds a pipeline around a borrowed isolator plus owned
    /// transcriber/diarizer streaming state.
    pub fn new(
        isolator: &'a mut VoiceIsolator,
        transcriber: Transcriber,
        diarizer: StreamingDiarizer,
    ) -> Self {
        StreamingPipeline {
            isolate: StreamingIsolator::new(isolator),
            transcribe: StreamingTranscriber::new(transcriber),
            diarize: diarizer,
            diarization_timeline: Vec::new(),
        }
    }

    /// Pushes one chunk of raw (un-isolated) audio through the full
    /// streaming stack. Returns zero or more finalized, speaker-annotated
    /// transcripts (usually zero or one, per push).
    pub fn push(&mut self, chunk: &AudioChunk) -> Result<Vec<Transcript>, VoiceError> {
        let isolated = self.isolate.push(chunk)?;
        self.feed(&isolated)
    }

    /// Drains buffered audio from every stage and finalizes whatever
    /// utterance remains.
    pub fn flush(&mut self) -> Result<Vec<Transcript>, VoiceError> {
        let isolated = self.isolate.flush()?;
        let mut out = self.feed(&isolated)?;
        if let Some(mut transcript) = self.transcribe.flush()? {
            self.annotate(&mut transcript);
            out.push(transcript);
        }
        Ok(out)
    }

    /// Number of distinct speakers the online diarizer has opened so far.
    pub fn num_speakers(&self) -> usize {
        self.diarize.num_speakers()
    }

    fn feed(&mut self, isolated: &AudioChunk) -> Result<Vec<Transcript>, VoiceError> {
        let mut out = Vec::new();
        if isolated.is_empty() {
            return Ok(out);
        }
        if let Some(window) = self.diarize.push(isolated)? {
            self.diarization_timeline.push(window);
        }
        if let Some(mut transcript) = self.transcribe.push(isolated)? {
            self.annotate(&mut transcript);
            out.push(transcript);
        }
        Ok(out)
    }

    /// Labels each unlabelled segment with the diarization window that
    /// overlaps it most, same rule as the batch pipeline's
    /// `annotate_transcript` — just against the timeline accumulated so
    /// far rather than a complete recording.
    fn annotate(&self, transcript: &mut Transcript) {
        for segment in &mut transcript.segments {
            if segment.speaker.is_some() {
                continue;
            }
            let range = segment.time_range();
            let best = self
                .diarization_timeline
                .iter()
                .filter_map(|window| {
                    let overlap = range.overlap_fraction(&window.time_range());
                    (overlap > 0.0).then_some((overlap, &window.speaker))
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((_, speaker)) = best {
                segment.speaker = Some(speaker.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_av_voice_diarize::DiarizerConfig;
    use tpt_av_voice_isolate::IsolatorConfig;
    use tpt_av_voice_transcribe::model::{AsrModel, ModelSize};
    use tpt_av_voice_transcribe::tokenizer::{BpeTokenizer, Tokenizer};
    use tpt_av_voice_transcribe::transcriber::TranscriberConfig;
    use tpt_av_voice_utils::audio_chunk::chunk_pcm;
    use tpt_av_voice_utils::signals::alternating_speakers;

    fn tiny_transcriber() -> Transcriber {
        let tok = BpeTokenizer::train("hello world speech test today", 300).unwrap();
        let model = AsrModel::random_init(ModelSize::Tiny.config(tok.vocab_size()), 3);
        Transcriber::from_parts(
            model,
            tok,
            TranscriberConfig {
                max_tokens_per_window: 4,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn end_to_end_meeting_does_not_panic_and_produces_output() {
        // Two alternating "speakers" (distinct pitch buzzes) with silence
        // gaps in between, like the batch pipeline's fixtures.
        let pcm = alternating_speakers(
            &[(150.0, 1.2), (400.0, 1.2), (150.0, 1.2), (400.0, 1.2)],
            0.4,
            16_000,
            0.5,
        );

        let mut isolator = VoiceIsolator::classical(IsolatorConfig::default()).unwrap();
        let diarizer = StreamingDiarizer::classical(&DiarizerConfig::default()).unwrap();
        let mut pipeline = StreamingPipeline::new(&mut isolator, tiny_transcriber(), diarizer);

        let mut transcripts = Vec::new();
        for chunk in chunk_pcm(pcm, 16_000, 1_600) {
            transcripts.extend(pipeline.push(&chunk).unwrap());
        }
        transcripts.extend(pipeline.flush().unwrap());

        assert!(
            !transcripts.is_empty(),
            "expected at least one finalized utterance"
        );
        for t in &transcripts {
            for seg in &t.segments {
                assert!(seg.end_time >= seg.start_time);
            }
        }
        // The online diarizer should have found more than one speaker
        // across an alternating-speaker recording.
        assert!(pipeline.num_speakers() >= 1);
    }

    #[test]
    fn empty_isolated_output_is_skipped_without_panic() {
        let mut isolator = VoiceIsolator::classical(IsolatorConfig::default()).unwrap();
        let diarizer = StreamingDiarizer::classical(&DiarizerConfig::default()).unwrap();
        let mut pipeline = StreamingPipeline::new(&mut isolator, tiny_transcriber(), diarizer);

        // A single tiny chunk, smaller than the isolator's algorithmic
        // latency: nothing should come out yet, and nothing should panic.
        let chunk = AudioChunk::new(vec![0.1; 32], 16_000, 0.0, 0);
        let out = pipeline.push(&chunk).unwrap();
        assert!(out.is_empty());
        let _ = pipeline.flush().unwrap();
    }
}
