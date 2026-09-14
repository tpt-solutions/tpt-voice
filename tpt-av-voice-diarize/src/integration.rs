//! Multi-speaker transcription integration: links diarization output to
//! `Transcript.segments[].speaker` (spec Phase 3).

use tpt_av_voice_utils::{Transcript, VoiceError};

use crate::diarizer::DiarizationResult;

/// Assigns each transcript segment the speaker whose diarization
/// segment overlaps it the most. Segments with no overlapping speech
/// keep `speaker: None`.
///
/// Returns the number of segments annotated.
pub fn annotate_transcript(
    transcript: &mut Transcript,
    diarization: &DiarizationResult,
) -> Result<usize, VoiceError> {
    let mut annotated = 0usize;
    for segment in &mut transcript.segments {
        let range = segment.time_range();
        let best = diarization
            .segments
            .iter()
            .filter_map(|spk| {
                let overlap = range.overlap_fraction(&spk.time_range());
                (overlap > 0.0).then_some((overlap, &spk.speaker))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((_, speaker)) = best {
            segment.speaker = Some(speaker.clone());
            annotated += 1;
        }
    }
    Ok(annotated)
}

/// Convenience wrapper: diarizes and annotates in one call.
pub fn diarize_and_annotate(
    diarizer: &mut crate::diarizer::Diarizer,
    samples: &[f32],
    sample_rate: u32,
    transcript: &mut Transcript,
) -> Result<DiarizationResult, VoiceError> {
    let result = diarizer.diarize_samples(samples, sample_rate)?;
    annotate_transcript(transcript, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_av_voice_utils::{Transcript, TranscriptSegment};

    fn result() -> DiarizationResult {
        DiarizationResult {
            segments: vec![
                crate::diarizer::SpeakerSegment {
                    speaker: tpt_av_voice_utils::SpeakerLabel::named("SPEAKER_00", "Alice"),
                    start_time: 0.0,
                    end_time: 2.0,
                    confidence: 0.9,
                },
                crate::diarizer::SpeakerSegment {
                    speaker: tpt_av_voice_utils::SpeakerLabel::named("SPEAKER_01", "Bob"),
                    start_time: 2.5,
                    end_time: 4.0,
                    confidence: 0.9,
                },
            ],
            num_speakers: 2,
        }
    }

    #[test]
    fn annotates_by_overlap() {
        let mut t = Transcript::new("en");
        t.push_segment(TranscriptSegment::new(0.5, 1.5, "hello", 1.0)); // → Alice
        t.push_segment(TranscriptSegment::new(2.6, 3.9, "hi", 1.0)); // → Bob
        t.push_segment(TranscriptSegment::new(5.0, 6.0, "?", 1.0)); // no overlap
        let n = annotate_transcript(&mut t, &result()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(t.segments[0].speaker.as_ref().unwrap().display_name(), "Alice");
        assert_eq!(t.segments[1].speaker.as_ref().unwrap().display_name(), "Bob");
        assert!(t.segments[2].speaker.is_none());
    }

    #[test]
    fn empty_result_leaves_transcript() {
        let mut t = Transcript::new("en");
        t.push_segment(TranscriptSegment::new(0.0, 1.0, "hello", 1.0));
        let n = annotate_transcript(&mut t, &DiarizationResult::default()).unwrap();
        assert_eq!(n, 0);
        assert!(t.segments[0].speaker.is_none());
    }
}
