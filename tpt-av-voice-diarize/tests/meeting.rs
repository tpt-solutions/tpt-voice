//! End-to-end diarization integration: VAD → classical embeddings →
//! AHC clustering → transcript annotation.

use tpt_av_voice_diarize::diarizer::DiarizerConfig;
use tpt_av_voice_diarize::{annotate_transcript, Diarizer};
use tpt_av_voice_utils::{Transcript, TranscriptSegment};

fn buzz(hz: f32, secs: f64, amp: f32) -> Vec<f32> {
    let n = (secs * 16_000.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let mut v = 0.0f32;
            for k in 1..=8 {
                v += (std::f32::consts::TAU * hz * k as f32 * t).sin() / k as f32;
            }
            v * amp
        })
        .collect()
}

#[test]
fn meeting_is_segmented_by_speaker_and_annotates_transcript() {
    // Timeline: A (1.5 s) — silence — B (1.5 s) — silence — A (1.5 s).
    let mut pcm = Vec::new();
    pcm.extend(buzz(150.0, 1.5, 0.4));
    pcm.extend(vec![0.0; 12_000]);
    pcm.extend(buzz(400.0, 1.5, 0.4));
    pcm.extend(vec![0.0; 12_000]);
    pcm.extend(buzz(151.0, 1.5, 0.4));

    let mut diarizer = Diarizer::classical(DiarizerConfig::default()).unwrap();
    let result = diarizer.diarize_samples(&pcm, 16_000).unwrap();
    assert_eq!(result.num_speakers, 2, "got: {result:?}");
    assert!(result.segments.len() >= 3, "alternation must produce ≥3 segments");
    assert_eq!(result.segments[0].speaker.id, result.segments.last().unwrap().speaker.id);

    // Link to a transcript with one segment per speech burst.
    let mut transcript = Transcript::new("en");
    transcript.push_segment(TranscriptSegment::new(0.2, 1.4, "good morning everyone", 1.0));
    transcript.push_segment(TranscriptSegment::new(2.2, 3.4, "morning! coffee is ready", 1.0));
    transcript.push_segment(TranscriptSegment::new(4.4, 5.6, "thanks for coming", 1.0));
    let annotated = annotate_transcript(&mut transcript, &result).unwrap();
    assert_eq!(annotated, 3);

    let a = transcript.segments[0].speaker.as_ref().unwrap().id.clone();
    let b = transcript.segments[1].speaker.as_ref().unwrap().id.clone();
    let a_again = transcript.segments[2].speaker.as_ref().unwrap().id.clone();
    assert_eq!(a, a_again);
    assert_ne!(a, b);
}

#[test]
fn spectral_clustering_agrees_on_two_speakers() {
    use tpt_av_voice_diarize::embedding::{ClassicalEmbedding, SpeakerEmbedder};
    use tpt_av_voice_utils::TimeRange;

    let mut embedder = ClassicalEmbedding;
    let a = embedder.embed(&buzz(150.0, 1.0, 0.4), 16_000).unwrap();
    let b = embedder.embed(&buzz(400.0, 1.0, 0.4), 16_000).unwrap();
    let regions = [
        TimeRange::new(0.0, 1.0),
        TimeRange::new(2.0, 3.0),
        TimeRange::new(4.0, 5.0),
        TimeRange::new(6.0, 7.0),
    ];
    let mut pcm = Vec::new();
    pcm.extend(buzz(150.0, 1.0, 0.4));
    pcm.extend(vec![0.0; 16_000]);
    pcm.extend(buzz(400.0, 1.0, 0.4));
    pcm.extend(vec![0.0; 16_000]);
    pcm.extend(buzz(152.0, 1.0, 0.4));
    pcm.extend(vec![0.0; 16_000]);
    pcm.extend(buzz(402.0, 1.0, 0.4));

    let embeddings: Vec<Vec<f32>> = regions
        .iter()
        .map(|r| {
            let s = (r.start * 16_000.0) as usize;
            let e = (r.end * 16_000.0) as usize;
            embedder.embed(&pcm[s..e], 16_000).unwrap()
        })
        .collect();
    let _ = (a, b);

    let cfg = tpt_av_voice_diarize::ClusteringConfig {
        algorithm: tpt_av_voice_diarize::ClusteringAlgorithm::Spectral,
        num_speakers: Some(2),
        ..Default::default()
    };
    let labels = tpt_av_voice_diarize::cluster(&embeddings, &cfg).unwrap();
    assert_eq!(labels[0], labels[2], "A regions cluster together");
    assert_eq!(labels[1], labels[3], "B regions cluster together");
    assert_ne!(labels[0], labels[1]);
}
