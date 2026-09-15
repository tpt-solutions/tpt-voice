//! Identify "who spoke when" in a meeting recording.
//!
//! Usage:
//! ```text
//! cargo run -p tpt-voice-examples --example diarize_meeting -- [meeting.wav]
//! ```
//!
//! Runs the weight-free pipeline (energy VAD → MFCC+pitch embeddings →
//! AHC clustering). With no input, a synthetic two-speaker meeting is
//! generated: low-pitch speaker, pause, high-pitch speaker, pause,
//! low-pitch speaker again.

use tpt_av_voice_diarize::{annotate_transcript, Diarizer, DiarizerConfig};
use tpt_av_voice_utils::{wav, Transcript, TranscriptSegment};

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (samples, sample_rate, source) = match args.first() {
        Some(path) => {
            let wav = wav::read_wav_file(path).expect("read wav");
            (wav.to_mono(), wav.sample_rate, path.clone())
        }
        None => {
            let path = std::env::temp_dir().join("tpt-voice-demo-meeting.wav");
            let mut pcm = Vec::new();
            pcm.extend(buzz(150.0, 1.5, 0.4));
            pcm.extend(vec![0.0; 12_000]);
            pcm.extend(buzz(400.0, 1.5, 0.4));
            pcm.extend(vec![0.0; 12_000]);
            pcm.extend(buzz(152.0, 1.5, 0.4));
            wav::write_wav_file(&path, &pcm, 16_000, 1).expect("write wav");
            (pcm, 16_000, path.display().to_string())
        }
    };

    println!(
        "diarizing {source} ({} samples @ {sample_rate} Hz)...",
        samples.len()
    );
    let mut diarizer = Diarizer::classical(DiarizerConfig::default()).expect("diarizer");
    let result = diarizer
        .diarize_samples(&samples, sample_rate)
        .expect("diarization");

    println!("detected {} speaker(s):", result.num_speakers);
    for seg in &result.segments {
        println!(
            "  {} {}: {:.2} confidence",
            seg.speaker,
            seg.time_range(),
            seg.confidence
        );
    }

    // Show the transcript-integration hook: label transcript segments
    // by the overlapping speaker.
    let mut transcript = Transcript::new("en");
    for seg in &result.segments {
        transcript.push_segment(TranscriptSegment::new(
            seg.start_time,
            seg.end_time,
            format!("<speech by {}>", seg.speaker),
            seg.confidence,
        ));
    }
    let annotated = annotate_transcript(&mut transcript, &result).expect("annotation");
    println!("annotated {annotated} transcript segments with speaker labels:");
    for seg in &transcript.segments {
        let speaker = seg
            .speaker
            .as_ref()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "?".to_string());
        println!("  [{}] {speaker}: {}", seg.time_range(), seg.text);
    }
}
