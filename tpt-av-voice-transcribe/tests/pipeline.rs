//! End-to-end pipeline tests using a tiny randomly-initialized model.
//! These verify the machinery (shapes, determinism, file round-trips),
//! not transcription accuracy — that requires trained weights.

use std::path::PathBuf;

use tpt_av_voice_transcribe::tokenizer::{BpeTokenizer, Tokenizer};
use tpt_av_voice_transcribe::transcriber::TranscriberConfig;
use tpt_av_voice_transcribe::Transcriber;
use tpt_av_voice_utils::wav;

fn demo_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tpt-voice-it-{tag}-{}", std::process::id()));
    tpt_av_voice_transcribe::weights::write_demo_model(
        &dir,
        "the quick brown fox jumps over the lazy dog silence hello world",
        42,
    )
    .unwrap();
    dir
}

#[test]
fn transcriber_runs_end_to_end_and_is_deterministic() {
    let dir = demo_model_dir("e2e");
    let transcriber = Transcriber::from_model_dir(&dir)
        .unwrap()
        .with_config(TranscriberConfig { max_tokens_per_window: 8, ..Default::default() });

    // 2 s of modulated tone ("speech-like").
    let pcm: Vec<f32> = (0..32_000)
        .map(|i| 0.6 * (i as f32 * 0.01).sin() * (i as f32 * 0.0002).sin())
        .collect();

    let a = transcriber.transcribe_samples(&pcm, 16_000).unwrap();
    let b = transcriber.transcribe_samples(&pcm, 16_000).unwrap();
    assert_eq!(a, b, "identical input → identical transcript");
    assert_eq!(a.language, "en");
    assert!(a.duration_secs() >= 2.0);
    // Untrained weights still produce *some* decode; structure must hold.
    for seg in &a.segments {
        assert!(seg.end_time > seg.start_time);
        assert!((0.0..=1.0).contains(&seg.confidence));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn transcriber_reads_wav_files_and_resamples() {
    let dir = demo_model_dir("wav");
    let transcriber = Transcriber::from_model_dir(&dir)
        .unwrap()
        .with_config(TranscriberConfig { max_tokens_per_window: 8, ..Default::default() });

    let path = std::env::temp_dir().join(format!("tpt-voice-it-{}.wav", std::process::id()));
    let pcm: Vec<f32> = (0..8_000).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
    wav::write_wav_file(&path, &pcm, 8_000, 1).unwrap(); // non-16k rate

    let t = transcriber.transcribe_file(&path).unwrap();
    assert_eq!(t.language, "en");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tokenizer_round_trips_through_demo_model() {
    let dir = demo_model_dir("tok");
    let json = std::fs::read_to_string(dir.join("tokenizer.json")).unwrap();
    let tok = BpeTokenizer::from_json(&json).unwrap();
    let ids = tok.encode("the quick brown fox");
    assert_eq!(tok.decode(&ids), "the quick brown fox");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn streaming_finalizes_silence_bounded_utterances() {
    let dir = demo_model_dir("stream");
    let transcriber = Transcriber::from_model_dir(&dir)
        .unwrap()
        .with_config(TranscriberConfig { max_tokens_per_window: 6, ..Default::default() });
    let mut stream = tpt_av_voice_transcribe::StreamingTranscriber::new(transcriber);

    let mut pcm = Vec::new();
    pcm.extend((0..16_000).map(|i| (i as f32 * 0.02).sin() * 0.5));
    pcm.extend(std::iter::repeat(0.0).take(10_000));
    pcm.extend((0..16_000).map(|i| (i as f32 * 0.02).sin() * 0.5));
    pcm.extend(std::iter::repeat(0.0).take(10_000));

    let chunks = tpt_av_voice_utils::audio_chunk::chunk_pcm(pcm, 16_000, 8_000);
    let count = stream.process(chunks).filter_map(Result::ok).count();
    assert!(count >= 1, "silence-bounded audio should finalize utterances");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn custom_decode_config_is_respected() {
    let dir = demo_model_dir("cfg");
    let transcriber = Transcriber::from_model_dir(&dir)
        .unwrap()
        .with_config(TranscriberConfig { min_confidence: 2.0, ..Default::default() });
    let pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.02).sin()).collect();
    let t = transcriber.transcribe_samples(&pcm, 16_000).unwrap();
    // Impossible confidence threshold filters every segment out.
    assert!(t.segments.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
