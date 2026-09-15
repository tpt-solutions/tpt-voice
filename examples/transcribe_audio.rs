//! Transcribe an audio file end-to-end.
//!
//! Usage:
//! ```text
//! cargo run -p tpt-voice-examples --example transcribe_audio -- --make-demo-model <dir>
//! cargo run -p tpt-voice-examples --example transcribe_audio -- <model-dir> [audio.wav]
//! ```
//!
//! `--make-demo-model` exports a tiny, randomly initialized model that
//! exercises the full pipeline. Transcripts from untrained weights are
//! gibberish by construction — real accuracy needs trained weights
//! (see docs/PUBLISHING.md). With no audio argument a synthetic WAV is
//! generated so the demo runs anywhere.

use tpt_av_voice_transcribe::Transcriber;
use tpt_av_voice_utils::wav;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: transcribe_audio --make-demo-model <dir>");
        eprintln!("       transcribe_audio <model-dir> [audio.wav]");
        std::process::exit(2);
    }

    if args[0] == "--make-demo-model" {
        let dir = args.get(1).map(String::as_str).unwrap_or("models/demo");
        let corpus = "the quick brown fox jumps over the lazy dog hello world \
                      silence speech recognition is fun the dog runs fast";
        let config = tpt_av_voice_transcribe::weights::write_demo_model(dir, corpus, 42)
            .expect("demo model export");
        println!("wrote demo model to {dir} (n_vocab = {})", config.n_vocab);
        println!("note: weights are randomly initialized — transcripts will be gibberish");
        return;
    }

    let model_dir = &args[0];
    let transcriber = Transcriber::from_model_dir(model_dir)
        .unwrap_or_else(|e| panic!("failed to load model from {model_dir}: {e}"));

    // Load the given file or synthesize a demo clip.
    let (samples, sample_rate, source) = match args.get(1) {
        Some(path) => {
            let wav = wav::read_wav_file(path).expect("read wav");
            (wav.to_mono(), wav.sample_rate, path.clone())
        }
        None => {
            let path = std::env::temp_dir().join("tpt-voice-demo-speech.wav");
            // 2 s of speech-like modulated tone.
            let pcm: Vec<f32> = (0..32_000)
                .map(|i| 0.6 * (i as f32 * 0.01).sin() * (i as f32 * 0.0003).sin())
                .collect();
            wav::write_wav_file(&path, &pcm, 16_000, 1).expect("write wav");
            (pcm, 16_000, path.display().to_string())
        }
    };

    println!(
        "transcribing {source} ({} samples @ {sample_rate} Hz)...",
        samples.len()
    );
    let transcript = transcriber
        .transcribe_samples(&samples, sample_rate)
        .expect("transcription");
    println!(
        "language: {}  confidence: {:.2}",
        transcript.language, transcript.confidence
    );
    for seg in &transcript.segments {
        println!(
            "  [{}] {}",
            tpt_av_voice_utils::time::TimeRange::new(seg.start_time, seg.end_time),
            seg.text
        );
    }
    println!("full text: {}", transcript.full_text);
}
