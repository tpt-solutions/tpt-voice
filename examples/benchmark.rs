//! Real-time factor (RTF) benchmarks against the spec §6 targets.
//!
//! Usage:
//! ```text
//! cargo run -p tpt-voice-examples --example benchmark -- [model-dir]
//! ```
//!
//! RTF = processing time ÷ audio duration. Lower is better; spec
//! targets: transcription < 0.5×, isolation < 1.0×, diarization < 0.3×
//! (all CPU). Run a release build for meaningful numbers:
//!
//! ```text
//! cargo run --release -p tpt-voice-examples --example benchmark
//! ```

use std::time::Instant;

use tpt_av_voice_diarize::{Diarizer, DiarizerConfig};
use tpt_av_voice_isolate::{IsolatorConfig, VoiceIsolator};
use tpt_av_voice_tts::{FormantSynth, SynthConfig};
use tpt_av_voice_utils::signals;
use tpt_av_voice_utils::time::TARGET_SAMPLE_RATE;

/// Runs `f` once on a warmed copy, then times it and prints the RTF.
fn measure(name: &str, target: Option<f64>, audio_secs: f64, mut f: impl FnMut()) {
    f(); // warm-up (FFT plans, caches, page-ins)
    let start = Instant::now();
    f();
    let elapsed = start.elapsed().as_secs_f64();
    let rtf = elapsed / audio_secs;
    let verdict = match target {
        Some(t) if rtf <= t => "PASS",
        Some(_) => "MISS",
        None => "  — ",
    };
    let target_str = target
        .map(|t| format!("< {t:.1}×"))
        .unwrap_or_else(|| "n/a".into());
    println!(
        "{name:<28} {rtf:>7.3}×   target {target_str:<8} [{verdict}]  ({elapsed:.3}s for {audio_secs:.1}s audio)"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let secs = 10.0;
    println!(
        "tpt-voice RTF benchmark — {secs:.0}s of synthetic audio @ {TARGET_SAMPLE_RATE} Hz (dev-profile numbers are pessimistic; use --release)"
    );
    println!("{}", "-".repeat(92));

    let noisy: Vec<f32> = {
        let voice = signals::speech_like(180.0, secs, TARGET_SAMPLE_RATE, 0.4);
        let noise = signals::white_noise(secs, TARGET_SAMPLE_RATE, 0.05, 42);
        voice.iter().zip(noise).map(|(v, n)| v + n).collect()
    };

    measure("isolate (classical)", Some(1.0), secs, || {
        let mut iso = VoiceIsolator::classical(IsolatorConfig::default()).unwrap();
        let _ = iso.isolate_samples(&noisy, TARGET_SAMPLE_RATE).unwrap();
    });

    let meeting: Vec<f32> = signals::alternating_speakers(
        &[
            (150.0, secs / 4.0),
            (400.0, secs / 4.0),
            (155.0, secs / 4.0),
            (410.0, secs / 4.0),
        ],
        0.5,
        TARGET_SAMPLE_RATE,
        0.4,
    );
    let meeting_secs = meeting.len() as f64 / f64::from(TARGET_SAMPLE_RATE);
    measure("diarize (classical)", Some(0.3), meeting_secs, || {
        let mut d = Diarizer::classical(DiarizerConfig::default()).unwrap();
        let _ = d.diarize_samples(&meeting, TARGET_SAMPLE_RATE).unwrap();
    });

    let text = "hello world this is the tpt voice benchmark running on synthetic audio ";
    measure("tts (formant)", None, secs, || {
        let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
        let _ = synth.synthesize(&text.repeat(3)).unwrap();
    });

    // Transcription needs a model directory; the demo model works.
    let model_dir = args
        .first()
        .cloned()
        .unwrap_or_else(|| "target/demo-model".to_string());
    if !std::path::Path::new(&model_dir)
        .join("model.safetensors")
        .exists()
    {
        println!(
            "{:<28} skipped — no model at {model_dir:?} (run transcribe_audio --make-demo-model first)"
        , "transcribe");
    } else {
        let transcriber = tpt_av_voice_transcribe::Transcriber::from_model_dir(&model_dir).unwrap();
        measure("transcribe (demo model)", Some(0.5), secs, || {
            let _ = transcriber
                .transcribe_samples(&noisy, TARGET_SAMPLE_RATE)
                .unwrap();
        });
    }
}
