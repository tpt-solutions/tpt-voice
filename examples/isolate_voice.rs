//! Isolate the voice from background noise in an audio file.
//!
//! Usage:
//! ```text
//! cargo run -p tpt-voice-examples --example isolate_voice -- [input.wav] [voice-out.wav]
//! ```
//!
//! Runs in classical spectral-subtraction mode (no trained weights
//! needed). With no input, a noisy synthetic clip is generated: a
//! speech-like tone buried in wideband noise.

use tpt_av_voice_isolate::{IsolatorConfig, VoiceIsolator};
use tpt_av_voice_utils::nn::Rng;
use tpt_av_voice_utils::wav;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let input = args.first().cloned();
    let voice_out = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "voice.wav".to_string());
    let noise_out = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "noise.wav".to_string());

    let (samples, sample_rate, source) = match input {
        Some(path) => {
            let wav = wav::read_wav_file(&path).expect("read wav");
            (wav.to_mono(), wav.sample_rate, path)
        }
        None => {
            let path = std::env::temp_dir().join("tpt-voice-demo-noisy.wav");
            // "Voice": 300 Hz harmonic stack with vibrato.
            let mut rng = Rng::new(7);
            let pcm: Vec<f32> = (0..32_000)
                .map(|i| {
                    let t = i as f32 / 16_000.0;
                    let vibrato = 1.0 + 0.02 * (std::f32::consts::TAU * 5.0 * t).sin();
                    let mut v = 0.0f32;
                    for k in 1..=6 {
                        v += (std::f32::consts::TAU * 300.0 * k as f32 * vibrato * t).sin()
                            / k as f32;
                    }
                    v * 0.25 + rng.uniform_range(-1.0, 1.0) * 0.05 // + noise floor
                })
                .collect();
            wav::write_wav_file(&path, &pcm, 16_000, 1).expect("write wav");
            (pcm, 16_000, path.display().to_string())
        }
    };

    println!(
        "isolating voice from {source} ({} samples)...",
        samples.len()
    );
    let mut isolator =
        VoiceIsolator::classical(IsolatorConfig::default()).expect("classical isolator");
    let isolated = isolator
        .isolate_samples(&samples, sample_rate)
        .expect("isolation");

    let in_rms = tpt_av_voice_utils::dsp::rms(&samples);
    let voice_rms = tpt_av_voice_utils::dsp::rms(&isolated.voice);
    println!(
        "input rms {in_rms:.4} → voice rms {voice_rms:.4} (ratio {:.2})",
        voice_rms / in_rms.max(1e-9)
    );

    wav::write_wav_file(&voice_out, &isolated.voice, isolated.sample_rate, 1).expect("write voice");
    wav::write_wav_file(&noise_out, &isolated.noise, isolated.sample_rate, 1).expect("write noise");
    println!("wrote {voice_out} (isolated voice) and {noise_out} (residual noise)");
    println!(
        "tip: use a trained separation model via VoiceIsolator::from_model_dir for real separation"
    );
}
