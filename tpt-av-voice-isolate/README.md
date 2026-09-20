# tpt-av-voice-isolate

Pure-Rust real-time voice isolation: STFT masking with neural or classical
spectral subtraction.

Separates human speech from background noise: STFT / inverse STFT,
time-frequency masking, a U-Net separation model, or classical spectral
subtraction when no trained weights are available.

Pipeline: `audio → STFT → mask prediction → masked STFT → ISTFT → voice`.

## Usage

```rust,no_run
use tpt_av_voice_isolate::{VoiceIsolator, IsolatorConfig, SourceType};

// Classical mode needs no trained weights:
let mut isolator = VoiceIsolator::classical(IsolatorConfig::default())?;
let pcm = tpt_av_voice_utils::wav::read_wav_file("noisy.wav")?;
let isolated = isolator.isolate_samples(&pcm.samples, pcm.sample_rate)?;

// Isolated voice and residual noise, as f32 PCM:
tpt_av_voice_utils::wav::write_wav_file("voice.wav", &isolated.voice, isolated.sample_rate)?;
# Ok::<(), tpt_av_voice_utils::VoiceError>(())
```

Real-time streaming isolation with overlap-add is available via
`StreamingIsolator`.

## In the `tpt-voice` workspace

Built on [`tpt-av-voice-utils`](../tpt-av-voice-utils) for shared types, DSP
and weight I/O. See the [workspace README](../README.md) for the full crate
list and how they fit together.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
