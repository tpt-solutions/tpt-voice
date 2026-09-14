# tpt-voice

**A pure-Rust, memory-safe speech processing engine.** Transcription, voice
isolation, diarization, and text-based editing. The intelligence layer of the
TPT AV stack.

**Status:** Early-stage / Pre-1.0
**License:** [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)
**Ecosystem:** [TPT Solutions Open Source](https://opensource.tptsolutions.co.nz/)

[![CI](https://github.com/tpt-solutions/tpt-voice/actions/workflows/ci.yml/badge.svg)](https://github.com/tpt-solutions/tpt-voice/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

---

## Why?

Every existing Rust speech stack (`whisper-rs`, `vosk`, `deepspeech`) is a
binding to C/C++ code: GPL/LGPL licensing contamination, complex build
systems, cross-compilation pain, and no real-time safety guarantees.
`tpt-voice` is **pure Rust end to end** — no PyTorch, no TensorFlow, no
C/C++ bindings. Licensing is enforced by `cargo-deny` in CI.

## Crates

| Crate | Role |
| :--- | :--- |
| [`tpt-av-voice-utils`](tpt-av-voice-utils) | Shared types, DSP/NN kernels, WAV I/O, weight I/O |
| [`tpt-av-voice-transcribe`](tpt-av-voice-transcribe) | Speech-to-text (transformer encoder-decoder, streaming) |
| [`tpt-av-voice-isolate`](tpt-av-voice-isolate) | Real-time voice isolation (STFT masking, U-Net or classical) |
| [`tpt-av-voice-diarize`](tpt-av-voice-diarize) | Speaker diarization (VAD, embeddings, clustering) |
| [`tpt-av-voice-align`](tpt-av-voice-align) | Forced alignment + text-based editing (`AudioEdit`) |
| [`tpt-av-voice-tts`](tpt-av-voice-tts) | Text-to-speech (early stage, formant synthesis) |

Each sub-crate is independently useful — use just the transcriber, just the
isolator, or just the diarizer.

## Quickstart

```toml
[dependencies]
tpt-av-voice-isolate = "0.1"
```

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

End-to-end demos live in [`examples/`](examples):

```text
cargo run -p tpt-voice-examples --example isolate_voice -- input.wav voice.wav
cargo run -p tpt-voice-examples --example diarize_meeting -- meeting.wav
cargo run -p tpt-voice-examples --example text_based_edit -- audio.wav "original text" "edited text"
cargo run -p tpt-voice-examples --example transcribe_audio -- models/en-small audio.wav
```

> **Note on model weights:** the neural architectures (ASR transformer,
> separation U-Net, speaker embeddings) ship without pre-trained weights in
> this early stage. Isolation, diarization, alignment and TTS all have
> classical/deterministic fallbacks that work out of the box. See
> [docs/PUBLISHING.md](docs/PUBLISHING.md) for the weights distribution plan
> and `tpt-av-voice-transcribe` docs for exporting a pipeline-test model.

## Ecosystem

```text
tpt-cadence (decodes audio files → raw PCM f32)
  ↓
tpt-audio (manages timeline, plays audio)
  ↓
tpt-voice (processes PCM → transcript, speaker labels, isolated voices)
  ↓
Application (displays transcript, text-based editing, captions)
```

`tpt-voice` consumes decoded PCM `f32` samples from `tpt-cadence` and
returns [`AudioEdit`](tpt-av-voice-align) operations that `tpt-audio` applies
to timeline clips.

## Design

See [spec.txt](spec.txt) for the full design document: architecture, API
sketches, model architecture, performance targets (RTF < 0.5× transcription
on AVX2), and the dependency/licensing rules.

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The short
version: permissive-licensed dependencies only (no GPL/LGPL/AGPL/MPL), all
ML implemented in pure Rust, conformance tests for model code.

## Publishing

Crate naming, versioning, and the model-weights distribution plan are
documented in [docs/PUBLISHING.md](docs/PUBLISHING.md).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
