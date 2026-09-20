# tpt-av-voice-utils

Shared types, DSP/NN kernels, WAV I/O and error handling for the `tpt-voice`
speech processing engine.

Pure Rust: no C/C++ bindings, no ML runtime dependencies.

## What's in here

- Common vocabulary types: `Transcript`, `TranscriptSegment`, `SpeakerLabel`, `AudioChunk`
- `VoiceError` / `AudioError` error handling
- Time utilities (`TimeRange`, `Timestamp`)
- Minimal WAV I/O (`wav`)
- Shared DSP primitives: windows, FFT glue, mel filter banks, MFCC (`dsp`)
- Small pure-Rust neural-network building blocks (`nn`)
- Safetensors-compatible weight I/O (`safetensors`)
- SHA-256 model manifests (`manifest`)
- Runtime-dispatched SIMD kernels, AVX2/AVX-512/NEON with scalar fallback (`simd`)
- Deterministic synthetic audio fixtures for tests/benches (`signals`)

## Usage

```rust,no_run
let pcm = tpt_av_voice_utils::wav::read_wav_file("input.wav")?;
println!("{} samples @ {} Hz", pcm.samples.len(), pcm.sample_rate);
# Ok::<(), tpt_av_voice_utils::VoiceError>(())
```

## In the `tpt-voice` workspace

This is the foundation crate that every other `tpt-av-voice-*` crate depends
on. It is not usually depended on directly — see the
[workspace README](../README.md) for the full crate list and how they fit
together.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
