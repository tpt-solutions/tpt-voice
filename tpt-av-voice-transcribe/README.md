# tpt-av-voice-transcribe

Pure-Rust speech-to-text: transformer encoder-decoder ASR with streaming
support.

Implements a Whisper-style transformer encoder-decoder architecture entirely
in Rust: mel spectrogram feature extraction, the encoder-decoder model,
greedy/beam-search/CTC decoding, byte-level BPE tokenization, and
safetensors-compatible weight loading.

## Usage

```rust,no_run
use tpt_av_voice_transcribe::Transcriber;

# let pcm: Vec<f32> = Vec::new();
let transcriber = Transcriber::from_model_dir("models/en-small").unwrap();
let transcript = transcriber.transcribe_samples(&pcm, 16_000).unwrap();
println!("{}", transcript.full_text);
```

Streaming transcription is available via `StreamingTranscriber`.

> **Note:** the ASR transformer architecture is complete but ships without
> pre-trained weights at this stage; see the workspace
> [`docs/PUBLISHING.md`](../docs/PUBLISHING.md) for the weights plan. Use
> `write_model_weights` / the `--make-demo-model` flag on
> `examples/transcribe_audio` to export a tiny pipeline-test model.

## In the `tpt-voice` workspace

Built on [`tpt-av-voice-utils`](../tpt-av-voice-utils) for shared types, DSP
and weight I/O. See the [workspace README](../README.md) for the full crate
list and how they fit together.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
