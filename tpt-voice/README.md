# tpt-voice

Pure-Rust speech processing engine: one-crate facade over transcription,
isolation, diarization, alignment, editing and TTS.

The one-crate facade over the `tpt-av-voice-*` speech engine: every
capability, one dependency. Sub-crates are still published and usable
individually; this crate only re-exports them (`align`, `diarize`,
`isolate`, `transcribe`, `tts`, `utils`) and adds cross-cutting glue.

## Usage

```toml
[dependencies]
tpt-voice = "0.1"
```

```rust,no_run
use tpt_voice::prelude::*;

# let pcm: Vec<f32> = Vec::new();
let mut pipeline = SpeechPipeline::from_model_dir(
    "models/en-small",
    DiarizerConfig::default(),
    AlignerConfig::default(),
)
.unwrap();
let out = pipeline.run(&pcm, 16_000).unwrap();
for seg in &out.transcript.segments {
    let speaker = seg.speaker.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "?".into());
    println!("[{:.2}-{:.2}] {speaker}: {}", seg.start_time, seg.end_time, seg.text);
}
```

`pipeline::apply_text_edit_with_speech` synthesizes audio for inserted words
via the built-in TTS, ready to splice on a `tpt-audio` timeline.
`StreamingPipeline` composes streaming isolation, transcription and
diarization chunk-at-a-time.

## In the `tpt-voice` workspace

This is the top-level facade crate. See the
[workspace README](../README.md) for the full crate list, the underlying
sub-crates, and how they fit into the wider TPT AV stack.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
