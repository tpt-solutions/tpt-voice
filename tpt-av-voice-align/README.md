# tpt-av-voice-align

Pure-Rust text-audio alignment and text-based editing: forced alignment,
word timestamps, `AudioEdit` operations.

Maps transcript text onto audio time ranges: CTC forced alignment, rule-based
English grapheme-to-phoneme conversion, word-level alignment, and a
`TextBasedEditor` that diffs an edited transcript against the original and
emits `AudioEdit` operations consumable by a `tpt-audio` timeline.

## Usage

```rust,no_run
use tpt_av_voice_align::{Aligner, AlignerConfig, TextBasedEditor};

# let pcm: Vec<f32> = Vec::new();
let mut aligner = Aligner::new(AlignerConfig::default())?;
let aligned = aligner.align_samples(&pcm, 16_000, "hello brave new world")?;
for w in &aligned.words {
    println!("{} {}", w.word, w.start_time);
}

let mut editor = TextBasedEditor::new(aligned);
let edits = editor.apply_edits("hello brave old world");
for op in &edits {
    println!("{:?} [{} - {}]", op.edit_type, op.start_time, op.end_time);
}
# Ok::<(), tpt_av_voice_utils::VoiceError>(())
```

The energy-proportional alignment mode works with no trained weights;
`align_with_emissions` accepts CTC emissions from a trained acoustic model.

## In the `tpt-voice` workspace

Built on [`tpt-av-voice-utils`](../tpt-av-voice-utils) for shared types. See
the [workspace README](../README.md) for the full crate list and how they
fit together — in particular the `Application → tpt-audio` data flow that
consumes `AudioEdit`.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
