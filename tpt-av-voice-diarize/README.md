# tpt-av-voice-diarize

Pure-Rust speaker diarization: voice activity detection, speaker embeddings,
and clustering.

Answers "which speaker is talking at each moment": VAD & segmentation,
speaker embedding extraction (TDNN-style, with a classical MFCC statistics
fallback), and speaker clustering (agglomerative hierarchical or spectral).

## Usage

```rust,no_run
use tpt_av_voice_diarize::{Diarizer, DiarizerConfig};

# let pcm: Vec<f32> = Vec::new();
// Classical mode needs no trained weights:
let mut diarizer = Diarizer::classical(DiarizerConfig::default())?;
let result = diarizer.diarize_samples(&pcm, 16_000)?;

println!("detected {} speaker(s)", result.num_speakers);
for seg in &result.segments {
    println!("{} {}: {:.2} confidence", seg.speaker, seg.time_range(), seg.confidence);
}
# Ok::<(), tpt_av_voice_utils::VoiceError>(())
```

`annotate_transcript` / `diarize_and_annotate` link diarization output to a
`Transcript` so segments carry `speaker` labels. Streaming diarization is
available via `StreamingDiarizer`.

## In the `tpt-voice` workspace

Built on [`tpt-av-voice-utils`](../tpt-av-voice-utils) for shared types, DSP
and weight I/O. See the [workspace README](../README.md) for the full crate
list and how they fit together.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
