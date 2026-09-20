# tpt-av-voice-tts

Pure-Rust text-to-speech (early stage): rule-based grapheme-to-speech with a
formant synthesizer.

Provides a source-filter formant synthesizer (Klatt-lite: 3 resonators,
fricatives/stops/nasals) driven by rule-based letter-to-sound rules. It
produces intelligible (robotic) speech deterministically, with no external
dependencies or model downloads, and defines the API surface that neural
voices will implement later.

## Usage

```rust,no_run
use tpt_av_voice_tts::{FormantSynth, SynthConfig};

let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
let pcm = synth.synthesize("Hello world").unwrap();
// pcm.samples, pcm.sample_rate ready for playback / WAV export.
```

`VoiceProfile` selects voice characteristics (e.g. `VoiceProfile::default_male()`).

## In the `tpt-voice` workspace

Built on [`tpt-av-voice-utils`](../tpt-av-voice-utils) and
[`tpt-av-voice-align`](../tpt-av-voice-align) (for phoneme/prosody input).
Used by the `tpt-voice` facade to synthesize inserted/replaced words during
text-based editing. See the [workspace README](../README.md) for the full
crate list and how they fit together.

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE)
at your option.
