# Publishing plan (crates.io)

## Namespace reservation

Publish a placeholder of every `tpt-av-voice-*` crate name as soon as the
crate compiles, so the namespace is squatted before the 1.0 release:

| Crate | Role |
| :--- | :--- |
| `tpt-av-voice-utils` | shared types/kernels |
| `tpt-av-voice-transcribe` | ASR |
| `tpt-av-voice-isolate` | voice separation |
| `tpt-av-voice-diarize` | speaker diarization |
| `tpt-av-voice-align` | alignment + text-based editing |
| `tpt-av-voice-tts` | text-to-speech |

All crates share `version.workspace = true` and are released together in
lockstep (0.x while pre-1.0; breaking changes bump the minor, fixes bump the
patch, per Cargo's 0.x semantics).

## What gets published

- **Code crates:** published to crates.io from CI (tag-triggered
  `cargo publish` in dependency order: utils → transcribe/isolate/diarize/
  align/tts → nothing else; `examples/` is `publish = false`).
- **Model weights:** *never* published to crates.io. Weights are
  version-checked artifacts distributed via GitHub Releases
  (`https://github.com/tpt-solutions/tpt-voice/releases/tag/models-vN`),
  described by a `manifest.json` (SHA-256, format version, languages,
  model size class). Loaders accept any `safetensors`-compatible file
  written by `tpt_av_voice_utils::safetensors`.
- **Tokenizer data:** travels inside the model directory
  (`tokenizer.json` + `merges.txt`), same release channel as weights.

## Version/compatibility contract

- Weight format carries an integer `format_version`; loaders must accept
  all older versions within the same 0.x line.
- Until 1.0, MSRV and dependency bumps are minor releases.
- CI publishes only from tagged commits on `master`, after fmt/clippy/test/
  cargo-deny all pass.
