# tpt-voice — Project Todo

Tracks all work for `tpt-voice`, the pure-Rust speech processing engine (TPT Solutions).
License: dual **MIT OR Apache-2.0**.

---

## Phase 0: Project Setup & Infrastructure

- [x] Initialize git repository, add `.gitignore` (Rust/Cargo template)
- [x] Create workspace `Cargo.toml` (members, `workspace.package`, `workspace.dependencies` per spec.txt §9)
- [x] Create `LICENSE-MIT` and `LICENSE-APACHE` (dual license)
- [x] Set `license = "MIT OR Apache-2.0"` in workspace `Cargo.toml`
- [x] Update `spec.txt` / `DESIGN.md` license references from MIT-only to dual MIT/Apache-2.0
- [x] Create `deny.toml` (cargo-deny license enforcement per spec.txt §8)
- [x] Scaffold `tpt-av-voice-utils` crate (`Cargo.toml` + `src/lib.rs`)
- [x] Scaffold `tpt-av-voice-transcribe` crate (`Cargo.toml` + `src/lib.rs`)
- [x] Scaffold `tpt-av-voice-isolate` crate (`Cargo.toml` + `src/lib.rs`)
- [x] Scaffold `tpt-av-voice-diarize` crate (`Cargo.toml` + `src/lib.rs`)
- [x] Scaffold `tpt-av-voice-align` crate (`Cargo.toml` + `src/lib.rs`)
- [x] Scaffold `tpt-av-voice-tts` crate (`Cargo.toml` + `src/lib.rs`)
- [x] Create `README.md` (project overview, badges, quickstart)
- [x] Create `CONTRIBUTING.md` (contribution rules from spec.txt §10)
- [x] Set up CI (GitHub Actions): build, test, clippy, fmt check
- [x] Add `cargo-deny check` to CI
- [x] Create `examples/` directory with stub files (`transcribe_audio.rs`, `isolate_voice.rs`, `diarize_meeting.rs`, `text_based_edit.rs`)
- [x] Decide & document crates.io publishing plan (namespace reservation for `tpt-av-voice-*`) — see `docs/PUBLISHING.md`
  - Note: `tpt-cadence` is intentionally not a workspace dependency yet (external repo not
    resolvable); the integration point is documented in the spec/README data flow.

---

## Phase 1: Foundation & Transcription

- [x] `tpt-av-voice-utils`: `Transcript` / `TranscriptSegment` types (`transcript.rs`)
- [x] `tpt-av-voice-utils`: `SpeakerLabel` type (`speaker.rs`)
- [x] `tpt-av-voice-utils`: `AudioChunk` streaming type (`audio_chunk.rs`)
- [x] `tpt-av-voice-utils`: time alignment types (`time.rs`)
- [x] `tpt-av-voice-utils`: `VoiceError` enum (`error.rs`)
- [x] `tpt-av-voice-transcribe`: mel spectrogram feature extraction (`encoder.rs`)
- [x] `tpt-av-voice-transcribe`: transformer encoder-decoder ASR model architecture (`model.rs`)
- [x] `tpt-av-voice-transcribe`: text decoding — beam search / CTC (`decoder.rs`)
- [x] `tpt-av-voice-transcribe`: tokenizer — BPE/WordPiece (`tokenizer.rs`) — byte-level BPE with trainer
- [x] `tpt-av-voice-transcribe`: model weight loading — custom binary format or safetensors (`weights.rs`) — safetensors-compatible
- [x] `tpt-av-voice-transcribe`: high-level `Transcriber` API (`transcriber.rs`)
- [x] `tpt-av-voice-transcribe`: streaming transcription support (`streaming.rs`)
- [ ] Source or train pre-trained English small-model weights
  - **Blocked on training infrastructure.** The architecture is complete and a
    `write_demo_model` export exists for pipeline smoke tests; `examples/transcribe_audio`
    prints instructions. Real accuracy requires a trained checkpoint (safetensors format
    is ready for it).
- [ ] Conformance tests against LibriSpeech dataset
  - **Harness added** (`tests/conformance.rs`, WER via word-level Levenshtein, env-gated on
    `TPT_VOICE_CONFORMANCE_MODEL` + `TPT_VOICE_CONFORMANCE_DATA`, skips cleanly when absent).
    Executing it needs the dataset and a trained model (above).
- [x] `examples/transcribe_audio.rs` working end-to-end
  - `--make-demo-model` exports a tiny pipeline-test model; transcripts from untrained
    weights are gibberish by construction (documented in the example output).

---

## Phase 2: Voice Isolation

- [x] `tpt-av-voice-isolate`: STFT implementation (`stft.rs`)
- [x] `tpt-av-voice-isolate`: inverse STFT (`istft.rs`) — weighted overlap-add, round-trips
- [x] `tpt-av-voice-isolate`: time-frequency masking (`mask.rs`)
- [x] `tpt-av-voice-isolate`: U-Net/transformer separation model (`model.rs`) — U-Net-lite with skips
- [x] `tpt-av-voice-isolate`: high-level `VoiceIsolator` API — `isolate_file` (`isolator.rs`)
  - Ships a weight-free classical spectral-subtraction mode so it works out of the box.
- [x] `tpt-av-voice-isolate`: real-time streaming isolation with overlap-add (`streaming.rs`)
- [ ] Source or train pre-trained separation model weights — same blocker as Phase 1 weights
- [x] `examples/isolate_voice.rs` working end-to-end

---

## Phase 3: Diarization

- [x] `tpt-av-voice-diarize`: voice activity detection & segmentation (`segmentation.rs`)
- [x] `tpt-av-voice-diarize`: speaker embedding extraction — TDNN/ECAPA-TDNN (`embedding.rs`)
  - TDNN-style network + a weight-free classical embedding (MFCC stats + autocorrelation F0)
    so diarization works without trained weights.
- [x] `tpt-av-voice-diarize`: speaker clustering — spectral clustering / AHC (`clustering.rs`)
- [x] `tpt-av-voice-diarize`: high-level `Diarizer` API — `diarize_file` (`diarizer.rs`)
- [x] `tpt-av-voice-diarize`: streaming diarization support (`streaming.rs`) — online centroids
  with two-window hysteresis for new-speaker confirmation
- [x] Multi-speaker transcription integration (link diarization output to `Transcript.segments[].speaker`) — `integration.rs`
- [x] `examples/diarize_meeting.rs` working end-to-end

---

## Phase 4: Text-Based Editing

- [x] `tpt-av-voice-align`: forced alignment — CTC-based (`forced_align.rs`)
- [x] `tpt-av-voice-align`: phoneme-level alignment (`phoneme.rs`) — rule-based English G2P
- [x] `tpt-av-voice-align`: word-level alignment (`word_align.rs`)
- [x] `tpt-av-voice-align`: high-level `Aligner` API (`aligner.rs`)
  - Energy-proportional mode works with no weights; `align_with_emissions` accepts
    CTC emissions from a trained acoustic model.
- [x] `TextBasedEditor`: diff original vs. edited transcript — word-level LCS
- [x] `TextBasedEditor`: map text changes to `AudioEdit` operations — Delete/Insert/Replace
- [x] Integration with `tpt-audio` timeline (consume `AudioEdit` to mutate timeline clips)
  - The `AudioEdit` contract + consumption docs live in `align/src/editor.rs`; the
    tpt-audio-side mutation belongs to the tpt-audio repository.
- [x] `examples/text_based_edit.rs` working end-to-end (synthesizes the original via TTS when no file is given)

---

## Phase 5: Advanced Features

- [x] Multi-language support (Spanish, French, German, etc.) — `Language` type, prompt/config
  plumbing, open extension via `Language::other`; per-language acoustic resources ship with
  trained models
- [x] Larger, more accurate ASR models — `ModelSize` presets Tiny→Large (accuracy depends on
  trained weights)
- [ ] GPU-accelerated inference via `wgpu` — transformer attention matmul
  - **Deferred.** The CPU path (SIMD dispatch below) meets the Phase-1 RTF targets for the
    small models; the wgpu backend is planned as a feature-gated module so default builds
    stay dependency-light.
- [ ] GPU-accelerated inference via `wgpu` — STFT/ISTFT FFT
- [ ] GPU-accelerated inference via `wgpu` — convolution
- [x] SIMD optimization pass — AVX2/AVX-512 via `std::arch` (x86_64) — runtime-dispatched
  (`utils/src/simd.rs`); AVX-512 behind the `avx512` cargo feature, requires Rust ≥ 1.89
- [x] SIMD optimization pass — NEON via `std::arch` (ARM) — aarch64 baseline; compile-checked
  via `cargo check --target aarch64-pc-windows-msvc`
- [x] Pure Rust scalar fallback path for all SIMD code — reference implementations with
  parity tests against the vector paths
- [x] `tpt-av-voice-tts` (text-to-speech) — initial design — API surface + `VoiceProfile`s +
  prosody model (`g2p.rs` sentences with terminal contours)
- [x] `tpt-av-voice-tts` (text-to-speech) — initial implementation — Klatt-lite formant
  synthesizer (source-filter, 3 resonators, fricatives/stops/nasals), intelligible robotic
  output, deterministic

---

## Phase 6: Review, Hardening, Adoption & Security (review pass)

### Review & hardening

- [x] Audit codebase for stubs / `todo!()` / dead ends — none found; `VoiceError::NotImplemented`
  remains as an API surface variant (unused by current code)
- [x] Fix `wav::write_wav` silent truncation for outputs near the 32-bit RIFF limit
  (now `AudioError::InvalidFormat`)
- [x] Fix safetensors reader overflow on hostile shape headers (checked element counts —
  error, not debug-panic/release-wraparound)
- [x] Panic-free model loading: every loader (ASR transformer, separation U-Net, TDNN
  embeddings) now cross-validates weights/biases/layernorms/embedding dimensions and
  returns `VoiceError::Model` on corrupt checkpoints — regression tests included
  (`corrupt_model_file_errors_instead_of_panic`, `hostile_shape_rejected_without_panic`)

### Security audit

- [x] RustSec advisory scan: `cargo deny check advisories` + `cargo audit` — **clean**
  (no known vulnerabilities in the 30-crate dependency tree)
- [x] `docs/SECURITY.md`: hostile-input threat model, `unsafe` inventory (SIMD kernels
  only, runtime-detected, with invariants), supply-chain policy, reporting process
- [x] Fuzz targets for the hostile-input parsers (`fuzz/`: `wav_parse`,
  `safetensors_parse`, `tokenizer_json`) + scheduled fuzz workflow (`.github/workflows/fuzz.yml`)

### Adoption & ergonomics

- [x] **`tpt-voice` facade crate** — one dependency for everything: `prelude` module and
  re-exports of all six engine crates
- [x] `pipeline::SpeechPipeline` — transcribe → diarize → speaker-annotate → word-align
  in one call (`PipelineOutput`)
- [x] `pipeline::apply_text_edit_with_speech` / `fill_insertions` — `AudioEdit` gains a
  `words` field and inserted/replaced text is synthesized via the TTS engine into
  `new_audio`, ready to splice on a `tpt-audio` timeline
- [x] Convenience re-exports in every sub-crate (e.g. `tpt_av_voice_transcribe::Transcript`)
  so users need fewer direct dependencies
- [x] `utils::signals` — deterministic synthetic audio fixtures (sine, seeded noise,
  harmonic stacks, speech-like, alternating-speaker meeting pattern)
- [x] `examples/benchmark.rs` — RTF benchmark vs spec §6 targets
  (isolation < 1.0×, diarization < 0.3×, transcription < 0.5×); all pass in dev builds
- [x] Model integrity: `utils::manifest` (SHA-256 via `sha2`, schema-versioned,
  traversal-safe) — demo model export pins a manifest, `Transcriber::from_model_dir`
  verifies it automatically when present

### Tooling & CI

- [x] WASM support verified: `cargo check --target wasm32-unknown-unknown` is clean for
  all library crates (browser-capable); CI job added
- [x] CI: license + bans + advisories audit, WASM job, MSRV 1.75 job, weekly fuzz workflow

### Follow-ups (tracked, not started)

- [ ] Continuous fuzzing with crash corpora committed as regression tests
- [ ] Criterion-based benches for kernel-level microbenchmarks (matmul, FFT, mel)
- [ ] Streaming pipeline composition (isolate → stream-transcribe → online diarize)
- [ ] `wgpu` GPU backend (see Phase 5 deferral)
- [ ] Trained model weights + LibriSpeech/VoxCeleb conformance runs (Phase 1/2 blockers)
- [ ] docs.rs-friendly feature flags (`--no-default-features` docs pass) + crate
  categories/README doc scraping on publish

