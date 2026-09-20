# Contributing to tpt-voice

This project does not accept pull requests. Please open a
[GitHub issue](../../issues) for bug reports, feature requests, or questions
instead.

The rules below describe the standards the codebase itself is held to; they
are provided for context, not as a PR checklist.

## Non-negotiable rules

1. **Permissive dependencies only.** Do not introduce any dependency whose
   license is GPL, LGPL, AGPL, or MPL — including transitive dependencies.
   `cargo-deny` enforces this (`deny.toml`); CI will fail on violations.
   Allowed licenses: MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib, Unicode-3.0.
2. **Pure Rust.** All ML models must be implemented in pure Rust. No C/C++
   bindings, no PyTorch/TensorFlow/onnxruntime, no `build.rs` that compiles
   external native code.
3. **Conformance tests.** Model code must include conformance tests against
   standard datasets (LibriSpeech, VoxCeleb, …). Dataset-gated tests should
   activate via an environment variable and skip cleanly when the data is
   absent (see `tpt-av-voice-transcribe/tests/conformance.rs`).
4. **No trained-weight blobs in git.** Weights are distributed via GitHub
   releases (see `docs/PUBLISHING.md`), not committed to the repository.

## Development workflow

```bash
cargo fmt --all                  # format (CI runs --check)
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace           # unit + integration tests
cargo deny check licenses bans   # license audit
```

## Code style

- Rust edition 2021, MSRV 1.75. Do not use features requiring a newer
  compiler in library code (dev-only code may target CI's stable toolchain).
- All heavy numerical kernels need a portable scalar fallback. SIMD paths
  (`std::arch` AVX2/AVX-512/NEON) must be runtime-dispatched and must have
  parity tests against the scalar path.
- Public APIs take/return `f32` PCM at explicit sample rates and timestamps
  in `f64` seconds. Streaming APIs consume `AudioChunk`s.
- Errors go through `tpt-av-voice-utils::VoiceError`; no panics on invalid
  input (return errors), panics only for programmer errors.

## Commit messages

Plain, imperative subject lines ("Add AHC clustering parity test"), body
explaining *why* when non-obvious.
