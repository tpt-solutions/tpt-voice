# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- Shared vocabulary types: `Transcript`, `TranscriptSegment`, `SpeakerLabel`, `AudioChunk`, time utilities
- `VoiceError` / `AudioError` error handling
- Minimal WAV I/O with hostile-input handling (malformed/truncated headers rejected, not panicking)
- Shared DSP primitives: windows, FFT glue, mel filter banks, MFCC
- Small pure-Rust neural-network building blocks (matmul, layers)
- Safetensors-compatible weight I/O with checked arithmetic against hostile shape headers
- SHA-256 model manifests, schema-versioned and traversal-safe
- Runtime-dispatched SIMD kernels (AVX2 by default, AVX-512 behind the `avx512` feature, NEON on aarch64) with pure-Rust scalar fallback
- Deterministic synthetic audio fixtures (sine, seeded noise, harmonic stacks, speech-like, alternating-speaker) for tests and benchmarks
