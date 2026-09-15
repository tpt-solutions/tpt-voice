# Security Policy

This document describes tpt-voice's security posture: what is trusted,
what is treated as hostile, where the `unsafe` code lives, and how to
report vulnerabilities.

## Reporting a vulnerability

Please use [GitHub private security advisories]
(https://github.com/tpt-solutions/tpt-voice/security/advisories) rather
than a public issue. We aim to acknowledge reports within one week.

## Threat model

| Input | Trust level | Handling |
| :--- | :--- | :--- |
| WAV / PCM bytes | **Hostile** | Pure-Rust RIFF parser validates magic, chunk bounds, word alignment, and encoding tags; truncated/hostile input yields `VoiceError`, never a panic. Writes refuse inputs that would overflow the 32-bit RIFF size. |
| Model weight files (safetensors) | **Hostile** | Header length, JSON shape, dtype, and per-tensor byte ranges are validated with checked arithmetic (a shape like `[usize::MAX, 2]` is rejected, not wrapped). Every layer cross-checks dimensions and bias/layernorm lengths at load time, so a corrupt checkpoint returns `VoiceError::Model` instead of panicking at inference. |
| `manifest.json` | **Hostile** | Schema-versioned; pins SHA-256 of sibling files only (path traversal rejected). Verification runs before inference when the file is present. |
| Tokenizer JSON | **Hostile** | serde + explicit duplicate-token and merge-format validation. |
| Sample rates / configs | **Programmer-supplied** | Validated with `VoiceError::Config`; mismatched sample rates are errors, not silent resamples. |
| Audio samples (`f32` slice) | Trusted content, untrusted length | NaN/Inf values flow through DSP without panicking (outputs degrade); empty buffers are `AudioError::Empty`. |

There is no dynamic memory limit on the size of a single input *file*
beyond what the caller hands us; `read_wav_file` loads whole files.
Callers processing untrusted multi-GB inputs should stream or pre-check
file sizes.

## `unsafe` inventory

All `unsafe` lives in `tpt-av-voice-utils/src/simd.rs` and is limited to
SIMD intrinsics behind runtime feature detection:

- `avx2_dot` / `avx2_axpy` / `avx2_sum_squares` — require `avx2` (+ `fma`
  where used), checked with `is_x86_feature_detected!` immediately before
  the call. Pointers come from local slices; tails are handled scalar.
- `avx512_dot` / `avx512_axpy` — behind the `avx512` cargo feature and
  `is_x86_feature_detected!("avx512f")`; requires a toolchain with stable
  AVX-512 intrinsics (Rust ≥ 1.89).
- `neon_*` (aarch64) — NEON is architecture-baseline on aarch64, so no
  runtime check is needed; compile-checked in CI via
  `cargo check --target aarch64-pc-windows-msvc` and linux-gnu targets.

No `unsafe` exists in parsers, models, or I/O. The deterministic `Rng`
(xorshift) is **not** cryptographically secure and is never used for
security-relevant decisions.

## Supply chain

- `cargo-deny` enforces the allow-list of permissive licenses
  (MIT / Apache-2.0 / BSD / ISC / Zlib / Unicode); GPL, LGPL, AGPL and
  MPL are banned by policy (spec §8) and by the default-deny license
  audit in CI.
- `cargo-deny check advisories` (RustSec database) runs in CI;
  vulnerabilities fail the build.
- Model artifacts are distributed via GitHub Releases with SHA-256
  manifests; loaders verify them when present (see
  `tpt_av_voice_utils::manifest`).

## Fuzzing

`fuzz/` contains libFuzzer targets for the hostile-input parsers
(`wav_parse`, `safetensors_parse`, `tokenizer_json`). Run locally with
[cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz):

```bash
cargo +nightly fuzz run wav_parse -- -max_total_time=300
```

The dedicated fuzz workflow builds these on a schedule; failures are
triaged as crashes (panics/aborts are bugs by policy).
