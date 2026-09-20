# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- Mel spectrogram feature extraction (`encoder`)
- Transformer encoder-decoder ASR model architecture, with `ModelSize` presets Tiny through Large (`model`)
- Greedy, beam-search and CTC text decoding (`decoder`)
- Byte-level BPE tokenizer with trainer (`tokenizer`)
- Safetensors-compatible model weight loading and export, with panic-free validation of corrupt checkpoints (`weights`)
- High-level `Transcriber` API and streaming transcription (`transcriber`, `streaming`)
- Multi-language support via the shared `Language` type
- Conformance test harness (WER via word-level Levenshtein) against LibriSpeech, env-gated pending trained weights
