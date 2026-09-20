# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- CTC forced alignment over arbitrary unit sequences (`forced_align`)
- Rule-based English grapheme-to-phoneme conversion (`phoneme`)
- Word-level alignment (`word_align`)
- High-level `Aligner` API, with an energy-proportional mode requiring no trained weights and an `align_with_emissions` path for trained acoustic models (`aligner`)
- `TextBasedEditor`: word-level LCS diff of an edited transcript against the original, emitting `Delete`/`Insert`/`Replace` `AudioEdit` operations for a `tpt-audio` timeline (`editor`)
