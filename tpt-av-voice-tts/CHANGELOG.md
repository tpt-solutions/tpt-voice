# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- Rule-based letter-to-sound (grapheme-to-phoneme) conversion, including sentence-level terminal contours (`g2p`)
- Klatt-lite source-filter formant synthesizer: 3 resonators, fricatives/stops/nasals, deterministic output (`synth`)
- `VoiceProfile` voice characteristics (`voice`)
- `SynthConfig` configuration (`config`)
- High-level `FormantSynth` API producing intelligible robotic speech with no model downloads
