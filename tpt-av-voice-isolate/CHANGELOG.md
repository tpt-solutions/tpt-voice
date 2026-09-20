# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- STFT and inverse STFT (weighted overlap-add, round-trips) (`stft`, `istft`)
- Time-frequency masking (`mask`)
- U-Net-lite separation model with skip connections (`model`)
- High-level `VoiceIsolator` API, including a weight-free classical spectral-subtraction mode that works out of the box (`isolator`)
- Real-time streaming isolation with overlap-add (`streaming`)
- Criterion benchmark for the STFT forward transform (`benches/stft`)
