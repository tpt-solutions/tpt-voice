# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- Voice activity detection and segmentation (`segmentation`)
- TDNN-style speaker embedding extraction, with a weight-free classical embedding (MFCC statistics + autocorrelation F0) (`embedding`)
- Speaker clustering: agglomerative hierarchical clustering and spectral clustering (`clustering`)
- High-level `Diarizer` API (`diarizer`)
- Streaming diarization with online centroids and two-window hysteresis for new-speaker confirmation (`streaming`)
- `annotate_transcript` / `diarize_and_annotate` to link diarization output to `Transcript.segments[].speaker` (`integration`)
