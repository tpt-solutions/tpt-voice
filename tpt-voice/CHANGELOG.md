# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-21

### Added

- One-crate facade re-exporting all six `tpt-av-voice-*` engine crates plus a `prelude` module
- `pipeline::SpeechPipeline`: transcribe → diarize → speaker-annotate → word-align in one call, returning `PipelineOutput`
- `pipeline::apply_text_edit_with_speech` / `fill_insertions`: text-based editing that synthesizes inserted/replaced audio via the TTS engine, ready to splice on a `tpt-audio` timeline
- `streaming_pipeline::StreamingPipeline`: chunk-at-a-time isolate → stream-transcribe → online-diarize composition
