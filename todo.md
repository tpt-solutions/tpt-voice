# tpt-voice — Project Todo

Tracks all work for `tpt-voice`, the pure-Rust speech processing engine (TPT Solutions).
License: dual **MIT OR Apache-2.0**.

---

## Phase 0: Project Setup & Infrastructure

- [ ] Initialize git repository, add `.gitignore` (Rust/Cargo template)
- [ ] Create workspace `Cargo.toml` (members, `workspace.package`, `workspace.dependencies` per spec.txt §9)
- [ ] Create `LICENSE-MIT` and `LICENSE-APACHE` (dual license)
- [ ] Set `license = "MIT OR Apache-2.0"` in workspace `Cargo.toml`
- [ ] Update `spec.txt` / `DESIGN.md` license references from MIT-only to dual MIT/Apache-2.0
- [ ] Create `deny.toml` (cargo-deny license enforcement per spec.txt §8)
- [ ] Scaffold `tpt-av-voice-utils` crate (`Cargo.toml` + `src/lib.rs`)
- [ ] Scaffold `tpt-av-voice-transcribe` crate (`Cargo.toml` + `src/lib.rs`)
- [ ] Scaffold `tpt-av-voice-isolate` crate (`Cargo.toml` + `src/lib.rs`)
- [ ] Scaffold `tpt-av-voice-diarize` crate (`Cargo.toml` + `src/lib.rs`)
- [ ] Scaffold `tpt-av-voice-align` crate (`Cargo.toml` + `src/lib.rs`)
- [ ] Scaffold `tpt-av-voice-tts` crate (`Cargo.toml` + `src/lib.rs`)
- [ ] Create `README.md` (project overview, badges, quickstart)
- [ ] Create `CONTRIBUTING.md` (contribution rules from spec.txt §10)
- [ ] Set up CI (GitHub Actions): build, test, clippy, fmt check
- [ ] Add `cargo-deny check` to CI
- [ ] Create `examples/` directory with stub files (`transcribe_audio.rs`, `isolate_voice.rs`, `diarize_meeting.rs`, `text_based_edit.rs`)
- [ ] Decide & document crates.io publishing plan (namespace reservation for `tpt-av-voice-*`)

---

## Phase 1: Foundation & Transcription

- [ ] `tpt-av-voice-utils`: `Transcript` / `TranscriptSegment` types (`transcript.rs`)
- [ ] `tpt-av-voice-utils`: `SpeakerLabel` type (`speaker.rs`)
- [ ] `tpt-av-voice-utils`: `AudioChunk` streaming type (`audio_chunk.rs`)
- [ ] `tpt-av-voice-utils`: time alignment types (`time.rs`)
- [ ] `tpt-av-voice-utils`: `VoiceError` enum (`error.rs`)
- [ ] `tpt-av-voice-transcribe`: mel spectrogram feature extraction (`encoder.rs`)
- [ ] `tpt-av-voice-transcribe`: transformer encoder-decoder ASR model architecture (`model.rs`)
- [ ] `tpt-av-voice-transcribe`: text decoding — beam search / CTC (`decoder.rs`)
- [ ] `tpt-av-voice-transcribe`: tokenizer — BPE/WordPiece (`tokenizer.rs`)
- [ ] `tpt-av-voice-transcribe`: model weight loading — custom binary format or safetensors (`weights.rs`)
- [ ] `tpt-av-voice-transcribe`: high-level `Transcriber` API (`transcriber.rs`)
- [ ] `tpt-av-voice-transcribe`: streaming transcription support (`streaming.rs`)
- [ ] Source or train pre-trained English small-model weights
- [ ] Conformance tests against LibriSpeech dataset
- [ ] `examples/transcribe_audio.rs` working end-to-end

---

## Phase 2: Voice Isolation

- [ ] `tpt-av-voice-isolate`: STFT implementation (`stft.rs`)
- [ ] `tpt-av-voice-isolate`: inverse STFT (`istft.rs`)
- [ ] `tpt-av-voice-isolate`: time-frequency masking (`mask.rs`)
- [ ] `tpt-av-voice-isolate`: U-Net/transformer separation model (`model.rs`)
- [ ] `tpt-av-voice-isolate`: high-level `VoiceIsolator` API — `isolate_file` (`isolator.rs`)
- [ ] `tpt-av-voice-isolate`: real-time streaming isolation with overlap-add (`streaming.rs`)
- [ ] Source or train pre-trained separation model weights
- [ ] `examples/isolate_voice.rs` working end-to-end

---

## Phase 3: Diarization

- [ ] `tpt-av-voice-diarize`: voice activity detection & segmentation (`segmentation.rs`)
- [ ] `tpt-av-voice-diarize`: speaker embedding extraction — TDNN/ECAPA-TDNN (`embedding.rs`)
- [ ] `tpt-av-voice-diarize`: speaker clustering — spectral clustering / AHC (`clustering.rs`)
- [ ] `tpt-av-voice-diarize`: high-level `Diarizer` API — `diarize_file` (`diarizer.rs`)
- [ ] `tpt-av-voice-diarize`: streaming diarization support (`streaming.rs`)
- [ ] Multi-speaker transcription integration (link diarization output to `Transcript.segments[].speaker`)
- [ ] `examples/diarize_meeting.rs` working end-to-end

---

## Phase 4: Text-Based Editing

- [ ] `tpt-av-voice-align`: forced alignment — CTC-based (`forced_align.rs`)
- [ ] `tpt-av-voice-align`: phoneme-level alignment (`phoneme.rs`)
- [ ] `tpt-av-voice-align`: word-level alignment (`word_align.rs`)
- [ ] `tpt-av-voice-align`: high-level `Aligner` API (`aligner.rs`)
- [ ] `TextBasedEditor`: diff original vs. edited transcript
- [ ] `TextBasedEditor`: map text changes to `AudioEdit` operations
- [ ] Integration with `tpt-audio` timeline (consume `AudioEdit` to mutate timeline clips)
- [ ] `examples/text_based_edit.rs` working end-to-end

---

## Phase 5: Advanced Features

- [ ] Multi-language support (Spanish, French, German, etc.)
- [ ] Larger, more accurate ASR models
- [ ] GPU-accelerated inference via `wgpu` — transformer attention matmul
- [ ] GPU-accelerated inference via `wgpu` — STFT/ISTFT FFT
- [ ] GPU-accelerated inference via `wgpu` — convolution
- [ ] SIMD optimization pass — AVX2/AVX-512 via `std::arch` (x86_64)
- [ ] SIMD optimization pass — NEON via `std::arch` (ARM)
- [ ] Pure Rust scalar fallback path for all SIMD code
- [ ] `tpt-av-voice-tts` (text-to-speech) — initial design
- [ ] `tpt-av-voice-tts` (text-to-speech) — initial implementation
