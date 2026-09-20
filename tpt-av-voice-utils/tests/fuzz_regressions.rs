//! Regression tests for hostile/edge-case inputs to the parsers exercised
//! by `fuzz/fuzz_targets/wav_parse.rs` and `fuzz/fuzz_targets/safetensors_parse.rs`.
//!
//! These are not crashes found by an actual fuzzing run in this environment
//! (cargo-fuzz's libFuzzer runtime is not available on Windows here — see
//! `fuzz/README.md`), but they capture the same durable value: known-tricky
//! malformed inputs must return `Err(...)`, never panic.

use tpt_av_voice_utils::safetensors::read_safetensors;
use tpt_av_voice_utils::wav::read_wav_bytes;

// --- WAV ---

#[test]
fn wav_empty_input() {
    assert!(read_wav_bytes(&[]).is_err());
}

#[test]
fn wav_riff_magic_only_no_size() {
    assert!(read_wav_bytes(b"RIFF").is_err());
}

#[test]
fn wav_size_claims_beyond_eof() {
    let mut buf = b"RIFFxxxxWAVEfmt ".to_vec();
    // Claim a huge fmt chunk size that goes way past EOF.
    buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    assert!(read_wav_bytes(&buf).is_err());
}

#[test]
fn wav_zero_channels_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&36u32.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&0u16.to_le_bytes()); // zero channels
    buf.extend_from_slice(&16_000u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&0u32.to_le_bytes());
    assert!(read_wav_bytes(&buf).is_err());
}

#[test]
fn wav_zero_sample_rate_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&36u32.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // zero sample rate
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&0u32.to_le_bytes());
    assert!(read_wav_bytes(&buf).is_err());
}

#[test]
fn wav_missing_data_chunk() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&24u32.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&16_000u32.to_le_bytes());
    buf.extend_from_slice(&32_000u32.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    // No "data" chunk at all.
    assert!(read_wav_bytes(&buf).is_err());
}

#[test]
fn wav_unsupported_format_tag_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&36u32.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&7u16.to_le_bytes()); // u-law, unsupported
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&16_000u32.to_le_bytes());
    buf.extend_from_slice(&16_000u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&8u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&0u32.to_le_bytes());
    assert!(read_wav_bytes(&buf).is_err());
}

#[test]
fn wav_extensible_fmt_too_small_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&36u32.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // claims 16 bytes but fmt tag is EXTENSIBLE
    buf.extend_from_slice(&0xFFFEu16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&16_000u32.to_le_bytes());
    buf.extend_from_slice(&16_000u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&8u16.to_le_bytes());
    assert!(read_wav_bytes(&buf).is_err());
}

#[test]
fn wav_all_255_bytes_does_not_panic() {
    // Arbitrary hostile bytes: no panics regardless of the (expected) error.
    let buf = vec![0xFFu8; 128];
    let _ = read_wav_bytes(&buf);
}

#[test]
fn wav_arbitrary_short_inputs_never_panic() {
    for len in 0..64 {
        let buf = vec![0x41u8; len];
        let _ = read_wav_bytes(&buf);
    }
}

// --- safetensors ---

#[test]
fn safetensors_empty_input() {
    assert!(read_safetensors(&[]).is_err());
}

#[test]
fn safetensors_header_len_zero_rejected() {
    assert!(read_safetensors(&0u64.to_le_bytes()).is_err());
}

#[test]
fn safetensors_header_len_exceeds_buffer() {
    let mut buf = (1000u64).to_le_bytes().to_vec();
    buf.extend_from_slice(b"short");
    assert!(read_safetensors(&buf).is_err());
}

#[test]
fn safetensors_header_len_max_u64_no_panic() {
    // Would overflow `8 + header_len` on naive addition; must not panic.
    let buf = u64::MAX.to_le_bytes().to_vec();
    assert!(read_safetensors(&buf).is_err());
}

#[test]
fn safetensors_malformed_json_header() {
    let header = b"{not valid json";
    let mut buf = (header.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(header);
    assert!(read_safetensors(&buf).is_err());
}

#[test]
fn safetensors_negative_range_end_before_start() {
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#;
    let mut buf = (header.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&[0u8; 4]);
    assert!(read_safetensors(&buf).is_err());
}

#[test]
fn safetensors_offsets_out_of_bounds() {
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,1000000]}}"#;
    let mut buf = (header.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&[0u8; 4]);
    assert!(read_safetensors(&buf).is_err());
}

#[test]
fn safetensors_arbitrary_short_inputs_never_panic() {
    for len in 0..40 {
        let buf = vec![0x00u8; len];
        let _ = read_safetensors(&buf);
    }
    for len in 0..40 {
        let buf = vec![0xFFu8; len];
        let _ = read_safetensors(&buf);
    }
}
