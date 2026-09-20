//! Regression tests for hostile/edge-case inputs to `BpeTokenizer::from_json`,
//! exercised by `fuzz/fuzz_targets/tokenizer_json.rs`.
//!
//! These are not crashes found by an actual fuzzing run in this environment
//! (cargo-fuzz's libFuzzer runtime is not available on Windows here — see
//! `fuzz/README.md`), but they capture the same durable value: known-tricky
//! malformed tokenizer JSON must return `Err(...)`, never panic.

use tpt_av_voice_transcribe::BpeTokenizer;

#[test]
fn empty_input_rejected() {
    assert!(BpeTokenizer::from_json("").is_err());
}

#[test]
fn not_json_rejected() {
    assert!(BpeTokenizer::from_json("not json at all").is_err());
}

#[test]
fn missing_fields_rejected() {
    assert!(BpeTokenizer::from_json("{}").is_err());
    assert!(BpeTokenizer::from_json(r#"{"vocab": []}"#).is_err());
}

#[test]
fn missing_specials_rejected() {
    let json = r#"{"vocab":["a","b"],"merges":[],"specials":[]}"#;
    assert!(BpeTokenizer::from_json(json).is_err());
}

#[test]
fn duplicate_token_rejected() {
    let json = r#"{"vocab":["a","a","<|endoftext|>","<|startoftranscript|>"],"merges":[],"specials":[]}"#;
    assert!(BpeTokenizer::from_json(json).is_err());
}

#[test]
fn malformed_merge_rejected() {
    let json = r#"{"vocab":["a","b","<|endoftext|>","<|startoftranscript|>"],"merges":["nospacehere"],"specials":[]}"#;
    assert!(BpeTokenizer::from_json(json).is_err());
}

#[test]
fn wrong_types_rejected() {
    // vocab entries must be strings, not numbers.
    let json = r#"{"vocab":[1,2,3],"merges":[],"specials":[]}"#;
    assert!(BpeTokenizer::from_json(json).is_err());
}

#[test]
fn truncated_json_rejected() {
    let json = r#"{"vocab":["a","<|endoftext|>","<|startoftranscript|>"],"merges":["#;
    assert!(BpeTokenizer::from_json(json).is_err());
}

#[test]
fn huge_vocab_declared_size_mismatch_does_not_panic() {
    // A vocab array that is internally consistent but very large should
    // parse without panicking, blowing the stack, or hanging.
    let mut vocab: Vec<String> = (0..5000).map(|i| format!("tok{i}")).collect();
    vocab.push("<|endoftext|>".to_string());
    vocab.push("<|startoftranscript|>".to_string());
    let json = serde_json::json!({
        "vocab": vocab,
        "merges": Vec::<String>::new(),
        "specials": Vec::<String>::new(),
    })
    .to_string();
    assert!(BpeTokenizer::from_json(&json).is_ok());
}

#[test]
fn lossy_utf8_never_panics() {
    // Mirrors the fuzz target: arbitrary bytes decoded lossily as UTF-8.
    let hostile: &[&[u8]] = &[
        &[0xFF, 0xFE, 0x00, 0x01],
        &[0x7B, 0x22, 0x76, 0x22], // `{"v` truncated
        &[0x00; 16],
        &[0xC0, 0x80], // overlong encoding
    ];
    for bytes in hostile {
        let text = String::from_utf8_lossy(bytes);
        let _ = BpeTokenizer::from_json(&text);
    }
}
