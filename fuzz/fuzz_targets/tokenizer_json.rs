//! Fuzzes tokenizer deserialization with arbitrary bytes (decoded as
//! UTF-8 lossily): duplicate tokens, bad merges and huge vocabs must
//! error cleanly.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = tpt_av_voice_transcribe::BpeTokenizer::from_json(&text);
});
