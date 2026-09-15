//! Fuzzes the WAV parser with arbitrary bytes: malformed RIFF headers,
//! truncated chunks, absurd sizes and hostile encodings must return
//! errors — never panic, hang, or allocate unboundedly.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Errors and successes are both fine; panics/aborts are the bug.
    let _ = tpt_av_voice_utils::wav::read_wav_bytes(data);
});
