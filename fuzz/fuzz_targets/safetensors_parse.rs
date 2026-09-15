//! Fuzzes the safetensors reader: hostile headers (overflowing shapes,
//! out-of-bounds offsets, unknown dtypes) must be rejected with errors.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = tpt_av_voice_utils::safetensors::read_safetensors(data);
});
