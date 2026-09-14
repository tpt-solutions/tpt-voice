//! Transcribe an audio file end-to-end.
//!
//! Usage: cargo run --example transcribe_audio -- <model-dir> <audio.wav>
//!
//! Run with `--make-demo-model <dir>` first to export a tiny, randomly
//! initialized model that exercises the full pipeline (transcripts from
//! untrained weights are gibberish; real accuracy requires trained
//! weights — see docs/PUBLISHING.md for the weights distribution plan).

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("transcribe_audio: not yet implemented (args: {args:?})");
}
