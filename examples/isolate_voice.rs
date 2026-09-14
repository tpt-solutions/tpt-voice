//! Isolate the voice from background noise in an audio file.
//!
//! Usage: cargo run --example isolate_voice -- <input.wav> <output.wav>

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("isolate_voice: not yet implemented (args: {args:?})");
}
