//! Edit audio by editing its transcript: aligns text to audio, applies a
//! text edit, and prints the resulting AudioEdit operations.
//!
//! Usage: cargo run --example text_based_edit -- <audio.wav> "<original>" "<edited>"

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("text_based_edit: not yet implemented (args: {args:?})");
}
