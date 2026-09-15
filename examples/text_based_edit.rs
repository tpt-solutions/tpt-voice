//! Edit audio by editing its transcript.
//!
//! Usage:
//! ```text
//! cargo run -p tpt-voice-examples --example text_based_edit -- \
//!     [audio.wav "<original transcript>"] "<edited transcript>"
//! ```
//!
//! With only the edited text, the original is synthesized with the TTS
//! crate (so the whole pipeline is observable in one run), then
//! word-aligned and diffed against the edit. With an audio file and its
//! original transcript, alignment is energy-based on that file.

use tpt_av_voice_align::{Aligner, AlignerConfig, EditType, TextBasedEditor};
use tpt_av_voice_tts::{FormantSynth, SynthConfig};
use tpt_av_voice_utils::wav;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (samples, sample_rate, original, edited, source_desc) = match args.as_slice() {
        [edited] => {
            // Synthesize the original sentence so the demo is self-contained.
            let original = "hello brave new world";
            let mut synth = FormantSynth::new(SynthConfig::default()).expect("synth");
            let audio = synth.synthesize(original).expect("synthesis");
            (
                audio.samples,
                audio.sample_rate,
                original.to_string(),
                edited.clone(),
                "synthesized speech".to_string(),
            )
        }
        [audio, original, edited] => {
            let wav = wav::read_wav_file(audio).expect("read wav");
            (
                wav.to_mono(),
                wav.sample_rate,
                original.clone(),
                edited.clone(),
                audio.clone(),
            )
        }
        _ => {
            eprintln!("usage: text_based_edit [audio.wav \"original\"] \"edited\"");
            eprintln!("       text_based_edit \"edited\"   (synthesizes the original)");
            std::process::exit(2);
        }
    };

    println!("aligning {source_desc} to: {original:?}");
    let mut aligner = Aligner::new(AlignerConfig::default()).expect("aligner");
    let aligned = aligner
        .align_samples(&samples, sample_rate, &original)
        .expect("alignment");
    for w in &aligned.words {
        println!("  {:<8} {}", w.word, w.start_time);
    }

    let mut editor = TextBasedEditor::new(aligned);
    let edits = editor.apply_edits(&edited);
    println!(
        "
edited transcript: {:?} → {} operation(s)",
        editor.edited_transcript(),
        edits.len()
    );
    for op in &edits {
        let kind = match op.edit_type {
            EditType::Delete => "DELETE ",
            EditType::Insert => "INSERT ",
            EditType::Replace => "REPLACE",
        };
        println!(
            "  {kind} [{} → {}] new_audio: {}",
            op.start_time,
            op.end_time,
            op.new_audio.is_some()
        );
    }
    println!(
        "
these AudioEdit operations are what tpt-audio consumes to mutate its timeline
         (insert/replace audio can be synthesized via tpt-av-voice-tts)"
    );
}
