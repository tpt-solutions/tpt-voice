//! Conformance harness against standard datasets (spec §10).
//!
//! Runs real accuracy evaluation when both environment variables are set:
//!
//! * `TPT_VOICE_CONFORMANCE_MODEL` — model directory (trained weights)
//! * `TPT_VOICE_CONFORMANCE_DATA` — dataset directory of `NNN.wav` +
//!   `NNN.txt` pairs (LibriSpeech `test-clean` layout works: wav/txt
//!   files in the same tree)
//!
//! Without them the test skips — CI and fresh clones have no dataset.

use std::path::Path;

use tpt_av_voice_transcribe::Transcriber;
use tpt_av_voice_utils::wav;

fn collect_pairs(root: &Path, out: &mut Vec<(std::path::PathBuf, std::path::PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_pairs(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("wav") {
            let txt = path.with_extension("txt");
            if txt.exists() {
                out.push((path, txt));
            }
        }
    }
}

/// Word error rate over the dataset (greedy decoding).
#[test]
fn librispeech_conformance_when_dataset_present() {
    let (Some(model_dir), Some(data_dir)) = (
        std::env::var("TPT_VOICE_CONFORMANCE_MODEL").ok(),
        std::env::var("TPT_VOICE_CONFORMANCE_DATA").ok(),
    ) else {
        eprintln!("skipping conformance: set TPT_VOICE_CONFORMANCE_MODEL and \
                   TPT_VOICE_CONFORMANCE_DATA to run LibriSpeech evaluation");
        return;
    };

    let transcriber = Transcriber::from_model_dir(model_dir).expect("model loads");
    let mut pairs = Vec::new();
    collect_pairs(Path::new(&data_dir), &mut pairs);
    assert!(!pairs.is_empty(), "no wav/txt pairs found under {data_dir}");

    let mut total_words = 0usize;
    let mut total_errors = 0usize;
    for (wav_path, txt_path) in &pairs {
        let reference = std::fs::read_to_string(txt_path).unwrap().to_lowercase();
        let reference_words: Vec<&str> = reference.split_whitespace().collect();
        let audio = wav::read_wav_file(wav_path).unwrap();
        let transcript = transcriber.transcribe_samples(&audio.to_mono(), audio.sample_rate).unwrap();
        let hyp_words: Vec<&str> = transcript.full_text.split_whitespace().collect();
        total_errors += levenshtein_words(&hyp_words, &reference_words);
        total_words += reference_words.len();
    }
    let wer = total_errors as f64 / total_words.max(1) as f64;
    println!("LibriSpeech WER: {wer:.4} over {total_words} words");
    // Gate: acceptable accuracy for the shipped checkpoint. Tighten as
    // models improve.
    assert!(wer < 0.25, "WER {wer:.4} exceeded 0.25 gate");
}

fn levenshtein_words(hyp: &[&str], reference: &[&str]) -> usize {
    let mut dp: Vec<usize> = (0..=hyp.len()).collect();
    for (i, r) in reference.iter().enumerate() {
        let mut prev = dp[0];
        dp[0] = i + 1;
        for (j, h) in hyp.iter().enumerate() {
            let tmp = dp[j + 1];
            dp[j + 1] = (dp[j + 1] + 1).min(dp[j] + 1).min(prev + usize::from(h != r));
            prev = tmp;
        }
    }
    dp[hyp.len()]
}

#[test]
fn levenshtein_known_values() {
    assert_eq!(levenshtein_words(&["a", "b"], &["a", "b"]), 0);
    assert_eq!(levenshtein_words(&["a", "x"], &["a", "b"]), 1);
    assert_eq!(levenshtein_words(&["a"], &["a", "b", "c"]), 2);
}
