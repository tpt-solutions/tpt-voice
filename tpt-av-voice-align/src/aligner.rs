//! High-level alignment API.
//!
//! Two alignment modes:
//!
//! * [`AlignmentMode::Energy`] — deterministic, weight-free: words are
//!   distributed over detected speech frames proportionally to their
//!   phoneme counts, with boundaries snapped to local energy minima.
//!   Works out of the box.
//! * [`AlignmentMode::Neural`] — CTC forced alignment over G2P
//!   phonemes, given a frame emission matrix from a trained acoustic
//!   model ([`Aligner::align_with_emissions`]).

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::dsp::hann_window;
use tpt_av_voice_utils::wav;
use tpt_av_voice_utils::{TimeRange, VoiceError, WordTimestamp};

use crate::forced_align::{ctc_align, AlignmentUnit};
use crate::phoneme::{g2p, g2p_word, Phoneme};
use crate::word_align::words_from_phoneme_alignment;

/// Which alignment backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AlignmentMode {
    /// Energy-proportional alignment, no trained weights (default).
    #[default]
    Energy,
    /// CTC forced alignment through [`Aligner::align_with_emissions`].
    Neural,
}

/// Aligner settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignerConfig {
    pub mode: AlignmentMode,
    /// Analysis hop in seconds (0.01 = 10 ms).
    pub hop_secs: f64,
    /// Analysis window in seconds (0.025 = 25 ms).
    pub window_secs: f64,
}

impl Default for AlignerConfig {
    fn default() -> Self {
        AlignerConfig {
            mode: AlignmentMode::Energy,
            hop_secs: 0.01,
            window_secs: 0.025,
        }
    }
}

impl AlignerConfig {
    fn validate(&self) -> Result<(), VoiceError> {
        if self.hop_secs <= 0.0 || self.window_secs < self.hop_secs {
            return Err(VoiceError::config("require 0 < hop_secs <= window_secs"));
        }
        Ok(())
    }
}

/// A transcript with word-level timestamps (spec §4.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AlignedTranscript {
    /// Words with timestamps.
    pub words: Vec<WordTimestamp>,
    /// Full text.
    pub text: String,
}

impl AlignedTranscript {
    /// Original word strings in order.
    pub fn word_strings(&self) -> Vec<&str> {
        self.words.iter().map(|w| w.word.as_str()).collect()
    }
}

/// Text-audio alignment engine (spec §4.5).
pub struct Aligner {
    config: AlignerConfig,
}

impl Aligner {
    /// Creates an aligner.
    pub fn new(config: AlignerConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        Ok(Aligner { config })
    }

    /// Aligns a transcript to an audio file (word-level timestamps).
    pub fn align(
        &mut self,
        audio_path: &Path,
        transcript: &str,
    ) -> Result<AlignedTranscript, VoiceError> {
        let audio = wav::read_wav_file(audio_path)?;
        self.align_samples(&audio.to_mono(), audio.sample_rate, transcript)
    }

    /// Aligns a transcript to mono PCM samples.
    pub fn align_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        transcript: &str,
    ) -> Result<AlignedTranscript, VoiceError> {
        match self.config.mode {
            AlignmentMode::Energy => {
                let words = energy_align(samples, sample_rate, transcript, &self.config)?;
                Ok(AlignedTranscript {
                    words,
                    text: transcript.to_string(),
                })
            }
            AlignmentMode::Neural => Err(VoiceError::config(
                "neural alignment requires emissions; call align_with_emissions",
            )),
        }
    }

    /// CTC forced alignment given per-frame phoneme emissions from a
    /// trained acoustic model.
    ///
    /// `vocab` maps phonemes to emission columns; `blank` is the CTC
    /// blank label id.
    pub fn align_with_emissions(
        &mut self,
        emissions: &ndarray::Array2<f32>,
        transcript: &str,
        vocab: &HashMap<Phoneme, usize>,
        blank: usize,
    ) -> Result<AlignedTranscript, VoiceError> {
        let word_list: Vec<&str> = transcript.split_whitespace().collect();
        let mut units = Vec::new();
        let label_of = |p: Phoneme, vocab: &HashMap<Phoneme, usize>| -> Result<usize, VoiceError> {
            vocab
                .get(&p)
                .copied()
                .ok_or_else(|| VoiceError::decode(format!("phoneme {p} missing from vocab")))
        };
        for (w, word) in word_list.iter().enumerate() {
            for p in g2p_word(word) {
                units.push(AlignmentUnit::new(label_of(p, vocab)?, w));
            }
        }
        let alignment = ctc_align(emissions, &units, blank)?;
        let frame_secs = 0.01; // documented emission convention; matches hop
        let words = words_from_phoneme_alignment(&alignment, &word_list, frame_secs);
        Ok(AlignedTranscript {
            words,
            text: transcript.to_string(),
        })
    }

    /// Borrows the configuration.
    pub fn config(&self) -> &AlignerConfig {
        &self.config
    }
}

/// Energy-proportional word alignment.
///
/// 1. Frame the audio; compute band-limited energy in dB.
/// 2. Speech = frames above (10th-percentile floor + margin).
/// 3. Assign words to speech frames proportionally to their phoneme
///    counts, snapping boundaries to local energy minima.
fn energy_align(
    samples: &[f32],
    sample_rate: u32,
    transcript: &str,
    config: &AlignerConfig,
) -> Result<Vec<WordTimestamp>, VoiceError> {
    let words: Vec<&str> = transcript.split_whitespace().collect();
    if words.is_empty() {
        return Ok(Vec::new());
    }
    if samples.is_empty() {
        return Err(VoiceError::from(tpt_av_voice_utils::AudioError::Empty));
    }
    let hop = (config.hop_secs * sample_rate as f64).round() as usize;
    let window = (config.window_secs * sample_rate as f64).round() as usize;
    if hop == 0 || window == 0 || samples.len() < window {
        return Err(VoiceError::config("audio too short for the aligner window"));
    }
    let win = hann_window(window);
    let n_frames = (samples.len() - window) / hop + 1;

    // Per-frame speech-band energy in dB.
    let mut frame_db = vec![0.0f32; n_frames];
    // Cheap band weight: emphasize 200–4000 Hz via a rough spectral tilt
    // using FFT every frame is overkill here; RMS suffices for placement.
    for (t, frame_db_v) in frame_db.iter_mut().enumerate() {
        let frame = &samples[t * hop..t * hop + window];
        let energy: f32 = frame.iter().zip(&win).map(|(s, w)| (s * w).powi(2)).sum();
        *frame_db_v = 10.0 * (energy.max(1e-12)).log10();
    }
    let mut sorted = frame_db.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[n_frames / 10];
    // Speech-dominated clips have a speech-level floor; cap the
    // threshold at the median so at least half the frames stay
    // eligible.
    let threshold = (floor + 8.0).min(sorted[n_frames / 2]);
    let mut speech: Vec<bool> = frame_db.iter().map(|&db| db > threshold).collect();
    // Bridge short silence gaps (< 150 ms) so words aren't fragmented
    // across micro-runs by within-word dips (stops, envelope ramps).
    let bridge = ((0.15 / config.hop_secs) as usize).max(1);
    let mut i = 0usize;
    while i < n_frames {
        if !speech[i] {
            let mut j = i;
            while j < n_frames && !speech[j] {
                j += 1;
            }
            if i > 0 && j < n_frames && j - i < bridge {
                for v in speech.iter_mut().take(j).skip(i) {
                    *v = true;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    let speech_frames: Vec<usize> = speech
        .iter()
        .enumerate()
        .filter(|(_, s)| **s)
        .map(|(i, _)| i)
        .collect();
    if speech_frames.is_empty() {
        // No speech detected: distribute uniformly so callers still get
        // monotone timestamps.
        return Ok(uniform_words(&words, samples.len(), sample_rate, config));
    }

    // Weights from phoneme counts.
    let weights: Vec<usize> = words.iter().map(|w| g2p_word(w).len().max(1)).collect();
    let total_weight: usize = weights.iter().sum();
    let total_speech = speech_frames.len();
    let scale = total_speech as f64 / total_weight as f64;

    // Contiguous speech runs — words never span a silence gap.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &f in &speech_frames {
        match runs.last_mut() {
            Some((_, e)) if f == *e + 1 => *e = f,
            _ => runs.push((f, f)),
        }
    }

    let mut out: Vec<WordTimestamp> = Vec::new();
    let mut w = 0usize;
    for (ri, &(run_start, run_end)) in runs.iter().enumerate() {
        let last_run = ri + 1 == runs.len();
        let mut cursor = run_start;
        while w < words.len() {
            let mut cost = ((weights[w] as f64 * scale).round() as usize).max(1);
            let remaining_run = run_end + 1 - cursor;
            if cost > remaining_run {
                if last_run {
                    cost = remaining_run.max(1); // squeeze the tail here
                } else {
                    break; // this word belongs to a later run
                }
            }
            let start = cursor;
            let end = cursor + cost - 1;
            let snapped_start = if w == 0 {
                start
            } else {
                snap_to_minimum(&frame_db, start, 3)
            };
            let snapped_end = if w == words.len() - 1 {
                end
            } else {
                snap_to_minimum(&frame_db, end, 3)
            };
            out.push(WordTimestamp::new(
                words[w],
                snapped_start as f64 * config.hop_secs,
                (snapped_end + 1) as f64 * config.hop_secs,
                1.0,
            ));
            cursor = end + 1;
            w += 1;
            if cursor > run_end {
                break;
            }
        }
        if w == words.len() {
            break;
        }
    }
    // Leftovers (pathological: far more speech weight than frames).
    let tail = out.last().map(|t| t.end_time).unwrap_or(0.0);
    while w < words.len() {
        out.push(WordTimestamp::new(
            words[w],
            tail,
            tail + config.hop_secs,
            0.0,
        ));
        w += 1;
    }
    if out.is_empty() {
        return Ok(uniform_words(&words, samples.len(), sample_rate, config));
    }

    // Enforce monotonicity after snapping.
    for i in 1..out.len() {
        if out[i].start_time < out[i - 1].end_time {
            out[i].start_time = out[i - 1].end_time;
        }
        if out[i].end_time < out[i].start_time {
            out[i].end_time = out[i].start_time + config.hop_secs;
        }
    }
    Ok(out)
}

/// Snaps a frame index to the lowest-energy frame within ±radius.
fn snap_to_minimum(frame_db: &[f32], idx: usize, radius: usize) -> usize {
    let lo = idx.saturating_sub(radius);
    let hi = (idx + radius).min(frame_db.len() - 1);
    let mut best = idx;
    let mut best_v = f32::INFINITY;
    for (i, &v) in frame_db[lo..=hi].iter().enumerate() {
        if v < best_v {
            best_v = v;
            best = lo + i;
        }
    }
    best
}

/// Fallback: evenly spread words across the whole clip.
fn uniform_words(
    words: &[&str],
    n_samples: usize,
    sample_rate: u32,
    config: &AlignerConfig,
) -> Vec<WordTimestamp> {
    let total = tpt_av_voice_utils::time::samples_to_secs(n_samples, sample_rate);
    let per = (total / words.len() as f64).max(config.hop_secs);
    words
        .iter()
        .enumerate()
        .map(|(i, w)| WordTimestamp::new(*w, i as f64 * per, (i + 1) as f64 * per, 0.0))
        .collect()
}

/// Exposed for tests/examples: phoneme count weight of a word.
pub fn word_weight(word: &str) -> usize {
    g2p(word).first().map(Vec::len).unwrap_or(1)
}

/// Time range helper for editors.
pub fn word_range(word: &WordTimestamp) -> TimeRange {
    TimeRange::new(word.start_time, word.end_time)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, secs: f64) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| (std::f32::consts::TAU * hz * i as f32 / 16_000.0).sin() * 0.5)
            .collect()
    }

    /// "hello" burst (1 s) — silence (1 s) — "world" burst (1 s).
    fn two_word_audio() -> Vec<f32> {
        let mut pcm = Vec::new();
        pcm.extend(tone(220.0, 1.0));
        pcm.extend(vec![0.0; 16_000]);
        pcm.extend(tone(330.0, 1.0));
        pcm
    }

    #[test]
    fn energy_align_places_words_in_their_bursts() {
        let mut aligner = Aligner::new(AlignerConfig::default()).unwrap();
        let pcm = two_word_audio();
        let aligned = aligner.align_samples(&pcm, 16_000, "hello world").unwrap();
        assert_eq!(aligned.words.len(), 2);
        assert_eq!(aligned.text, "hello world");
        let w0 = &aligned.words[0];
        let w1 = &aligned.words[1];
        // Word 1 lives in the first burst, word 2 in the last.
        assert!(w0.start_time < 1.2, "w0 start {w0:?}");
        assert!(w0.end_time <= 2.0, "w0 end {w0:?}");
        assert!(w1.start_time >= 1.6, "w1 start {w1:?}");
        assert!(w1.end_time <= 3.05, "w1 end {w1:?}");
        assert!(w1.start_time >= w0.end_time - 0.15);
    }

    #[test]
    fn empty_transcript_aligns_to_nothing() {
        let mut aligner = Aligner::new(AlignerConfig::default()).unwrap();
        let aligned = aligner
            .align_samples(&tone(200.0, 1.0), 16_000, "")
            .unwrap();
        assert!(aligned.words.is_empty());
    }

    #[test]
    fn all_silence_falls_back_to_uniform() {
        let mut aligner = Aligner::new(AlignerConfig::default()).unwrap();
        let aligned = aligner
            .align_samples(&vec![0.0; 32_000], 16_000, "a b c d")
            .unwrap();
        assert_eq!(aligned.words.len(), 4);
        let times: Vec<f64> = aligned.words.iter().map(|w| w.start_time).collect();
        assert!(times.windows(2).all(|p| p[0] < p[1]), "monotone");
    }

    #[test]
    fn align_file_round_trip() {
        let mut aligner = Aligner::new(AlignerConfig::default()).unwrap();
        let path = std::env::temp_dir().join(format!("tpt-align-{}.wav", std::process::id()));
        tpt_av_voice_utils::wav::write_wav_file(&path, &two_word_audio(), 16_000, 1).unwrap();
        let aligned = aligner.align(&path, "hello world").unwrap();
        assert_eq!(aligned.words.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn neural_mode_requires_emissions() {
        let cfg = AlignerConfig {
            mode: AlignmentMode::Neural,
            ..Default::default()
        };
        let mut aligner = Aligner::new(cfg).unwrap();
        assert!(aligner
            .align_samples(&tone(200.0, 1.0), 16_000, "cat")
            .is_err());
    }

    #[test]
    fn ctc_neural_mode_with_synthetic_emissions() {
        use ndarray::Array2;
        // Phoneme vocab: K=1, AE=2, T=3, blank=0. "cat".
        let vocab: HashMap<Phoneme, usize> = [(Phoneme::K, 1), (Phoneme::AE, 2), (Phoneme::T, 3)]
            .into_iter()
            .collect();
        // 3 units × 5 frames each, blanks between.
        let mut e = Array2::from_elem((17, 4), -5.0);
        for t in 0..5 {
            e[[t, 1]] = 5.0;
        }
        for t in 5..10 {
            e[[t, 0]] = 5.0;
        }
        for t in 10..12 {
            e[[t, 2]] = 5.0;
        }
        for t in 12..17 {
            e[[t, 3]] = 5.0;
        }
        let cfg = AlignerConfig {
            mode: AlignmentMode::Neural,
            ..Default::default()
        };
        let mut aligner = Aligner::new(cfg).unwrap();
        let aligned = aligner.align_with_emissions(&e, "cat", &vocab, 0).unwrap();
        assert_eq!(aligned.words.len(), 1);
        let w = &aligned.words[0];
        assert!(w.start_time < 0.2 && w.end_time > 0.15, "word {w:?}");
    }

    #[test]
    fn config_validation() {
        assert!(Aligner::new(AlignerConfig::default()).is_ok());
        assert!(Aligner::new(AlignerConfig {
            hop_secs: 0.0,
            ..Default::default()
        })
        .is_err());
        assert!(Aligner::new(AlignerConfig {
            hop_secs: 0.05,
            window_secs: 0.01,
            ..Default::default()
        })
        .is_err());
    }
}
