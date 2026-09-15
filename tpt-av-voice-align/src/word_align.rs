//! Word-level alignment: grouping aligned unit timings into word
//! timestamps.

use tpt_av_voice_utils::WordTimestamp;

use crate::forced_align::ForcedAlignment;

/// Groups aligned unit timings into word timestamps.
///
/// `UnitTiming::word_index` partitions the units into words in order;
/// a word's time span runs from its first unit's start to its last
/// unit's end. `frame_secs` converts frames to seconds.
pub fn words_from_phoneme_alignment(
    alignment: &ForcedAlignment,
    words: &[&str],
    frame_secs: f64,
) -> Vec<WordTimestamp> {
    let mut out: Vec<WordTimestamp> = Vec::new();
    let mut current_word: Option<usize> = None;
    for timing in &alignment.units {
        let surface = words.get(timing.word_index).copied().unwrap_or("");
        let start = timing.start_frame as f64 * frame_secs;
        let end = (timing.end_frame + 1) as f64 * frame_secs;
        match (current_word, out.last_mut()) {
            (Some(w), Some(last)) if w == timing.word_index => {
                // Another unit of the current word: extend it.
                last.end_time = end;
            }
            _ => {
                out.push(WordTimestamp::new(surface, start, end, 1.0));
            }
        }
        current_word = Some(timing.word_index);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forced_align::{ctc_align, AlignmentUnit, UnitTiming};
    use ndarray::Array2;

    fn manual_alignment() -> ForcedAlignment {
        ForcedAlignment {
            units: vec![
                UnitTiming {
                    unit_index: 0,
                    label: 1,
                    word_index: 0,
                    start_frame: 0,
                    end_frame: 3,
                },
                UnitTiming {
                    unit_index: 1,
                    label: 2,
                    word_index: 0,
                    start_frame: 4,
                    end_frame: 6,
                },
                UnitTiming {
                    unit_index: 2,
                    label: 3,
                    word_index: 1,
                    start_frame: 8,
                    end_frame: 9,
                },
                UnitTiming {
                    unit_index: 3,
                    label: 4,
                    word_index: 1,
                    start_frame: 10,
                    end_frame: 14,
                },
            ],
            score: -1.0,
        }
    }

    #[test]
    fn groups_units_into_words() {
        let timings = words_from_phoneme_alignment(&manual_alignment(), &["hi", "there"], 0.01);
        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].word, "hi");
        assert!((timings[0].start_time - 0.0).abs() < 1e-9);
        assert!((timings[0].end_time - 0.07).abs() < 1e-9);
        assert_eq!(timings[1].word, "there");
        assert!((timings[1].start_time - 0.08).abs() < 1e-9);
        assert!((timings[1].end_time - 0.15).abs() < 1e-9);
    }

    #[test]
    fn repeated_words_stay_separate() {
        let mut a = manual_alignment();
        a.units[2].word_index = 1; // "and"
        a.units[3].word_index = 2; // "hi" again
        let timings = words_from_phoneme_alignment(&a, &["hi", "and", "hi"], 0.01);
        assert_eq!(timings.len(), 3);
        assert_eq!(timings[0].word, "hi");
        assert_eq!(timings[1].word, "and");
        assert_eq!(timings[2].word, "hi");
        assert!(timings[2].start_time > timings[0].end_time);
    }

    #[test]
    fn end_to_end_with_ctc_align() {
        // Two units with clean emission blocks; word boundary between.
        let mut e = Array2::from_elem((15, 3), -5.0);
        for t in 0..5 {
            e[[t, 1]] = 5.0;
        }
        for t in 5..10 {
            e[[t, 0]] = 5.0;
        }
        for t in 10..15 {
            e[[t, 2]] = 5.0;
        }
        let units = vec![AlignmentUnit::new(1, 0), AlignmentUnit::new(2, 1)];
        let aligned = ctc_align(&e, &units, 0).unwrap();
        let timings = words_from_phoneme_alignment(&aligned, &["cat", "dog"], 0.01);
        assert_eq!(timings.len(), 2);
        assert!(timings[0].end_time <= timings[1].start_time + 1e-9);
    }
}
