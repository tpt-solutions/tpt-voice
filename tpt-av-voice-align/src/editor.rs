//! Text-based editing: diff an edited transcript against the aligned
//! original and emit [`AudioEdit`] operations (spec §4.5).
//!
//! The [`AudioEdit`] list is the integration contract with the
//! `tpt-audio` timeline: a `Delete` removes `[start_time, end_time)`
//! from the clip; an `Insert` splices `new_audio` (when present) at
//! `start_time`; a `Replace` removes the range and splices in the new
//! audio. `tpt-audio` consumes these by mutating its timeline clips —
//! delete = split/remove the time range; insert/replace = resize clips
//! and drop in replacement PCM (synthesized via `tpt-av-voice-tts` when
//! the edit adds new words).

use serde::{Deserialize, Serialize};

use crate::aligner::AlignedTranscript;

/// Edit type (spec §4.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditType {
    /// Remove the time range.
    Delete,
    /// Splice new audio in at `start_time`.
    Insert,
    /// Remove the range and splice new audio in its place.
    Replace,
}

/// An audio edit operation derived from text editing (spec §4.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEdit {
    /// Edit type (delete, insert, replace).
    pub edit_type: EditType,
    /// Start time (in seconds).
    pub start_time: f64,
    /// End time (in seconds).
    pub end_time: f64,
    /// New audio content (for insert/replace; `None` until a
    /// synthesizer fills it).
    pub new_audio: Option<Vec<f32>>,
    /// The inserted/replacement words (empty for [`EditType::Delete`]).
    /// A synthesizer (e.g. `tpt-av-voice-tts`) renders these into
    /// [`AudioEdit::new_audio`]; see the `tpt-voice` facade crate's
    /// `pipeline::fill_insertions`.
    #[serde(default)]
    pub words: Vec<String>,
}

/// Text-based editor over an aligned transcript (spec §4.5).
pub struct TextBasedEditor {
    /// Original transcript.
    transcript: AlignedTranscript,
    /// Edited transcript (user modifications).
    edited_transcript: String,
}

impl TextBasedEditor {
    /// Creates an editor over an aligned transcript.
    pub fn new(transcript: AlignedTranscript) -> Self {
        TextBasedEditor {
            transcript,
            edited_transcript: String::new(),
        }
    }

    /// The last edited text set via [`TextBasedEditor::apply_edits`].
    pub fn edited_transcript(&self) -> &str {
        &self.edited_transcript
    }

    /// Borrows the original aligned transcript.
    pub fn transcript(&self) -> &AlignedTranscript {
        &self.transcript
    }

    /// Applies text edits and returns the corresponding audio edit
    /// operations, in timeline order.
    ///
    /// The diff is word-level (whitespace tokens) via longest common
    /// subsequence: deletions map to the original words' time ranges,
    /// insertions land between their new neighbors, and an adjacent
    /// delete+insert pair merges into a `Replace`.
    pub fn apply_edits(&mut self, edited_text: &str) -> Vec<AudioEdit> {
        self.edited_transcript = edited_text.to_string();
        let original = self.transcript.word_strings();
        let edited: Vec<&str> = edited_text.split_whitespace().collect();
        diff_to_edits(&original, &edited, &self.transcript.words)
    }
}

/// Word-level LCS diff → ordered [`AudioEdit`]s.
fn diff_to_edits(
    original: &[&str],
    edited: &[&str],
    timings: &[tpt_av_voice_utils::WordTimestamp],
) -> Vec<AudioEdit> {
    let n = original.len();
    let m = edited.len();
    // lcs[i][j] = LCS length of original[i..] and edited[j..].
    let mut lcs = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if original[i] == edited[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    // Walk both sequences, batching contiguous runs.
    #[derive(Debug)]
    enum Run {
        Delete { start: f64, end: f64 },
        Insert { at: f64, words: Vec<String> },
    }
    let mut runs: Vec<Run> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    let time_of = |idx: usize, end: bool| -> f64 {
        match timings.get(idx) {
            Some(w) => {
                if end {
                    w.end_time
                } else {
                    w.start_time
                }
            }
            None => timings.last().map(|w| w.end_time).unwrap_or(0.0),
        }
    };
    while i < n || j < m {
        if i < n && j < m && original[i] == edited[j] {
            i += 1;
            j += 1;
        } else if i < n && (j >= m || lcs[i + 1][j] >= lcs[i][j + 1]) {
            // Deletion run of original[i..k).
            let start = time_of(i, false);
            let mut k = i;
            while k < n
                && (j >= m || lcs[k + 1][j] >= lcs[k][j + 1])
                && (j >= m || original[k] != edited[j])
            {
                k += 1;
            }
            let end = time_of(k - 1, true);
            runs.push(Run::Delete { start, end });
            i = k;
        } else {
            // Insertion run of edited[j..k).
            let at = time_of(i.min(n), false);
            let mut k = j;
            let mut words = Vec::new();
            while k < m && (i >= n || lcs[i][k + 1] < lcs[i + 1][k] || original[i] != edited[k]) {
                words.push(edited[k].to_string());
                k += 1;
            }
            runs.push(Run::Insert { at, words });
            j = k;
        }
    }

    // Merge adjacent delete+insert into replace, then emit ops.
    let mut ops: Vec<AudioEdit> = Vec::new();
    let mut idx = 0usize;
    while idx < runs.len() {
        let is_delete = matches!(&runs[idx], Run::Delete { .. });
        let next_is_insert = matches!(runs.get(idx + 1), Some(Run::Insert { .. }));
        if is_delete && next_is_insert {
            let (start, end) = match &runs[idx] {
                Run::Delete { start, end } => (*start, *end),
                _ => unreachable!(),
            };
            let at = match &runs[idx + 1] {
                Run::Insert { at, .. } => *at,
                _ => unreachable!(),
            };
            if at <= end + 1e-9 {
                let insert_words = match &runs[idx + 1] {
                    Run::Insert { words, .. } => words.clone(),
                    _ => unreachable!(),
                };
                ops.push(AudioEdit {
                    edit_type: EditType::Replace,
                    start_time: start,
                    end_time: end,
                    new_audio: None,
                    words: insert_words,
                });
                idx += 2;
                continue;
            }
        }
        match &runs[idx] {
            Run::Delete { start, end } => ops.push(AudioEdit {
                edit_type: EditType::Delete,
                start_time: *start,
                end_time: *end,
                new_audio: None,
                words: Vec::new(),
            }),
            Run::Insert { at, words } => ops.push(AudioEdit {
                edit_type: EditType::Insert,
                start_time: *at,
                end_time: *at,
                new_audio: None,
                words: words.clone(),
            }),
        }
        idx += 1;
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aligner::AlignedTranscript;
    use tpt_av_voice_utils::WordTimestamp;

    fn transcript() -> AlignedTranscript {
        let words = ["hello", "brave", "new", "world", "today"]
            .iter()
            .enumerate()
            .map(|(i, w)| WordTimestamp::new(*w, i as f64, (i + 1) as f64, 1.0))
            .collect();
        AlignedTranscript {
            words,
            text: "hello brave new world today".into(),
        }
    }

    #[test]
    fn delete_maps_to_word_range() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("hello brave world today");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].edit_type, EditType::Delete);
        assert!((ops[0].start_time - 2.0).abs() < 1e-9, "{:?}", ops[0]);
        assert!((ops[0].end_time - 3.0).abs() < 1e-9);
        assert!(ops[0].new_audio.is_none());
        assert_eq!(editor.edited_transcript(), "hello brave world today");
    }

    #[test]
    fn delete_run_spans_words() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("hello today");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].edit_type, EditType::Delete);
        assert!((ops[0].start_time - 1.0).abs() < 1e-9);
        assert!((ops[0].end_time - 4.0).abs() < 1e-9);
    }

    #[test]
    fn insert_lands_between_neighbors() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("hello brave big new world today");
        let ins: Vec<&AudioEdit> = ops
            .iter()
            .filter(|o| o.edit_type == EditType::Insert)
            .collect();
        assert_eq!(ins.len(), 1);
        assert!(
            (ins[0].start_time - 2.0).abs() < 1e-9,
            "insert at {:?}",
            ins[0]
        );
        assert!(ins[0].end_time >= ins[0].start_time);
    }

    #[test]
    fn replace_word() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("hello huge new world today");
        let rep: Vec<&AudioEdit> = ops
            .iter()
            .filter(|o| o.edit_type == EditType::Replace)
            .collect();
        assert_eq!(rep.len(), 1, "ops: {ops:?}");
        assert!((rep[0].start_time - 1.0).abs() < 1e-9);
        assert!((rep[0].end_time - 2.0).abs() < 1e-9);
    }

    #[test]
    fn no_change_no_ops() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("hello brave new world today");
        assert!(ops.is_empty());
    }

    #[test]
    fn delete_everything_and_clear() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].edit_type, EditType::Delete);
        assert!((ops[0].start_time - 0.0).abs() < 1e-9);
        assert!((ops[0].end_time - 5.0).abs() < 1e-9);
    }

    #[test]
    fn append_at_end() {
        let mut editor = TextBasedEditor::new(transcript());
        let ops = editor.apply_edits("hello brave new world today folks");
        let ins: Vec<&AudioEdit> = ops
            .iter()
            .filter(|o| o.edit_type == EditType::Insert)
            .collect();
        assert_eq!(ins.len(), 1);
        assert!((ins[0].start_time - 5.0).abs() < 1e-9);
    }
}
