//! Transcript data structures — the shared output vocabulary of the engine.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a [`TranscriptSegment`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct SegmentId(pub u64);

impl fmt::Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seg-{:06}", self.0)
    }
}

impl From<u64> for SegmentId {
    fn from(v: u64) -> Self {
        SegmentId(v)
    }
}

/// A complete transcript of an audio file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Transcript {
    /// All segments in the transcript.
    pub segments: Vec<TranscriptSegment>,
    /// Full text (concatenation of all segments).
    pub full_text: String,
    /// Language code (e.g. `"en"`, `"es"`, `"fr"`).
    pub language: String,
    /// Aggregate confidence score (0.0 to 1.0).
    pub confidence: f32,
}

impl Transcript {
    /// Creates an empty transcript in the given language.
    pub fn new(language: impl Into<String>) -> Self {
        Transcript {
            segments: Vec::new(),
            full_text: String::new(),
            language: language.into(),
            confidence: 0.0,
        }
    }

    /// Appends a segment, assigns it the next free [`SegmentId`], and
    /// refreshes [`Transcript::full_text`].
    pub fn push_segment(&mut self, mut segment: TranscriptSegment) {
        segment.id = SegmentId(self.segments.len() as u64);
        self.segments.push(segment);
        self.refresh_full_text();
        self.recompute_confidence();
    }

    /// Recomputes the aggregate confidence as the duration-weighted mean
    /// of segment confidences.
    pub fn recompute_confidence(&mut self) {
        let mut weighted = 0.0f64;
        let mut total = 0.0f64;
        for s in &self.segments {
            let w = (s.end_time - s.start_time).max(0.0);
            weighted += f64::from(s.confidence) * w;
            total += w;
        }
        self.confidence = if total > 0.0 {
            (weighted / total) as f32
        } else {
            0.0
        };
    }

    /// Total duration covered by the transcript (end of last segment).
    pub fn duration_secs(&self) -> f64 {
        self.segments
            .iter()
            .map(|s| s.end_time)
            .fold(0.0_f64, f64::max)
    }

    /// Flattens all word-level timestamps, if present.
    pub fn words(&self) -> impl Iterator<Item = &WordTimestamp> {
        self.segments
            .iter()
            .filter_map(|s| s.words.as_ref())
            .flatten()
    }

    /// Sorts segments by start time and refreshes derived fields.
    pub fn sort_by_time(&mut self) {
        self.segments.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        self.refresh_full_text();
    }

    /// Rebuilds [`Transcript::full_text`] from the current segments.
    pub fn refresh_full_text(&mut self) {
        self.full_text = self
            .segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
    }
}

/// A single segment of the transcript (sentence or phrase).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Unique segment identifier.
    pub id: SegmentId,
    /// Start time (in seconds).
    pub start_time: f64,
    /// End time (in seconds).
    pub end_time: f64,
    /// Text content of the segment.
    pub text: String,
    /// Speaker label (if diarization was performed).
    pub speaker: Option<crate::SpeakerLabel>,
    /// Confidence score for this segment (0.0 to 1.0).
    pub confidence: f32,
    /// Word-level timestamps (optional).
    pub words: Option<Vec<WordTimestamp>>,
}

impl TranscriptSegment {
    /// Creates a new segment without speaker or word timestamps.
    pub fn new(start_time: f64, end_time: f64, text: impl Into<String>, confidence: f32) -> Self {
        TranscriptSegment {
            id: SegmentId::default(),
            start_time,
            end_time,
            text: text.into(),
            speaker: None,
            confidence,
            words: None,
        }
    }

    /// Time range covered by this segment.
    pub fn time_range(&self) -> crate::TimeRange {
        crate::TimeRange::new(self.start_time, self.end_time)
    }
}

/// Word-level timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// The word.
    pub word: String,
    /// Start time (in seconds).
    pub start_time: f64,
    /// End time (in seconds).
    pub end_time: f64,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

impl WordTimestamp {
    /// Creates a word timestamp.
    pub fn new(word: impl Into<String>, start_time: f64, end_time: f64, confidence: f32) -> Self {
        WordTimestamp {
            word: word.into(),
            start_time,
            end_time,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_segment_assigns_ids_and_text() {
        let mut t = Transcript::new("en");
        t.push_segment(TranscriptSegment::new(0.0, 1.0, "Hello", 0.9));
        t.push_segment(TranscriptSegment::new(1.0, 2.0, " world.", 0.8));
        assert_eq!(t.segments[0].id, SegmentId(0));
        assert_eq!(t.segments[1].id, SegmentId(1));
        assert_eq!(t.full_text, "Hello world.");
        assert!((t.confidence - 0.85).abs() < 1e-6);
        assert!((t.duration_secs() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn words_flattening() {
        let mut t = Transcript::new("en");
        let mut seg = TranscriptSegment::new(0.0, 1.0, "hi there", 1.0);
        seg.words = Some(vec![
            WordTimestamp::new("hi", 0.0, 0.4, 1.0),
            WordTimestamp::new("there", 0.4, 1.0, 1.0),
        ]);
        t.push_segment(seg);
        assert_eq!(t.words().count(), 2);
        assert_eq!(t.words().next().unwrap().word, "hi");
    }

    #[test]
    fn segment_id_formatting() {
        assert_eq!(SegmentId(42).to_string(), "seg-000042");
    }

    #[test]
    fn sort_by_time_and_refresh() {
        let mut t = Transcript::new("en");
        let a = TranscriptSegment::new(2.0, 3.0, "second", 1.0);
        let b = TranscriptSegment::new(0.0, 1.0, "first", 1.0);
        t.segments = vec![a, b];
        t.sort_by_time();
        assert_eq!(t.full_text, "first second");
    }
}
