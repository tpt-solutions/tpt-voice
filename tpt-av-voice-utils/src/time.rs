//! Time and sample-index utilities.
//!
//! Timestamps throughout `tpt-voice` are `f64` seconds; sample positions
//! are `usize`/`u32` indices. These helpers convert between them and
//! provide the [`TimeRange`] primitive used by segmentation and editing.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Sample rate used by the models in this workspace (Whisper-style).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Converts a sample index to seconds.
#[inline]
pub fn samples_to_secs(samples: usize, sample_rate: u32) -> f64 {
    samples as f64 / f64::from(sample_rate)
}

/// Converts seconds to a (rounded) sample index.
#[inline]
pub fn secs_to_samples(secs: f64, sample_rate: u32) -> usize {
    (secs.max(0.0) * f64::from(sample_rate)).round() as usize
}

/// A half-open time range `[start, end)` in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time in seconds (inclusive).
    pub start: f64,
    /// End time in seconds (exclusive; `start == end` for an empty range).
    pub end: f64,
}

impl TimeRange {
    /// Creates a new range. `start` is clamped so it never exceeds `end`.
    pub fn new(start: f64, end: f64) -> Self {
        TimeRange {
            start: start.min(end),
            end,
        }
    }

    /// Duration of the range in seconds.
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// True when `t` lies inside the range.
    pub fn contains(&self, t: f64) -> bool {
        t >= self.start && t < self.end
    }

    /// Fraction of `self` covered by `other` (0.0 when `self` is empty).
    pub fn overlap_fraction(&self, other: &TimeRange) -> f64 {
        let dur = self.duration();
        if dur <= 0.0 {
            return 0.0;
        }
        let overlap = self.end.min(other.end) - self.start.max(other.start);
        (overlap.max(0.0) / dur).min(1.0)
    }

    /// Linear intersection of two ranges, if any.
    pub fn intersect(&self, other: &TimeRange) -> Option<TimeRange> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (end > start).then_some(TimeRange { start, end })
    }

    /// Union (bounding range) of two ranges.
    pub fn union(&self, other: &TimeRange) -> TimeRange {
        TimeRange {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl fmt::Display for TimeRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} → {}]", Timestamp(self.start), Timestamp(self.end))
    }
}

/// A formatted wall-clock timestamp (`mm:ss.mmm`, or `h:mm:ss.mmm` past
/// one hour), used for human-readable output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timestamp(pub f64);

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_ms = (self.0.max(0.0) * 1000.0).round() as u64;
        let ms = total_ms % 1000;
        let total_s = total_ms / 1000;
        let s = total_s % 60;
        let m = (total_s / 60) % 60;
        let h = total_s / 3600;
        if h > 0 {
            write!(f, "{h}:{m:02}:{s:02}.{ms:03}")
        } else {
            write!(f, "{m:02}:{s:02}.{ms:03}")
        }
    }
}

impl From<Duration> for Timestamp {
    fn from(d: Duration) -> Self {
        Timestamp(d.as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_conversions_round_trip() {
        assert!((samples_to_secs(16_000, 16_000) - 1.0).abs() < 1e-12);
        assert_eq!(secs_to_samples(2.5, 16_000), 40_000);
        assert_eq!(secs_to_samples(-1.0, 16_000), 0);
    }

    #[test]
    fn time_range_ops() {
        let a = TimeRange::new(1.0, 3.0);
        let b = TimeRange::new(2.0, 5.0);
        assert_eq!(a.duration(), 2.0);
        assert_eq!(a.intersect(&b).unwrap(), TimeRange::new(2.0, 3.0));
        assert_eq!(a.union(&b), TimeRange::new(1.0, 5.0));
        assert!((a.overlap_fraction(&b) - 0.5).abs() < 1e-9);
        assert!(a.contains(1.0) && !a.contains(3.0));

        // start > end is clamped
        assert_eq!(TimeRange::new(4.0, 1.0), TimeRange::new(1.0, 1.0));
        assert!(TimeRange::new(4.0, 1.0).intersect(&a).is_none());
    }

    #[test]
    fn timestamp_formatting() {
        assert_eq!(Timestamp(0.0).to_string(), "00:00.000");
        assert_eq!(Timestamp(61.5).to_string(), "01:01.500");
        assert_eq!(Timestamp(3661.25).to_string(), "1:01:01.250");
        assert_eq!(Timestamp(-3.0).to_string(), "00:00.000");
    }
}
