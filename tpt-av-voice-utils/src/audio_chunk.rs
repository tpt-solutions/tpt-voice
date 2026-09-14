//! Streaming audio chunk types.
//!
//! All streaming APIs in `tpt-voice` consume and produce
//! [`Iterator<Item = AudioChunk>`]: mono `f32` PCM at an explicit sample
//! rate, with the chunk's position on the source timeline.

use serde::{Deserialize, Serialize};
/// A contiguous block of mono `f32` PCM audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioChunk {
    /// Interleaved-free (mono) PCM samples, nominally in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Position of the first sample on the source timeline, in seconds.
    pub start_time: f64,
    /// Monotonic sequence number assigned by the producer.
    pub sequence: u64,
}

impl AudioChunk {
    /// Creates a chunk starting at `start_time`.
    pub fn new(samples: Vec<f32>, sample_rate: u32, start_time: f64, sequence: u64) -> Self {
        AudioChunk { samples, sample_rate, start_time, sequence }
    }

    /// Number of samples in the chunk.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True when the chunk carries no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        crate::time::samples_to_secs(self.samples.len(), self.sample_rate)
    }

    /// Time at which this chunk ends on the source timeline.
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration()
    }

    /// Splits the chunk at the given sample offset. Returns
    /// `(head, tail)`; either side may be empty. `offset > len` is
    /// treated as `len`. The tail's `start_time` reflects its position
    /// on the source timeline.
    pub fn split_at(mut self, offset: usize) -> (AudioChunk, AudioChunk) {
        let offset = offset.min(self.samples.len());
        let mut head_samples = std::mem::take(&mut self.samples);
        let tail_samples = head_samples.split_off(offset);
        let tail_start = self.start_time + crate::time::samples_to_secs(offset, self.sample_rate);
        let head = AudioChunk {
            samples: head_samples,
            sample_rate: self.sample_rate,
            start_time: self.start_time,
            sequence: self.sequence,
        };
        let tail = AudioChunk {
            samples: tail_samples,
            sample_rate: self.sample_rate,
            start_time: tail_start,
            sequence: self.sequence,
        };
        (head, tail)
    }

    /// Appends another chunk's samples (same timeline position/rate).
    ///
    /// # Panics
    /// Panics if sample rates differ — mixing rates is a programmer error.
    pub fn append(&mut self, other: AudioChunk) {
        assert_eq!(self.sample_rate, other.sample_rate, "cannot append chunks with different sample rates");
        self.samples.extend_from_slice(&other.samples);
    }
}

impl Default for AudioChunk {
    fn default() -> Self {
        AudioChunk { samples: Vec::new(), sample_rate: crate::time::TARGET_SAMPLE_RATE, start_time: 0.0, sequence: 0 }
    }
}

/// Splits a PCM buffer into fixed-size [`AudioChunk`]s (the last chunk may
/// be shorter). Useful to feed file-based audio through streaming APIs.
pub fn chunk_pcm(
    samples: Vec<f32>,
    sample_rate: u32,
    chunk_samples: usize,
) -> impl Iterator<Item = AudioChunk> {
    assert!(chunk_samples > 0, "chunk_samples must be > 0");
    let mut start = 0usize;
    let mut seq = 0u64;
    std::iter::from_fn(move || {
        if start >= samples.len() {
            return None;
        }
        let end = (start + chunk_samples).min(samples.len());
        let slice = &samples[start..end];
        let chunk = AudioChunk::new(
            slice.to_vec(),
            sample_rate,
            crate::time::samples_to_secs(start, sample_rate),
            seq,
        );
        start = end;
        seq += 1;
        Some(chunk)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_chunk() {
        let chunk = AudioChunk::new((0..10).map(|i| i as f32).collect(), 100, 0.5, 3);
        let (head, tail) = chunk.split_at(4);
        assert_eq!(head.samples, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(head.start_time, 0.5);
        assert_eq!(tail.samples, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(tail.start_time, 0.54);
        assert_eq!(head.sequence, 3);
    }

    #[test]
    fn chunking_covers_everything() {
        let pcm: Vec<f32> = (0..2500).map(|i| (i as f32 * 0.01).sin()).collect();
        let chunks: Vec<AudioChunk> = chunk_pcm(pcm.clone(), 16_000, 1000).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2].len(), 500);
        assert!((chunks[1].start_time - 0.0625).abs() < 1e-12);
        let rejoined: Vec<f32> = chunks.into_iter().flat_map(|c| c.samples).collect();
        assert_eq!(rejoined, pcm);
    }

    #[test]
    fn append_preserves_rate() {
        let mut a = AudioChunk::new(vec![1.0], 8000, 0.0, 0);
        a.append(AudioChunk::new(vec![2.0], 8000, 0.000125, 1));
        assert_eq!(a.samples, vec![1.0, 2.0]);
    }

    #[test]
    fn empty_chunk() {
        let c = AudioChunk::default();
        assert!(c.is_empty());
        assert_eq!(c.sample_rate, 16_000);
        assert_eq!(c.duration(), 0.0);
        let (h, t) = c.split_at(10);
        assert!(h.is_empty() && t.is_empty());
    }
}
