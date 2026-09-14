//! Streaming diarization: online speaker labeling over windowed chunks.
//!
//! [`StreamingDiarizer`] buffers audio into overlapping windows, embeds
//! each window, and assigns it to the nearest known speaker centroid or
//! opens a new speaker when nothing is close enough (and the speaker
//! budget allows). Centroids are running means, so labels stabilize as
//! evidence accumulates.

use tpt_av_voice_utils::{AudioChunk, SpeakerLabel, VoiceError};

use crate::diarizer::{DiarizerConfig, SpeakerSegment};
use crate::embedding::{cosine_distance, l2_normalize, SpeakerEmbedder};

/// Online speaker-assignment tuning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamDiarizerConfig {
    /// Window length in seconds.
    pub window_secs: f64,
    /// Hop between windows in seconds (overlap smooths boundaries).
    pub hop_secs: f64,
    /// Cosine distance at which a window joins an existing speaker.
    pub join_threshold: f32,
}

impl Default for StreamDiarizerConfig {
    fn default() -> Self {
        StreamDiarizerConfig { window_secs: 1.5, hop_secs: 0.75, join_threshold: 0.9 }
    }
}

/// Incremental "who is speaking" tracker.
pub struct StreamingDiarizer {
    embedder: Box<dyn SpeakerEmbedder>,
    stream: StreamDiarizerConfig,
    limits: (Option<usize>, Option<usize>), // (min, max)
    buffer: Vec<f32>,
    sample_rate: u32,
    buffer_start: f64,
    /// Per-speaker centroid embeddings.
    centroids: Vec<Vec<f32>>,
    /// Total windows assigned per speaker.
    counts: Vec<usize>,
    timeline: Vec<SpeakerSegment>,
    /// Window index → label, for downstream smoothing.
    window_labels: Vec<usize>,
    windows_seen: usize,
    /// Distance from the most recent window to its assigned centroid.
    last_distance: f32,
    /// A window that was too far from every centroid; a new speaker is
    /// only opened once a second consecutive window agrees with it.
    pending: Option<PendingWindow>,
}

/// Candidate for a new speaker awaiting confirmation.
struct PendingWindow {
    embedding: Vec<f32>,
    distance: f32,
}

impl StreamingDiarizer {
    /// Builds a streaming diarizer from an embedder.
    pub fn new(
        embedder: Box<dyn SpeakerEmbedder>,
        config: StreamDiarizerConfig,
        diarizer: &DiarizerConfig,
    ) -> Result<Self, VoiceError> {
        if config.window_secs <= 0.0 || config.hop_secs <= 0.0 {
            return Err(VoiceError::config("window_secs and hop_secs must be > 0"));
        }
        if config.hop_secs > config.window_secs {
            return Err(VoiceError::config("hop_secs must not exceed window_secs"));
        }
        Ok(StreamingDiarizer {
            embedder,
            stream: config,
            limits: (diarizer.min_speakers, diarizer.max_speakers),
            buffer: Vec::new(),
            sample_rate: 16_000,
            buffer_start: 0.0,
            centroids: Vec::new(),
            counts: Vec::new(),
            timeline: Vec::new(),
            window_labels: Vec::new(),
            windows_seen: 0,
            last_distance: 1.0,
            pending: None,
        })
    }

    /// Weight-free shortcut mirroring [`crate::Diarizer::classical`].
    pub fn classical(diarizer: &DiarizerConfig) -> Result<Self, VoiceError> {
        // Same calibration as the batch classical mode: classical
        // embeddings sit at ~0.001 within a speaker and ~0.05+ across.
        let stream = StreamDiarizerConfig { join_threshold: 0.03, ..Default::default() };
        Self::new(Box::new(crate::embedding::ClassicalEmbedding), stream, diarizer)
    }

    /// Speakers discovered so far.
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }

    /// Labels assigned per window so far.
    pub fn window_labels(&self) -> &[usize] {
        &self.window_labels
    }

    /// Pushes a chunk; when a full window (+ hop) has accumulated, the
    /// window is embedded and labelled. Returns the window's speaker
    /// segment once processed.
    pub fn push(&mut self, chunk: &AudioChunk) -> Result<Option<SpeakerSegment>, VoiceError> {
        if chunk.sample_rate != 16_000 {
            return Err(VoiceError::config("streaming diarizer expects 16 kHz chunks"));
        }
        if self.buffer.is_empty() {
            self.buffer_start = chunk.start_time;
        }
        self.buffer.extend_from_slice(&chunk.samples);
        let window_samples = (self.stream.window_secs * f64::from(self.sample_rate)) as usize;
        if self.buffer.len() < window_samples {
            return Ok(None);
        }
        let segment = self.process_window()?;
        Ok(Some(segment))
    }

    fn process_window(&mut self) -> Result<SpeakerSegment, VoiceError> {
        let embedding = match self.embedder.embed(&self.buffer, self.sample_rate) {
            Ok(v) => v,
            Err(VoiceError::Audio(tpt_av_voice_utils::AudioError::Empty)) => {
                // All-zero window: keep going with a silent label.
                vec![0.0; 1]
            }
            Err(e) => return Err(e),
        };

        let label = self.assign(&embedding);
        let start = self.buffer_start;
        let end = start + self.stream.window_secs;

        let segment = SpeakerSegment {
            speaker: SpeakerLabel::new(label),
            start_time: start,
            end_time: end,
            confidence: (1.0 - self.last_distance).clamp(0.0, 1.0),
        };
        self.timeline.push(segment.clone());
        self.window_labels.push(label);
        self.windows_seen += 1;

        // Advance by hop.
        let hop_samples = (self.stream.hop_secs * f64::from(self.sample_rate)) as usize;
        self.buffer.drain(..hop_samples.min(self.buffer.len()));
        self.buffer_start = end - self.stream.hop_secs;
        Ok(segment)
    }

    fn assign(&mut self, embedding: &[f32]) -> usize {
        let max_speakers = self.limits.1.unwrap_or(usize::MAX);
        let nearest = self.nearest_centroid(embedding);
        match nearest {
            // Close enough to a known speaker: join it.
            Some((label, d)) if d <= self.stream.join_threshold => {
                self.pending = None;
                self.last_distance = d;
                self.update_centroid(label, embedding);
                label
            }
            // Far from everything, but a previous window was far in the
            // same way: confirm a new speaker.
            _ if self.pending.as_ref().map_or(false, |p| {
                cosine_distance(embedding, &p.embedding) <= self.stream.join_threshold
            }) && self.centroids.len() < max_speakers =>
            {
                let mut v = embedding.to_vec();
                l2_normalize(&mut v);
                self.centroids.push(v);
                self.counts.push(0);
                let label = self.centroids.len() - 1;
                self.last_distance = 0.0;
                self.pending = None;
                self.update_centroid(label, embedding);
                label
            }
            // Far from everything: stash as a pending candidate and
            // provisionally stick with the nearest known speaker.
            _ => {
                self.pending = Some(PendingWindow {
                    embedding: embedding.to_vec(),
                    distance: nearest.map_or(f32::INFINITY, |(_, d)| d),
                });
                let (label, d) = nearest.unwrap_or((0, 1.0));
                self.last_distance = d;
                label
            }
        }
    }

    fn nearest_centroid(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_distance(embedding, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn update_centroid(&mut self, label: usize, embedding: &[f32]) {
        self.counts[label] += 1;
        let n = self.counts[label] as f32;
        let centroid_len = self.centroids[label].len();
        {
            let c = &mut self.centroids[label];
            for (ci, &x) in c.iter_mut().zip(embedding).take(centroid_len) {
                *ci += (x - *ci) / n;
            }
        }
        l2_normalize(&mut self.centroids[label]);
    }

    /// Flushed timeline (all windows so far).
    pub fn timeline(&self) -> &[SpeakerSegment] {
        &self.timeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarizer::DiarizerConfig;

    fn buzz(hz: f32, secs: f64) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let mut v = 0.0f32;
                for k in 1..=8 {
                    v += (std::f32::consts::TAU * hz * k as f32 * t).sin() / k as f32;
                }
                v * 0.3
            })
            .collect()
    }

    #[test]
    fn online_assignment_finds_two_speakers() {
        let cfg = DiarizerConfig::default();
        let mut stream = StreamingDiarizer::classical(&cfg).unwrap();
        let mut pcm = Vec::new();
        pcm.extend(buzz(150.0, 3.0)); // speaker 0
        pcm.extend(buzz(400.0, 3.0)); // speaker 1
        let mut labels = Vec::new();
        for (s, c) in pcm.chunks(8_000).enumerate() {
            let chunk = AudioChunk::new(c.to_vec(), 16_000, 0.0, s as u64);
            if let Some(seg) = stream.push(&chunk).unwrap() {
                labels.push(seg.speaker.index().unwrap());
            }
        }
        assert!(stream.num_speakers() == 2, "speakers: {}", stream.num_speakers());
        // First windows are speaker 0, later windows speaker 1.
        assert_eq!(labels[0], 0);
        assert_eq!(*labels.last().unwrap(), 1);
    }

    #[test]
    fn rejects_bad_config() {
        let cfg = DiarizerConfig::default();
        let bad = StreamDiarizerConfig { hop_secs: 2.0, window_secs: 1.0, ..Default::default() };
        assert!(StreamingDiarizer::new(Box::new(crate::embedding::ClassicalEmbedding), bad, &cfg).is_err());
    }

    #[test]
    fn wrong_rate_rejected() {
        let cfg = DiarizerConfig::default();
        let mut stream = StreamingDiarizer::classical(&cfg).unwrap();
        let bad = AudioChunk::new(vec![0.0; 100], 22_050, 0.0, 0);
        assert!(stream.push(&bad).is_err());
    }
}
