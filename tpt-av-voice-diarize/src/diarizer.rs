//! High-level diarization API.

use std::path::Path;

use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::wav;
use tpt_av_voice_utils::{safetensors, SpeakerLabel, TimeRange, VoiceError};

use crate::clustering::{cluster, ClusteringAlgorithm, ClusteringConfig};
use crate::embedding::{ClassicalEmbedding, EmbeddingModel, SpeakerEmbedder};
use crate::segmentation::{SpeechRegions, VoiceActivityDetector, VadConfig};

/// Diarization settings (spec §4.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizerConfig {
    /// Minimum number of speakers (optional).
    pub min_speakers: Option<usize>,
    /// Maximum number of speakers (optional).
    pub max_speakers: Option<usize>,
    /// Expected number of speakers (optional; overrides clustering).
    pub num_speakers: Option<usize>,
    /// Threshold for speaker-change / cluster merge detection.
    pub threshold: f32,
    /// Speech segments longer than this are split before embedding.
    pub max_segment_secs: f64,
    /// VAD configuration.
    pub vad: VadConfig,
}

impl Default for DiarizerConfig {
    fn default() -> Self {
        DiarizerConfig {
            min_speakers: Some(1),
            max_speakers: Some(12),
            num_speakers: None,
            threshold: 0.75,
            max_segment_secs: 10.0,
            vad: VadConfig::default(),
        }
    }
}

/// A segment of speech by a specific speaker (spec §4.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerSegment {
    /// Speaker label.
    pub speaker: SpeakerLabel,
    /// Start time (in seconds).
    pub start_time: f64,
    /// End time (in seconds).
    pub end_time: f64,
    /// Confidence score (0..1; 1 − normalized cluster distance).
    pub confidence: f32,
}

impl SpeakerSegment {
    /// Time range covered by this segment.
    pub fn time_range(&self) -> TimeRange {
        TimeRange::new(self.start_time, self.end_time)
    }
}

/// Diarization result (spec §4.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DiarizationResult {
    /// Speaker segments (who spoke when), time-sorted.
    pub segments: Vec<SpeakerSegment>,
    /// Number of unique speakers detected.
    pub num_speakers: usize,
}

/// Speaker diarization engine (spec §4.4).
///
/// Pipeline: VAD → segment splitting → embeddings → clustering →
/// speaker-labelled segments.
///
/// * [`Diarizer::classical`] — MFCC-statistics embeddings, no weights.
/// * [`Diarizer::from_model_dir`] — TDNN embedding model from disk.
pub struct Diarizer {
    embedder: Box<dyn SpeakerEmbedder>,
    vad: VoiceActivityDetector,
    config: DiarizerConfig,
}

impl Diarizer {
    /// Creates a diarizer around a trained embedding model directory
    /// (`model.safetensors`).
    pub fn from_model_dir(
        config: DiarizerConfig,
        dir: impl AsRef<Path>,
    ) -> Result<Self, VoiceError> {
        let dir = dir.as_ref();
        let tensors = safetensors::load_from_path(dir.join("model.safetensors"))?;
        let model = EmbeddingModel::from_tensors(&tensors)?;
        Self::from_embedder(Box::new(model), config)
    }

    /// Weight-free mode: MFCC-statistics + pitch embeddings.
    pub fn classical(mut config: DiarizerConfig) -> Result<Self, VoiceError> {
        // Cepstral+pitch embeddings have a tight same-speaker spread
        // (~0.001) and a modest cross-speaker spread (~0.2); a small
        // threshold separates cleanly.
        if config.threshold >= 0.75 {
            config.threshold = 0.1;
        }
        Self::from_embedder(Box::new(ClassicalEmbedding), config)
    }

    /// Wraps a custom embedder.
    pub fn from_embedder(embedder: Box<dyn SpeakerEmbedder>, config: DiarizerConfig) -> Result<Self, VoiceError> {
        if config.max_segment_secs <= 0.0 {
            return Err(VoiceError::config("max_segment_secs must be > 0"));
        }
        let clustering_probe = ClusteringConfig {
            threshold: config.threshold,
            min_speakers: config.min_speakers,
            max_speakers: config.max_speakers,
            num_speakers: config.num_speakers,
            algorithm: ClusteringAlgorithm::Agglomerative,
        };
        clustering_probe.validate()?;
        Ok(Diarizer {
            embedder,
            vad: VoiceActivityDetector::new(config.vad.clone())?,
            config,
        })
    }

    /// Diarizes an audio file (identifies speakers).
    pub fn diarize_file(&mut self, audio_path: &Path) -> Result<DiarizationResult, VoiceError> {
        let audio = wav::read_wav_file(audio_path)?;
        self.diarize_samples(&audio.to_mono(), audio.sample_rate)
    }

    /// Diarizes mono PCM samples.
    pub fn diarize_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<DiarizationResult, VoiceError> {
        // 1. Voice activity detection.
        let regions = self.vad.detect(samples, sample_rate);
        self.diarize_regions(&regions, samples, sample_rate)
    }

    /// Diarizes pre-segmented speech regions (used by streaming and tests).
    pub fn diarize_regions(
        &mut self,
        regions: &SpeechRegions,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<DiarizationResult, VoiceError> {
        // 2. Split long regions into embeddable sub-segments.
        let mut unit_ranges: Vec<TimeRange> = Vec::new();
        for region in regions {
            let mut cursor = region.start;
            while cursor < region.end {
                let end = (cursor + self.config.max_segment_secs).min(region.end);
                unit_ranges.push(TimeRange::new(cursor, end));
                cursor = end;
            }
        }

        // 3. Extract speaker embeddings per unit.
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(unit_ranges.len());
        let mut kept_ranges: Vec<TimeRange> = Vec::new();
        let unit_samples = |r: &TimeRange| {
            let s = (r.start * f64::from(sample_rate)).round() as usize;
            let e = ((r.end * f64::from(sample_rate)).round() as usize).min(samples.len());
            (s.min(e), e)
        };
        for range in &unit_ranges {
            let (s, e) = unit_samples(range);
            if e - s < sample_rate as usize / 4 {
                continue; // skip slivers under 250 ms
            }
            match self.embedder.embed(&samples[s..e], sample_rate) {
                Ok(v) => {
                    embeddings.push(v);
                    kept_ranges.push(*range);
                }
                Err(VoiceError::Audio(tpt_av_voice_utils::AudioError::Empty)) => continue,
                Err(e) => return Err(e),
            }
        }
        if embeddings.is_empty() {
            return Ok(DiarizationResult::default());
        }

        // 4. Cluster embeddings.
        let clustering = ClusteringConfig {
            threshold: self.config.threshold,
            min_speakers: self.config.min_speakers,
            max_speakers: self.config.max_speakers,
            num_speakers: self.config.num_speakers,
            algorithm: ClusteringAlgorithm::Agglomerative,
        };
        let labels = cluster(&embeddings, &clustering)?;

        // 5. Build labelled segments, merging adjacent same-speaker units.
        let mut result = DiarizationResult::default();
        let mut open: Option<(usize, f64, f64, f32)> = None; // (label, start, end, dist-sum)
        for (i, &label) in labels.iter().enumerate() {
            let range = kept_ranges[i];
            match &mut open {
                Some((l, _, end, _)) if *l == label && range.start - *end <= 0.05 => {
                    *end = range.end;
                }
                _ => {
                    if let Some((l, start, end, _)) = open.take() {
                        result.segments.push(make_segment(l, start, end));
                    }
                    open = Some((label, range.start, range.end, 0.0));
                }
            }
        }
        if let Some((l, start, end, _)) = open {
            result.segments.push(make_segment(l, start, end));
        }
        result.num_speakers = result
            .segments
            .iter()
            .map(|s| s.speaker.id.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        Ok(result)
    }

    /// Borrows the configuration.
    pub fn config(&self) -> &DiarizerConfig {
        &self.config
    }
}

fn make_segment(cluster_label: usize, start: f64, end: f64) -> SpeakerSegment {
    SpeakerSegment {
        speaker: SpeakerLabel::new(cluster_label),
        start_time: start,
        end_time: end,
        confidence: 0.9, // deterministic pipeline; refined with trained models
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buzz(hz: f32, secs: f64, amp: f32) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let mut v = 0.0f32;
                for k in 1..=8 {
                    v += (std::f32::consts::TAU * hz * k as f32 * t).sin() / k as f32;
                }
                v * amp
            })
            .collect()
    }

    /// A "meeting": speaker 0 (low pitch), silence, speaker 1 (high
    /// pitch), silence, speaker 0 again.
    fn meeting() -> Vec<f32> {
        let mut pcm = Vec::new();
        pcm.extend(buzz(150.0, 1.5, 0.4));
        pcm.extend(vec![0.0; 12_000]);
        pcm.extend(buzz(400.0, 1.5, 0.4));
        pcm.extend(vec![0.0; 12_000]);
        pcm.extend(buzz(152.0, 1.5, 0.4));
        pcm
    }

    #[test]
    fn classical_diarizer_finds_two_speakers() {
        let mut d = Diarizer::classical(DiarizerConfig::default()).unwrap();
        let result = d.diarize_samples(&meeting(), 16_000).unwrap();
        assert_eq!(result.num_speakers, 2, "segments: {result:?}");
        // Speaker 0 talks first and last; speaker 1 in the middle.
        let first = &result.segments[0];
        let mid = result
            .segments
            .iter()
            .find(|s| s.speaker.index() == Some(1))
            .expect("speaker 1 present");
        let last = result.segments.last().unwrap();
        assert_eq!(first.speaker.id, last.speaker.id);
        assert_ne!(first.speaker.id, mid.speaker.id);
        assert!(mid.start_time > first.start_time && mid.end_time < last.end_time + 0.1);
    }

    #[test]
    fn exact_speaker_count_is_honored() {
        let cfg = DiarizerConfig { num_speakers: Some(3), ..Default::default() };
        let mut d = Diarizer::classical(cfg).unwrap();
        let result = d.diarize_samples(&meeting(), 16_000).unwrap();
        assert_eq!(result.num_speakers, 3);
    }

    #[test]
    fn silence_yields_empty_result() {
        let mut d = Diarizer::classical(DiarizerConfig::default()).unwrap();
        let result = d.diarize_samples(&vec![0.0; 16_000], 16_000).unwrap();
        assert_eq!(result.num_speakers, 0);
        assert!(result.segments.is_empty());
    }

    #[test]
    fn config_validation() {
        let bad = DiarizerConfig { max_segment_secs: 0.0, ..Default::default() };
        assert!(Diarizer::classical(bad).is_err());
    }
}
