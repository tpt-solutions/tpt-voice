//! Synthesizer configuration.

use serde::{Deserialize, Serialize};

use crate::voice::VoiceProfile;

/// Text-to-speech synthesis settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthConfig {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Global speaking-rate multiplier (2.0 = twice as fast).
    pub speaking_rate: f32,
    /// Output gain (linear; 1.0 = unity).
    pub volume: f32,
    /// Sentence-final pitch drop in semitones (prosody).
    pub sentence_drop_semitones: f32,
    /// Question-final pitch rise in semitones.
    pub question_rise_semitones: f32,
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig {
            sample_rate: 22_050,
            speaking_rate: 1.0,
            volume: 0.9,
            sentence_drop_semitones: 3.0,
            question_rise_semitones: 4.0,
        }
    }
}

impl SynthConfig {
    /// Validates rates and gains.
    pub fn validate(&self) -> Result<(), tpt_av_voice_utils::VoiceError> {
        if self.sample_rate < 8_000 || self.sample_rate > 96_000 {
            return Err(tpt_av_voice_utils::VoiceError::config(
                "sample_rate must be between 8 kHz and 96 kHz",
            ));
        }
        if self.speaking_rate <= 0.0 || self.speaking_rate > 4.0 {
            return Err(tpt_av_voice_utils::VoiceError::config(
                "speaking_rate must be in (0, 4]",
            ));
        }
        if !(0.0..=4.0).contains(&self.volume) {
            return Err(tpt_av_voice_utils::VoiceError::config(
                "volume must be in [0, 4]",
            ));
        }
        Ok(())
    }

    /// Sample period in seconds.
    pub(crate) fn sample_period(&self) -> f32 {
        1.0 / self.sample_rate as f32
    }

    /// Combines a voice profile into an effective F0 baseline.
    pub(crate) fn base_f0(&self, voice: &VoiceProfile) -> f32 {
        voice.base_f0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation() {
        assert!(SynthConfig::default().validate().is_ok());
        assert!(SynthConfig {
            sample_rate: 1_000,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SynthConfig {
            speaking_rate: 0.0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SynthConfig {
            volume: 9.0,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
