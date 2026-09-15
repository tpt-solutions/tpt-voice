//! Voice profiles: who is speaking.

use serde::{Deserialize, Serialize};

/// A synthesized voice's characteristics. Neural voices will extend
/// this; the formant synthesizer uses F0 and formant scaling today.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceProfile {
    /// Profile name (e.g. `"default-male"`).
    pub name: String,
    /// Baseline pitch in Hz.
    pub base_f0: f32,
    /// Vocal-tract length factor: >1 lengthens the tract (lower
    /// formants, larger-sounding speaker), <1 raises formants.
    pub formant_scale: f32,
    /// 0..1 mix of aspiration noise into voiced frames.
    pub breathiness: f32,
}

impl VoiceProfile {
    /// Default lower-pitched voice.
    pub fn default_male() -> Self {
        VoiceProfile {
            name: "default-male".into(),
            base_f0: 110.0,
            formant_scale: 1.0,
            breathiness: 0.05,
        }
    }

    /// Default higher-pitched voice.
    pub fn default_female() -> Self {
        VoiceProfile {
            name: "default-female".into(),
            base_f0: 200.0,
            formant_scale: 1.15,
            breathiness: 0.07,
        }
    }

    /// Classic robotic voice (no vibrato, no breath).
    pub fn robot() -> Self {
        VoiceProfile {
            name: "robot".into(),
            base_f0: 95.0,
            formant_scale: 1.0,
            breathiness: 0.0,
        }
    }

    /// All built-in profiles.
    pub fn presets() -> Vec<VoiceProfile> {
        vec![Self::default_male(), Self::default_female(), Self::robot()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid() {
        for p in VoiceProfile::presets() {
            assert!(p.base_f0 > 50.0 && p.base_f0 < 400.0);
            assert!((0.5..2.0).contains(&p.formant_scale));
            assert!((0.0..=1.0).contains(&p.breathiness));
            assert!(!p.name.is_empty());
        }
    }
}
