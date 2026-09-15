//! Source-filter formant synthesis (Klatt-lite).
//!
//! Every phoneme maps to formant targets (F1/F2/F3), an excitation
//! type (voiced pulse train, fricative noise, or stop closure), and a
//! duration. The synthesizer glides the formants between phoneme
//! targets while driving three cascaded resonators, adds a pause at
//! punctuation, and applies a sentence-final pitch contour.
//!
//! The output is intelligible robotic speech — a working baseline that
//! neural vocoders will later replace behind the same API.

use tpt_av_voice_align::Phoneme;
use tpt_av_voice_utils::VoiceError;

use crate::config::SynthConfig;
use crate::g2p::{parse_text, Sentence};
use crate::voice::VoiceProfile;

/// Synthesized audio.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedAudio {
    /// Mono PCM in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Per-phoneme articulation spec.
#[derive(Debug, Clone, Copy)]
struct Segment {
    f1: f32,
    f2: f32,
    f3: f32,
    amplitude: f32,
    /// Voiced excitation fraction (1 = fully voiced, 0 = pure noise).
    voicing: f32,
    /// Closure (silence) fraction before the sound — stops.
    closure: f32,
    duration_secs: f32,
}

fn segment_for(p: Phoneme, voice: &VoiceProfile) -> Segment {
    use Phoneme::*;
    let s = voice.formant_scale;
    let base = Segment {
        f1: 500.0,
        f2: 1500.0,
        f3: 2500.0,
        amplitude: 1.0,
        voicing: 1.0,
        closure: 0.0,
        duration_secs: p.typical_frames() as f32 * 0.01,
    };
    match p {
        // Vowels (formants from Peterson–Barney-ish targets, scaled by
        // vocal-tract length).
        AA => Segment {
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            ..base
        },
        AE => Segment {
            f1: 660.0,
            f2: 1720.0,
            f3: 2410.0,
            ..base
        },
        AH => Segment {
            f1: 620.0,
            f2: 1220.0,
            f3: 2550.0,
            ..base
        },
        AO => Segment {
            f1: 570.0,
            f2: 840.0,
            f3: 2410.0,
            ..base
        },
        EH => Segment {
            f1: 530.0,
            f2: 1840.0,
            f3: 2480.0,
            ..base
        },
        ER => Segment {
            f1: 490.0,
            f2: 1350.0,
            f3: 1690.0,
            ..base
        },
        IH => Segment {
            f1: 390.0,
            f2: 1990.0,
            f3: 2550.0,
            ..base
        },
        IY => Segment {
            f1: 270.0,
            f2: 2290.0,
            f3: 3010.0,
            ..base
        },
        UH => Segment {
            f1: 440.0,
            f2: 1020.0,
            f3: 2240.0,
            ..base
        },
        UW => Segment {
            f1: 300.0,
            f2: 870.0,
            f3: 2240.0,
            ..base
        },
        // Diphthongs lean toward their offset vowel mid-segment (the
        // linear glide below handles the motion).
        AY => Segment {
            f1: 660.0,
            f2: 1500.0,
            f3: 2500.0,
            ..base
        },
        EY => Segment {
            f1: 450.0,
            f2: 1900.0,
            f3: 2500.0,
            ..base
        },
        OW => Segment {
            f1: 480.0,
            f2: 900.0,
            f3: 2400.0,
            ..base
        },
        OY => Segment {
            f1: 480.0,
            f2: 1100.0,
            f3: 2400.0,
            ..base
        },
        AW => Segment {
            f1: 650.0,
            f2: 1100.0,
            f3: 2400.0,
            ..base
        },
        // Sonorants.
        L => Segment {
            f1: 360.0,
            f2: 1300.0,
            f3: 2900.0,
            ..base
        },
        R => Segment {
            f1: 350.0,
            f2: 1100.0,
            f3: 1600.0,
            ..base
        },
        W => Segment {
            f1: 290.0,
            f2: 610.0,
            f3: 2150.0,
            ..base
        },
        Y => Segment {
            f1: 270.0,
            f2: 2300.0,
            f3: 3000.0,
            ..base
        },
        M => Segment {
            f1: 280.0,
            f2: 900.0,
            f3: 2200.0,
            ..base
        },
        N => Segment {
            f1: 280.0,
            f2: 1700.0,
            f3: 2600.0,
            ..base
        },
        NG => Segment {
            f1: 280.0,
            f2: 2300.0,
            f3: 2750.0,
            ..base
        },
        // Fricatives: noise-dominated.
        S => Segment {
            f1: 5500.0,
            f2: 5500.0,
            f3: 6500.0,
            amplitude: 0.35,
            voicing: 0.0,
            ..base
        },
        SH => Segment {
            f1: 2800.0,
            f2: 3200.0,
            f3: 4000.0,
            amplitude: 0.35,
            voicing: 0.0,
            ..base
        },
        F => Segment {
            f1: 1500.0,
            f2: 4000.0,
            f3: 6000.0,
            amplitude: 0.18,
            voicing: 0.0,
            ..base
        },
        TH => Segment {
            f1: 1600.0,
            f2: 4500.0,
            f3: 6500.0,
            amplitude: 0.15,
            voicing: 0.0,
            ..base
        },
        Z => Segment {
            f1: 5000.0,
            f2: 5500.0,
            f3: 6500.0,
            amplitude: 0.3,
            voicing: 0.4,
            ..base
        },
        ZH => Segment {
            f1: 2600.0,
            f2: 3200.0,
            f3: 4000.0,
            amplitude: 0.3,
            voicing: 0.4,
            ..base
        },
        V => Segment {
            f1: 350.0,
            f2: 1200.0,
            f3: 2500.0,
            amplitude: 0.5,
            voicing: 0.7,
            ..base
        },
        HH => Segment {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            amplitude: 0.25,
            voicing: 0.0,
            ..base
        },
        DH => Segment {
            f1: 350.0,
            f2: 1400.0,
            f3: 2600.0,
            amplitude: 0.4,
            voicing: 0.7,
            ..base
        },
        // Stops: brief closure then voiced release.
        P => Segment {
            closure: 0.05,
            f1: 400.0,
            f2: 1000.0,
            f3: 2200.0,
            amplitude: 0.5,
            ..base
        },
        T => Segment {
            closure: 0.05,
            f1: 350.0,
            f2: 1700.0,
            f3: 2600.0,
            amplitude: 0.5,
            ..base
        },
        K => Segment {
            closure: 0.05,
            f1: 350.0,
            f2: 1900.0,
            f3: 2300.0,
            amplitude: 0.5,
            ..base
        },
        B => Segment {
            closure: 0.04,
            f1: 350.0,
            f2: 1000.0,
            f3: 2300.0,
            ..base
        },
        D => Segment {
            closure: 0.04,
            f1: 300.0,
            f2: 1700.0,
            f3: 2700.0,
            ..base
        },
        G => Segment {
            closure: 0.04,
            f1: 300.0,
            f2: 1900.0,
            f3: 2400.0,
            ..base
        },
        JH => Segment {
            closure: 0.03,
            f1: 300.0,
            f2: 2000.0,
            f3: 2800.0,
            ..base
        },
        CH => Segment {
            closure: 0.05,
            f1: 2800.0,
            f2: 3200.0,
            f3: 4000.0,
            amplitude: 0.4,
            voicing: 0.0,
            ..base
        },
        SIL => Segment {
            amplitude: 0.0,
            ..base
        },
    }
    .scaled(s)
}

impl Segment {
    /// Scales all formant targets by the vocal-tract length factor.
    fn scaled(mut self, s: f32) -> Self {
        self.f1 /= s;
        self.f2 /= s;
        self.f3 /= s;
        self
    }
}

/// A second-order resonator (two-pole IIR band-pass).
#[derive(Debug, Clone, Copy, Default)]
struct Resonator {
    a: f32,
    b: f32,
    c: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    /// Recomputes coefficients for center `freq` Hz, bandwidth `bw` Hz.
    fn retune(&mut self, freq: f32, bw: f32, sample_rate: f32) {
        let r = (-std::f32::consts::PI * bw / sample_rate).exp();
        let theta = std::f32::consts::TAU * freq / sample_rate;
        self.c = -r * r;
        self.b = 2.0 * r * theta.cos();
        self.a = 1.0 - self.b - self.c;
        // Normalize peak gain to ~1.
        let gain = (1.0 - self.b - self.c).abs().max(1e-6);
        self.a /= gain;
    }

    fn tick(&mut self, x: f32) -> f32 {
        let y = self.a * x + self.b * self.y1 + self.c * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// The formant synthesizer.
pub struct FormantSynth {
    config: SynthConfig,
    voice: VoiceProfile,
    rng: tpt_av_voice_utils::nn::Rng,
    r1: Resonator,
    r2: Resonator,
    r3: Resonator,
    phase: f32,
}

impl FormantSynth {
    /// Creates a synthesizer with the default voice.
    pub fn new(config: SynthConfig) -> Result<Self, VoiceError> {
        Self::with_voice(config, VoiceProfile::default_male())
    }

    /// Creates a synthesizer with a specific voice profile.
    pub fn with_voice(config: SynthConfig, voice: VoiceProfile) -> Result<Self, VoiceError> {
        config.validate()?;
        Ok(FormantSynth {
            config,
            voice,
            rng: tpt_av_voice_utils::nn::Rng::new(0x7EE7),
            r1: Resonator::default(),
            r2: Resonator::default(),
            r3: Resonator::default(),
            phase: 0.0,
        })
    }

    /// Borrows the configuration.
    pub fn config(&self) -> &SynthConfig {
        &self.config
    }

    /// Borrows the voice profile.
    pub fn voice(&self) -> &VoiceProfile {
        &self.voice
    }

    /// Synthesizes text into PCM.
    pub fn synthesize(&mut self, text: &str) -> Result<SynthesizedAudio, VoiceError> {
        let sentences = parse_text(text);
        if sentences.is_empty() {
            return Ok(SynthesizedAudio {
                samples: Vec::new(),
                sample_rate: self.config.sample_rate,
            });
        }
        let mut samples = Vec::new();
        for sentence in &sentences {
            self.synthesize_sentence(sentence, &mut samples);
        }
        // Normalize to the configured volume.
        let peak = samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let gain = self.config.volume / peak * 0.95;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        Ok(SynthesizedAudio {
            samples,
            sample_rate: self.config.sample_rate,
        })
    }

    fn synthesize_sentence(&mut self, sentence: &Sentence, out: &mut Vec<f32>) {
        let n_words = sentence.words.len().max(1);
        for (w, word) in sentence.words.iter().enumerate() {
            let n_phones = word.phonemes.len().max(1);
            for (p_idx, &phone) in word.phonemes.iter().enumerate() {
                // Phrase-final position droops the pitch.
                let final_position = w + 1 == n_words && p_idx + 1 == n_phones;
                let end_semitones = if final_position {
                    sentence.end_semitones
                } else {
                    0.0
                };
                self.synthesize_segment(phone, end_semitones, out);
            }
            if word.pause_secs > 0.0 {
                self.append_silence(word.pause_secs / self.config.speaking_rate, out);
            }
        }
    }

    fn synthesize_segment(&mut self, phone: Phoneme, end_semitones: f32, out: &mut Vec<f32>) {
        let seg = segment_for(phone, &self.voice);
        let sr = self.config.sample_rate as f32;
        let duration = (seg.duration_secs / self.config.speaking_rate).max(0.02);
        let closure = seg.closure / self.config.speaking_rate;
        self.append_silence(closure, out);

        let n = (duration * sr) as usize;
        let f0_start = self.config.base_f0(&self.voice);
        let f0_end = f0_start * 2f32.powf(end_semitones / 12.0);

        for i in 0..n {
            let t = i as f32 / n as f32;
            // Amplitude envelope: 20 ms attack/release.
            let envelope = (t * 10.0).min(1.0).min((1.0 - t) * 10.0).clamp(0.0, 1.0);
            // Formants glide toward this segment's targets.
            let glide = if i < n / 3 { t * 3.0f32.min(1.0) } else { 1.0 } * 0.3 + 0.7;
            let f1 = seg.f1 * glide;
            let f2 = seg.f2 * glide;
            let f3 = seg.f3 * glide;
            self.r1.retune(f1, 70.0, sr);
            self.r2.retune(f2, 110.0, sr);
            self.r3.retune(f3, 170.0, sr);

            // Excitation: glottal pulses + breath noise + fricative noise.
            let f0 = f0_start + (f0_end - f0_start) * t;
            self.phase += f0 * self.config.sample_period();
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
            let pulse = if self.phase < 0.4 {
                (self.phase / 0.4) * 2.0 - 1.0
            } else {
                1.0 - (self.phase - 0.4) / 0.6
            };
            let breath = (self.rng.uniform() * 2.0 - 1.0) * self.voice.breathiness;
            let fricative = (self.rng.uniform() * 2.0 - 1.0) * (1.0 - seg.voicing);
            let source = (pulse + breath) * seg.voicing + fricative;

            let voiced_path = self.r1.tick(source);
            let filtered = self.r2.tick(voiced_path);
            let filtered = self.r3.tick(filtered);
            out.push(filtered * envelope * seg.amplitude);
        }
    }

    fn append_silence(&mut self, secs: f32, out: &mut Vec<f32>) {
        let n = (secs * self.config.sample_rate as f32) as usize;
        // Reset the resonators so bursts start clean.
        self.r1 = Resonator::default();
        self.r2 = Resonator::default();
        self.r3 = Resonator::default();
        out.extend(std::iter::repeat(0.0).take(n));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    #[test]
    fn synthesizes_nonempty_audio() {
        let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
        let audio = synth.synthesize("Hello world").unwrap();
        assert_eq!(audio.sample_rate, 22_050);
        assert!(audio.samples.len() > 12_000, "over half a second of audio");
        assert!(audio.samples.iter().all(|s| s.is_finite()));
        assert!(rms(&audio.samples) > 0.01, "audible content");
        // Normalized near the target volume.
        let peak = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((peak - 0.855).abs() < 0.1, "peak {peak}");
    }

    #[test]
    fn empty_text_is_silent() {
        let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
        let audio = synth.synthesize("").unwrap();
        assert!(audio.samples.is_empty());
    }

    #[test]
    fn longer_text_is_longer_audio() {
        let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
        let short = synth.synthesize("hi").unwrap().samples.len();
        let long = synth
            .synthesize("hello brave new world")
            .unwrap()
            .samples
            .len();
        assert!(long > short * 2);
    }

    #[test]
    fn rate_and_volume_have_effects() {
        let base = SynthConfig::default();
        let mut a = FormantSynth::new(base.clone()).unwrap();
        let normal = a.synthesize("hello there").unwrap().samples.len();
        let mut b = FormantSynth::new(SynthConfig {
            speaking_rate: 2.0,
            ..base
        })
        .unwrap();
        let fast = b.synthesize("hello there").unwrap().samples.len();
        assert!(fast < normal / 2 + 2_000, "fast {fast} vs normal {normal}");
    }

    #[test]
    fn voices_differ() {
        let mut m =
            FormantSynth::with_voice(SynthConfig::default(), VoiceProfile::default_male()).unwrap();
        let mut f =
            FormantSynth::with_voice(SynthConfig::default(), VoiceProfile::default_female())
                .unwrap();
        let male = m.synthesize("ah").unwrap().samples;
        let female = f.synthesize("ah").unwrap().samples;
        // Same length (same phonemes), different content.
        assert_eq!(male.len(), female.len());
        let diff: f32 = male.iter().zip(&female).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1.0, "voices should differ, diff {diff}");
    }

    #[test]
    fn determinism() {
        let mut a = FormantSynth::new(SynthConfig::default()).unwrap();
        let mut b = FormantSynth::new(SynthConfig::default()).unwrap();
        assert_eq!(
            a.synthesize("the cat").unwrap(),
            b.synthesize("the cat").unwrap()
        );
    }

    #[test]
    fn silence_between_sentences() {
        let mut synth = FormantSynth::new(SynthConfig::default()).unwrap();
        let audio = synth.synthesize("stop. look.").unwrap();
        // Find the inter-sentence pause (250 ms of zeros).
        let zero_run = audio
            .samples
            .split(|s| s.abs() > 1e-6)
            .map(|run| run.len())
            .max()
            .unwrap_or(0);
        assert!(
            zero_run > 4_000,
            "expected ≥180 ms silence gap, got {zero_run} samples"
        );
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(FormantSynth::new(SynthConfig {
            sample_rate: 500,
            ..Default::default()
        })
        .is_err());
    }
}
