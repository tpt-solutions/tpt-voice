//! Cross-crate pipelines: the composed workflows that make the engine
//! one call deep.
//!
//! * [`SpeechPipeline`] — transcribe → diarize → speaker-annotate →
//!   word-align a recording.
//! * [`apply_text_edit_with_speech`] / [`fill_insertions`] — close the
//!   text-based-editing loop by synthesizing audio for inserted words.

use tpt_av_voice_align::{AlignedTranscript, Aligner, AlignerConfig, TextBasedEditor};
use tpt_av_voice_diarize::{annotate_transcript, DiarizationResult, Diarizer, DiarizerConfig};
use tpt_av_voice_transcribe::Transcriber;
use tpt_av_voice_tts::{FormantSynth, SynthConfig, VoiceProfile};
use tpt_av_voice_utils::{Transcript, VoiceError, WordTimestamp};

use std::path::Path;

/// Output of [`SpeechPipeline::run`].
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Speaker-annotated transcript.
    pub transcript: Transcript,
    /// Raw diarization (who spoke when).
    pub diarization: DiarizationResult,
    /// Word-level timestamps for the recognized text.
    pub words: Vec<WordTimestamp>,
}

/// Transcription + diarization + alignment in one object.
pub struct SpeechPipeline {
    transcriber: Transcriber,
    diarizer: Diarizer,
    aligner: Aligner,
}

impl SpeechPipeline {
    /// Builds a pipeline around a loaded [`Transcriber`].
    pub fn with_parts(
        transcriber: Transcriber,
        diarizer: Diarizer,
        aligner: Aligner,
    ) -> Result<Self, VoiceError> {
        Ok(SpeechPipeline {
            transcriber,
            diarizer,
            aligner,
        })
    }

    /// Loads the ASR model from `model_dir` and uses the weight-free
    /// classical diarizer and energy aligner — one call, one directory.
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        diarizer: DiarizerConfig,
        aligner: AlignerConfig,
    ) -> Result<Self, VoiceError> {
        let transcriber = Transcriber::from_model_dir(model_dir)?;
        let diarizer = Diarizer::classical(diarizer)?;
        let aligner = Aligner::new(aligner)?;
        Self::with_parts(transcriber, diarizer, aligner)
    }

    /// Runs the full pipeline over mono PCM.
    ///
    /// 1. Speech-to-text.
    /// 2. Speaker diarization.
    /// 3. Speaker labels merged into the transcript segments.
    /// 4. The recognized text is word-aligned back onto the audio.
    pub fn run(&mut self, samples: &[f32], sample_rate: u32) -> Result<PipelineOutput, VoiceError> {
        let mut transcript = self.transcriber.transcribe_samples(samples, sample_rate)?;
        let diarization = self.diarizer.diarize_samples(samples, sample_rate)?;
        annotate_transcript(&mut transcript, &diarization)?;

        let words = if transcript.full_text.trim().is_empty() {
            Vec::new()
        } else {
            self.aligner
                .align_samples(samples, sample_rate, &transcript.full_text)?
                .words
        };

        Ok(PipelineOutput {
            transcript,
            diarization,
            words,
        })
    }

    /// Borrows the transcriber (to tweak decode settings mid-flight).
    pub fn transcriber(&self) -> &Transcriber {
        &self.transcriber
    }

    /// Replaces the transcription settings (e.g. decode budget).
    pub fn set_transcriber_config(
        &mut self,
        config: tpt_av_voice_transcribe::transcriber::TranscriberConfig,
    ) {
        self.transcriber.set_config(config);
    }
}

/// Text-based editing that fills inserted/replacement words with
/// synthesized speech (see [`fill_insertions`]).
///
/// Returns the canonical edited transcript plus the audio edit
/// operations; `Insert`/`Replace` ops arrive with `new_audio` ready to
/// splice at the source sample rate.
pub fn apply_text_edit_with_speech(
    aligned: &AlignedTranscript,
    edited: &str,
    voice: &VoiceProfile,
    sample_rate: u32,
) -> Result<(String, Vec<tpt_av_voice_align::AudioEdit>), VoiceError> {
    let mut editor = TextBasedEditor::new(aligned.clone());
    let mut ops = editor.apply_edits(edited);
    fill_insertions(&mut ops, voice, sample_rate)?;
    Ok((editor.edited_transcript().to_string(), ops))
}

/// Synthesizes [`AudioEdit::words`] into [`AudioEdit::new_audio`] for
/// every insert/replace op that has none yet. Idempotent: ops whose
/// audio is already filled are left untouched.
pub fn fill_insertions(
    ops: &mut [tpt_av_voice_align::AudioEdit],
    voice: &VoiceProfile,
    sample_rate: u32,
) -> Result<(), VoiceError> {
    for op in ops.iter_mut() {
        if op.new_audio.is_some() || op.words.is_empty() {
            continue;
        }
        let mut synth = FormantSynth::with_voice(
            SynthConfig {
                sample_rate,
                ..SynthConfig::default()
            },
            voice.clone(),
        )?;
        let audio = synth.synthesize(&op.words.join(" "))?;
        op.new_audio = Some(audio.samples);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_av_voice_transcribe::transcriber::TranscriberConfig;

    fn demo_model() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tpt-voice-pipeline-{}", std::process::id()));
        tpt_av_voice_transcribe::weights::write_demo_model(&dir, "hello world speech test", 11)
            .unwrap();
        dir
    }

    /// Speech-like clip with two loudness-separated "utterances".
    fn clip() -> Vec<f32> {
        let mut pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
        pcm.extend(vec![0.0; 12_000]);
        pcm.extend((0..16_000).map(|i| (i as f32 * 0.033).sin() * 0.5));
        pcm
    }

    #[test]
    fn one_call_pipeline_produces_all_outputs() {
        let dir = demo_model();
        let mut pipeline = SpeechPipeline::from_model_dir(
            &dir,
            DiarizerConfig::default(),
            AlignerConfig::default(),
        )
        .unwrap();
        // Keep the decode budget tiny for test speed.
        pipeline.set_transcriber_config(TranscriberConfig {
            max_tokens_per_window: 4,
            ..Default::default()
        });
        let pcm = clip();
        let out = pipeline.run(&pcm, 16_000).unwrap();
        assert!(!out.transcript.full_text.is_empty());
        assert!((0.5..3.5).contains(&out.transcript.duration_secs()));
        assert!(out.diarization.num_speakers >= 1);
        for seg in &out.transcript.segments {
            assert!(seg.speaker.is_some(), "diarization must annotate segments");
        }
        // Words are aligned onto the same timeline.
        for w in &out.words {
            assert!(w.end_time <= 3.0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fill_insertions_synthesizes_speech() {
        let words = ["hello", "brave", "new", "world", "today"]
            .iter()
            .enumerate()
            .map(|(i, w)| WordTimestamp::new(*w, i as f64, (i + 1) as f64, 1.0))
            .collect();
        let aligned = AlignedTranscript {
            words,
            text: "hello brave new world today".into(),
        };

        let (edited, ops) = apply_text_edit_with_speech(
            &aligned,
            "hello huge world today",
            &VoiceProfile::default_male(),
            16_000,
        )
        .unwrap();
        assert_eq!(edited, "hello huge world today");
        let rep = ops
            .iter()
            .find(|o| o.edit_type == tpt_av_voice_align::EditType::Replace)
            .expect("word swap must produce a Replace op");
        assert_eq!(rep.words, vec!["huge"]);
        let audio = rep.new_audio.as_ref().expect("synthesized insertion");
        assert!(!audio.is_empty());
        assert!(audio.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn fill_insertions_is_idempotent() {
        let mut ops = vec![tpt_av_voice_align::AudioEdit {
            edit_type: tpt_av_voice_align::EditType::Insert,
            start_time: 1.0,
            end_time: 1.0,
            new_audio: Some(vec![0.5]),
            words: vec!["hi".into()],
        }];
        fill_insertions(&mut ops, &VoiceProfile::robot(), 16_000).unwrap();
        assert_eq!(
            ops[0].new_audio,
            Some(vec![0.5]),
            "already-filled op untouched"
        );
    }
}
