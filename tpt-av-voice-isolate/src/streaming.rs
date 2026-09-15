//! Streaming isolation with overlap-add.
//!
//! [`StreamingIsolator`] consumes 16 kHz [`AudioChunk`]s of any size and
//! emits isolated samples with a fixed algorithmic latency of
//! `window_size − hop_size` samples. Frames are masked one at a time
//! (through [`VoiceIsolator::mask_frame`]) and synthesized with
//! weighted overlap-add, so classical and model-backed isolation stream
//! identically.
//!
//! ```no_run
//! use tpt_av_voice_isolate::{VoiceIsolator, IsolatorConfig, StreamingIsolator};
//!
//! # let chunks: Vec<tpt_av_voice_utils::AudioChunk> = Vec::new();
//! # let mut isolator = VoiceIsolator::classical(IsolatorConfig::default()).unwrap();
//! let mut stream = StreamingIsolator::new(&mut isolator);
//! for chunk in chunks {
//!     let isolated = stream.push(&chunk).unwrap();
//!     play(isolated);
//! }
//! let tail = stream.flush().unwrap(); // emit the buffered remainder
//! # fn play(_: tpt_av_voice_utils::AudioChunk) {}
//! ```

use rustfft::num_complex::Complex;

use tpt_av_voice_utils::dsp::{hann_window, RealFft};
use tpt_av_voice_utils::{AudioChunk, VoiceError};

use crate::isolator::VoiceIsolator;

/// Incremental isolator over a chunk stream.
pub struct StreamingIsolator<'a> {
    isolator: &'a mut VoiceIsolator,
    n_fft: usize,
    hop: usize,
    fft: RealFft,
    window: Vec<f32>,
    /// Unprocessed input samples.
    input: Vec<f32>,
    /// Overlap-add accumulators; `acc[0]` sits at absolute sample
    /// `acc_start` of the output timeline.
    acc: Vec<f32>,
    wsum: Vec<f32>,
    acc_start: usize,
    /// Absolute output index of the next sample to emit.
    next_out: usize,
    /// Number of frames processed so far.
    frames: usize,
    /// Total real input samples seen (caps the flush output).
    total_in: usize,
    start_time: Option<f64>,
    seq: u64,
}

impl<'a> StreamingIsolator<'a> {
    /// Wraps a mutably-borrowed isolator.
    pub fn new(isolator: &'a mut VoiceIsolator) -> Self {
        let n_fft = isolator.config().window_size;
        let hop = isolator.config().hop_size;
        StreamingIsolator {
            isolator,
            fft: RealFft::new(n_fft),
            window: hann_window(n_fft),
            n_fft,
            hop,
            input: Vec::new(),
            acc: Vec::new(),
            wsum: Vec::new(),
            acc_start: 0,
            next_out: 0,
            frames: 0,
            total_in: 0,
            start_time: None,
            seq: 0,
        }
    }

    /// Algorithmic latency in samples (`window_size − hop_size`).
    pub fn latency_samples(&self) -> usize {
        self.n_fft - self.hop
    }

    /// Pushes one chunk and returns whatever isolated samples became
    /// final (initially fewer than the input, by the latency amount).
    pub fn push(&mut self, chunk: &AudioChunk) -> Result<AudioChunk, VoiceError> {
        if chunk.sample_rate != 16_000 {
            return Err(VoiceError::config(
                "streaming isolator expects 16 kHz chunks",
            ));
        }
        if self.start_time.is_none() {
            self.start_time = Some(chunk.start_time);
        }
        self.seq = self.seq.max(chunk.sequence);
        self.total_in += chunk.len();
        self.input.extend_from_slice(&chunk.samples);
        self.process_ready_frames()?;
        Ok(self.take_output(usize::MAX, false))
    }

    /// Pads the remaining input and emits everything still buffered.
    pub fn flush(&mut self) -> Result<AudioChunk, VoiceError> {
        // Zero-pad past the last partial frame so every real sample has
        // been through a complete window.
        if !self.input.is_empty() || self.acc_start + self.acc.len() < self.total_in {
            self.input.resize(self.input.len().max(self.n_fft), 0.0);
            self.process_ready_frames()?;
        }
        Ok(self.take_output(usize::MAX, true))
    }

    fn process_ready_frames(&mut self) -> Result<(), VoiceError> {
        while self.input.len() >= self.n_fft {
            let frame: Vec<f32> = self.input[..self.n_fft].to_vec();
            let windowed: Vec<f32> = frame.iter().zip(&self.window).map(|(s, w)| s * w).collect();
            let spectrum = self.fft.forward(&windowed);
            let half = &spectrum[..self.n_fft / 2 + 1];
            let mag: Vec<f32> = half.iter().map(|c| c.norm()).collect();
            let mask = self.isolator.mask_frame(&mag)?;
            let masked: Vec<Complex<f32>> = half
                .iter()
                .zip(&mask)
                .map(|(c, m)| Complex::new(c.re * m, c.im * m))
                .collect();
            let synthesized = self.fft.inverse_half(&masked);

            // Overlap-add at frame start = frames·hop.
            let frame_start = self.frames * self.hop;
            let needed = frame_start + self.n_fft - self.acc_start;
            self.acc.resize(needed.max(self.acc.len()), 0.0);
            self.wsum.resize(needed.max(self.wsum.len()), 0.0);
            for (i, s) in synthesized.iter().enumerate() {
                let idx = frame_start - self.acc_start + i;
                self.acc[idx] += s * self.window[i];
                self.wsum[idx] += self.window[i] * self.window[i];
            }
            self.frames += 1;
            self.input.drain(..self.hop);
        }
        Ok(())
    }

    /// Drains normalized output samples. Samples are final once no
    /// future frame can touch them, i.e. index < frames·hop (or
    /// everything, on the final pass).
    fn take_output(&mut self, max: usize, final_pass: bool) -> AudioChunk {
        let horizon = if final_pass {
            usize::MAX
        } else {
            self.frames * self.hop
        };
        let ready = horizon.saturating_sub(self.next_out).min(self.acc.len());
        let n = ready.min(max);
        // Never emit more than the real input we were given (flush pads).
        let n = n.min(self.total_in.saturating_sub(self.next_out));
        let out: Vec<f32> = self.acc[..n]
            .iter()
            .zip(&self.wsum[..n])
            .map(|(s, w)| if *w > 1e-8 { s / w } else { 0.0 })
            .collect();
        self.acc.drain(..n);
        self.wsum.drain(..n);
        self.acc_start += n;
        self.next_out += n;
        let start = self.start_time.unwrap_or(0.0)
            + tpt_av_voice_utils::time::samples_to_secs(self.next_out - out.len(), 16_000);
        AudioChunk::new(out, 16_000, start, self.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isolator::{IsolatorConfig, VoiceIsolator};

    fn isolator() -> VoiceIsolator {
        VoiceIsolator::classical(IsolatorConfig::default()).unwrap()
    }

    fn chunk(samples: Vec<f32>, start: f64, seq: u64) -> AudioChunk {
        AudioChunk::new(samples, 16_000, start, seq)
    }

    #[test]
    fn output_count_matches_input_after_flush() {
        let mut iso = isolator();
        let mut stream = StreamingIsolator::new(&mut iso);
        let input: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let mut out = Vec::new();
        for (s, c) in input.chunks(1_600).enumerate() {
            out.extend(
                stream
                    .push(&chunk(c.to_vec(), 0.0, s as u64))
                    .unwrap()
                    .samples,
            );
        }
        out.extend(stream.flush().unwrap().samples);
        assert_eq!(out.len(), input.len(), "streaming must preserve length");
    }

    #[test]
    fn streaming_matches_batch_closely() {
        let input: Vec<f32> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * 250.0 * i as f32 / 16_000.0).sin() * 0.4)
            .collect();

        let mut batch_iso = isolator();
        let batch = batch_iso.isolate_samples(&input, 16_000).unwrap();

        let mut stream_iso = isolator();
        let mut stream = StreamingIsolator::new(&mut stream_iso);
        let mut streamed = Vec::new();
        for (s, c) in input.chunks(2_000).enumerate() {
            streamed.extend(
                stream
                    .push(&chunk(c.to_vec(), 0.0, s as u64))
                    .unwrap()
                    .samples,
            );
        }
        streamed.extend(stream.flush().unwrap().samples);

        // Same pipeline, different framing → tiny numerical drift only.
        // Compare interior regions (both paths taper their very edges).
        let (start, end) = (1_000, 15_000);
        let max_diff = batch.voice[start..end]
            .iter()
            .zip(&streamed[start..end])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max diff {max_diff}");
    }

    #[test]
    fn latency_is_window_minus_hop() {
        let mut iso = isolator();
        let stream = StreamingIsolator::new(&mut iso);
        assert_eq!(stream.latency_samples(), 256);
    }

    #[test]
    fn wrong_rate_rejected() {
        let mut iso = isolator();
        let mut stream = StreamingIsolator::new(&mut iso);
        let bad = AudioChunk::new(vec![0.0; 256], 44_100, 0.0, 0);
        assert!(stream.push(&bad).is_err());
    }
}
