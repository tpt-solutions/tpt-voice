//! Minimal pure-Rust WAV (RIFF) reader/writer.
//!
//! Supports the encodings the TPT AV stack actually produces: 8/16/24/32-bit
//! integer PCM, 32/64-bit IEEE float, and `WAVE_FORMAT_EXTENSIBLE`
//! wrappers of those. Samples are returned as mono-capable `f32` in
//! `[-1.0, 1.0]`; multi-channel data stays interleaved (see [`WavData::to_mono`]).
//!
//! This exists so the engine has zero mandatory audio-I/O dependencies;
//! decode from other containers is `tpt-cadence`'s job (spec §2).

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::{AudioError, VoiceError};

/// Decoded WAV audio: interleaved `f32` samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    /// Interleaved samples (`channels` frames of `channels` samples).
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
}

impl WavData {
    /// Number of frames (sample sets) in the data.
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f64 {
        crate::time::samples_to_secs(self.frames(), self.sample_rate)
    }

    /// Downmixes to mono by averaging channels.
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let ch = self.channels as usize;
        self.samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    }
}

const FMT_PCM: u16 = 1;
const FMT_FLOAT: u16 = 3;
const FMT_EXTENSIBLE: u16 = 0xFFFE;

/// Reads a WAV file from a byte stream.
pub fn read_wav(reader: &mut impl Read) -> Result<WavData, VoiceError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    read_wav_bytes(&bytes)
}

/// Reads a WAV file from disk.
pub fn read_wav_file(path: impl AsRef<Path>) -> Result<WavData, VoiceError> {
    let mut file = File::open(path)?;
    read_wav(&mut file)
}

/// Parses WAV bytes. Only the `fmt ` and `data` chunks are interpreted;
/// every other chunk is skipped.
pub fn read_wav_bytes(bytes: &[u8]) -> Result<WavData, VoiceError> {
    let ensure = |cond: bool, msg: &str| -> Result<(), VoiceError> {
        if cond {
            Ok(())
        } else {
            Err(AudioError::InvalidFormat(msg.to_string()).into())
        }
    };
    ensure(bytes.len() >= 12, "file too short for RIFF header")?;
    ensure(&bytes[0..4] == b"RIFF", "missing RIFF magic")?;
    ensure(&bytes[8..12] == b"WAVE", "not a WAVE file")?;

    let mut pos = 12usize;
    let mut format: Option<(u16, u16, u32, u16)> = None; // (fmt, channels, rate, bits)
    let mut data_range: Option<(usize, usize)> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let payload = pos + 8;
        let end = payload.saturating_add(size);
        if end > bytes.len() {
            return Err(AudioError::Truncated(format!(
                "chunk {id:?} claims {size} bytes past EOF"
            ))
            .into());
        }
        match id {
            b"fmt " => {
                ensure(size >= 16, "fmt chunk too small")?;
                let audio_format = u16::from_le_bytes([bytes[payload], bytes[payload + 1]]);
                let channels = u16::from_le_bytes([bytes[payload + 2], bytes[payload + 3]]);
                let sample_rate = u32::from_le_bytes([
                    bytes[payload + 4],
                    bytes[payload + 5],
                    bytes[payload + 6],
                    bytes[payload + 7],
                ]);
                let bits = u16::from_le_bytes([bytes[payload + 14], bytes[payload + 15]]);
                let mut fmt = audio_format;
                if audio_format == FMT_EXTENSIBLE {
                    ensure(size >= 40, "extensible fmt chunk too small")?;
                    fmt = u16::from_le_bytes([bytes[payload + 24], bytes[payload + 25]]);
                }
                ensure(
                    matches!(fmt, FMT_PCM | FMT_FLOAT),
                    &format!("unsupported WAV format tag {audio_format:#x}"),
                )?;
                ensure(channels > 0, "zero channels")?;
                ensure(sample_rate > 0, "zero sample rate")?;
                format = Some((fmt, channels, sample_rate, bits));
            }
            b"data" => data_range = Some((payload, end)),
            _ => {}
        }
        pos = payload + size + (size & 1); // chunks are word-aligned
    }

    let (fmt, channels, sample_rate, bits) =
        format.ok_or_else(|| AudioError::InvalidFormat("missing fmt chunk".into()))?;
    let (start, end) =
        data_range.ok_or_else(|| AudioError::InvalidFormat("missing data chunk".into()))?;

    let bytes_per_sample = (bits / 8) as usize;
    ensure(bytes_per_sample > 0, "zero bits per sample")?;
    let raw = &bytes[start..end];
    let n_samples = raw.len() / bytes_per_sample;
    let mut samples = Vec::with_capacity(n_samples);

    match (fmt, bits) {
        (FMT_PCM, 8) => {
            for &b in raw {
                samples.push((f32::from(b) - 128.0) / 128.0);
            }
        }
        (FMT_PCM, 16) => {
            for chunk in raw.chunks_exact(2) {
                samples.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
            }
        }
        (FMT_PCM, 24) => {
            for chunk in raw.chunks_exact(3) {
                let v = i32::from_le_bytes([0, chunk[0], chunk[1], chunk[2]]);
                samples.push(v as f32 / 8_388_608.0);
            }
        }
        (FMT_PCM, 32) => {
            for chunk in raw.chunks_exact(4) {
                samples
                    .push(i32::from_le_bytes(chunk.try_into().unwrap()) as f32 / 2_147_483_648.0);
            }
        }
        (FMT_FLOAT, 32) => {
            for chunk in raw.chunks_exact(4) {
                samples.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        (FMT_FLOAT, 64) => {
            for chunk in raw.chunks_exact(8) {
                samples.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
        }
        (fmt, bits) => {
            return Err(
                AudioError::UnsupportedEncoding(format!("format {fmt}, {bits} bits")).into(),
            );
        }
    }

    Ok(WavData {
        samples,
        sample_rate,
        channels,
    })
}

/// Writes `samples` (interleaved) as 16-bit PCM WAV to a stream.
pub fn write_wav(
    writer: &mut impl Write,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), VoiceError> {
    if channels == 0 {
        return Err(AudioError::InvalidFormat("zero channels".into()).into());
    }
    // RIFF sizes are 32-bit; refuse inputs that would wrap instead of
    // silently writing a corrupt header.
    let data_len_u64 = samples.len() as u64 * 2;
    if data_len_u64 + 36 > u32::MAX as u64 {
        return Err(AudioError::InvalidFormat(format!(
            "WAV output of {} samples exceeds the 32-bit RIFF limit",
            samples.len()
        ))
        .into());
    }
    let data_len = data_len_u64 as u32;
    let bits = 16u16;
    let block_align = channels * bits / 8;
    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_len).to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16_u32.to_le_bytes())?;
    writer.write_all(&FMT_PCM.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&(sample_rate * u32::from(block_align)).to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Writes a 16-bit PCM WAV file to disk.
pub fn write_wav_file(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), VoiceError> {
    let mut file = File::create(path)?;
    write_wav(&mut file, samples, sample_rate, channels)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(samples: Vec<f32>, rate: u32, channels: u16) {
        let mut buf = Vec::new();
        write_wav(&mut buf, &samples, rate, channels).unwrap();
        let wav = read_wav_bytes(&buf).unwrap();
        assert_eq!(wav.sample_rate, rate);
        assert_eq!(wav.channels, channels);
        assert_eq!(wav.frames(), samples.len() / channels as usize);
        for (a, b) in wav.samples.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 3.1e-4, "{a} vs {b}"); // 16-bit quantization
        }
    }

    #[test]
    fn write_read_16bit() {
        roundtrip(vec![0.0, 0.5, -0.5, 1.0, -1.0], 16_000, 1);
        roundtrip(vec![0.1; 100], 44_100, 2);
    }

    #[test]
    fn read_float32_wav() {
        // Hand-build a float32 WAV.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        let data: Vec<u8> = (0.25f32)
            .to_le_bytes()
            .iter()
            .chain((-0.75f32).to_le_bytes().iter())
            .copied()
            .collect();
        let data_len = data.len() as u32;
        let riff_len: u32 = 36 + data_len;
        buf.extend_from_slice(&riff_len.to_le_bytes());
        buf.extend_from_slice(b"WAVEfmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes());
        buf.extend_from_slice(&3_u16.to_le_bytes()); // IEEE float
        buf.extend_from_slice(&1_u16.to_le_bytes()); // mono
        buf.extend_from_slice(&16_000_u32.to_le_bytes());
        buf.extend_from_slice(&(16_000_u32 * 4).to_le_bytes());
        buf.extend_from_slice(&4_u16.to_le_bytes());
        buf.extend_from_slice(&32_u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        buf.extend_from_slice(&data);

        let wav = read_wav_bytes(&buf).unwrap();
        assert!((wav.samples[0] - 0.25).abs() < 1e-9);
        assert!((wav.samples[1] + 0.75).abs() < 1e-9);
    }

    #[test]
    fn skips_odd_sized_foreign_chunk() {
        let mut buf = Vec::new();
        write_wav(&mut buf, &[0.0; 4], 8000, 1).unwrap();
        // Insert a LIST chunk with odd size after the WAVE tag: rebuild manually.
        let mut full = Vec::new();
        full.extend_from_slice(b"RIFF");
        let list_payload: [u8; 5] = *b"hello"; // odd size
        let body_len = 4 + (8 + list_payload.len() + 1) + (buf.len() - 12);
        full.extend_from_slice(&(body_len as u32).to_le_bytes());
        full.extend_from_slice(b"WAVE");
        full.extend_from_slice(b"LIST");
        full.extend_from_slice(&(list_payload.len() as u32).to_le_bytes());
        full.extend_from_slice(&list_payload);
        full.push(0); // pad byte
                      // copy original fmt+data from buf
        full.extend_from_slice(&buf[12..]);

        let wav = read_wav_bytes(&full).unwrap();
        assert_eq!(wav.frames(), 4);
    }

    #[test]
    fn truncated_and_invalid() {
        assert!(read_wav_bytes(&[0; 8]).is_err());
        assert!(read_wav_bytes(b"NOTAWAV....").is_err());
        let mut buf = Vec::new();
        write_wav(&mut buf, &[0.5], 8000, 1).unwrap();
        buf.truncate(30);
        assert!(matches!(
            read_wav_bytes(&buf),
            Err(VoiceError::Audio(AudioError::Truncated(_)))
        ));
    }

    #[test]
    fn stereo_to_mono() {
        let wav = WavData {
            samples: vec![0.0, 1.0, 0.5, 0.5],
            sample_rate: 8000,
            channels: 2,
        };
        assert_eq!(wav.to_mono(), vec![0.5, 0.5]);
        assert!((wav.duration() - 0.00025).abs() < 1e-9);
    }
}
