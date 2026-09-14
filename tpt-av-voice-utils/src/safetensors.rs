//! `safetensors`-compatible model weight I/O.
//!
//! Reads and writes the [safetensors](https://github.com/huggingface/safetensors)
//! format (8-byte little-endian header length, JSON header, raw tensor
//! data) in pure Rust, so models trained in any framework can be loaded
//! without a serialization runtime. `F32`, `F16`, `F64` and `BF16`
//! tensors are decoded to `f32`; writes are always `F32`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::error::VoiceError;

/// A dense tensor decoded to `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Tensor shape (row-major).
    pub shape: Vec<usize>,
    /// Tensor data, `shape` elements in row-major order.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Element count of the shape.
    pub fn num_elements(shape: &[usize]) -> usize {
        shape.iter().product()
    }
}

/// Writes tensors as an `F32` safetensors file. Keys are sorted for
/// deterministic output (BTreeMap).
pub fn write_safetensors(tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let mut data_offset = 0usize;
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    // First pass to compute offsets — collect serialized entries first.
    let mut entries: Vec<(String, Vec<usize>, usize)> = Vec::new();
    for (name, (shape, data)) in tensors {
        entries.push((name.clone(), shape.clone(), data.len() * 4));
    }
    for (name, shape, len) in &entries {
        let start = data_offset;
        data_offset += len;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, start + len],
            }),
        );
    }
    for (_, data) in tensors.values() {
        for v in data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }

    let header_json = serde_json::Value::Object(header).to_string();
    let header_len = header_json.len();
    // Pad header with spaces to keep payload 8-byte aligned.
    let padding = (8 - (8 + header_len) % 8) % 8;
    let mut out = Vec::with_capacity(8 + header_len + padding + payload.len());
    out.extend_from_slice(&((header_len + padding) as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    out.extend(std::iter::repeat(b' ').take(padding));
    out.extend_from_slice(&payload);
    out
}

/// Saves tensors to a file.
pub fn save_to_path(
    path: impl AsRef<Path>,
    tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
) -> Result<(), VoiceError> {
    fs::write(path, write_safetensors(tensors))?;
    Ok(())
}

/// Parses a safetensors byte buffer into decoded `f32` tensors.
pub fn read_safetensors(bytes: &[u8]) -> Result<BTreeMap<String, Tensor>, VoiceError> {
    if bytes.len() < 8 {
        return Err(VoiceError::weights("file too short for safetensors header"));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if header_len == 0 || 8 + header_len > bytes.len() {
        return Err(VoiceError::weights(format!("invalid header length {header_len}")));
    }
    let header_json = &bytes[8..8 + header_len];
    let header: BTreeMap<String, HeaderEntry> = serde_json::from_slice(header_json)
        .map_err(|e| VoiceError::weights(format!("malformed header JSON: {e}")))?;
    let data = &bytes[8 + header_len..];

    let mut out = BTreeMap::new();
    for (name, entry) in header {
        let (start, end) = entry.data_offsets;
        if end < start || end > data.len() {
            return Err(VoiceError::weights(format!("tensor {name:?} data range [{start},{end}) out of bounds")));
        }
        let n = end - start;
        let expected = Tensor::num_elements(&entry.shape);
        let raw = &data[start..end];
        let decoded = match entry.dtype.as_str() {
            "F32" => {
                if n != expected * 4 {
                    return Err(VoiceError::weights(format!("tensor {name:?} size mismatch")));
                }
                raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            }
            "F64" => {
                if n != expected * 8 {
                    return Err(VoiceError::weights(format!("tensor {name:?} size mismatch")));
                }
                raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
            }
            "F16" => {
                if n != expected * 2 {
                    return Err(VoiceError::weights(format!("tensor {name:?} size mismatch")));
                }
                raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect()
            }
            "BF16" => {
                if n != expected * 2 {
                    return Err(VoiceError::weights(format!("tensor {name:?} size mismatch")));
                }
                raw.chunks_exact(2)
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(c.try_into().unwrap())) << 16))
                    .collect()
            }
            other => {
                return Err(VoiceError::weights(format!(
                    "tensor {name:?} has unsupported dtype {other} (only F32/F64/F16/BF16 supported)"
                )));
            }
        };
        out.insert(name, Tensor { shape: entry.shape, data: decoded });
    }
    Ok(out)
}

/// Loads tensors from a file.
pub fn load_from_path(path: impl AsRef<Path>) -> Result<BTreeMap<String, Tensor>, VoiceError> {
    read_safetensors(&fs::read(path)?)
}

#[derive(Debug, serde::Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// IEEE 754 half-precision → f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;
    let f32_bits = if exponent == 0 {
        // Subnormal (or zero): scale into f32 subnormals/normal range.
        if mantissa == 0 {
            sign << 31
        } else {
            let mut e = 0u32;
            let mut m = mantissa;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x03FF;
            let exp = 127 - 15 - e + 1;
            (sign << 31) | (exp << 23) | (m << 13)
        }
    } else if exponent == 0x1F {
        // Inf / NaN
        (sign << 31) | 0x7F80_0000 | (mantissa << 13)
    } else {
        (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        BTreeMap::from([
            ("encoder.weight".to_string(), (vec![2, 3], vec![1.0, -2.0, 3.5, 0.25, -0.125, 7.0])),
            ("decoder.bias".to_string(), (vec![3], vec![0.1, 0.2, 0.3])),
        ])
    }

    #[test]
    fn round_trip() {
        let bytes = write_safetensors(&sample());
        let decoded = read_safetensors(&bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        let w = &decoded["encoder.weight"];
        assert_eq!(w.shape, vec![2, 3]);
        assert_eq!(w.data, vec![1.0, -2.0, 3.5, 0.25, -0.125, 7.0]);
        assert_eq!(decoded["decoder.bias"].data, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn payload_is_8_byte_aligned() {
        // Find a header length that would misalign without padding.
        let mut tensors = sample();
        tensors.insert("x".to_string(), (vec![1], vec![1.0]));
        let bytes = write_safetensors(&tensors);
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!((8 + header_len) % 8, 0);
        // Padding chars must be spaces inside the declared header.
        let header_end = 8 + header_len;
        assert!(bytes[8..header_end].iter().all(|&b| b.is_ascii_graphic() || b == b' '));
    }

    #[test]
    fn truncated_input_rejected() {
        let bytes = write_safetensors(&sample());
        assert!(read_safetensors(&bytes[..4]).is_err());
        assert!(read_safetensors(&bytes[..12]).is_err()); // header_len > available
        let mut evil = bytes.clone();
        evil[0] = 0xFF; // absurd header length
        assert!(read_safetensors(&evil).is_err());
    }

    #[test]
    fn f16_decoding() {
        // 1.0 = 0x3C00, -2.0 = 0xC000, +inf = 0x7C00, subnormal min = 0x0001
        let f32_val = f16_to_f32(0x3C00);
        assert_eq!(f32_val, 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        let smallest = f16_to_f32(0x0001);
        assert!(smallest > 0.0 && smallest < 1e-6);
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn unsupported_dtype_rejected() {
        // Build a header with INT8 dtype.
        let header = r#"{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#;
        let hl = header.len() as u64;
        let mut bytes = hl.to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[7u8]); // payload
        let err = read_safetensors(&bytes).unwrap_err();
        assert!(err.to_string().contains("I8"));
    }
}
