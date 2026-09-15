//! Model integrity manifests: SHA-256 checksums for model directories.
//!
//! A `manifest.json` next to the model files pins their content:
//!
//! ```json
//! { "format_version": 1, "files": { "model.safetensors": "<hex sha256>" } }
//! ```
//!
//! [`write_manifest`] pins a set of files after export; [`verify_manifest`]
//! re-hashes them and fails on any drift. [`Transcriber`](tpt_av_voice_transcribe::Transcriber)
//! verifies automatically when a manifest is present, so a corrupted or
//! tampered download is rejected before it reaches inference.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::VoiceError;

/// Manifest file name inside a model directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Current manifest schema version.
pub const FORMAT_VERSION: u32 = 1;

/// A model directory manifest.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelManifest {
    /// Schema version (see [`FORMAT_VERSION`]).
    pub format_version: u32,
    /// Relative file name → lowercase hex SHA-256.
    pub files: BTreeMap<String, String>,
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Hashes the given files (relative to `dir`) and writes
/// `manifest.json`. Errors if any listed file is missing.
pub fn write_manifest(dir: impl AsRef<Path>, files: &[&str]) -> Result<ModelManifest, VoiceError> {
    let dir = dir.as_ref();
    let mut entries = BTreeMap::new();
    for name in files {
        // Refuse path traversal: manifests only pin siblings.
        if name.contains("..") || name.contains('/') || name.contains('\\') {
            return Err(VoiceError::weights(format!(
                "manifest file {name:?} must be a sibling"
            )));
        }
        let bytes = fs::read(dir.join(name))
            .map_err(|e| VoiceError::weights(format!("cannot hash {name:?} for manifest: {e}")))?;
        entries.insert((*name).to_string(), sha256_hex(&bytes));
    }
    let manifest = ModelManifest {
        format_version: FORMAT_VERSION,
        files: entries,
    };
    fs::write(
        dir.join(MANIFEST_FILE),
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| VoiceError::weights(format!("manifest serialization: {e}")))?,
    )?;
    Ok(manifest)
}

/// Loads `manifest.json` from a model directory.
pub fn read_manifest(dir: impl AsRef<Path>) -> Result<ModelManifest, VoiceError> {
    let bytes = fs::read(dir.as_ref().join(MANIFEST_FILE))
        .map_err(|e| VoiceError::weights(format!("no readable manifest ({MANIFEST_FILE}): {e}")))?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes)
        .map_err(|e| VoiceError::weights(format!("malformed manifest: {e}")))?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(VoiceError::weights(format!(
            "manifest format_version {} unsupported (expected {FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    Ok(manifest)
}

/// Verifies every file pinned by the directory's manifest. Returns the
/// manifest on success; errors on the first mismatch or missing file.
pub fn verify_manifest(dir: impl AsRef<Path>) -> Result<ModelManifest, VoiceError> {
    let manifest = read_manifest(&dir)?;
    let dir = dir.as_ref();
    for (name, expected) in &manifest.files {
        let bytes = fs::read(dir.join(name)).map_err(|e| {
            VoiceError::weights(format!("manifest pins missing file {name:?}: {e}"))
        })?;
        let actual = sha256_hex(&bytes);
        if actual != *expected {
            return Err(VoiceError::weights(format!(
                "integrity check failed for {name:?}: expected sha256 {expected}, got {actual}"
            )));
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tpt-manifest-{tag}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_then_verify_round_trip() {
        let dir = dir("roundtrip");
        fs::write(dir.join("model.bin"), b"weights-bytes").unwrap();
        fs::write(dir.join("config.json"), b"{}").unwrap();
        let manifest = write_manifest(&dir, &["model.bin", "config.json"]).unwrap();
        assert_eq!(manifest.format_version, FORMAT_VERSION);
        assert_eq!(manifest.files.len(), 2);
        assert_eq!(verify_manifest(&dir).unwrap(), manifest);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_tampering() {
        let dir = dir("tamper");
        fs::write(dir.join("model.bin"), b"original").unwrap();
        write_manifest(&dir, &["model.bin"]).unwrap();
        fs::write(dir.join("model.bin"), b"tampered!").unwrap();
        let err = verify_manifest(&dir).unwrap_err();
        assert!(err.to_string().contains("integrity check failed"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_pinned_file_fails() {
        let dir = dir("missing");
        fs::write(dir.join("a.bin"), b"x").unwrap();
        write_manifest(&dir, &["a.bin"]).unwrap();
        fs::remove_file(dir.join("a.bin")).unwrap();
        assert!(verify_manifest(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn traversal_rejected() {
        let dir = dir("traversal");
        assert!(write_manifest(&dir, &["../secret.bin"]).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sha256_known_vector() {
        // sha256("abc") per FIPS 180-4.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn bad_format_version_rejected() {
        let dir = dir("badver");
        fs::write(
            dir.join(MANIFEST_FILE),
            r#"{ "format_version": 99, "files": {} }"#,
        )
        .unwrap();
        assert!(read_manifest(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
