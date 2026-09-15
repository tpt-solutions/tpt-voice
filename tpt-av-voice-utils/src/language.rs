//! Language codes and multi-language scaffolding (spec Phase 5).
//!
//! Models accept a [`Language`] which is embedded in the decoding prompt
//! (transformer ASR) or used to pick tokenizer/acoustic resources. The
//! type is intentionally open: unknown codes round-trip through
//! [`Language::other`] so new languages work without a crate release.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A language identifier (ISO 639-1 code where available).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Language(pub String);

impl Default for Language {
    fn default() -> Self {
        Language::english()
    }
}

impl Language {
    /// English — the default language.
    pub fn english() -> Language {
        Language("en".to_string())
    }

    /// Well-known languages with first-class support targets (spec §7,
    /// Phase 5: Spanish, French, German, …).
    pub fn spanish() -> Language {
        Language("es".to_string())
    }

    pub fn french() -> Language {
        Language("fr".to_string())
    }

    pub fn german() -> Language {
        Language("de".to_string())
    }

    /// Creates a language for an arbitrary code.
    pub fn other(code: impl Into<String>) -> Language {
        Language(code.into())
    }

    /// The ISO 639-1 (or nearest) code.
    pub fn code(&self) -> &str {
        &self.0
    }

    /// Languages the workspace targets for first-class support.
    pub fn supported() -> Vec<Language> {
        vec![
            Language::english(),
            Language::spanish(),
            Language::french(),
            Language::german(),
            Language::other("it"),
            Language::other("pt"),
            Language::other("nl"),
            Language::other("ru"),
            Language::other("ja"),
            Language::other("zh"),
            Language::other("ko"),
        ]
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for Language {
    type Err = crate::VoiceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let code = s.trim().to_ascii_lowercase();
        if code.is_empty() || code.len() > 8 || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(crate::VoiceError::config(format!(
                "invalid language code: {s:?}"
            )));
        }
        Ok(Language(code))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display() {
        let l: Language = " FR ".parse().unwrap();
        assert_eq!(l, Language("fr".to_string()));
        assert_eq!(l.to_string(), "fr");
        assert_eq!(Language::default().code(), "en");
    }

    #[test]
    fn invalid_codes_rejected() {
        assert!("".parse::<Language>().is_err());
        assert!("not a language!".parse::<Language>().is_err());
    }

    #[test]
    fn supported_list() {
        assert!(Language::supported().contains(&Language::german()));
    }
}
