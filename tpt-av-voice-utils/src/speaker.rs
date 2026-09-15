//! Speaker identification types.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A speaker label assigned by the diarizer or provided by the user.
///
/// Machine-assigned ids follow the `SPEAKER_00`, `SPEAKER_01`, …
/// convention; a human-readable `name` can be attached later (e.g. after
/// enrollment or manual identification).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SpeakerLabel {
    /// Speaker identifier (e.g. `"SPEAKER_00"`).
    pub id: String,
    /// Display name, if known.
    pub name: Option<String>,
}

impl SpeakerLabel {
    /// Creates the canonical machine label for a 0-based speaker index
    /// (`SPEAKER_00` for index 0).
    pub fn new(index: usize) -> Self {
        SpeakerLabel {
            id: format!("SPEAKER_{index:02}"),
            name: None,
        }
    }

    /// Creates a label from an arbitrary id string without a display name.
    pub fn unnamed(id: impl Into<String>) -> Self {
        SpeakerLabel {
            id: id.into(),
            name: None,
        }
    }

    /// Creates a label with both id and display name.
    pub fn named(id: impl Into<String>, name: impl Into<String>) -> Self {
        SpeakerLabel {
            id: id.into(),
            name: Some(name.into()),
        }
    }

    /// Returns the display name if present, else the raw id.
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }

    /// Parses the trailing index of a `SPEAKER_NN` label, if it matches
    /// the convention.
    pub fn index(&self) -> Option<usize> {
        self.id
            .strip_prefix("SPEAKER_")
            .and_then(|n| n.parse().ok())
    }
}

impl fmt::Display for SpeakerLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_labels() {
        assert_eq!(SpeakerLabel::new(0).id, "SPEAKER_00");
        assert_eq!(SpeakerLabel::new(7).id, "SPEAKER_07");
        assert_eq!(SpeakerLabel::new(42).id, "SPEAKER_42");
        assert_eq!(SpeakerLabel::new(3).index(), Some(3));
        assert_eq!(SpeakerLabel::unnamed("host").index(), None);
    }

    #[test]
    fn display_prefers_name() {
        let s = SpeakerLabel::named("SPEAKER_00", "Alice");
        assert_eq!(s.to_string(), "Alice");
        assert_eq!(SpeakerLabel::new(1).to_string(), "SPEAKER_01");
    }

    #[test]
    fn ordering_is_by_id() {
        let mut v = [
            SpeakerLabel::new(2),
            SpeakerLabel::new(0),
            SpeakerLabel::new(10),
        ];
        v.sort();
        let ids: Vec<&str> = v.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["SPEAKER_00", "SPEAKER_02", "SPEAKER_10"]);
    }
}
