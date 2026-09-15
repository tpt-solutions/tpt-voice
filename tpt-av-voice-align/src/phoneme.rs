//! Rule-based English grapheme-to-phoneme conversion.
//!
//! A pragmatic two-stage G2P: a small exception lexicon for the most
//! common irregular words, then ordered longest-suffix rules
//! (CMU-style digraphs, vowel contexts, silent-e). Output uses the
//! ARPAbet phoneme set so aligned units match standard acoustic
//! vocabularies. Good enough for alignment scaffolding; trained G2P
//! models can replace it behind the same API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// ARPAbet phoneme (subset used by the aligner and TTS crate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phoneme {
    // Monophthongs
    AA,
    AE,
    AH,
    AO,
    AW,
    AY,
    EH,
    ER,
    EY,
    IH,
    IY,
    OW,
    OY,
    UH,
    UW,
    // Consonants
    B,
    CH,
    D,
    DH,
    F,
    G,
    HH,
    JH,
    K,
    L,
    M,
    N,
    NG,
    P,
    R,
    S,
    SH,
    T,
    TH,
    V,
    W,
    Y,
    Z,
    ZH,
    // Special
    SIL,
}

impl Phoneme {
    /// ARPAbet spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            Phoneme::AA => "AA",
            Phoneme::AE => "AE",
            Phoneme::AH => "AH",
            Phoneme::AO => "AO",
            Phoneme::AW => "AW",
            Phoneme::AY => "AY",
            Phoneme::EH => "EH",
            Phoneme::ER => "ER",
            Phoneme::EY => "EY",
            Phoneme::IH => "IH",
            Phoneme::IY => "IY",
            Phoneme::OW => "OW",
            Phoneme::OY => "OY",
            Phoneme::UH => "UH",
            Phoneme::UW => "UW",
            Phoneme::B => "B",
            Phoneme::CH => "CH",
            Phoneme::D => "D",
            Phoneme::DH => "DH",
            Phoneme::F => "F",
            Phoneme::G => "G",
            Phoneme::HH => "HH",
            Phoneme::JH => "JH",
            Phoneme::K => "K",
            Phoneme::L => "L",
            Phoneme::M => "M",
            Phoneme::N => "N",
            Phoneme::NG => "NG",
            Phoneme::P => "P",
            Phoneme::R => "R",
            Phoneme::S => "S",
            Phoneme::SH => "SH",
            Phoneme::T => "T",
            Phoneme::TH => "TH",
            Phoneme::V => "V",
            Phoneme::W => "W",
            Phoneme::Y => "Y",
            Phoneme::Z => "Z",
            Phoneme::ZH => "ZH",
            Phoneme::SIL => "SIL",
        }
    }

    /// True for vowel phonemes (used by rule context matching).
    pub fn is_vowel(self) -> bool {
        matches!(
            self,
            Phoneme::AA
                | Phoneme::AE
                | Phoneme::AH
                | Phoneme::AO
                | Phoneme::AW
                | Phoneme::AY
                | Phoneme::EH
                | Phoneme::ER
                | Phoneme::EY
                | Phoneme::IH
                | Phoneme::IY
                | Phoneme::OW
                | Phoneme::OY
                | Phoneme::UH
                | Phoneme::UW
        )
    }

    /// Rough intrinsic duration in frames (10 ms) for fallback spacing.
    pub fn typical_frames(self) -> usize {
        if self.is_vowel() {
            if matches!(
                self,
                Phoneme::AW | Phoneme::AY | Phoneme::EY | Phoneme::OW | Phoneme::OY
            ) {
                12 // diphthongs
            } else {
                9
            }
        } else if matches!(self, Phoneme::SIL) {
            5
        } else {
            6
        }
    }
}

impl fmt::Display for Phoneme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Common irregular words (lowercase key → ARPAbet, spaces separate).
const LEXICON: &[(&str, &str)] = &[
    ("a", "AH"),
    ("the", "DH AH"),
    ("of", "AH V"),
    ("to", "T UW"),
    ("do", "D UW"),
    ("does", "D AH Z"),
    ("said", "S EH D"),
    ("one", "W AH N"),
    ("two", "T UW"),
    ("who", "HH UW"),
    ("was", "W AH Z"),
    ("were", "W ER"),
    ("are", "AA R"),
    ("have", "HH AE V"),
    ("give", "G IH V"),
    ("live", "L IH V"),
    ("where", "W EH R"),
    ("there", "DH EH R"),
    ("their", "DH EH R"),
    ("here", "HH IY R"),
    ("what", "W AH T"),
    ("want", "W AA N T"),
    ("come", "K AH M"),
    ("some", "S AH M"),
    ("done", "D AH N"),
    ("gone", "G AO N"),
    ("love", "L AH V"),
    ("above", "AH B AH V"),
    ("could", "K UH D"),
    ("would", "W UH D"),
    ("should", "SH UH D"),
    ("again", "AH G EH N"),
    ("people", "P IY P AH L"),
    ("because", "B IH K AO Z"),
    ("thought", "TH AO T"),
    ("enough", "IH N AH F"),
    ("through", "TH R UW"),
    ("though", "DH OW"),
    ("once", "W AH N S"),
    ("know", "N OW"),
    ("knew", "N UW"),
    ("laugh", "L AE F"),
    ("eye", "AY"),
    ("eyes", "AY Z"),
    ("only", "OW N L IY"),
    ("woman", "W UH M AH N"),
    ("women", "W IH M IH N"),
    ("money", "M AH N IY"),
    ("young", "Y AH NG"),
    ("bus", "B AH S"),
    ("put", "P UH T"),
    ("full", "F UH L"),
    ("pull", "P UH L"),
    ("be", "B IY"),
    ("he", "HH IY"),
    ("we", "W IY"),
    ("me", "M IY"),
    ("she", "SH IY"),
    ("go", "G OW"),
    ("no", "N OW"),
    ("so", "S OW"),
    ("my", "M AY"),
    ("by", "B AY"),
    ("oh", "OW"),
];

fn lexicon() -> HashMap<&'static str, Vec<Phoneme>> {
    LEXICON
        .iter()
        .map(|(w, phones)| (*w, phones.split(' ').map(parse_phoneme).collect::<Vec<_>>()))
        .collect()
}

fn parse_phoneme(s: &str) -> Phoneme {
    // Lexicon strings are trusted; unknown entries fall back to AH.
    PHONEME_NAMES
        .iter()
        .find(|(name, _)| *name == s)
        .map(|(_, p)| *p)
        .unwrap_or(Phoneme::AH)
}

const PHONEME_NAMES: &[(&str, Phoneme)] = &[
    ("AA", Phoneme::AA),
    ("AE", Phoneme::AE),
    ("AH", Phoneme::AH),
    ("AO", Phoneme::AO),
    ("AW", Phoneme::AW),
    ("AY", Phoneme::AY),
    ("EH", Phoneme::EH),
    ("ER", Phoneme::ER),
    ("EY", Phoneme::EY),
    ("IH", Phoneme::IH),
    ("IY", Phoneme::IY),
    ("OW", Phoneme::OW),
    ("OY", Phoneme::OY),
    ("UH", Phoneme::UH),
    ("UW", Phoneme::UW),
    ("B", Phoneme::B),
    ("CH", Phoneme::CH),
    ("D", Phoneme::D),
    ("DH", Phoneme::DH),
    ("F", Phoneme::F),
    ("G", Phoneme::G),
    ("HH", Phoneme::HH),
    ("JH", Phoneme::JH),
    ("K", Phoneme::K),
    ("L", Phoneme::L),
    ("M", Phoneme::M),
    ("N", Phoneme::N),
    ("NG", Phoneme::NG),
    ("P", Phoneme::P),
    ("R", Phoneme::R),
    ("S", Phoneme::S),
    ("SH", Phoneme::SH),
    ("T", Phoneme::T),
    ("TH", Phoneme::TH),
    ("V", Phoneme::V),
    ("W", Phoneme::W),
    ("Y", Phoneme::Y),
    ("Z", Phoneme::Z),
    ("ZH", Phoneme::ZH),
    ("SIL", Phoneme::SIL),
];

/// Converts one word to phonemes. Non-alphabetic characters are skipped.
pub fn g2p_word(word: &str) -> Vec<Phoneme> {
    let lower = word.to_ascii_lowercase();
    if lower.is_empty() {
        return Vec::new();
    }
    if let Some(phones) = lexicon().get(lower.as_str()) {
        return phones.clone();
    }
    let letters: Vec<char> = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut igh = false;
    let n = letters.len();
    let at = |idx: usize| -> Option<char> { letters.get(idx).copied() };
    let prev_is_vowel_letter =
        |idx: usize| idx > 0 && matches!(letters[idx - 1], 'a' | 'e' | 'i' | 'o' | 'u');
    let next_letter_is_vowel =
        |idx: usize| at(idx + 1).is_some_and(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y'));

    while i < n {
        let c = letters[i];
        // Doubled consonants (pp, ll, ss, …) sound once; doubled
        // vowels at the word end (see) were consumed by their first
        // half's rule.
        let is_vowel_letter = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u');
        if i > 0 && letters[i - 1] == c && (!is_vowel_letter || i == n - 1) {
            i += 1;
            continue;
        }
        let next = at(i + 1);
        let two: Option<String> = next.map(|n2| format!("{c}{n2}"));

        // Longest-match digraphs first.
        if let Some(digraph) = two.as_deref() {
            let mapped = match digraph {
                "sh" => Some(vec![Phoneme::SH]),
                "ch" | "tch" => Some(vec![Phoneme::CH]),
                "th" => Some(vec![Phoneme::TH]),
                "ph" => Some(vec![Phoneme::F]),
                "ck" => Some(vec![Phoneme::K]),
                "ng" => Some(vec![Phoneme::NG]),
                "qu" => Some(vec![Phoneme::K, Phoneme::W]),
                "wh" => Some(vec![Phoneme::W]),
                "gh" => Some(vec![]), // silent (light, high)
                _ => None,
            };
            if let Some(m) = mapped {
                out.extend(m);
                i += 2;
                continue;
            }
        }

        let emitted: Vec<Phoneme> = match c {
            'a' => {
                // magic-e ("cake") and open syllables ("paper") → long A.
                if (at(i + 2) == Some('e') && at(i + 1).is_some()) || next_letter_is_vowel(i) {
                    vec![Phoneme::EY]
                } else {
                    vec![Phoneme::AE]
                }
            }
            'e' => {
                if i == n.saturating_sub(1) && n > 2 && !prev_is_vowel_letter(i) {
                    vec![] // final silent e (time, table)
                } else if next_letter_is_vowel(i) {
                    vec![Phoneme::IY]
                } else {
                    vec![Phoneme::EH]
                }
            }
            'i' => {
                // igh → AY (light)
                if at(i + 1) == Some('g') && at(i + 2) == Some('h') {
                    igh = true;
                    vec![Phoneme::AY]
                } else if (i + 3 == n && at(i + 1) == Some('n') && at(i + 2) == Some('d'))
                    || at(i + 2) == Some('e')
                {
                    vec![Phoneme::AY] // -ind (find) and i_e (like)
                } else {
                    vec![Phoneme::IH]
                }
            }
            'o' => {
                if at(i + 2) == Some('e') || next == Some('w') || next == Some('a') {
                    vec![Phoneme::OW] // o_e (home), ow/oa digraphs
                } else {
                    vec![Phoneme::AA]
                }
            }
            'u' => {
                if at(i + 2) == Some('e') {
                    vec![Phoneme::UW] // u_e (rule) — flute-fluke ambiguity accepted
                } else {
                    vec![Phoneme::AH]
                }
            }
            'y' => {
                if i == 0 {
                    vec![Phoneme::Y]
                } else if i == n - 1 {
                    if n == 3 && !prev_is_vowel_letter(i) {
                        vec![Phoneme::AY] // monosyllable (fly, sky)
                    } else {
                        vec![Phoneme::IY] // final y (happy)
                    }
                } else {
                    vec![Phoneme::IH]
                }
            }
            'b' => vec![Phoneme::B],
            'c' => match next {
                Some('e') | Some('i') | Some('y') => vec![Phoneme::S],
                _ => vec![Phoneme::K],
            },
            'd' => vec![Phoneme::D],
            'f' => vec![Phoneme::F],
            'g' => match next {
                Some('e') | Some('i') | Some('y') => vec![Phoneme::JH],
                _ => vec![Phoneme::G],
            },
            'h' => vec![Phoneme::HH],
            'j' => vec![Phoneme::JH],
            'k' => vec![Phoneme::K],
            'l' => vec![Phoneme::L],
            'm' => vec![Phoneme::M],
            'n' => vec![Phoneme::N],
            'p' => vec![Phoneme::P],
            'q' => vec![Phoneme::K],
            'r' => vec![Phoneme::R],
            's' => {
                if i == 0 && next == Some('s') && n > 2 {
                    vec![] // "ss" cluster start — second s handled next
                } else {
                    vec![Phoneme::S]
                }
            }
            't' => vec![Phoneme::T],
            'v' => vec![Phoneme::V],
            'w' => vec![Phoneme::W],
            'x' => vec![Phoneme::K, Phoneme::S],
            'z' => vec![Phoneme::Z],
            _ => vec![],
        };
        out.extend(emitted);
        if igh {
            // The AY came from the "igh" trigraph: skip g and h.
            igh = false;
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// Converts whitespace-separated text into per-word phoneme lists.
pub fn g2p(text: &str) -> Vec<Vec<Phoneme>> {
    text.split_whitespace().map(g2p_word).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(p: &[Phoneme]) -> Vec<&'static str> {
        p.iter().map(Phoneme::as_str).collect()
    }

    #[test]
    fn lexicon_words() {
        assert_eq!(names(&g2p_word("the")), ["DH", "AH"]);
        assert_eq!(names(&g2p_word("one")), ["W", "AH", "N"]);
        assert_eq!(names(&g2p_word("THROUGH")), ["TH", "R", "UW"]);
    }

    #[test]
    fn regular_words() {
        assert_eq!(names(&g2p_word("cat")), ["K", "AE", "T"]);
        assert_eq!(names(&g2p_word("ship")), ["SH", "IH", "P"]);
        assert_eq!(names(&g2p_word("cake")), ["K", "EY", "K"]);
        assert_eq!(names(&g2p_word("like")), ["L", "AY", "K"]);
        assert_eq!(names(&g2p_word("home")), ["HH", "OW", "M"]);
        assert_eq!(names(&g2p_word("happy")), ["HH", "AE", "P", "IY"]);
        assert_eq!(names(&g2p_word("fox")), ["F", "AA", "K", "S"]);
    }

    #[test]
    fn silent_e_and_digraphs() {
        assert_eq!(names(&g2p_word("make")), ["M", "EY", "K"]);
        // "e" in "the" is lexicon; check "time"
        assert_eq!(names(&g2p_word("time")), ["T", "AY", "M"]);
        assert_eq!(names(&g2p_word("chat")), ["CH", "AE", "T"]);
        assert_eq!(names(&g2p_word("when")), ["W", "EH", "N"]);
    }

    #[test]
    fn punctuation_ignored() {
        assert_eq!(g2p_word("Hello,"), g2p_word("hello"));
        assert!(g2p_word("!!!").is_empty());
        assert!(g2p_word("").is_empty());
    }

    #[test]
    fn g2p_splits_words() {
        let w = g2p("the cat");
        assert_eq!(w.len(), 2);
        assert_eq!(names(&w[0]), ["DH", "AH"]);
        assert_eq!(names(&w[1]), ["K", "AE", "T"]);
    }

    #[test]
    fn durations_are_positive() {
        for (_, p) in PHONEME_NAMES {
            assert!((*p).typical_frames() > 0);
        }
    }
}
