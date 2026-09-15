//! Text normalization and phoneme parsing for synthesis.
//!
//! Sentences are split into segments carrying prosody intent (statement
//! vs question vs clause pause), words into phonemes via the align
//! crate's rule-based G2P.

use tpt_av_voice_align::Phoneme;

/// A word plus trailing punctuation intent.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub phonemes: Vec<Phoneme>,
    /// Pause after this word, in seconds, before rate scaling.
    pub pause_secs: f32,
}

/// One sentence: words plus its terminal contour.
#[derive(Debug, Clone, PartialEq)]
pub struct Sentence {
    pub words: Vec<Word>,
    /// Pitch offset at the sentence end, in semitones.
    pub end_semitones: f32,
}

/// Parses text into sentences of phoneme-converted words.
pub fn parse_text(text: &str) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut current: Vec<Word> = Vec::new();

    for raw in text.split_whitespace() {
        // Strip and classify trailing punctuation.
        let (core, punct) = split_punctuation(raw);
        let phonemes = tpt_av_voice_align::g2p_word(core);
        if !phonemes.is_empty() {
            current.push(Word {
                phonemes,
                pause_secs: 0.0,
            });
        }
        match punct {
            b'?' => {
                if !current.is_empty() {
                    current.last_mut().unwrap().pause_secs = 0.25;
                    sentences.push(Sentence {
                        words: std::mem::take(&mut current),
                        end_semitones: 4.0,
                    });
                }
            }
            b'.' | b'!' | b';' => {
                if !current.is_empty() {
                    if let Some(w) = current.last_mut() {
                        w.pause_secs = 0.25;
                    }
                    sentences.push(Sentence {
                        words: std::mem::take(&mut current),
                        end_semitones: -3.0,
                    });
                }
            }
            b',' | b':' => {
                if let Some(w) = current.last_mut() {
                    w.pause_secs = 0.15;
                }
            }
            _ => {}
        }
    }
    if !current.is_empty() {
        sentences.push(Sentence {
            words: current,
            end_semitones: -3.0,
        });
    }
    sentences
}

fn split_punctuation(word: &str) -> (&str, u8) {
    let trimmed = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    let bytes = word.as_bytes();
    let last = bytes.last().copied().unwrap_or(b'.');
    match last {
        b'?' | b'!' | b'.' | b',' | b';' | b':' if word.ends_with(last as char) => {
            let _ = bytes;
            (trimmed, last)
        }
        _ => (trimmed, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences_on_terminal_punctuation() {
        let s = parse_text("Hello world. How are you?");
        assert_eq!(s.len(), 2);
        assert!(s[0].end_semitones < 0.0, "statement falls");
        assert!(s[1].end_semitones > 0.0, "question rises");
        assert_eq!(s[0].words.len(), 2);
        assert!(s[0].words.last().unwrap().pause_secs > 0.0);
    }

    #[test]
    fn commas_pause_without_splitting() {
        let s = parse_text("one, two three");
        assert_eq!(s.len(), 1);
        assert!(s[0].words[0].pause_secs > 0.0);
        assert_eq!(s[0].words[1].pause_secs, 0.0);
    }

    #[test]
    fn punctuation_is_stripped_from_words() {
        let s = parse_text("(wait)...");
        assert_eq!(s[0].words.len(), 1);
        assert_eq!(s[0].words[0].phonemes, tpt_av_voice_align::g2p_word("wait"));
    }

    #[test]
    fn empty_text_gives_no_sentences() {
        assert!(parse_text("   ...  ").is_empty());
    }
}
