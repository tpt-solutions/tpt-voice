//! Byte-level BPE tokenization (GPT-2 style), with a trainer so pipelines
//! can build small vocabularies offline.
//!
//! Serialization is plain JSON: `{"vocab": {token: id}, "merges":
//! ["a b", ...], "specials": ["<|endoftext|>", ...]}`. Byte-level
//! encoding maps arbitrary UTF-8 into printable tokens, so any input
//! round-trips losslessly.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::VoiceError;

/// Behavior shared by tokenizer implementations.
pub trait Tokenizer {
    /// Encodes text into token ids.
    fn encode(&self, text: &str) -> Vec<usize>;
    /// Decodes token ids back into text (unknown ids are dropped).
    fn decode(&self, tokens: &[usize]) -> String;
    /// Vocabulary size (including specials).
    fn vocab_size(&self) -> usize;
    /// Id of the end-of-text special, if any.
    fn eot_token(&self) -> Option<usize>;
}

/// Special-token names used by the engine.
pub const EOT: &str = "<|endoftext|>";
pub const SOT: &str = "<|startoftranscript|>";

/// Byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: HashMap<String, usize>,
    ids: Vec<String>,
    merges: HashMap<(String, String), usize>,
    specials: Vec<String>,
    eot: usize,
    sot: usize,
    /// Byte ↔ printable-unicode tables.
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

/// Serializable on-disk representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BpeFile {
    vocab: Vec<String>,
    merges: Vec<String>,
    specials: Vec<String>,
}

impl BpeTokenizer {
    /// Loads a tokenizer from the JSON representation produced by
    /// [`BpeTokenizer::to_json`] / [`BpeTokenizer::train`].
    pub fn from_json(json: &str) -> Result<Self, VoiceError> {
        let file: BpeFile = serde_json::from_str(json)
            .map_err(|e| VoiceError::tokenizer(format!("bad tokenizer JSON: {e}")))?;
        Self::build(file.vocab, file.merges, file.specials)
    }

    /// Serializes the tokenizer to JSON.
    pub fn to_json(&self) -> String {
        let file = BpeFile {
            vocab: self.ids.clone(),
            merges: self
                .merges
                .iter()
                .map(|((a, b), _)| format!("{a} {b}"))
                .collect(),
            specials: self.specials.clone(),
        };
        serde_json::to_string(&file).expect("tokenizer serialization cannot fail")
    }

    fn build(
        vocab: Vec<String>,
        merges: Vec<String>,
        specials: Vec<String>,
    ) -> Result<Self, VoiceError> {
        let mut ids_map = HashMap::new();
        let mut ids = Vec::new();
        for (id, tok) in vocab.into_iter().enumerate() {
            if ids_map.insert(tok.clone(), id).is_some() {
                return Err(VoiceError::tokenizer(format!("duplicate token {tok:?}")));
            }
            ids.push(tok);
        }
        let mut merges_map = HashMap::new();
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = m
                .split_once(' ')
                .ok_or_else(|| VoiceError::tokenizer(format!("bad merge {m:?}")))?;
            merges_map.insert((a.to_string(), b.to_string()), rank);
        }
        let (byte_encoder, byte_decoder) = byte_tables();
        let eot = ids_map
            .get(EOT)
            .copied()
            .ok_or_else(|| VoiceError::tokenizer("missing <|endoftext|> special token"))?;
        let sot = ids_map
            .get(SOT)
            .copied()
            .ok_or_else(|| VoiceError::tokenizer("missing <|startoftranscript|> special token"))?;
        Ok(BpeTokenizer {
            vocab: ids_map,
            ids,
            merges: merges_map,
            specials,
            eot,
            sot,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Id of the start-of-transcript token.
    pub fn sot_token(&self) -> usize {
        self.sot
    }

    /// Looks up a token's id (including specials).
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }

    /// Looks up the token string for an id.
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.ids.get(id).map(String::as_str)
    }

    /// Trains a byte-level BPE tokenizer on `corpus` down to
    /// `vocab_size` tokens (two specials reserved).
    pub fn train(corpus: &str, vocab_size: usize) -> Result<Self, VoiceError> {
        if vocab_size < 300 {
            return Err(VoiceError::config(
                "vocab_size must reserve 256 byte tokens + specials",
            ));
        }
        // Word frequency over whitespace-split words.
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for word in corpus.split_whitespace() {
            *word_freqs.entry(word.to_string()).or_default() += 1;
        }
        if word_freqs.is_empty() {
            return Err(VoiceError::config("empty training corpus"));
        }

        // Initial symbols: printable byte chars; a leading 'Ġ' marks the
        // space that separated this word in the corpus.
        let (byte_encoder, _) = byte_tables();
        let mut words: Vec<(Vec<String>, usize)> = word_freqs
            .into_iter()
            .map(|(w, f)| {
                let mut syms: Vec<String> = vec!["Ġ".to_string()];
                syms.extend(
                    w.as_bytes()
                        .iter()
                        .map(|&b| byte_encoder[b as usize].to_string()),
                );
                (syms, f)
            })
            .collect();

        let mut vocab: Vec<String> = Vec::new();
        vocab.push(EOT.to_string());
        vocab.push(SOT.to_string());
        // All 256 byte tokens; space maps to 'Ġ' and newline to 'Ċ'.
        for b in 0..256u16 {
            vocab.push(byte_encoder[b as usize].to_string());
        }
        let mut merges: Vec<String> = Vec::new();

        while vocab.len() < vocab_size {
            // Count adjacent pairs.
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
            for (syms, freq) in &words {
                for pair in syms.windows(2) {
                    *pair_freqs
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_default() += *freq;
                }
            }
            // Deterministic pick: highest frequency, ties broken
            // lexicographically by the pair.
            let mut candidates: Vec<((String, String), usize)> =
                pair_freqs.into_iter().filter(|(_, f)| *f > 1).collect();
            candidates.sort_by(|(a, fa), (b, fb)| fb.cmp(fa).then_with(|| a.cmp(b)));
            let Some(((a, b), _)) = candidates.first() else {
                break;
            };
            let (a, b) = (a.clone(), b.clone());
            // Merge the pair everywhere.
            for (syms, _) in words.iter_mut() {
                let mut i = 0;
                while i + 1 < syms.len() {
                    if syms[i] == a && syms[i + 1] == b {
                        let merged = format!("{a}{b}");
                        syms.splice(i..i + 2, std::iter::once(merged));
                    }
                    i += 1;
                }
            }
            merges.push(format!("{a} {b}"));
            vocab.push(format!("{a}{b}"));
        }

        Self::build(vocab, merges, vec![EOT.to_string(), SOT.to_string()])
    }

    /// Applies the learned merge ranks to one word (already
    /// byte-encoded, with `Ġ` space marker).
    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for (i, pair) in symbols.windows(2).enumerate() {
                if let Some(&rank) = self.merges.get(&(pair[0].clone(), pair[1].clone())) {
                    if best.map_or(true, |(br, _)| rank < br) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, std::iter::once(merged));
        }
        symbols
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        // Special-token pass first.
        let mut rest = text;
        for special in &self.specials {
            // Simple scan: any occurrence becomes one id.
            while let Some(idx) = rest.find(special.as_str()) {
                let (before, after_full) = rest.split_at(idx);
                out.extend(self.encode_plain(before));
                if let Some(id) = self.vocab.get(special) {
                    out.push(*id);
                }
                rest = &after_full[special.len()..];
            }
        }
        out.extend(self.encode_plain(rest));
        out
    }

    fn decode(&self, tokens: &[usize]) -> String {
        // Reassemble raw bytes from the byte-level tokens, then decode
        // UTF-8 — this round-trips any original text exactly.
        let mut bytes: Vec<u8> = Vec::new();
        for &id in tokens {
            let Some(tok) = self.ids.get(id) else {
                continue;
            };
            if self.specials.iter().any(|sp| sp == tok) {
                continue;
            }
            for ch in tok.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.ids.len()
    }

    fn eot_token(&self) -> Option<usize> {
        Some(self.eot)
    }
}

impl BpeTokenizer {
    /// Plain (no-specials) encoding. Text is chunked on whitespace
    /// boundaries (`split_inclusive`), so every whitespace byte survives
    /// round-trip; each chunk's bytes go through the GPT-2 byte table
    /// and BPE merges.
    fn encode_plain(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for chunk in text.split_inclusive(char::is_whitespace) {
            if chunk.is_empty() {
                continue;
            }
            let symbols: Vec<String> = chunk
                .bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect();
            for sym in self.bpe(symbols) {
                match self.vocab.get(&sym) {
                    Some(&id) => out.push(id),
                    None => {
                        // Fall back to raw byte tokens (always in vocab).
                        for ch in sym.chars() {
                            if let Some(&id) = self.vocab.get(&ch.to_string()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// GPT-2 byte↔unicode tables (maps all 256 bytes to printable chars).
fn byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=0xFFu8);
    let mut cs: Vec<u16> = bs.iter().map(|&b| b as u16).collect();
    let mut n = 0u16;
    for b in 0..256u16 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_encoder = ['\0'; 256];
    let mut byte_decoder = HashMap::new();
    for (b, c) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(c as u32).unwrap_or('\u{FFFD}');
        byte_encoder[b as usize] = ch;
        byte_decoder.insert(ch, b);
    }
    (byte_encoder, byte_decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "the quick brown fox jumps over the lazy dog \
        the dog barks and the fox runs the quick fox is quick";

    fn trained() -> BpeTokenizer {
        BpeTokenizer::train(CORPUS, 300).unwrap()
    }

    #[test]
    fn round_trip_arbitrary_text() {
        let tok = trained();
        for text in [
            "",
            "the quick fox",
            "héllo wörld 你好",
            "  leading spaces",
            "tabs\tand\nnewlines",
        ] {
            let ids = tok.encode(text);
            assert!(ids.iter().all(|&i| i < tok.vocab_size()));
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, text, "round trip of {text:?}");
        }
    }

    #[test]
    fn specials_encode_atomically() {
        let tok = trained();
        let ids = tok.encode(&format!("hello {EOT} world"));
        assert!(ids.contains(&tok.token_to_id(EOT).unwrap()));
        assert!(
            !tok.decode(&ids).contains('<'),
            "specials are not spelled out"
        );
    }

    #[test]
    fn merge_happens_on_frequent_words() {
        let tok = BpeTokenizer::train("abababab abababab abababab", 320).unwrap();
        // "ab" must become a merged token and encode shorter than bytes.
        let ids = tok.encode("abababab");
        assert!(ids.len() < 8, "expected merges, got {ids:?}");
        assert_eq!(tok.decode(&ids), "abababab");
    }

    #[test]
    fn json_serialization_round_trip() {
        let tok = trained();
        let json = tok.to_json();
        let tok2 = BpeTokenizer::from_json(&json).unwrap();
        assert_eq!(tok.encode("the quick fox"), tok2.encode("the quick fox"));
        assert_eq!(tok2.decode(&tok2.encode("the lazy dog")), "the lazy dog");
    }

    #[test]
    fn train_validates() {
        assert!(BpeTokenizer::train(CORPUS, 100).is_err());
        assert!(BpeTokenizer::train("   ", 300).is_err());
    }

    #[test]
    fn eot_sot_present() {
        let tok = trained();
        assert_eq!(tok.token_to_id(SOT), Some(tok.sot_token()));
        assert_eq!(tok.eot_token(), Some(tok.token_to_id(EOT).unwrap()));
        assert!(tok.vocab_size() >= 258);
    }
}
