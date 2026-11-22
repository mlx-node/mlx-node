//! Direct byte-level suffix-automaton constrained masker for grounded text
//! extraction (`GrammarSpec::GroundedText`).
//!
//! Matches the exact language of Clairic's `grounding.py:_text_states`:
//! - `"null"` (if `nullable == true`), OR
//! - `"<json-escaped non-empty substring of source>"`, OR
//! - `"<json-escaped non-empty substring of normalized_source>"` (if provided).
//!
//! Building a character-level suffix automaton (O(N) states/edges) and
//! expanding each state's outgoing JSON-escaped character atoms into a flat
//! deterministic byte DFA avoids CFG rule explosion entirely: a 4,096-char
//! source compiles in <1 ms and masks a 150k+ vocabulary via `TokTrie::add_bias`
//! in microseconds.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use napi::{Error, Result};
use toktrie::{Recognizer, SimpleVob, TokEnv, TokTrie, TokenId};

/// JSON-escape a single Unicode scalar value (matching RFC 8259 / `json.dumps(ch)[1:-1]`).
fn json_escape_char(ch: char, out: &mut Vec<u8>) {
    out.clear();
    match ch {
        '"' => out.extend_from_slice(b"\\\""),
        '\\' => out.extend_from_slice(b"\\\\"),
        '\x08' => out.extend_from_slice(b"\\b"),
        '\x0C' => out.extend_from_slice(b"\\f"),
        '\n' => out.extend_from_slice(b"\\n"),
        '\r' => out.extend_from_slice(b"\\r"),
        '\t' => out.extend_from_slice(b"\\t"),
        c if (c as u32) < 0x20 => {
            let mut buf = String::with_capacity(6);
            let _ = write!(buf, "\\u{:04x}", c as u32);
            out.extend_from_slice(buf.as_bytes());
        }
        c => {
            let mut utf8 = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut utf8).as_bytes());
        }
    }
}

/// Character-level suffix automaton state (standard Blumer et al. / cp-algorithms).
struct SamState {
    len: usize,
    link: Option<usize>,
    next: HashMap<char, usize>,
}

fn build_char_sam(text: &str) -> Vec<SamState> {
    let char_count = text.chars().count();
    let mut states = Vec::with_capacity(char_count.saturating_mul(2).max(1));
    states.push(SamState {
        len: 0,
        link: None,
        next: HashMap::new(),
    });
    let mut last: usize = 0;

    for ch in text.chars() {
        let cur = states.len();
        states.push(SamState {
            len: states[last].len + 1,
            link: None,
            next: HashMap::new(),
        });

        let mut p = Some(last);
        while let Some(pp) = p {
            if states[pp].next.contains_key(&ch) {
                break;
            }
            states[pp].next.insert(ch, cur);
            p = states[pp].link;
        }

        if let Some(pp) = p {
            if let Some(&q) = states[pp].next.get(&ch) {
                if states[pp].len + 1 == states[q].len {
                    states[cur].link = Some(q);
                } else {
                    let clone = states.len();
                    let q_link = states[q].link;
                    let q_next = states[q].next.clone();
                    states.push(SamState {
                        len: states[pp].len + 1,
                        link: q_link,
                        next: q_next,
                    });
                    let mut ppp = p;
                    while let Some(idx) = ppp {
                        if states[idx].next.get(&ch) == Some(&q) {
                            states[idx].next.insert(ch, clone);
                            ppp = states[idx].link;
                        } else {
                            break;
                        }
                    }
                    states[q].link = Some(clone);
                    states[cur].link = Some(clone);
                }
            }
        } else {
            states[cur].link = Some(0);
        }
        last = cur;
    }

    states
}

/// Compact flat node in a byte-level DFA expanded from a character SAM.
#[derive(Clone, Copy, Debug)]
struct ByteDfaNode {
    edge_start: u32,
    edge_len: u16,
    /// True iff this node is a valid end-of-substring character boundary
    /// where a closing `"` may be emitted.
    can_close: bool,
}

/// Deterministic byte-level automaton for all JSON-escaped substrings of a text.
struct CharSamByteDfa {
    nodes: Vec<ByteDfaNode>,
    /// Flat sorted `(byte, target_node_id)` slices referenced by `nodes`.
    edges: Vec<(u8, u32)>,
    is_empty_source: bool,
}

impl CharSamByteDfa {
    fn build(text: &str, allow_empty_string: bool) -> Self {
        let sam = build_char_sam(text);
        let is_empty_source = sam.len() == 1 && sam[0].next.is_empty();

        // Node IDs 0..sam.len() correspond 1-to-1 with SAM character-boundary states.
        // Intermediate nodes for multi-byte JSON escapes / UTF-8 chars are appended after.
        let mut adj: Vec<Vec<(u8, u32)>> = vec![Vec::new(); sam.len()];
        let mut can_close: Vec<bool> = sam
            .iter()
            .enumerate()
            .map(|(idx, _)| idx > 0 || (allow_empty_string && is_empty_source))
            .collect();

        let mut esc_buf = Vec::with_capacity(8);

        for (sam_idx, state) in sam.iter().enumerate() {
            for (&ch, &target_sam) in &state.next {
                json_escape_char(ch, &mut esc_buf);
                if esc_buf.is_empty() {
                    continue;
                }
                let mut cur_node = sam_idx;
                let last_pos = esc_buf.len() - 1;
                for (pos, &b) in esc_buf.iter().enumerate() {
                    if pos == last_pos {
                        // Leaf of this char's byte sequence lands on `target_sam`.
                        // JSON escapes and UTF-8 are prefix-free, so no collision occurs.
                        adj[cur_node].push((b, target_sam as u32));
                    } else if let Some(&(_, existing)) =
                        adj[cur_node].iter().find(|&&(edge_b, _)| edge_b == b)
                    {
                        cur_node = existing as usize;
                    } else {
                        let new_node = adj.len();
                        adj.push(Vec::new());
                        can_close.push(false);
                        adj[cur_node].push((b, new_node as u32));
                        cur_node = new_node;
                    }
                }
            }
        }

        let total_edges: usize = adj.iter().map(Vec::len).sum();
        let mut nodes = Vec::with_capacity(adj.len());
        let mut edges = Vec::with_capacity(total_edges);

        for (mut node_edges, closable) in adj.into_iter().zip(can_close) {
            node_edges.sort_unstable_by_key(|&(b, _)| b);
            let edge_start = edges.len() as u32;
            let edge_len = node_edges.len() as u16;
            edges.extend(node_edges);
            nodes.push(ByteDfaNode {
                edge_start,
                edge_len,
                can_close: closable,
            });
        }

        Self {
            nodes,
            edges,
            is_empty_source,
        }
    }

    #[inline]
    fn can_close(&self, node: u32) -> bool {
        self.nodes.get(node as usize).is_some_and(|n| n.can_close)
    }

    #[inline]
    fn step(&self, node: u32, byte: u8) -> Option<u32> {
        let info = self.nodes.get(node as usize)?;
        let start = info.edge_start as usize;
        let len = info.edge_len as usize;
        let slice = &self.edges[start..start + len];
        if len <= 8 {
            for &(b, target) in slice {
                if b == byte {
                    return Some(target);
                }
                if b > byte {
                    return None;
                }
            }
            None
        } else {
            slice
                .binary_search_by_key(&byte, |&(b, _)| b)
                .ok()
                .map(|idx| slice[idx].1)
        }
    }
}

/// Deterministic state of the grounded-text JSON matcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Initial state before any output byte.
    Start,
    /// Matched `1..=4` bytes of literal `null`. `Null(4)` is accepting.
    Null(u8),
    /// Inside `"..."`, tracking active byte-DFA node in raw & optional normalized SAM.
    InString { raw: Option<u32>, norm: Option<u32> },
    /// Matched closing `"`. Accepting (only EOS allowed next).
    ClosedQuote,
    /// Consumed EOS after an accepting state.
    Done,
    /// Invalid byte encountered.
    Dead,
}

struct SuffixDfa {
    raw: CharSamByteDfa,
    norm: Option<CharSamByteDfa>,
    nullable: bool,
    has_string_branch: bool,
}

impl SuffixDfa {
    fn new(source: &str, normalized_source: Option<&str>, nullable: bool) -> Self {
        let norm_str = normalized_source.filter(|&n| n != source && !n.is_empty());
        let both_empty = source.is_empty() && norm_str.is_none();
        let allow_empty_string = both_empty && !nullable;

        let raw = CharSamByteDfa::build(source, allow_empty_string);
        let norm = norm_str.map(|n| CharSamByteDfa::build(n, false));
        let has_string_branch = !raw.is_empty_source
            || norm.as_ref().is_some_and(|n| !n.is_empty_source)
            || allow_empty_string;

        Self {
            raw,
            norm,
            nullable,
            has_string_branch,
        }
    }

    #[inline]
    fn step_phase(&self, phase: Phase, byte: u8) -> Phase {
        match phase {
            Phase::Start => {
                if byte == b'"' && self.has_string_branch {
                    let raw_init =
                        (!self.raw.is_empty_source || self.raw.can_close(0)).then_some(0);
                    let norm_init = self
                        .norm
                        .as_ref()
                        .and_then(|n| (!n.is_empty_source).then_some(0));
                    Phase::InString {
                        raw: raw_init,
                        norm: norm_init,
                    }
                } else if self.nullable && byte == b'n' {
                    Phase::Null(1)
                } else {
                    Phase::Dead
                }
            }
            Phase::Null(k) => {
                const NULL_BYTES: &[u8; 4] = b"null";
                if (k as usize) < NULL_BYTES.len() && byte == NULL_BYTES[k as usize] {
                    Phase::Null(k + 1)
                } else {
                    Phase::Dead
                }
            }
            Phase::InString { raw, norm } => {
                // First follow explicit byte edges (including `b'"'` as the 2nd byte of `\"`).
                let next_raw = raw.and_then(|s| self.raw.step(s, byte));
                let next_norm = norm.and_then(|s| self.norm.as_ref().and_then(|n| n.step(s, byte)));
                if next_raw.is_some() || next_norm.is_some() {
                    return Phase::InString {
                        raw: next_raw,
                        norm: next_norm,
                    };
                }
                // Otherwise, `b'"'` closes the JSON string if either branch is at a closable boundary.
                if byte == b'"' {
                    let closable = raw.is_some_and(|s| self.raw.can_close(s))
                        || norm.is_some_and(|s| self.norm.as_ref().is_some_and(|n| n.can_close(s)));
                    if closable {
                        return Phase::ClosedQuote;
                    }
                }
                Phase::Dead
            }
            Phase::ClosedQuote | Phase::Done | Phase::Dead => Phase::Dead,
        }
    }

    #[inline]
    fn is_accepting(phase: Phase) -> bool {
        matches!(phase, Phase::Null(4) | Phase::ClosedQuote)
    }

    #[inline]
    fn is_terminated(phase: Phase) -> bool {
        matches!(phase, Phase::Null(4) | Phase::ClosedQuote | Phase::Done)
    }
}

struct SuffixRecognizer<'a> {
    dfa: &'a SuffixDfa,
    stack: Vec<Phase>,
}

impl<'a> SuffixRecognizer<'a> {
    fn new(dfa: &'a SuffixDfa, initial: Phase) -> Self {
        let mut stack = Vec::with_capacity(64);
        stack.push(initial);
        Self { dfa, stack }
    }
}

impl Recognizer for SuffixRecognizer<'_> {
    #[inline]
    fn pop_bytes(&mut self, num: usize) {
        let new_len = self.stack.len().saturating_sub(num).max(1);
        self.stack.truncate(new_len);
    }

    #[inline]
    fn collapse(&mut self) {
        let top = self.stack.last().copied().unwrap_or(Phase::Dead);
        self.stack.clear();
        self.stack.push(top);
    }

    #[inline]
    fn trie_finished(&mut self) {
        self.stack.truncate(1);
    }

    #[inline]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if byte == TokTrie::SPECIAL_TOKEN_MARKER {
            return false;
        }
        let top = self.stack.last().copied().unwrap_or(Phase::Dead);
        let next = self.dfa.step_phase(top, byte);
        if next != Phase::Dead {
            self.stack.push(next);
            true
        } else {
            false
        }
    }
}

/// Fast, exact constrained decoder for `GrammarSpec::GroundedText`.
///
/// Intersects a deterministic byte-level suffix automaton directly with
/// `TokTrie::add_bias` — zero CFG rule blowup, O(N) compile time, microsecond
/// per-step masking.
#[derive(Clone)]
pub struct SuffixAutomatonMasker {
    tok_env: TokEnv,
    dfa: Arc<SuffixDfa>,
    phase: Phase,
    stop_token_ids: Vec<TokenId>,
}

impl SuffixAutomatonMasker {
    /// Compile a grounded-text suffix automaton over `source` and optional
    /// `normalized_source`.
    pub fn new(
        tok_env: TokEnv,
        source: &str,
        normalized_source: Option<&str>,
        nullable: bool,
        stop_token_ids: Vec<TokenId>,
    ) -> Result<Self> {
        let dfa = Arc::new(SuffixDfa::new(source, normalized_source, nullable));
        let mut stops = stop_token_ids;
        let primary_eos = tok_env.tok_trie().eos_token();
        if !stops.contains(&primary_eos) {
            stops.insert(0, primary_eos);
        }
        Ok(Self {
            tok_env,
            dfa,
            phase: Phase::Start,
            stop_token_ids: stops,
        })
    }

    /// Returns `true` when the grammar has reached a terminal accepting state
    /// (`"..."` closed quote, `"null"`, or EOS consumed).
    pub fn is_terminated(&self) -> bool {
        SuffixDfa::is_terminated(self.phase)
    }

    /// Returns `true` when EOS is currently allowed.
    pub fn is_accepting(&self) -> bool {
        SuffixDfa::is_accepting(self.phase) || self.phase == Phase::Done
    }

    /// Reset matcher back to initial state.
    pub fn reset(&mut self) {
        self.phase = Phase::Start;
    }

    /// Compute the `SimpleVob` bitmask of allowed tokens at the current state.
    pub fn compute_mask(&self) -> SimpleVob {
        let trie = self.tok_env.tok_trie();
        let vocab_size = trie.vocab_size();
        let mut vob = trie.alloc_token_set();

        if self.phase == Phase::Dead {
            return vob;
        }

        if !self.is_terminated() {
            let mut recog = SuffixRecognizer::new(&self.dfa, self.phase);
            trie.add_bias(&mut recog, &mut vob, &[]);
        }

        if self.is_accepting() {
            for &stop_id in &self.stop_token_ids {
                if (stop_id as usize) < vocab_size {
                    vob.allow_token(stop_id);
                }
            }
        }

        vob
    }

    /// Fill `allowed` slice with `true` for allowed token IDs and `false` otherwise.
    pub fn fill_allowed_mask(&self, allowed: &mut [bool]) -> Result<()> {
        let vocab_size = self.tok_env.tok_trie().vocab_size();
        if allowed.len() < vocab_size {
            return Err(Error::from_reason(format!(
                "allowed mask length {} is smaller than vocab_size {vocab_size}",
                allowed.len()
            )));
        }
        let vob = self.compute_mask();
        allowed.fill(false);
        vob.iter_set_entries(|idx| {
            if idx < vocab_size {
                allowed[idx] = true;
            }
        });
        Ok(())
    }

    /// Advance the masker by one token. Returns `Ok(true)` if the token is valid
    /// and accepted, or `Ok(false)` if disallowed (leaving state unchanged).
    pub fn accept_token(&mut self, token_id: TokenId) -> Result<bool> {
        let trie = self.tok_env.tok_trie();
        if (token_id as usize) >= trie.vocab_size() {
            return Ok(false);
        }

        if self.stop_token_ids.contains(&token_id) {
            if self.is_accepting() {
                self.phase = Phase::Done;
                return Ok(true);
            }
            return Ok(false);
        }

        if self.is_terminated() || self.phase == Phase::Dead {
            return Ok(false);
        }

        let bytes = trie.token(token_id);
        if bytes.is_empty() || bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER {
            return Ok(false);
        }

        let mut next_phase = self.phase;
        for &b in bytes {
            next_phase = self.dfa.step_phase(next_phase, b);
            if next_phase == Phase::Dead {
                return Ok(false);
            }
        }

        self.phase = next_phase;
        Ok(true)
    }
}

/// Emit Clairic's GBNF suffix-automaton DAG (`root ::= ...`, `s0 ::= ...`, `n0 ::= ...`)
/// for parity testing and benchmarking against `llguidance` Lark compilation.
pub fn build_clairic_ebnf_for_text(
    source: &str,
    normalized_source: Option<&str>,
    nullable: bool,
) -> String {
    let mut out = String::new();
    let norm_str = normalized_source.filter(|&n| n != source && !n.is_empty());

    let mut root_alts: Vec<&str> = Vec::new();
    if nullable {
        root_alts.push("\"null\"");
    }
    if !source.is_empty() {
        root_alts.push("\"\\\"\" s0");
    }
    if norm_str.is_some() {
        root_alts.push("\"\\\"\" n0");
    }
    if root_alts.is_empty() {
        root_alts.push("\"\\\"\\\"\"");
    }
    let _ = writeln!(out, "root ::= {}", root_alts.join(" | "));

    if !source.is_empty() {
        emit_sam_gbnf_rules(&mut out, "s", source);
    }
    if let Some(norm) = norm_str {
        emit_sam_gbnf_rules(&mut out, "n", norm);
    }

    out
}

fn emit_sam_gbnf_rules(out: &mut String, prefix: &str, text: &str) {
    let sam = build_char_sam(text);
    let mut esc_buf = Vec::with_capacity(8);

    for (idx, state) in sam.iter().enumerate() {
        let mut alts: Vec<String> = Vec::with_capacity(state.next.len() + 1);
        if idx > 0 {
            alts.push("\"\\\"\"".to_string());
        }
        let mut edges: Vec<(char, usize)> = state.next.iter().map(|(&c, &t)| (c, t)).collect();
        edges.sort_unstable_by_key(|&(c, _)| c);
        for (ch, target) in edges {
            json_escape_char(ch, &mut esc_buf);
            // `esc_buf` is valid UTF-8 representing JSON-string content; escape `\` and `"`
            // once more for the enclosing GBNF string literal.
            let esc_str = String::from_utf8_lossy(&esc_buf);
            let mut gbnf_lit = String::with_capacity(esc_str.len() + 4);
            gbnf_lit.push('"');
            for c in esc_str.chars() {
                match c {
                    '\\' => gbnf_lit.push_str("\\\\"),
                    '"' => gbnf_lit.push_str("\\\""),
                    other => gbnf_lit.push(other),
                }
            }
            gbnf_lit.push('"');
            alts.push(format!("{gbnf_lit} {prefix}{target}"));
        }
        if !alts.is_empty() {
            let _ = writeln!(out, "{prefix}{idx} ::= {}", alts.join(" | "));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::tok_env::build_tok_env;
    use std::path::PathBuf;
    use tokenizers::Tokenizer;

    fn load_clef_env() -> TokEnv {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json");
        let tok = Arc::new(Tokenizer::from_file(&path).expect("load clef tokenizer"));
        let vocab_size = tok.get_vocab_size(true);
        build_tok_env(tok, vocab_size, 0, &[0]).expect("build tok env")
    }

    fn accepts_text(masker: &SuffixAutomatonMasker, env: &TokEnv, text: &str) -> bool {
        let mut m = masker.clone();
        let tokens = env.tokenize(text);
        let vocab_size = env.tok_trie().vocab_size();
        let mut allowed = vec![false; vocab_size];
        for tok in tokens {
            m.fill_allowed_mask(&mut allowed).expect("fill mask");
            if !allowed[tok as usize] {
                return false;
            }
            if !m.accept_token(tok).expect("accept") {
                return false;
            }
        }
        m.is_accepting() && m.accept_token(0).expect("eos")
    }

    #[test]
    fn test_grounded_text_basic_escapes_unicode_and_normalized() {
        let env = load_clef_env();
        let source = "Invoice #42:\n  \"Acme\\Co\" — café 日本語";
        let normalized = "Invoice #42: \"Acme\\Co\" — café 日本語";
        let masker =
            SuffixAutomatonMasker::new(env.clone(), source, Some(normalized), true, vec![0])
                .expect("build masker");

        // Exact substrings (raw & normalized) + null
        assert!(accepts_text(&masker, &env, "null"));
        assert!(accepts_text(&masker, &env, r#""Invoice #42""#));
        assert!(accepts_text(
            &masker,
            &env,
            r#""Invoice #42:\n  \"Acme\\Co\"""#
        ));
        assert!(accepts_text(
            &masker,
            &env,
            r#""Invoice #42: \"Acme\\Co\"""#
        ));
        assert!(accepts_text(&masker, &env, r#""café 日本語""#));
        assert!(accepts_text(&masker, &env, r#""\"Acme\\Co\"""#));

        // Disallowed strings
        assert!(!accepts_text(&masker, &env, r#""""#)); // empty string disallowed when source non-empty
        assert!(!accepts_text(&masker, &env, r#""Invoice #99""#)); // hallucinated substring
        assert!(!accepts_text(&masker, &env, r#"Invoice #42"#)); // missing JSON quotes
        assert!(!accepts_text(&masker, &env, r#""Acme"extra"#)); // trailing bytes after quote
        assert!(!accepts_text(&masker, &env, "nullify")); // trailing bytes after null
    }

    #[test]
    fn test_grounded_text_non_nullable() {
        let env = load_clef_env();
        let masker = SuffixAutomatonMasker::new(env.clone(), "hello world", None, false, vec![0])
            .expect("build masker");
        assert!(!accepts_text(&masker, &env, "null"));
        assert!(accepts_text(&masker, &env, r#""hello""#));
        assert!(accepts_text(&masker, &env, r#""lo wo""#));
        assert!(!accepts_text(&masker, &env, r#""hello!""#));
    }

    #[test]
    fn test_split_escape_across_single_byte_tokens() {
        let env = load_clef_env();
        let trie = env.tok_trie();
        let masker_template = SuffixAutomatonMasker::new(env.clone(), "a\"b", None, false, vec![0])
            .expect("build masker");

        // Feed `"`, `a`, `\`, `"`, `b`, `"` as individual single-byte tokens!
        let bytes = br#""a\"b""#;
        let mut m = masker_template;
        let mut allowed = vec![false; trie.vocab_size()];
        for &b in bytes {
            let tid = trie.token_id(&[b]).expect("single byte token");
            m.fill_allowed_mask(&mut allowed).expect("fill mask");
            assert!(
                allowed[tid as usize],
                "expected byte {:?} (tok {tid}) to be allowed",
                b as char
            );
            assert!(m.accept_token(tid).expect("accept"));
        }
        assert!(m.is_terminated());
        assert!(m.accept_token(0).expect("eos"));
    }
}
