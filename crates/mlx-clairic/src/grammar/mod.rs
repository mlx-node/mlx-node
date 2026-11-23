//! Pure-Rust constrained decoding engine backed by `llguidance` + `toktrie`
//! and a direct byte-level suffix-automaton masker for grounded text.

pub mod ebnf_to_lark;
pub mod suffix_masker;
pub mod tok_env;

use std::sync::Arc;

use llguidance::api::{ParserLimits, TopLevelGrammar};
use llguidance::{Matcher, ParserFactory};
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokEnv};

use crate::errors::{ClairicError, Result};
use ebnf_to_lark::clairic_ebnf_to_lark;
use suffix_masker::SuffixAutomatonMasker;
use tok_env::build_tok_env;

/// Grammar specification supported by [`GrammarEngine`].
#[derive(Debug, Clone)]
pub enum GrammarSpec {
    /// JSON Schema constraint compiled by `llguidance`.
    JsonSchema(serde_json::Value),
    /// Native `llguidance` Lark grammar.
    Lark(String),
    /// Clairic / llama.cpp GBNF (`root ::= ...`) translated to Lark and compiled by `llguidance`.
    ClairicEbnf(String),
    /// Direct byte-level JSON-quoted substring suffix automaton over `source`
    /// (and optional `normalized_source`), optionally allowing `"null"`.
    GroundedText {
        source: String,
        normalized_source: Option<String>,
        nullable: bool,
    },
}

/// Shared grammar compilation engine bound to a model's tokenizer and stop tokens.
///
/// Construct once per model/tokenizer and reuse across requests via [`GrammarEngine::compile`].
#[derive(Clone)]
pub struct GrammarEngine {
    factory: Arc<ParserFactory>,
    tok_env: TokEnv,
    vocab_size: usize,
    eos_token_id: u32,
    stop_token_ids: Vec<u32>,
}

impl GrammarEngine {
    /// Create a [`GrammarEngine`] directly from a HuggingFace `tokenizers::Tokenizer`.
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        vocab_size: usize,
        eos_token_id: u32,
        stop_token_ids: Vec<u32>,
    ) -> Result<Self> {
        let mut stops = Vec::with_capacity(stop_token_ids.len() + 1);
        stops.push(eos_token_id);
        for &id in &stop_token_ids {
            if (id as usize) < vocab_size && !stops.contains(&id) {
                stops.push(id);
            }
        }

        let tok_env = build_tok_env(tokenizer, vocab_size, eos_token_id, &stops)?;
        let mut factory = ParserFactory::new_simple(&tok_env).map_err(|e| {
            ClairicError::from_reason(format!("Failed to build llguidance factory: {e}"))
        })?;
        factory.quiet();

        Ok(Self {
            factory: Arc::new(factory),
            tok_env,
            vocab_size,
            eos_token_id,
            stop_token_ids: stops,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    pub fn tok_env(&self) -> &TokEnv {
        &self.tok_env
    }

    /// Compile a [`GrammarSpec`] into an independent per-sequence [`GrammarMatcher`].
    pub fn compile(&self, spec: &GrammarSpec) -> Result<GrammarMatcher> {
        self.compile_with_limits(spec, None)
    }

    /// Compile a [`GrammarSpec`] with optional custom `llguidance::api::ParserLimits`.
    pub fn compile_with_limits(
        &self,
        spec: &GrammarSpec,
        limits: Option<ParserLimits>,
    ) -> Result<GrammarMatcher> {
        let backend = match spec {
            GrammarSpec::JsonSchema(schema) => {
                let grm = TopLevelGrammar::from_json_schema(schema.clone());
                MatcherBackend::LlGuidance(Box::new(self.build_llg_matcher(grm, limits)?))
            }
            GrammarSpec::Lark(lark) => {
                let grm = TopLevelGrammar::from_lark(lark.clone());
                MatcherBackend::LlGuidance(Box::new(self.build_llg_matcher(grm, limits)?))
            }
            GrammarSpec::ClairicEbnf(ebnf) => {
                let lark = clairic_ebnf_to_lark(ebnf)?;
                let grm = TopLevelGrammar::from_lark(lark);
                MatcherBackend::LlGuidance(Box::new(self.build_llg_matcher(grm, limits)?))
            }
            GrammarSpec::GroundedText {
                source,
                normalized_source,
                nullable,
            } => {
                let masker = SuffixAutomatonMasker::new(
                    self.tok_env.clone(),
                    source,
                    normalized_source.as_deref(),
                    *nullable,
                    self.stop_token_ids.clone(),
                )?;
                MatcherBackend::Suffix(masker)
            }
        };

        Ok(GrammarMatcher {
            backend,
            vocab_size: self.vocab_size,
            stop_token_ids: self.stop_token_ids.clone(),
        })
    }

    fn build_llg_matcher(
        &self,
        grammar: TopLevelGrammar,
        limits: Option<ParserLimits>,
    ) -> Result<Matcher> {
        let parser = match limits {
            Some(custom_limits) => self.factory.create_parser_from_init_ext(
                llguidance::api::GrammarInit::Serialized(grammar),
                llguidance::Logger::new(0, 0),
                toktrie::InferenceCapabilities::default(),
                custom_limits,
            ),
            None => self.factory.create_parser(grammar),
        }
        .map_err(|e| ClairicError::from_reason(format!("Grammar compilation failed: {e}")))?;

        let matcher = Matcher::new(Ok(parser));
        if let Some(err) = matcher.get_error() {
            return Err(ClairicError::from_reason(format!(
                "Grammar matcher initialization failed: {err}"
            )));
        }
        Ok(matcher)
    }
}

#[derive(Clone)]
enum MatcherBackend {
    LlGuidance(Box<Matcher>),
    Suffix(SuffixAutomatonMasker),
}

/// Stateful per-sequence constrained decoding matcher.
#[derive(Clone)]
pub struct GrammarMatcher {
    backend: MatcherBackend,
    vocab_size: usize,
    stop_token_ids: Vec<u32>,
}

impl GrammarMatcher {
    /// Vocabulary size this matcher produces masks for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns `true` when the sequence has reached a terminal state (no further
    /// grammar bytes are possible or EOS was already consumed).
    pub fn is_terminated(&self) -> bool {
        match &self.backend {
            MatcherBackend::LlGuidance(m) => m.is_stopped(),
            MatcherBackend::Suffix(s) => s.is_terminated(),
        }
    }

    /// Compute the compact `u32` bitmask (bit `i` = token `i` allowed).
    pub fn compute_allowed_bitmask(&mut self) -> Result<Vec<u32>> {
        let vob = self.compute_vob()?;
        Ok(Vec::from(vob))
    }

    /// Fill `allowed` with `true` for every token allowed at the current grammar state
    /// and `false` for disallowed tokens.
    pub fn fill_allowed_mask(&mut self, allowed: &mut [bool]) -> Result<()> {
        if allowed.len() < self.vocab_size {
            return Err(ClairicError::from_reason(format!(
                "allowed mask length {} is smaller than vocab_size {}",
                allowed.len(),
                self.vocab_size
            )));
        }
        match &mut self.backend {
            MatcherBackend::LlGuidance(m) => {
                let vob = m
                    .compute_mask_or_eos()
                    .map_err(|e| ClairicError::from_reason(format!("compute_mask failed: {e}")))?;
                allowed.fill(false);
                let vocab_size = self.vocab_size;
                vob.iter_set_entries(|idx| {
                    if idx < vocab_size {
                        allowed[idx] = true;
                    }
                });
                Ok(())
            }
            MatcherBackend::Suffix(s) => s.fill_allowed_mask(allowed),
        }
    }

    fn compute_vob(&mut self) -> Result<SimpleVob> {
        match &mut self.backend {
            MatcherBackend::LlGuidance(m) => m
                .compute_mask_or_eos()
                .map_err(|e| ClairicError::from_reason(format!("compute_mask failed: {e}"))),
            MatcherBackend::Suffix(s) => Ok(s.compute_mask()),
        }
    }

    /// Advance the matcher with a sampled token.
    ///
    /// Returns `Ok(true)` if the token was valid and accepted, `Ok(false)` if
    /// disallowed (leaving parser state intact), or `Err` on internal failure.
    pub fn accept_token(&mut self, token_id: u32) -> Result<bool> {
        if (token_id as usize) >= self.vocab_size {
            return Ok(false);
        }
        match &mut self.backend {
            MatcherBackend::LlGuidance(m) => {
                if m.is_stopped() {
                    if m.stop_reason().is_ok() && self.stop_token_ids.contains(&token_id) {
                        return Ok(true);
                    }
                    return Ok(false);
                }
                let consumed = m
                    .try_consume_tokens(&[token_id])
                    .map_err(|e| ClairicError::from_reason(format!("accept_token failed: {e}")))?;
                Ok(consumed == 1)
            }
            MatcherBackend::Suffix(s) => s.accept_token(token_id),
        }
    }
}

/// Compute allowed-token boolean masks for a batch of matchers.
///
/// Uses scoped multi-threading when `matchers.len() > 1` so batch mask computation
/// scales across CPU cores without requiring an extra thread-pool dependency.
pub fn fill_batch_allowed_masks(
    matchers: &mut [GrammarMatcher],
    vocab_size: usize,
) -> Result<Vec<Vec<bool>>> {
    if matchers.is_empty() {
        return Ok(Vec::new());
    }
    if matchers.len() == 1 {
        let mut mask = vec![false; vocab_size];
        matchers[0].fill_allowed_mask(&mut mask)?;
        return Ok(vec![mask]);
    }

    let mut masks: Vec<Vec<bool>> = (0..matchers.len())
        .map(|_| vec![false; vocab_size])
        .collect();

    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(matchers.len());

    if max_threads <= 1 {
        for (matcher, mask) in matchers.iter_mut().zip(masks.iter_mut()) {
            matcher.fill_allowed_mask(mask)?;
        }
        return Ok(masks);
    }

    let chunk_size = matchers.len().div_ceil(max_threads);
    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(max_threads);
        for (m_chunk, mask_chunk) in matchers
            .chunks_mut(chunk_size)
            .zip(masks.chunks_mut(chunk_size))
        {
            handles.push(scope.spawn(move || -> Result<()> {
                for (matcher, mask) in m_chunk.iter_mut().zip(mask_chunk.iter_mut()) {
                    matcher.fill_allowed_mask(mask)?;
                }
                Ok(())
            }));
        }
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(r) => r,
                Err(_) => Err(ClairicError::from_reason(
                    "Worker thread panicked during fill_batch_allowed_masks",
                )),
            })
            .collect()
    });

    for res in results {
        res?;
    }

    Ok(masks)
}

/// Compute compact `u32` allowed-token bitmasks for active rows in a batch.
///
/// Rows with `active[row] == false` are skipped and receive an empty `Vec`,
/// avoiding wasted grammar work for already-finished sequences.
pub fn fill_batch_allowed_bitmasks(
    matchers: &mut [GrammarMatcher],
    active: &[bool],
) -> Result<Vec<Vec<u32>>> {
    if matchers.is_empty() {
        return Ok(Vec::new());
    }
    if active.len() != matchers.len() {
        return Err(ClairicError::from_reason(format!(
            "active length {} != matchers length {}",
            active.len(),
            matchers.len()
        )));
    }
    let active_count = active.iter().filter(|&&a| a).count();
    if active_count == 0 {
        return Ok(vec![Vec::new(); matchers.len()]);
    }
    if active_count == 1 {
        let mut out = vec![Vec::new(); matchers.len()];
        for (i, (&is_active, matcher)) in active.iter().zip(matchers.iter_mut()).enumerate() {
            if is_active {
                out[i] = matcher.compute_allowed_bitmask()?;
            }
        }
        return Ok(out);
    }

    let mut out: Vec<Vec<u32>> = vec![Vec::new(); matchers.len()];
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(active_count);

    if max_threads <= 1 {
        for ((&is_active, matcher), dst) in
            active.iter().zip(matchers.iter_mut()).zip(out.iter_mut())
        {
            if is_active {
                *dst = matcher.compute_allowed_bitmask()?;
            }
        }
        return Ok(out);
    }

    let chunk_size = matchers.len().div_ceil(max_threads);
    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(max_threads);
        for ((m_chunk, a_chunk), dst_chunk) in matchers
            .chunks_mut(chunk_size)
            .zip(active.chunks(chunk_size))
            .zip(out.chunks_mut(chunk_size))
        {
            handles.push(scope.spawn(move || -> Result<()> {
                for ((matcher, &is_active), dst) in m_chunk
                    .iter_mut()
                    .zip(a_chunk.iter())
                    .zip(dst_chunk.iter_mut())
                {
                    if is_active {
                        *dst = matcher.compute_allowed_bitmask()?;
                    }
                }
                Ok(())
            }));
        }
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(r) => r,
                Err(_) => Err(ClairicError::from_reason(
                    "Worker thread panicked during fill_batch_allowed_bitmasks",
                )),
            })
            .collect()
    });

    for res in results {
        res?;
    }

    Ok(out)
}

/// Apply compact `u32` `bitmask` directly over `logits` without allocating a
/// full-vocab copy or `log_probs` buffer.
///
/// Iterates only set bits in `bitmask`, returning
/// `(argmax_token_id, argmax_logprob, scored_tokens)` where `scored_tokens`
/// has one `(logprob, token_id)` entry per ID in `score_token_ids`.
/// Produces bit-identical results to [`mask_and_log_softmax_row`].
pub fn mask_and_score_row_bitmask(
    logits: &[f32],
    bitmask: &[u32],
    score_token_ids: &[u32],
) -> (u32, f32, Vec<(f32, u32)>) {
    let n = logits.len();
    let max_words = bitmask.len().min(n.div_ceil(32));

    let mut max_val = f32::NEG_INFINITY;
    let mut best_id: u32 = 0;

    for (word_idx, &word) in bitmask[..max_words].iter().enumerate() {
        let mut w = word;
        let base = word_idx * 32;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            w &= w - 1;
            let idx = base + tz;
            if idx >= n {
                break;
            }
            let x = logits[idx];
            if x.is_finite() && x > max_val {
                max_val = x;
                best_id = idx as u32;
            }
        }
    }

    if !max_val.is_finite() {
        let scored = score_token_ids
            .iter()
            .map(|&tok| (f32::NEG_INFINITY, tok))
            .collect();
        return (0, f32::NEG_INFINITY, scored);
    }

    let max_f64 = f64::from(max_val);
    let mut sum_exp = 0.0_f64;
    for (word_idx, &word) in bitmask[..max_words].iter().enumerate() {
        let mut w = word;
        let base = word_idx * 32;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            w &= w - 1;
            let idx = base + tz;
            if idx >= n {
                break;
            }
            let x = logits[idx];
            if x.is_finite() {
                sum_exp += (f64::from(x) - max_f64).exp();
            }
        }
    }

    let log_sum_exp = max_f64 + sum_exp.ln();
    let best_lp = (max_f64 - log_sum_exp) as f32;

    let scored = if score_token_ids.is_empty() {
        Vec::new()
    } else {
        score_token_ids
            .iter()
            .map(|&tok| {
                let idx = tok as usize;
                let allowed = idx < n
                    && (idx / 32) < bitmask.len()
                    && ((bitmask[idx / 32] >> (idx & 31)) & 1) != 0;
                let lp = if allowed && logits[idx].is_finite() {
                    (f64::from(logits[idx]) - log_sum_exp) as f32
                } else {
                    f32::NEG_INFINITY
                };
                (lp, tok)
            })
            .collect()
    };

    (best_id, best_lp, scored)
}

/// In-place logit masking: sets disallowed positions to `-inf`.
pub fn mask_logits(logits: &mut [f32], allowed: &[bool]) {
    let common = logits.len().min(allowed.len());
    for i in 0..common {
        if !allowed[i] {
            logits[i] = f32::NEG_INFINITY;
        }
    }
    for logit in &mut logits[common..] {
        *logit = f32::NEG_INFINITY;
    }
}

/// Numerically stable `log_softmax` over a logit row (uses `f64` reduction
/// for accuracy across 150k-250k vocabularies). Non-finite inputs map to `-inf`.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    for &x in logits {
        if x.is_finite() && x > max_val {
            max_val = x;
        }
    }
    if !max_val.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }

    let max_f64 = f64::from(max_val);
    let mut sum_exp = 0.0_f64;
    for &x in logits {
        if x.is_finite() {
            sum_exp += (f64::from(x) - max_f64).exp();
        }
    }
    let log_sum_exp = max_f64 + sum_exp.ln();

    logits
        .iter()
        .map(|&x| {
            if x.is_finite() {
                (f64::from(x) - log_sum_exp) as f32
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// Apply `allowed` mask to `logits`, compute numerically stable `log_softmax`,
/// and return `(argmax_token_id, argmax_logprob, full_log_probs)`.
///
/// Argmax is taken over raw masked logits (lowest index on exact ties),
/// matching upstream `mlx_engine.py:804` (`masked.argmax(axis=-1)`) and
/// [`mask_and_score_row_bitmask`].
///
/// Callers can index `full_log_probs[score_id as usize]` directly to read
/// step-0 constrained logprobs for arbitrary candidate token IDs.
pub fn mask_and_log_softmax_row(logits: &[f32], allowed: &[bool]) -> (u32, f32, Vec<f32>) {
    let mut masked = logits.to_vec();
    mask_logits(&mut masked, allowed);
    let log_probs = log_softmax(&masked);

    let mut best_id: u32 = 0;
    let mut best_raw = f32::NEG_INFINITY;
    for (idx, &x) in masked.iter().enumerate() {
        if x.is_finite() && x > best_raw {
            best_raw = x;
            best_id = idx as u32;
        }
    }
    let best_lp = if best_raw.is_finite() {
        log_probs
            .get(best_id as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY)
    } else {
        f32::NEG_INFINITY
    };

    (best_id, best_lp, log_probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Instant;

    fn load_clef_engine() -> GrammarEngine {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json");
        let tok = Arc::new(Tokenizer::from_file(&path).expect("load clef tokenizer.json"));
        let vocab_size = tok.get_vocab_size(true);
        GrammarEngine::new(tok, vocab_size, 0, vec![0]).expect("build GrammarEngine")
    }

    fn run_sequence(engine: &GrammarEngine, spec: &GrammarSpec, text: &str) -> bool {
        let mut matcher = engine.compile(spec).expect("compile grammar");
        let tokens = engine.tok_env().tokenize(text);
        let mut allowed = vec![false; engine.vocab_size()];

        for tok in tokens {
            if matcher.is_terminated() {
                return false;
            }
            matcher.fill_allowed_mask(&mut allowed).expect("fill mask");
            if !allowed[tok as usize] {
                return false;
            }
            if !matcher.accept_token(tok).expect("accept_token") {
                return false;
            }
        }

        // Verify EOS is allowed at the end of the sequence.
        matcher.fill_allowed_mask(&mut allowed).expect("final mask");
        if !allowed[engine.eos_token_id() as usize] {
            return false;
        }
        matcher
            .accept_token(engine.eos_token_id())
            .expect("accept eos")
            && matcher.is_terminated()
    }

    #[test]
    fn test_json_schema_constraints() {
        let engine = load_clef_engine();

        // 1. boolean
        let bool_spec = GrammarSpec::JsonSchema(json!({ "type": "boolean" }));
        assert!(run_sequence(&engine, &bool_spec, "true"));
        assert!(run_sequence(&engine, &bool_spec, "false"));
        assert!(!run_sequence(&engine, &bool_spec, "null"));
        assert!(!run_sequence(&engine, &bool_spec, "\"true\""));

        // 2. number
        let num_spec = GrammarSpec::JsonSchema(json!({ "type": "number" }));
        assert!(run_sequence(&engine, &num_spec, "42"));
        assert!(run_sequence(&engine, &num_spec, "-12.5"));
        assert!(!run_sequence(&engine, &num_spec, "\"42\""));
        assert!(!run_sequence(&engine, &num_spec, "abc"));

        // 3. string
        let str_spec = GrammarSpec::JsonSchema(json!({ "type": "string" }));
        assert!(run_sequence(&engine, &str_spec, r#""hello world""#));
        assert!(!run_sequence(&engine, &str_spec, "123"));

        // 4. enum
        let enum_spec = GrammarSpec::JsonSchema(json!({
            "enum": ["low", "medium", "high"]
        }));
        assert!(run_sequence(&engine, &enum_spec, r#""medium""#));
        assert!(!run_sequence(&engine, &enum_spec, r#""critical""#));

        // 5. object
        let obj_spec = GrammarSpec::JsonSchema(json!({
            "type": "object",
            "properties": {
                "ok": { "type": "boolean" },
                "score": { "type": "number" }
            },
            "required": ["ok", "score"],
            "additionalProperties": false
        }));
        assert!(run_sequence(
            &engine,
            &obj_spec,
            r#"{"ok":true,"score":0.95}"#
        ));
        assert!(!run_sequence(
            &engine,
            &obj_spec,
            r#"{"ok":"yes","score":1}"#
        ));

        // 6. anyOf (nullable string)
        let any_of_spec = GrammarSpec::JsonSchema(json!({
            "anyOf": [{ "type": "string" }, { "type": "null" }]
        }));
        assert!(run_sequence(&engine, &any_of_spec, r#""extracted""#));
        assert!(run_sequence(&engine, &any_of_spec, "null"));
        assert!(!run_sequence(&engine, &any_of_spec, "true"));

        // 7. array of {label: string}
        let arr_spec = GrammarSpec::JsonSchema(json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": { "label": { "type": "string" } },
                "required": ["label"],
                "additionalProperties": false
            }
        }));
        assert!(run_sequence(
            &engine,
            &arr_spec,
            r#"[{"label":"alpha"},{"label":"beta"}]"#
        ));
        assert!(!run_sequence(&engine, &arr_spec, r#"[{"label":123}]"#));
    }

    #[test]
    fn test_all_four_clairic_ebnf_templates() {
        let engine = load_clef_engine();

        // Template 1: Number literal alternation
        let num_ebnf = GrammarSpec::ClairicEbnf(r#"root ::= "12.5" | "100" | "null""#.to_string());
        assert!(run_sequence(&engine, &num_ebnf, "12.5"));
        assert!(run_sequence(&engine, &num_ebnf, "100"));
        assert!(run_sequence(&engine, &num_ebnf, "null"));
        assert!(!run_sequence(&engine, &num_ebnf, "99"));

        // Template 2: ISO date alternation
        let date_ebnf =
            GrammarSpec::ClairicEbnf(r#"root ::= "\"2024-03-15\"" | "null""#.to_string());
        assert!(run_sequence(&engine, &date_ebnf, r#""2024-03-15""#));
        assert!(run_sequence(&engine, &date_ebnf, "null"));
        assert!(!run_sequence(&engine, &date_ebnf, r#""2024-03-16""#));

        // Template 3: Boolean-or-null
        let bool_ebnf =
            GrammarSpec::ClairicEbnf(r#"root ::= "true" | "false" | "null""#.to_string());
        assert!(run_sequence(&engine, &bool_ebnf, "true"));
        assert!(run_sequence(&engine, &bool_ebnf, "false"));
        assert!(run_sequence(&engine, &bool_ebnf, "null"));
        assert!(!run_sequence(&engine, &bool_ebnf, "maybe"));

        // Template 4: Suffix-automaton DAG
        let dag_ebnf = GrammarSpec::ClairicEbnf(suffix_masker::build_clairic_ebnf_for_text(
            "cat\ndog",
            Some("cat dog"),
            true,
        ));
        assert!(run_sequence(&engine, &dag_ebnf, "null"));
        assert!(run_sequence(&engine, &dag_ebnf, r#""cat""#));
        assert!(run_sequence(&engine, &dag_ebnf, r#""cat\ndog""#));
        assert!(run_sequence(&engine, &dag_ebnf, r#""cat dog""#));
        assert!(!run_sequence(&engine, &dag_ebnf, r#""bird""#));
    }

    #[test]
    fn test_batch_masks_and_log_softmax_row() {
        let engine = load_clef_engine();
        let mut matchers = vec![
            engine
                .compile(&GrammarSpec::ClairicEbnf(
                    r#"root ::= "true" | "false""#.to_string(),
                ))
                .expect("m0"),
            engine
                .compile(&GrammarSpec::GroundedText {
                    source: "hello".to_string(),
                    normalized_source: None,
                    nullable: true,
                })
                .expect("m1"),
            engine
                .compile(&GrammarSpec::JsonSchema(json!({ "type": "number" })))
                .expect("m2"),
        ];

        let masks =
            fill_batch_allowed_masks(&mut matchers, engine.vocab_size()).expect("batch masks");
        assert_eq!(masks.len(), 3);
        for mask in &masks {
            assert_eq!(mask.len(), engine.vocab_size());
            assert!(mask.iter().any(|&b| b));
        }

        // Verify mask_and_log_softmax_row
        let logits = vec![1.0_f32, 3.0, 2.0, 10.0];
        let allowed = vec![true, true, false, false];
        let (best_id, best_lp, lps) = mask_and_log_softmax_row(&logits, &allowed);
        assert_eq!(best_id, 1);
        assert!(lps[2].is_infinite() && lps[2].is_sign_negative());
        assert!(lps[3].is_infinite() && lps[3].is_sign_negative());
        // exp(lps[0]) + exp(lps[1]) == 1.0
        let prob_sum = lps[0].exp() + lps[1].exp();
        assert!((prob_sum - 1.0).abs() < 1e-6);
        assert!((best_lp - lps[1]).abs() < 1e-6);
    }

    #[test]
    fn test_4096_char_grounded_text_and_benchmark_vs_llguidance_lark() {
        let engine = load_clef_engine();

        // Build a realistic 4,096-character document with newlines, quotes, and numbers.
        let paragraph = "Section 1: The quick brown fox jumps over 42 lazy dogs.\n  \"Quoted text\" with extra   spaces, 2024-03-15, and $1,234.56.\n";
        let mut raw_source = String::with_capacity(4096 + paragraph.len());
        let mut idx = 0usize;
        while raw_source.len() < 4096 {
            raw_source.push_str(&format!("[{idx:03}] {paragraph}"));
            idx += 1;
        }
        raw_source.truncate(4096);
        let norm_source: String = raw_source.split_whitespace().collect::<Vec<_>>().join(" ");

        // 1. Direct SuffixAutomatonMasker timing & accuracy
        let spec = GrammarSpec::GroundedText {
            source: raw_source.clone(),
            normalized_source: Some(norm_source.clone()),
            nullable: true,
        };
        let t_compile = Instant::now();
        let mut direct_matcher = engine.compile(&spec).expect("compile 4096 GroundedText");
        let direct_compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;

        let target = r#""The quick brown fox jumps over 42 lazy dogs.""#;
        let target_tokens = engine.tok_env().tokenize(target);
        let mut allowed = vec![false; engine.vocab_size()];
        let t_steps = Instant::now();
        for &tok in &target_tokens {
            direct_matcher
                .fill_allowed_mask(&mut allowed)
                .expect("direct mask");
            assert!(allowed[tok as usize]);
            assert!(direct_matcher.accept_token(tok).expect("direct accept"));
        }
        let direct_step_us =
            (t_steps.elapsed().as_secs_f64() * 1_000_000.0) / (target_tokens.len() as f64);
        assert!(direct_matcher.is_terminated());

        // Verify normalized substring and null also accepted, hallucination rejected
        assert!(run_sequence(
            &engine,
            &spec,
            r#""\"Quoted text\" with extra spaces, 2024-03-15""#
        ));
        assert!(run_sequence(&engine, &spec, "null"));
        assert!(!run_sequence(
            &engine,
            &spec,
            r#""The quick brown unicorn""#
        ));

        // 2. Compare llguidance Lark on the equivalent 4,096-char suffix-automaton DAG.
        //    Note: llguidance's Lark compiler recurses over rule DAG depth, which overflows
        //    the default 2 MiB test thread stack on ~12k-rule DAGs; run on a 64 MiB stack
        //    to measure actual compile/step latency without crashing the test harness.
        let ebnf_4k =
            suffix_masker::build_clairic_ebnf_for_text(&raw_source, Some(&norm_source), true);
        let rule_count = ebnf_4k.lines().count();
        let engine_clone = engine.clone();
        let target_tokens_clone = target_tokens.clone();
        let llg_summary = std::thread::Builder::new()
            .name("llg-4k-bench".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let t_llg = Instant::now();
                let llg_res = engine_clone.compile(&GrammarSpec::ClairicEbnf(ebnf_4k));
                let llg_compile_ms = t_llg.elapsed().as_secs_f64() * 1000.0;
                match llg_res {
                    Ok(mut m) => {
                        let mut allowed_llg = vec![false; engine_clone.vocab_size()];
                        let t_m = Instant::now();
                        let mut ok = true;
                        for &tok in &target_tokens_clone {
                            if m.fill_allowed_mask(&mut allowed_llg).is_err()
                                || !allowed_llg[tok as usize]
                                || !m.accept_token(tok).unwrap_or(false)
                            {
                                ok = false;
                                break;
                            }
                        }
                        let step_us = (t_m.elapsed().as_secs_f64() * 1_000_000.0)
                            / (target_tokens_clone.len() as f64);
                        format!(
                            "OK (64MiB stack) compile={llg_compile_ms:.2}ms step={step_us:.1}us matched={ok}"
                        )
                    }
                    Err(e) => format!("ERR (64MiB stack) compile={llg_compile_ms:.2}ms ({e})"),
                }
            })
            .expect("spawn 64MiB bench thread")
            .join()
            .expect("join 64MiB bench thread");

        eprintln!(
            "[4096-char GroundedText benchmark] rules={rule_count} | \
             Direct SuffixAutomatonMasker: compile={direct_compile_ms:.2}ms step={direct_step_us:.1}us | \
             llguidance Lark: {llg_summary}"
        );
    }

    #[test]
    fn bitmask_scoring_matches_dense_mask_and_log_softmax_row_bit_for_bit() {
        let logits: Vec<f32> = (0..130)
            .map(|i| match i {
                7 => f32::NAN,
                19 => f32::NEG_INFINITY,
                33 | 64 => 4.25, // deliberate tie: lower index 33 must win
                97 => f32::INFINITY,
                _ => ((i as f32) * 0.37).sin() * 3.0 - 0.5,
            })
            .collect();

        let mut allowed = vec![false; logits.len()];
        let mut bitmask = vec![0u32; logits.len().div_ceil(32)];
        for idx in [0, 3, 7, 19, 31, 32, 33, 64, 65, 97, 127, 129] {
            allowed[idx] = true;
            bitmask[idx / 32] |= 1u32 << (idx & 31);
        }

        let score_ids = [0u32, 1, 7, 33, 64, 129, 999];
        let (dense_id, dense_lp, dense_lps) = mask_and_log_softmax_row(&logits, &allowed);
        let (bit_id, bit_lp, bit_scored) =
            mask_and_score_row_bitmask(&logits, &bitmask, &score_ids);

        assert_eq!(bit_id, dense_id);
        assert_eq!(bit_id, 33);
        assert_eq!(bit_lp.to_bits(), dense_lp.to_bits());
        assert_eq!(bit_scored.len(), score_ids.len());
        for (lp, tok) in bit_scored {
            let expected = dense_lps
                .get(tok as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            assert_eq!(lp.to_bits(), expected.to_bits(), "mismatch for token {tok}");
        }

        // Empty score_ids on step > 0 returns empty Vec while preserving argmax & logprob.
        let (bit_id2, bit_lp2, empty_scored) = mask_and_score_row_bitmask(&logits, &bitmask, &[]);
        assert_eq!(bit_id2, dense_id);
        assert_eq!(bit_lp2.to_bits(), dense_lp.to_bits());
        assert!(empty_scored.is_empty());

        // All-masked bitmask returns (0, -inf, all -inf).
        let (zero_id, zero_lp, zero_scored) =
            mask_and_score_row_bitmask(&logits, &[0u32; 5], &[3, 33]);
        assert_eq!(zero_id, 0);
        assert!(zero_lp.is_infinite() && zero_lp.is_sign_negative());
        assert!(
            zero_scored
                .iter()
                .all(|&(lp, _)| lp.is_infinite() && lp.is_sign_negative())
        );

        // Near-tie where both FP32 log-probabilities round to the same value:
        // both dense and sparse scoring must select token 1 (higher raw logit),
        // matching upstream `masked.argmax(axis=-1)`.
        let near_tie_logits = [0.0_f32, 1e-8_f32];
        let near_tie_allowed = [true, true];
        let near_tie_bits = [0b11_u32];
        let (nt_dense_id, nt_dense_lp, _) =
            mask_and_log_softmax_row(&near_tie_logits, &near_tie_allowed);
        let (nt_bit_id, nt_bit_lp, nt_scored) =
            mask_and_score_row_bitmask(&near_tie_logits, &near_tie_bits, &[0, 1]);
        assert_eq!(nt_dense_id, 1);
        assert_eq!(nt_bit_id, 1);
        assert_eq!(nt_bit_lp.to_bits(), nt_dense_lp.to_bits());
        assert_eq!(nt_scored.len(), 2);

        // fill_batch_allowed_bitmasks skips inactive rows.
        let engine = load_clef_engine();
        let spec = GrammarSpec::JsonSchema(json!({"type": ["boolean", "null"]}));
        let mut matchers = vec![
            engine.compile(&spec).expect("m0"),
            engine.compile(&spec).expect("m1"),
            engine.compile(&spec).expect("m2"),
        ];
        let batch =
            fill_batch_allowed_bitmasks(&mut matchers, &[true, false, true]).expect("batch masks");
        assert_eq!(batch.len(), 3);
        assert!(!batch[0].is_empty());
        assert!(batch[1].is_empty(), "inactive row must be skipped");
        assert_eq!(batch[0], batch[2]);
    }
}
