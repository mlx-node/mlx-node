//! Offline replay of route traces (`MLX_DSV41_ROUTE_TRACE` JSONL, schema
//! v1 of `qwen4_exp/route_replay.rs`) through the same [`Cache::plan`] and
//! staging-tag rules the bank uses, so a replayed hit count is the live one
//! for the same routes and cache size.

use std::io::BufRead;

use napi::{Error, Result};
use serde_json::Value;

use super::cache::{Cache, Source, decode_step};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("moe_stream replay: {msg}"))
}

/// One layer step: the expert ids of every row (`rows * top_k`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub layer: usize,
    pub rows: usize,
    pub ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteTrace {
    pub layers: usize,
    pub experts: usize,
    pub top_k: usize,
    pub prompt_tokens: usize,
    pub steps: Vec<Step>,
}

fn field(v: &Value, k: &str) -> Result<u64> {
    let f = &v[k];
    f.as_u64()
        .or_else(|| f.as_str().and_then(|s| s.parse().ok()))
        .ok_or_else(|| bad(format!("field '{k}' missing or not an integer")))
}

/// Reads a trace: a header line, then one `routes` event per layer step.
pub fn read_trace(r: impl BufRead) -> Result<RouteTrace> {
    let mut lines = r.lines();
    let head: Value = serde_json::from_str(
        &lines
            .next()
            .ok_or_else(|| bad("empty trace"))?
            .map_err(|e| bad(e.to_string()))?,
    )
    .map_err(|e| bad(format!("header: {e}")))?;
    if head["kind"] != "qwen4_exp_expert_route_trace" || field(&head, "schema_version")? != 1 {
        return Err(bad(format!("not a schema-1 route trace: {head}")));
    }
    let mut t = RouteTrace {
        layers: field(&head, "layer_count")? as usize,
        experts: field(&head, "expert_count")? as usize,
        top_k: field(&head, "top_k")? as usize,
        prompt_tokens: field(&head, "prompt_tokens")? as usize,
        steps: Vec::new(),
    };
    for (i, line) in lines.enumerate() {
        let line = line.map_err(|e| bad(e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value =
            serde_json::from_str(&line).map_err(|e| bad(format!("line {}: {e}", i + 2)))?;
        if v["kind"] != "routes" {
            continue;
        }
        let ids: Vec<u32> = v["expert_ids"]
            .as_array()
            .ok_or_else(|| bad(format!("line {}: no expert_ids", i + 2)))?
            .iter()
            .map(|x| x.as_u64().map(|x| x as u32))
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("line {}: bad expert id", i + 2)))?;
        let s = Step {
            layer: field(&v, "layer")? as usize,
            rows: field(&v, "rows")? as usize,
            ids,
        };
        if s.layer >= t.layers
            || s.ids.len() != s.rows * t.top_k
            || s.ids.iter().any(|&e| e as usize >= t.experts)
        {
            return Err(bad(format!("line {}: step out of range", i + 2)));
        }
        t.steps.push(s);
    }
    Ok(t)
}

/// Counts of one replay. All experts have the same size, so use and byte
/// hit rates are equal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Replayed {
    pub uses: u64,
    pub hits: u64,
    pub decode_uses: u64,
    pub decode_hits: u64,
    /// Experts read from SSD (staging and victims).
    pub reads: u64,
}

impl Replayed {
    pub fn h(&self) -> f64 {
        self.hits as f64 / self.uses.max(1) as f64
    }

    pub fn decode_h(&self) -> f64 {
        self.decode_hits as f64 / self.decode_uses.max(1) as f64
    }
}

/// Replays `steps` (a run from a fresh bank) on `cache`, with staging tags
/// as the bank keeps them.
pub fn replay(cache: &mut Cache, experts: usize, steps: &[Step]) -> Replayed {
    let mut tags: Vec<Option<(usize, u32)>> = vec![None; experts];
    let mut out = Replayed::default();
    for s in steps {
        let mut want = s.ids.clone();
        want.sort_unstable();
        want.dedup();
        let decode = decode_step(s.rows);
        let (epoch, src) = cache.plan(s.layer, &want, decode);
        for (&e, x) in want.iter().zip(&src) {
            match *x {
                Source::Hit(_) => out.hits += 1,
                Source::Victim(c) => {
                    cache.commit(c, s.layer, e);
                    out.reads += 1;
                }
                Source::Staging => {
                    if tags[e as usize] != Some((s.layer, e)) {
                        tags[e as usize] = Some((s.layer, e));
                        out.reads += 1;
                    }
                }
            }
        }
        if decode {
            out.decode_uses += want.len() as u64;
            out.decode_hits += src.iter().filter(|x| matches!(x, Source::Hit(_))).count() as u64;
        }
        out.uses += want.len() as u64;
        cache.complete(epoch);
    }
    out
}

/// `replay` of a whole trace on a fresh LRU cache of `slots` slots.
pub fn replay_lru(t: &RouteTrace, slots: usize) -> Replayed {
    let mut c = Cache::lru(slots, t.layers, t.experts);
    replay(&mut c, t.experts, &t.steps)
}

#[cfg(test)]
mod tests;
