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
    // the S14a traces written before S14a-2 carry the family's own kind
    let kind_ok = head["kind"] == "qwen4_exp_expert_route_trace"
        || head["kind"] == "deepseek_v41_expert_route_trace";
    if !kind_ok || field(&head, "schema_version")? != 1 {
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
    /// Of `reads`, those of decode steps.
    pub decode_reads: u64,
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
    replay_pinned(cache, experts, steps, &[])
}

/// `replay` with whole layers resident: every expert of a layer with
/// `pinned[layer]` is a hit and never enters `cache` (D-42 layer-pinned and
/// hybrid arms; `&[]` pins nothing).
pub fn replay_pinned(
    cache: &mut Cache,
    experts: usize,
    steps: &[Step],
    pinned: &[bool],
) -> Replayed {
    run(cache, experts, steps, pinned, None)
}

/// Expert reads per layer of a staging-only replay (cache 0): `(all,
/// decode)`. The layer-pinned arm pins the layers with the most decode reads.
pub fn reads_per_layer(t: &RouteTrace) -> Vec<(u64, u64)> {
    let mut out = vec![(0, 0); t.layers];
    let mut c = Cache::lru(0, t.layers, t.experts);
    run(&mut c, t.experts, &t.steps, &[], Some(&mut out));
    out
}

fn run(
    cache: &mut Cache,
    experts: usize,
    steps: &[Step],
    pinned: &[bool],
    mut per_layer: Option<&mut Vec<(u64, u64)>>,
) -> Replayed {
    let mut tags: Vec<Option<(usize, u32)>> = vec![None; experts];
    let mut out = Replayed::default();
    for s in steps {
        let mut want = s.ids.clone();
        want.sort_unstable();
        want.dedup();
        let decode = decode_step(s.rows);
        let n = want.len() as u64;
        out.uses += n;
        if decode {
            out.decode_uses += n;
        }
        if pinned.get(s.layer).copied().unwrap_or(false) {
            out.hits += n;
            if decode {
                out.decode_hits += n;
            }
            continue;
        }
        let (epoch, src) = cache.plan(s.layer, &want, decode);
        let mut reads = 0;
        for (&e, x) in want.iter().zip(&src) {
            match *x {
                Source::Hit(_) => out.hits += 1,
                Source::Victim(c) => {
                    cache.commit(c, s.layer, e);
                    reads += 1;
                }
                Source::Staging => {
                    if tags[e as usize] != Some((s.layer, e)) {
                        tags[e as usize] = Some((s.layer, e));
                        reads += 1;
                    }
                }
            }
        }
        out.reads += reads;
        if decode {
            out.decode_reads += reads;
            out.decode_hits += src.iter().filter(|x| matches!(x, Source::Hit(_))).count() as u64;
        }
        if let Some(p) = per_layer.as_deref_mut() {
            p[s.layer].0 += reads;
            if decode {
                p[s.layer].1 += reads;
            }
        }
        cache.complete(epoch);
    }
    out
}

/// `replay` of a whole trace on a fresh LRU cache of `slots` slots.
pub fn replay_lru(t: &RouteTrace, slots: usize) -> Replayed {
    let mut c = Cache::lru(slots, t.layers, t.experts);
    replay(&mut c, t.experts, &t.steps)
}

/// `replay` of a whole trace on a fresh LFU-decay cache (`half_life` in
/// expert uses).
pub fn replay_lfu(t: &RouteTrace, slots: usize, half_life: f64) -> Replayed {
    let mut c = Cache::lfu_decay(slots, t.layers, t.experts, half_life);
    replay(&mut c, t.experts, &t.steps)
}

/// D-42 hybrid: `pinned` layers fully resident (each costs `experts` slots),
/// the remaining slots a global cache (LFU-decay with `half_life`, else LRU).
/// `None` when the pinned layers alone do not fit in `slots`.
pub fn replay_hybrid(
    t: &RouteTrace,
    slots: usize,
    pinned: &[usize],
    half_life: Option<f64>,
) -> Option<Replayed> {
    let rest = slots.checked_sub(pinned.len() * t.experts)?;
    let mut mask = vec![false; t.layers];
    for &l in pinned {
        *mask.get_mut(l)? = true;
    }
    let mut c = match half_life {
        Some(h) => Cache::lfu_decay(rest, t.layers, t.experts, h),
        None => Cache::lru(rest, t.layers, t.experts),
    };
    Some(replay_pinned(&mut c, t.experts, &t.steps, &mask))
}

#[cfg(test)]
mod tests;
