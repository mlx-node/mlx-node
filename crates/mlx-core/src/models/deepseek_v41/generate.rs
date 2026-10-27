//! Greedy generation from token ids on the store, as one turn
//! (`Store::begin_turn` .. `end_turn`). Each call's routes go to the route
//! trace, and a stats line per generated token to the stats file (see
//! `stats`, `trace`).

use napi::{Error, Result};
use serde_json::Value;

use super::model::{Session, Trace};
use super::stats::{self, Counters};
use super::store::Store;
use super::trace::RouteTrace;
use crate::array::MxArray;

/// One finished call, handed to the caller's hook.
pub struct Call<'a> {
    pub index: usize,
    pub start: usize,
    pub ids: &'a [u32],
    /// f32 logits `[ids.len(), vocab]`.
    pub logits: &'a MxArray,
    pub trace: &'a Trace,
    /// Greedy pick from the last row (lowest id on ties, as `torch.argmax`).
    pub argmax: u32,
}

pub struct Generated {
    /// Greedy pick after every call: after the prompt, then after each step.
    pub argmax: Vec<u32>,
    pub ttft_s: f64,
    pub decode_s: Vec<f64>,
    pub counters: Counters,
    pub bytes_read_total: u64,
    pub stats: Value,
}

/// Lowest index of the largest value.
pub fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Runs `prompt` (in calls of `chunk` ids, or one call) and then `steps`
/// one-token calls on `sess`. A step feeds `force[i]` when given (teacher
/// forcing), else the previous greedy pick. `keep_out` keeps every layer's
/// output stream in the hook's trace.
pub fn generate(
    store: &Store,
    sess: &mut Session,
    prompt: &[u32],
    chunk: Option<usize>,
    steps: usize,
    force: Option<&[u32]>,
    keep_out: bool,
    hook: &mut dyn FnMut(&Call) -> Result<()>,
) -> Result<Generated> {
    if prompt.is_empty() || chunk == Some(0) || force.is_some_and(|f| f.len() < steps) {
        return Err(Error::from_reason("deepseek_v41 generate: bad arguments"));
    }
    store.begin_turn()?;
    let out = run(store, sess, prompt, chunk, steps, force, keep_out, hook);
    let end = store.end_turn();
    let out = out?;
    end?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run(
    store: &Store,
    sess: &mut Session,
    prompt: &[u32],
    chunk: Option<usize>,
    steps: usize,
    force: Option<&[u32]>,
    keep_out: bool,
    hook: &mut dyn FnMut(&Call) -> Result<()>,
) -> Result<Generated> {
    let m = &store.model;
    let n_experts = m.cfg.n_routed_experts;
    let k = m.cfg.n_activated_experts;
    let mut rt = RouteTrace::from_env(prompt.len(), m.cfg.n_layers, n_experts, k)
        .map_err(|e| Error::from_reason(format!("route trace: {e}")))?;
    let c0 = Counters::now(&store.bank);
    let (_, swap0) = stats::host_pressure();
    let mut level_min: Option<u64> = None;
    let t0 = std::time::Instant::now();
    let mut calls: Vec<Vec<u32>> = prompt
        .chunks(chunk.unwrap_or(prompt.len()))
        .map(<[u32]>::to_vec)
        .collect();
    let n_prompt_calls = calls.len();
    let (mut picks, mut decode_s, mut ttft_s) = (Vec::new(), Vec::new(), 0.0);
    let mut stats_line = Value::Null;
    let mut i = 0;
    while i < calls.len() {
        let tc = std::time::Instant::now();
        let ids = calls[i].clone();
        let start = sess.tokens.len();
        let mut trace = Trace {
            keep_out,
            ..Default::default()
        };
        let logits = m.forward(sess, &ids, None, Some(&mut trace))?;
        let v = logits.shape_at(1)?;
        let last = logits
            .slice(&[ids.len() as i64 - 1, 0], &[ids.len() as i64, v])?
            .to_float32()?
            .to_vec();
        let pick = argmax(&last);
        if let Some(r) = rt.as_mut() {
            r.call(&trace.routes, sess.tokens.len())
                .map_err(|e| Error::from_reason(format!("route trace: {e}")))?;
        }
        hook(&Call {
            index: i,
            start,
            ids: &ids,
            logits: &logits,
            trace: &trace,
            argmax: pick,
        })?;
        let dt = tc.elapsed().as_secs_f64();
        if i + 1 < n_prompt_calls {
            i += 1;
            continue;
        }
        picks.push(pick);
        if i + 1 == n_prompt_calls {
            ttft_s = t0.elapsed().as_secs_f64();
        } else {
            decode_s.push(dt);
        }
        let (lvl, swap) = stats::host_pressure();
        level_min = match (level_min, lvl) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let c = Counters::now(&store.bank).since(&c0);
        let (_, peak, _) = stats::rusage();
        let swap_d = swap.zip(swap0).map_or(0, |(a, b)| a as i64 - b as i64);
        stats_line = stats::line(
            "s14a",
            prompt.len(),
            picks.len(),
            ttft_s,
            &decode_s,
            &c,
            t0.elapsed().as_secs_f64(),
            store.dense_read + c.fill_bytes + c.engram_bytes,
            store.cache_slots(),
            peak,
            swap_d,
            level_min,
        );
        stats::emit(&stats_line).map_err(|e| Error::from_reason(format!("stats: {e}")))?;
        let done = picks.len() - 1;
        if done < steps {
            let next = force.map_or(pick, |f| f[done]);
            calls.push(vec![next]);
        }
        i += 1;
    }
    let counters = Counters::now(&store.bank).since(&c0);
    Ok(Generated {
        argmax: picks,
        ttft_s,
        decode_s,
        bytes_read_total: store.dense_read + counters.fill_bytes + counters.engram_bytes,
        counters,
        stats: stats_line,
    })
}
