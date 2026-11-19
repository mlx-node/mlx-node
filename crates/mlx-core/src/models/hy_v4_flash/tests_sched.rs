//! Paged mode hooks. Fakes (no Metal): the mode and lane rule, the NAPI
//! concurrency, the prefill ends (grid + state rows), the pass keys, the
//! busy period (open / join / close, exclusive turns), the safe point (seed
//! reset after a join, readmit rate limit), preempt = release without
//! registration, the request depth clamp. Real (`HY4_DIR`, GPU): two owners
//! decoded in one wave give the greedy tokens each gives alone.

use std::path::PathBuf;

use napi::Result;

use super::chat::{MTP_DEPTH_MAX, generation_defaults, resolve};
use super::prefill::{BLOCK_ROWS, Grid, NEAR_EVERY};
use super::sched::{
    Busy, Edge, Requests, SafePoint, lanes, napi_concurrency, paged_mode, pass_keys, preempt,
    prefill_ends, slice_breaks,
};
use crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for;
use crate::engine::types::ChatConfig;
use crate::transformer::paged_kv_cache_adapter::SeqId;

const OPEN: [u32; 3] = [120_000, 611, 10_372];

fn grid(chunk: usize) -> Grid {
    Grid::new(chunk, OPEN.to_vec()).unwrap()
}

/// `len` ids, the assistant opening at `open_at`.
fn prompt(len: usize, open_at: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..len as u32).map(|i| 1000 + i % 97).collect();
    p[open_at..open_at + OPEN.len()].copy_from_slice(&OPEN);
    p
}

#[test]
fn mode_is_paged_unless_serial_or_no_row_cap() {
    assert!(paged_mode(false, 12));
    assert!(paged_mode(false, 2));
    assert!(!paged_mode(false, 1));
    assert!(!paged_mode(true, 12));
}

#[test]
fn lanes_are_the_admitted_ones_within_the_row_cap() {
    assert_eq!(lanes(4, 12), 4);
    assert_eq!(lanes(4, 3), 3);
    assert_eq!(lanes(1, 12), 1);
    assert_eq!(lanes(0, 12), 1);
}

/// Force serial reads 1; otherwise the engine's cap over the lanes.
#[test]
fn napi_concurrency_is_one_under_force_serial() {
    assert_eq!(napi_concurrency(true, 4), 1);
    assert_eq!(
        napi_concurrency(false, 4) as usize,
        scheduler_max_num_seqs_for(4)
    );
    assert_eq!(napi_concurrency(false, 1), 1);
}

/// The slice ends are the grid's chunk ends and the prompt's state rows
/// past the cached prefix, the last at the prompt end; another prompt
/// length is not this turn's.
#[test]
fn prefill_ends_are_grid_ends_and_state_rows() {
    let g = grid(2048);
    let p = prompt(20_000, 19_000);
    let ends = prefill_ends(&g, &p, 20_000, 0).unwrap();
    let replay = 19_000 - 19_000 % BLOCK_ROWS;
    for row in [NEAR_EVERY, 2 * NEAR_EVERY, replay, 20_000] {
        assert!(ends.contains(&(row as u32)), "{row} in {ends:?}");
    }
    assert!(
        ends.iter()
            .all(|&e| e % 2048 == 0 || e as usize == replay || e == 20_000)
    );
    assert!(ends.windows(2).all(|w| w[0] < w[1]));

    let resumed = prefill_ends(&g, &p, 20_000, 8192).unwrap();
    assert!(resumed.iter().all(|&e| e > 8192));
    assert_eq!(
        resumed,
        ends.iter()
            .copied()
            .filter(|&e| e > 8192)
            .collect::<Vec<_>>()
    );
    assert!(prefill_ends(&g, &p, 19_999, 0).is_none());
}

/// The engine's slice ends are the turn's grid ends for its prompt; any
/// other prompt (or none) is cut at every chunk multiple past the cached
/// rows, never at the cached row itself.
/// Mutation: `slice_breaks` ignores `prompt` → red.
#[test]
fn slice_breaks_follow_the_turn_grid_else_the_chunk() {
    let g = grid(2048);
    let p = prompt(20_000, 19_000);
    assert_eq!(
        slice_breaks(&g, Some(&p), 20_000, 0),
        prefill_ends(&g, &p, 20_000, 0).unwrap()
    );
    assert_eq!(
        slice_breaks(&g, Some(&p), 20_000, 8192),
        prefill_ends(&g, &p, 20_000, 8192).unwrap()
    );
    let chunked: Vec<u32> = (1..=9).map(|k| k * 2048).collect();
    assert_eq!(slice_breaks(&g, Some(&p), 19_999, 0), chunked);
    assert_eq!(slice_breaks(&g, None, 20_000, 0), chunked);
    assert_eq!(slice_breaks(&g, None, 20_000, 4096), chunked[2..].to_vec());
    assert_eq!(slice_breaks(&g, None, 5000, 4500), Vec::<u32>::new());
}

/// A block's key is the chunk that wrote it: equal for two prompts that
/// share the chunk, different once a chunk end inside moves; the last
/// chunk's blocks get none.
#[test]
fn pass_keys_follow_the_writing_chunk() {
    let a = prompt(300, 250);
    let ends = [128, 256, 300];
    let keys = pass_keys(&a, &ends, 128);
    assert_eq!(keys.len(), 256 / BLOCK_ROWS);
    assert_eq!(keys[0], keys[3]);
    assert_ne!(keys[3], keys[4]);

    let mut b = a.clone();
    b[290] = 7;
    assert_eq!(pass_keys(&b, &ends, 128), keys);

    let moved = pass_keys(&a, &[64, 128, 256, 300], 128);
    assert_ne!(moved[0], keys[0]);
    assert_eq!(moved[4], keys[4]);
    assert!(pass_keys(&a, &[], 128).is_empty());
}

#[test]
fn busy_period_spans_the_first_open_to_the_last_close() {
    let mut b = Busy::default();
    assert_eq!(b.opened(1), Edge::Begin);
    assert_eq!(b.opened(2), Edge::Join);
    assert_eq!(b.opened(2), Edge::None);
    assert_eq!(b.flag(true), Edge::None);
    assert_eq!(b.closed(1), Edge::None);
    assert!(b.is_live());
    assert_eq!(b.closed(2), Edge::End);
    assert!(!b.is_live());
    assert_eq!(b.flag(false), Edge::None);
}

/// A whole turn with no scheduled turn open is a busy period of its own; a
/// scheduled turn admitted during it joins.
#[test]
fn exclusive_turn_is_a_busy_period() {
    let mut b = Busy::default();
    assert_eq!(b.flag(true), Edge::Begin);
    assert_eq!(b.flag(true), Edge::None);
    assert_eq!(b.opened(5), Edge::Join);
    assert_eq!(b.flag(false), Edge::None);
    assert_eq!(b.closed(5), Edge::End);
    assert_eq!(b.flag(true), Edge::Begin);
    assert_eq!(b.flag(false), Edge::End);
}

/// The seed quota starts over at the first step after a join, once; a
/// re-admission is due every `every` steps; a new period clears a pending
/// reset.
#[test]
fn safe_point_resets_seeds_after_a_join_and_rate_limits_readmit() {
    let mut s = SafePoint::new(3);
    let first = s.step();
    assert!(!first.seed_reset && !first.readmit);
    s.joined();
    assert!(s.step().seed_reset);
    let third = s.step();
    assert!(!third.seed_reset);
    assert!(third.readmit);
    assert!(!s.step().readmit);
    assert!(!s.step().readmit);
    assert!(s.step().readmit);

    s.joined();
    s.begun();
    let after = s.step();
    assert!(!after.seed_reset && !after.readmit);
}

#[derive(Default)]
struct FakePool {
    held: Vec<SeqId>,
    calls: Vec<String>,
}

impl Requests for FakePool {
    fn holds(&self, seq: SeqId) -> bool {
        self.held.contains(&seq)
    }

    fn release(&mut self, seq: SeqId) -> Result<()> {
        self.calls.push(format!("release {seq}"));
        self.held.retain(|&s| s != seq);
        Ok(())
    }

    fn register(&mut self, seq: SeqId, upto: usize, _: &[Vec<u64>], _: u64) -> Result<()> {
        self.calls.push(format!("register {seq} {upto}"));
        Ok(())
    }
}

/// Preemption gives the blocks back and registers none of them; a sequence
/// without a request is left alone.
#[test]
fn preempt_releases_without_publication() {
    let mut pool = FakePool {
        held: vec![3, 4],
        ..FakePool::default()
    };
    preempt(&mut pool, 3).unwrap();
    assert_eq!(pool.calls, vec!["release 3".to_string()]);
    assert_eq!(pool.held, vec![4]);
    preempt(&mut pool, 9).unwrap();
    assert_eq!(pool.calls.len(), 1);
}

/// A request may ask for adaptive depth and any depth: adaptive is pinned
/// off and the depth clamped to `MTP_DEPTH_MAX` (3).
#[test]
fn request_depth_is_clamped_and_never_adaptive() {
    let cfg = ChatConfig {
        mtp_depth: Some(5),
        mtp_adaptive_depth: Some(true),
        ..Default::default()
    };
    let p = resolve(&cfg, &generation_defaults());
    assert_eq!(MTP_DEPTH_MAX, 3);
    assert_eq!(p.mtp_depth, 3);
    assert!(!p.mtp_adaptive_depth);
    let p = resolve(&ChatConfig::default(), &generation_defaults());
    assert_eq!(p.mtp_depth, 1);
}

fn hy4_dir() -> PathBuf {
    PathBuf::from(std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards"))
}

/// Greedy ids of `logits [rows, 1, vocab]`, one per row.
fn greedy(logits: &crate::array::MxArray) -> Vec<u32> {
    let ids = logits.argmax(-1, Some(false)).unwrap();
    ids.eval();
    ids.to_int32().unwrap().iter().map(|&t| t as u32).collect()
}

/// Two owners prefilled, then decoded alone and together in one wave: each
/// owner's greedy tokens are the same both ways (lane == alone).
/// Mutation: drop the M >= 2 pad of a one-row wave → red.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_two_owners_in_one_wave_equal_alone() {
    use crate::engine::backend::{ChatBackend, PagedBackend, PagedPrefix};
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    use crate::stream::Stream;

    const STEPS: usize = 8;
    let mut inner =
        super::chat::Inner::open(&hy4_dir(), Some(16_384), Default::default(), false, None)
            .unwrap();
    assert!(inner.paged_adapter().is_some());
    let tok = inner.tokenizer().unwrap();
    let prompts: Vec<Vec<u32>> = ["The capital of France is", "List three prime numbers:"]
        .iter()
        .map(|t| tok.encode_sync(t, Some(false)).unwrap())
        .collect();
    let prefill = |inner: &mut super::chat::Inner, seq: SeqId, p: &[u32]| -> u32 {
        inner.activate_paged_seq(seq).unwrap();
        let prefix = inner.prime_prefix_state(p, true, 32, &[], 0).unwrap();
        let cached = prefix.effective_cached_prefix_len();
        let logits = inner
            .paged_prefill(&p[cached..], &prefix, Stream::generation())
            .unwrap();
        greedy(&logits.reshape(&[1, 1, -1]).unwrap())[0]
    };
    let mut alone = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let seq = i as SeqId + 1;
        let mut t = prefill(&mut inner, seq, p);
        let mut out = vec![t];
        for _ in 1..STEPS {
            t = greedy(&inner.run_paged_decode_step_batched(&[(seq, t)]).unwrap())[0];
            out.push(t);
        }
        alone.push(out);
        inner.release_scheduled_cache(seq).unwrap();
        inner.release_scheduled_recurrent_for(seq);
    }
    let mut last: Vec<u32> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| prefill(&mut inner, i as SeqId + 1, p))
        .collect();
    let mut together = vec![vec![last[0]], vec![last[1]]];
    for _ in 1..STEPS {
        last = greedy(
            &inner
                .run_paged_decode_step_batched(&[(1, last[0]), (2, last[1])])
                .unwrap(),
        );
        together[0].push(last[0]);
        together[1].push(last[1]);
    }
    assert_eq!(together, alone);
}
