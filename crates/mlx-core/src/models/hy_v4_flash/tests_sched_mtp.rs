//! Scheduled MTP in paged mode. Fakes (no Metal): the wave row budget
//! (decode rows first, drafts in lane order, the last lanes cut) against
//! the vLLM-shaped table, a lens model of reserve / verify / commit /
//! rollback at depth 1..3 x width 2..4, the commit and fit rules, the paged
//! plan. Real (`HY4_DIR`, GPU): two owners through propose / verify /
//! commit at depth 1..3 give the greedy AR tokens each gives alone.

use std::path::PathBuf;

use super::cache::WINDOW_SLACK;
use super::chat::{MTP_DEPTH_MAX, generation_defaults, resolve};
use super::sched::{Open, WaveBudget, commit_len, rows_fit};
use super::spec::{LOOKAHEAD_MAX, PAGED_PLAN, PLAN};
use crate::engine::mtp_turn::turn_lookahead_rows;
use crate::engine::plan::SpeculativeLane;
use crate::engine::types::ChatConfig;
use crate::transformer::paged_kv_cache_adapter::SeqId;

/// Drafts granted to `lanes` lanes that each ask `depth`, in lane order.
fn grants(lanes: usize, depth: usize, row_cap: usize) -> Vec<usize> {
    let mut b = WaveBudget::default();
    for seq in 0..lanes {
        b.reserved(seq as SeqId);
    }
    let g = (0..lanes)
        .map(|seq| b.grant(seq as SeqId, depth, row_cap))
        .collect();
    assert!(b.rows() <= row_cap, "{lanes} lanes at depth {depth}");
    g
}

/// 12-vllm-dmax §7.3, 4 lanes: depth 1 never cuts; depth 2 cuts one draft
/// on Ultra (11); depth 3 cuts 4 on Max (12) and 5 on Ultra, last lanes
/// first.
#[test]
fn drafts_fill_the_row_cap_after_the_decode_rows() {
    for cap in [12, 11] {
        assert_eq!(grants(4, 1, cap), vec![1, 1, 1, 1]);
    }
    assert_eq!(grants(4, 2, 12), vec![2, 2, 2, 2]);
    assert_eq!(grants(4, 2, 11), vec![2, 2, 2, 1]);
    assert_eq!(grants(4, 3, 12), vec![3, 3, 2, 0]);
    assert_eq!(grants(4, 3, 11), vec![3, 3, 1, 0]);
    assert_eq!(grants(1, 3, 2), vec![1]);
    assert_eq!(grants(2, 3, 2), vec![0, 0]);
}

/// Every lane keeps its decode row; a lane is cut only after every lane
/// before it got its full depth.
#[test]
fn only_the_last_lanes_lose_drafts() {
    for cap in [2, 5, 6, 11, 12] {
        for lanes in 1..=4usize.min(cap) {
            for depth in 1..=MTP_DEPTH_MAX {
                let g = grants(lanes, depth, cap);
                let mut left = cap - lanes;
                for (i, &n) in g.iter().enumerate() {
                    assert_eq!(
                        n,
                        depth.min(left),
                        "cap {cap} lanes {lanes} depth {depth} lane {i}"
                    );
                    left -= n;
                }
            }
        }
    }
}

/// A reserve after a proposal opens the next wave; a retry reserve of a
/// lane before any proposal counts it once.
#[test]
fn a_reserve_after_a_proposal_starts_the_next_wave() {
    let mut b = WaveBudget::default();
    b.reserved(1);
    b.reserved(1);
    b.reserved(2);
    assert_eq!(b.rows(), 2);
    assert_eq!(b.grant(1, 3, 6), 3);
    assert_eq!(b.grant(2, 3, 6), 1);
    b.reserved(2);
    assert_eq!(b.rows(), 1, "the next wave holds only lane 2 so far");
    assert_eq!(b.grant(2, 3, 6), 3);
    b.close();
    assert_eq!(b.rows(), 0);
}

#[test]
fn a_commit_keeps_at_most_the_verified_rows() {
    let open = Open { start: 40, rows: 4 };
    for keep in 0..=4 {
        assert_eq!(commit_len(open, keep).unwrap(), 40 + keep);
    }
    assert!(commit_len(open, 5).is_err());
}

/// Rows fit when one cut can drop all but the anchor, one wave holds them
/// and the context does.
#[test]
fn verify_rows_fit_the_slack_the_cap_and_the_context() {
    assert_eq!(LOOKAHEAD_MAX, WINDOW_SLACK + 1);
    assert!(!rows_fit(0, 10, 100, 12));
    for rows in 1..=LOOKAHEAD_MAX {
        assert!(rows_fit(rows, 10, 100, 12));
    }
    assert!(!rows_fit(LOOKAHEAD_MAX + 1, 10, 100, 12));
    assert!(rows_fit(2, 10, 100, 2));
    assert!(!rows_fit(3, 10, 100, 2));
    assert!(rows_fit(4, 96, 100, 12));
    assert!(!rows_fit(4, 97, 100, 12));
}

/// One owner's lengths: trunk (its cell), pool rows, MTP rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lens {
    trunk: usize,
    pool: usize,
    draft: usize,
}

impl Lens {
    /// A verify of `rows` rows: trunk and pool grow, the wave closes the
    /// MTP rows of every trunk row but the last.
    fn verify(&mut self, rows: usize) -> Open {
        assert_eq!(self.draft + 1, self.trunk, "settled before the wave");
        let open = Open {
            start: self.trunk,
            rows,
        };
        self.trunk += rows;
        self.pool += rows;
        self.draft = self.trunk - 1;
        open
    }

    /// A commit keeping `keep` rows: pool and cell cut to the same length.
    fn commit(&mut self, open: Open, keep: usize) -> usize {
        let len = commit_len(open, keep).unwrap();
        let cut = self.trunk - len;
        self.trunk = len;
        self.pool = len;
        self.draft = self.draft.min(len.saturating_sub(1));
        cut
    }
}

/// Depth 1..3 x width 2..4 on Max and Ultra row caps: every wave fits the
/// cap, every accept count (0..=drafts) leaves pool == trunk and the
/// draft settled, and no cut passes the window slack.
#[test]
fn lens_model_over_waves_at_every_depth_and_width() {
    const CTX: usize = 4096;
    for cap in [12, 11] {
        for width in 2..=4usize {
            for depth in 1..=MTP_DEPTH_MAX {
                let mut owners = vec![
                    Lens {
                        trunk: 33,
                        pool: 33,
                        draft: 32,
                    };
                    width
                ];
                let mut budget = WaveBudget::default();
                for wave in 0..24usize {
                    for (seq, o) in owners.iter().enumerate() {
                        assert!(rows_fit(1 + depth, o.trunk, CTX, cap));
                        budget.reserved(seq as SeqId);
                    }
                    let drafts: Vec<usize> = (0..width)
                        .map(|seq| budget.grant(seq as SeqId, depth, cap))
                        .collect();
                    let rows: usize = drafts.iter().map(|g| g + 1).sum();
                    assert!(rows <= cap, "cap {cap} width {width} depth {depth}");
                    budget.close();
                    for (seq, (o, &g)) in owners.iter_mut().zip(&drafts).enumerate() {
                        let before = o.trunk;
                        let open = o.verify(1 + g);
                        let accepted = (wave + seq) % (g + 1);
                        let cut = o.commit(open, 1 + accepted);
                        assert!(cut <= WINDOW_SLACK);
                        assert_eq!(o.trunk, before + 1 + accepted);
                        assert_eq!(o.pool, o.trunk);
                        assert_eq!(o.draft + 1, o.trunk);
                    }
                }
            }
        }
    }
}

/// Paged mode verifies on the pool, so its MTP turns take the scheduled
/// lane; the flat plan stays a barrier. The deepest request's lookahead
/// is the stepper's.
#[test]
fn the_paged_plan_takes_the_scheduled_lane() {
    assert_eq!(PAGED_PLAN.lane(true), SpeculativeLane::Scheduled);
    assert_eq!(PAGED_PLAN.lane(false), SpeculativeLane::Barrier);
    assert_eq!(PLAN.lane(true), SpeculativeLane::Barrier);
    let p = resolve(
        &ChatConfig {
            mtp_depth: Some(9),
            mtp_adaptive_depth: Some(true),
            ..Default::default()
        },
        &generation_defaults(),
    );
    assert_eq!(turn_lookahead_rows(PAGED_PLAN, &p), LOOKAHEAD_MAX);
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

/// Two owners prefilled, decoded AR alone, then together through propose /
/// verify / commit at depth 1..3 with greedy acceptance: each owner's
/// tokens equal its AR tokens, every verify wave fits the row cap.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_scheduled_mtp_greedy_equals_ar_alone() {
    use super::chat::Inner;
    use crate::engine::backend::{ChatBackend, PagedBackend, PagedPrefix};
    use crate::engine::hybrid_scheduler::{
        HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
    };
    use crate::stream::Stream;

    const STEPS: usize = 12;
    let mut inner = Inner::open(&hy4_dir(), Some(16_384), Default::default(), false, None).unwrap();
    assert!(inner.supports_scheduled_speculation());
    let row_cap = inner.paged().unwrap().row_cap;
    let tok = inner.tokenizer().unwrap();
    let prompts: Vec<Vec<u32>> = ["The capital of France is", "List three prime numbers:"]
        .iter()
        .map(|t| tok.encode_sync(t, Some(false)).unwrap())
        .collect();
    let prefill = |inner: &mut Inner, seq: SeqId, p: &[u32], mtp: bool| -> u32 {
        inner.activate_paged_seq(seq).unwrap();
        let prefix = inner.prime_prefix_state(p, true, 32, &[], 0).unwrap();
        let cached = prefix.effective_cached_prefix_len();
        if mtp {
            assert!(
                inner
                    .begin_scheduled_speculation(seq, cached as u32)
                    .unwrap()
            );
        }
        let logits = inner
            .paged_prefill(&p[cached..], &prefix, Stream::generation())
            .unwrap();
        greedy(&logits.reshape(&[1, 1, -1]).unwrap())[0]
    };
    let release = |inner: &mut Inner, seq: SeqId| {
        inner.release_scheduled_speculation(seq);
        inner.release_scheduled_cache(seq).unwrap();
        inner.release_scheduled_recurrent_for(seq);
    };
    let mut alone = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let seq = i as SeqId + 1;
        let mut t = prefill(&mut inner, seq, p, false);
        let mut out = vec![t];
        for _ in 1..STEPS {
            t = greedy(&inner.run_paged_decode_step_batched(&[(seq, t)]).unwrap())[0];
            out.push(t);
        }
        alone.push(out);
        release(&mut inner, seq);
    }
    let mut rng = rand::rng();
    for depth in 1..=MTP_DEPTH_MAX {
        let params = resolve(
            &ChatConfig {
                temperature: Some(0.0),
                mtp_depth: Some(depth as i32),
                ..Default::default()
            },
            &generation_defaults(),
        );
        let seqs: Vec<SeqId> = (1..=prompts.len() as SeqId).collect();
        let mut anchors: Vec<u32> = seqs
            .iter()
            .zip(&prompts)
            .map(|(&s, p)| prefill(&mut inner, s, p, true))
            .collect();
        let mut out: Vec<Vec<u32>> = anchors.iter().map(|&t| vec![t]).collect();
        while out.iter().any(|o| o.len() < STEPS) {
            for &s in &seqs {
                assert!(inner.reserve_scheduled_speculation(s, depth + 1).unwrap());
            }
            let mut rows = Vec::new();
            for (&s, &a) in seqs.iter().zip(&anchors) {
                let p = inner
                    .propose_scheduled(s, a, depth, &params, &mut rng, false)
                    .unwrap();
                let mut tokens = vec![a];
                tokens.extend(p.draft_ids.iter().map(|&t| t as u32));
                rows.push(ScheduledVerifyRow {
                    seq_id: s,
                    first_position: inner.owner_len(s) as u32,
                    tokens,
                    speculative: true,
                });
            }
            let total: usize = rows.iter().map(|r| r.tokens.len()).sum();
            assert!(total <= row_cap, "{total} rows for a cap of {row_cap}");
            let target = greedy(&inner.run_scheduled_verify(&rows).unwrap());
            let mut commits = Vec::new();
            let mut at = 0;
            for (i, r) in rows.iter().enumerate() {
                let drafts = &r.tokens[1..];
                let t = &target[at..at + r.tokens.len()];
                at += r.tokens.len();
                let accepted = drafts.iter().zip(t).take_while(|(d, t)| d == t).count();
                out[i].extend_from_slice(&drafts[..accepted]);
                out[i].push(t[accepted]);
                anchors[i] = t[accepted];
                commits.push(ScheduledVerifyCommit {
                    seq_id: r.seq_id,
                    keep: 1 + accepted,
                });
            }
            for r in inner.commit_scheduled_verify(&commits).unwrap() {
                r.unwrap();
            }
        }
        for o in &mut out {
            o.truncate(STEPS);
        }
        assert_eq!(out, alone, "depth {depth}");
        for &s in &seqs {
            release(&mut inner, s);
        }
    }
}
