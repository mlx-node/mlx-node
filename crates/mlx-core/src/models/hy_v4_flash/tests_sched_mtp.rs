//! Scheduled MTP in paged mode. Fakes (no Metal): the wave row budget
//! (decode rows first, drafts in lane order, the last lanes cut) against
//! the vLLM-shaped table, a lens model of reserve / verify / commit /
//! rollback at depth 1..3 x width 2..4, the commit and fit rules, the paged
//! plan. Real (`HY4_DIR`, GPU): two owners through propose / verify /
//! commit at depth 1..3 give the greedy AR tokens each gives alone.

use std::path::PathBuf;

use super::cache::WINDOW_SLACK;
use super::chat::{MTP_DEPTH_MAX, generation_defaults, resolve};
use super::sched::{Open, WaveBudget, commit_len, mtp_wave_row_cap, rows_fit};
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

#[test]
fn mtp_wave_row_cap_accommodates_multi_lane_depth_two_and_three() {
    for base in [11usize, 12] {
        let cap = mtp_wave_row_cap(base, 4, true, false);
        assert_eq!(cap, 16);
        assert_eq!(grants(4, 1, cap), vec![1, 1, 1, 1]);
        assert_eq!(grants(4, 2, cap), vec![2, 2, 2, 2]);
        assert_eq!(grants(4, 3, cap), vec![3, 3, 3, 3]);
        assert_eq!(mtp_wave_row_cap(base, 4, false, false), base);
        assert_eq!(mtp_wave_row_cap(base, 4, true, true), base);
    }
    assert_eq!(mtp_wave_row_cap(12, 2, true, false), 12);
    assert_eq!(mtp_wave_row_cap(11, 3, true, false), 12);
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

/// The pool rows a step's lanes reserve fit the wave row cap: every anchor
/// first, then drafts in plan order, so an early lane's draft never takes
/// the rows a later lane's anchor needs; a lane's grant stays within what it
/// reserved; a lane outside the plan reserves what it asks.
#[test]
fn reserved_rows_fit_the_row_cap_before_any_proposal() {
    let mut b = WaveBudget::default();
    let plan: Vec<(SeqId, usize)> = (1..=4).map(|s| (s, 2)).collect();
    b.plan(&plan, 5);
    let rows: Vec<usize> = (1..=4).map(|s| b.rows_for(s, 2)).collect();
    assert_eq!(rows, vec![2, 1, 1, 1]);
    assert!(rows.iter().sum::<usize>() <= 5);
    for (s, &r) in (1..=4).zip(&rows) {
        b.reserved_rows(s, r);
    }
    let g: Vec<usize> = (1..=4).map(|s| b.grant(s, 1, 5)).collect();
    assert_eq!(g, vec![1, 0, 0, 0]);
    assert!(b.rows() <= 5);

    b.close();
    let plan: Vec<(SeqId, usize)> = (1..=4).map(|s| (s, 4)).collect();
    b.plan(&plan, 12);
    let rows: Vec<usize> = (1..=4).map(|s| b.rows_for(s, 4)).collect();
    assert_eq!(rows, vec![4, 4, 3, 1]);
    b.reserved_rows(1, 1);
    assert_eq!(b.grant(1, 3, 12), 0, "a retry with the anchor alone");
    assert_eq!(b.rows_for(9, 4), 4);
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

/// Real tiny `Inner` + `Paged::open` + `MtpLayer` verifying:
/// 1. Multi-lane `Paged::open` scales `states.ram_cap()` to `lanes * RAM_STATES`.
/// 2. Speculative `verify_scheduled` does NOT overwrite `Turn::held` before commit:
///    - `start % 32 == 28` with `keep < 4` preserves previously committed `held` (64).
///    - `start % 32 == 28` with `keep == 4` captures `start + 4` (96) on commit.
///    - `start % 32 in {29, 30, 31}` landing on `32k` (`29+3`, `30+2`, `31+1`)
///      captures `96` after `cut_owner` rollback.
/// 3. Committed `decode_rows` (non-speculative) and `spec::Paged::forward_with_hidden`
///    still capture on `32k`, while `spec::Paged::verify_step` + `rollback`
///    preserves prior `held` on `keep < 4` and captures on `{29,30,31}` landing on `32k`.
fn build_paged_inner(
    with_mtp: bool,
    want_lanes: usize,
) -> (
    super::chat::Inner,
    super::config::HyV4FlashConfig,
    super::test_support::Temp,
) {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::bank::KqBank;
    use super::chat::Inner;
    use super::config::{Attn, LayerConfig};
    use super::model::Model;
    use super::moe::ExpertSource;
    use super::mtp::MtpLayer;
    use super::prefill::Grid;
    use super::rows::F8Table;
    use super::scan::{ENGRAM_TABLE, ExpertLayer, Located, Scan};
    use super::sched::Paged;
    use super::store::{Classes, Lease, Sizes, Store, profile};
    use super::test_support::{
        Rng, TINY_CLASSES, Temp, TinyWeightsOpts, kquant_layer, mini_mtp_weights, tiny_cfg,
        tiny_engram_table, tiny_weights_opts,
    };
    use crate::moe_stream::budget::ModelSize;
    use crate::moe_stream::io::{IoPool, ShardSet};
    use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

    let mut cfg = tiny_cfg();
    cfg.context_length = 256;
    // kv_row = kv_lora_rank + rope_dim = 56 + 8 = 64 -> half = 32 (valid paged head_size).
    cfg.mla.kv_lora_rank = 56;
    cfg.mla.rope_dim = 8;
    if with_mtp {
        cfg.layers.push(LayerConfig {
            attn: Attn::Indexed,
            n_heads: 2,
            rope_theta: 1e7,
            swiglu_limit: 0.5,
            dense_ffn: false,
            engram: None,
            mtp: true,
        });
    }
    let cfg_out = cfg.clone();
    let mut rng = Rng::new(77);
    let mut exp_file = rng.bytes(1024);
    let experts: Vec<Option<ExpertLayer>> = TINY_CLASSES
        .iter()
        .enumerate()
        .map(|(il, c)| c.map(|c| kquant_layer(&mut rng, &cfg.moe, il, c, 32, &mut exp_file)))
        .collect();
    let tmp = Temp::new("sched-mtp-turn");
    let exp_path = tmp.write("experts.bin", &exp_file);
    let engram_rows: u64 = cfg.engram.head_vocab_sizes.iter().sum();
    let engram_path = tmp.write(
        "engram.bin",
        &vec![0u8; engram_rows as usize * cfg.engram.head_dim],
    );
    let paths = vec![exp_path, engram_path];
    let shards = Arc::new(ShardSet::open(&paths).unwrap());
    let engram_loc = Located {
        shard: 1,
        data_offset: 0,
        info: GgufTensorInfo {
            name: ENGRAM_TABLE.into(),
            n_dims: 2,
            dims: vec![cfg.engram.head_dim as u64, engram_rows],
            tensor_type: GgufTensorType::MXFP8,
            offset: 0,
        },
    };
    let table = Arc::new(
        F8Table::open_with_options(Arc::clone(&shards), 1, &engram_loc, &cfg.engram, 1, 1).unwrap(),
    );
    let bank = Arc::new(
        KqBank::new_direct(
            IoPool::new(Arc::clone(&shards), 2).unwrap(),
            &cfg.moe,
            &experts,
        )
        .unwrap(),
    );
    let classes = Classes {
        layers: [2, 1],
        slot: bank.slot_bytes(),
    };
    let p = profile(15_812);
    let size = ModelSize {
        dense: 4 << 20,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: cfg.moe.n_experts as u64,
        ctx: cfg.context_length as u64,
    };
    let b_ref = Arc::clone(&bank);
    let model = Model::new(
        cfg.clone(),
        Box::new(tiny_weights_opts(&cfg, TinyWeightsOpts::default())),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b_ref))),
        Box::new(tiny_engram_table(&cfg.engram, 19)),
    )
    .unwrap();
    let mtp = with_mtp.then(|| {
        let mw = mini_mtp_weights(&cfg, 91, 24, 0);
        MtpLayer::new(&cfg, mw, model.weights().token_embd()).unwrap()
    });
    let lease = Lease::new(p, size, 64, classes, None, None, &bank);
    let store = Store {
        model,
        bank,
        table,
        scan: Scan {
            paths,
            cfg,
            tensors: HashMap::new(),
            experts,
            engram_table: engram_loc,
            mtp: None,
        },
        shards,
        dense_read: 0,
        dense_bytes: size.dense,
        sizes: Sizes {
            dense_trunk: size.dense,
            dense_mtp: 0,
            classes,
            n_experts: 8,
        },
        size,
        profile: p,
        ctx: 256,
        chunk: 64,
        lanes: want_lanes,
        row_cap: 12,
        mtp: with_mtp,
        bank_alloc_s: 0.0,
        max_memory: None,
        direct_budget: Some(1 << 30),
        rset_bounded: false,
        lease,
    };
    let paged = Paged::open(&store).unwrap();
    let grid = Grid::new(64, Vec::new()).unwrap();
    let inner = Inner::for_paged_test(store, mtp, grid, paged).unwrap();
    (inner, cfg_out, tmp)
}

#[test]
fn speculative_verify_and_cut_owner_preserve_and_capture_turn_states() {
    use super::chat::Inner;
    use super::ladder::RAM_STATES;
    use super::sched::paged_ram_states;
    use crate::array::MxArray;
    use crate::engine::backend::{MtpStepper, PagedBackend, PagedPrefix};
    use crate::engine::hybrid_scheduler::{
        HybridSchedulerBackend, ScheduledVerifyCommit, ScheduledVerifyRow,
    };
    use crate::stream::Stream;

    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }

    let build_inner = |with_mtp: bool, want_lanes: usize| {
        let (inner, _, tmp) = build_paged_inner(with_mtp, want_lanes);
        (inner, tmp)
    };

    // Helper: prefill a distinct prompt of `prompt_len` tokens on `seq`, then
    // advance via committed verify waves to `target_start`.
    let advance_to = |inner: &mut Inner, seq: SeqId, prompt_len: usize, target_start: usize| {
        let prompt: Vec<u32> = (0..prompt_len as u32)
            .map(|i| 1 + ((i + seq * 13) % 60))
            .collect();
        inner.activate_paged_seq(seq).unwrap();
        let prefix = inner.prime_prefix_state(&prompt, true, 32, &[], 0).unwrap();
        assert_eq!(prefix.effective_cached_prefix_len(), 0);
        assert!(inner.begin_scheduled_speculation(seq, 0).unwrap());
        inner
            .paged_prefill(&prompt, &prefix, Stream::generation())
            .unwrap();
        let mut pos = prompt_len;
        while pos < target_start {
            let step = (target_start - pos).min(4);
            let step = if pos < 64 && pos + step > 64 {
                64 - pos
            } else {
                step
            };
            let toks: Vec<u32> = (0..step).map(|i| 2 + ((pos + i) % 60) as u32).collect();
            assert!(inner.reserve_scheduled_speculation(seq, step).unwrap());
            inner
                .run_scheduled_verify(&[ScheduledVerifyRow {
                    seq_id: seq,
                    first_position: pos as u32,
                    tokens: toks,
                    speculative: true,
                }])
                .unwrap();
            for r in inner
                .commit_scheduled_verify(&[ScheduledVerifyCommit {
                    seq_id: seq,
                    keep: step,
                }])
                .unwrap()
            {
                r.unwrap();
            }
            pos += step;
        }
        assert_eq!(inner.owner_len(seq), target_start);
    };

    let release = |inner: &mut Inner, seq: SeqId| {
        inner.release_scheduled_speculation(seq);
        inner.release_scheduled_cache(seq).unwrap();
        inner.release_scheduled_recurrent_for(seq);
    };

    let held_row = |inner: &Inner, seq: SeqId| -> Option<usize> {
        inner
            .paged()
            .unwrap()
            .turns
            .get(&seq)
            .and_then(|t| t.turn.held_row())
    };

    // 1. Multi-lane `Paged::open` sets `states.ram_cap()` to `4 * RAM_STATES`.
    let (mut inner, _tmp) = build_inner(true, 4);
    assert_eq!(inner.paged().unwrap().lanes, 4);
    assert_eq!(inner.paged().unwrap().states.ram_cap(), paged_ram_states(4));
    assert_eq!(inner.paged().unwrap().states.ram_cap(), 4 * RAM_STATES);

    // 2. `start % 32 == 28` (92), `keep < 4` (keep = 2 -> 94):
    //    previously committed Turn state at 64 MUST survive verify + rollback!
    advance_to(&mut inner, 1, 40, 92);
    assert_eq!(held_row(&inner, 1), Some(64));
    assert!(inner.reserve_scheduled_speculation(1, 4).unwrap());
    inner
        .run_scheduled_verify(&[ScheduledVerifyRow {
            seq_id: 1,
            first_position: 92,
            tokens: vec![5, 6, 7, 8],
            speculative: true,
        }])
        .unwrap();
    assert_eq!(
        held_row(&inner, 1),
        Some(64),
        "speculative verify wave reaching row 96 must not overwrite held row 64 before commit"
    );
    inner
        .commit_scheduled_verify(&[ScheduledVerifyCommit { seq_id: 1, keep: 2 }])
        .unwrap()
        .into_iter()
        .for_each(|r| r.unwrap());
    assert_eq!(inner.owner_len(1), 94);
    assert_eq!(
        held_row(&inner, 1),
        Some(64),
        "rollback from 96 -> 94 must preserve previously committed Turn state at 64"
    );
    release(&mut inner, 1);

    // 3. `start % 32 == 28` (92), `keep == 4` (96):
    //    fast-path `cut_owner` captures 96 once all 4 rows commit.
    advance_to(&mut inner, 2, 40, 92);
    assert_eq!(held_row(&inner, 2), Some(64));
    assert!(inner.reserve_scheduled_speculation(2, 4).unwrap());
    inner
        .run_scheduled_verify(&[ScheduledVerifyRow {
            seq_id: 2,
            first_position: 92,
            tokens: vec![9, 10, 11, 12],
            speculative: true,
        }])
        .unwrap();
    assert_eq!(held_row(&inner, 2), Some(64));
    inner
        .commit_scheduled_verify(&[ScheduledVerifyCommit { seq_id: 2, keep: 4 }])
        .unwrap()
        .into_iter()
        .for_each(|r| r.unwrap());
    assert_eq!(inner.owner_len(2), 96);
    assert_eq!(
        held_row(&inner, 2),
        Some(96),
        "full commit at 92 + 4 must capture Turn state at 96"
    );
    release(&mut inner, 2);

    // 4. `start % 32 in {29, 30, 31}` landing on `96` (`93+3`, `94+2`, `95+1`)
    //    after a 4-row speculative verify wave: `cut_owner` rollback path captures 96!
    for (idx, (start, keep)) in [(93usize, 3usize), (94, 2), (95, 1)]
        .into_iter()
        .enumerate()
    {
        let seq = 10 + idx as SeqId;
        advance_to(&mut inner, seq, 40, start);
        assert_eq!(held_row(&inner, seq), Some(64));
        assert!(inner.reserve_scheduled_speculation(seq, 4).unwrap());
        inner
            .run_scheduled_verify(&[ScheduledVerifyRow {
                seq_id: seq,
                first_position: start as u32,
                tokens: vec![13, 14, 15, 16],
                speculative: true,
            }])
            .unwrap();
        assert_eq!(held_row(&inner, seq), Some(64));
        inner
            .commit_scheduled_verify(&[ScheduledVerifyCommit { seq_id: seq, keep }])
            .unwrap()
            .into_iter()
            .for_each(|r| r.unwrap());
        assert_eq!(inner.owner_len(seq), 96);
        assert_eq!(
            held_row(&inner, seq),
            Some(96),
            "start={start} keep={keep} landing on 96 must capture Turn state at 96"
        );
        release(&mut inner, seq);
    }

    // 5. Exclusive paged MTP stepper (`spec::Paged`):
    //    - `forward_with_hidden` (committed) captures 64 on `63 -> 64`;
    //    - `verify_step` at 92 (`92..96`) + `rollback(1, 3)` (`keep = 2 -> 94`) preserves 64;
    //    - `verify_step` at 94 (`94..98`) + `rollback(1, 3)` (`keep = 2 -> 96`) captures 96.
    advance_to(&mut inner, 20, 40, 63);
    assert_eq!(held_row(&inner, 20), None);
    inner.activate_paged_seq(20).unwrap();
    {
        let mut stepper = super::spec::Stepper::paged(&mut inner).unwrap();
        let emb = stepper.embedding().clone();
        assert!(stepper.reserve_cycle_lookahead(1).unwrap());
        let id = MxArray::from_int32(&[7], &[1]).unwrap();
        stepper.forward_with_hidden(&id, &emb).unwrap();
    }
    assert_eq!(inner.owner_len(20), 64);
    assert_eq!(
        held_row(&inner, 20),
        Some(64),
        "committed spec::Paged::forward_with_hidden must capture 64"
    );
    release(&mut inner, 20);

    advance_to(&mut inner, 21, 40, 92);
    assert_eq!(held_row(&inner, 21), Some(64));
    inner.activate_paged_seq(21).unwrap();
    {
        let mut stepper = super::spec::Stepper::paged(&mut inner).unwrap();
        let emb = stepper.embedding().clone();
        assert!(stepper.reserve_cycle_lookahead(4).unwrap());
        stepper.snapshot_main_linear();
        stepper.verify_step(&[3, 4, 5, 6], &emb, 3).unwrap();
        stepper.rollback(1, 3); // keep = 1 + 1 = 2 -> 94
        assert!(stepper.take_replay_error().is_none());
    }
    assert_eq!(inner.owner_len(21), 94);
    assert_eq!(
        held_row(&inner, 21),
        Some(64),
        "spec::Paged::verify_step + rollback(1, 3) at 92 must preserve held row 64"
    );
    {
        let mut stepper = super::spec::Stepper::paged(&mut inner).unwrap();
        let emb = stepper.embedding().clone();
        assert!(stepper.reserve_cycle_lookahead(4).unwrap());
        stepper.snapshot_main_linear();
        stepper.verify_step(&[7, 8, 9, 10], &emb, 3).unwrap();
        stepper.rollback(1, 3); // keep = 1 + 1 = 2 -> 96
        assert!(stepper.take_replay_error().is_none());
    }
    assert_eq!(inner.owner_len(21), 96);
    assert_eq!(
        held_row(&inner, 21),
        Some(96),
        "spec::Paged::verify_step + rollback(1, 3) at 94 -> 96 must capture held row 96"
    );
    release(&mut inner, 21);

    // 6. Non-speculative committed AR decode (`decode_rows`, no MTP loaded) still
    //    captures on `32k` (`63 -> 64`).
    let (mut ar_inner, _ar_tmp) = build_inner(false, 2);
    let prompt: Vec<u32> = (0..63u32).map(|i| 1 + (i % 60)).collect();
    ar_inner.activate_paged_seq(1).unwrap();
    let prefix = ar_inner
        .prime_prefix_state(&prompt, true, 32, &[], 0)
        .unwrap();
    ar_inner
        .paged_prefill(&prompt, &prefix, Stream::generation())
        .unwrap();
    assert_eq!(held_row(&ar_inner, 1), None);
    assert!(ar_inner.reserve_rows(1, 1).unwrap());
    ar_inner.run_paged_decode_step_batched(&[(1, 5)]).unwrap();
    assert_eq!(ar_inner.owner_len(1), 64);
    assert_eq!(
        held_row(&ar_inner, 1),
        Some(64),
        "committed AR decode_rows must still capture Turn state at 64"
    );
}

/// Pre-decode budget skips in `resume_from_disk` must not consume the 2-read
/// SSD attempt cap: when the two deepest disk-only candidates exceed the
/// remaining budget, a third shallower candidate that fits is still restored.
#[test]
fn resume_from_disk_budget_skips_do_not_exhaust_read_limit() {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use mlx_paged_attn::{ColdCacheFingerprint, ColdLayerBlock};

    use super::config::Attn;
    use super::indexer::{IndexerGeometry, PagedIndexerState, TokTail};
    use super::ladder::Kind;
    use super::owner::{CellLayer, OwnerCell};
    use super::paged::PageGeometry;
    use super::persist::{Blocks, Disk, MINIMUM_ROWS};
    use super::persist_codec::{StateShape, split_block};
    use super::persist_writer::Telemetry;
    use super::prefill::BLOCK_ROWS;
    use super::prefix_state::{Key, PrefixState, States, written_key};
    use super::sched::{RestoreBudget, resume_from_disk};
    use super::test_support::Temp;
    use crate::array::{DType, MxArray};
    use crate::disk_tier::{CacheDisk, DiskTierOptions, PersistentIdentity, open_disk_tier};

    // SAFETY: pure process-wide Metal availability probe.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }

    struct ZeroBlocks(PageGeometry);
    impl Blocks for ZeroBlocks {
        fn begin(&mut self, _: &[u32]) -> bool {
            true
        }
        fn block(&mut self, _: usize, _: u64) -> Option<Vec<ColdLayerBlock>> {
            let half = self.0.half as usize;
            let lat = vec![0u16; BLOCK_ROWS * 2 * half];
            let pool = vec![0u16; (self.0.index_rows * self.0.index_dim) as usize];
            Some(vec![split_block(&lat, &pool, half); self.0.layers as usize])
        }
    }

    let (mut inner, cfg, _model_tmp) = build_paged_inner(false, 1);
    let page = PageGeometry::of(&cfg).unwrap();
    let shape = StateShape::of(&cfg, false).unwrap();
    let root = Temp::new("sched-resume-budget-skip");
    let open_disk = || {
        let tmp = root.join("tmp");
        std::fs::create_dir_all(&tmp).unwrap();
        let opts = DiskTierOptions {
            cache_disk: CacheDisk::Bytes(2 << 30),
            persistent: true,
            cache_dir: Some(root.join("cache")),
            temp_root: Some(tmp),
        };
        let fp = ColdCacheFingerprint::from_components([b"hy4-resume-budget-skip".as_slice()]);
        let id = PersistentIdentity {
            namespace: "22223333444455556666777788889999".into(),
            state: fp,
            kv: fp,
        };
        let tier = open_disk_tier(&opts, || Some(id), shape.span_bytes()).unwrap();
        Disk::open(tier, Arc::new(Telemetry::default()), shape.clone(), page).unwrap()
    };

    let row_fit = MINIMUM_ROWS;
    let row_big1 = MINIMUM_ROWS + BLOCK_ROWS;
    let row_big2 = MINIMUM_ROWS + 2 * BLOCK_ROWS;
    let full: Vec<u32> = (0..row_big2 + 16).map(|i| 11 + (i % 251) as u32).collect();
    let keys: Vec<Vec<u64>> = (0..row_big2 / BLOCK_ROWS)
        .map(|b| written_key(&full, b, 77))
        .collect();

    let mk_state = |row: usize| -> PrefixState {
        let width = cfg.mla.kv_row();
        let il = cfg
            .trunk()
            .iter()
            .position(|l| l.attn == Attn::Indexed)
            .unwrap();
        let idx_g = IndexerGeometry::of(&cfg, il).unwrap();
        let mut cell = OwnerCell::new(&cfg, None).unwrap();
        let kv = MxArray::zeros(&[row as i64, width as i64], Some(DType::Float16)).unwrap();
        for part in cell.layers_mut() {
            match part {
                CellLayer::Window(ring) => {
                    ring.append(&kv, 0).unwrap();
                }
                CellLayer::Index(ix) => {
                    let held = (idx_g.kpool - 1 + WINDOW_SLACK).min(row);
                    let r =
                        MxArray::zeros(&[held as i64, idx_g.head_dim as i64], Some(DType::Float32))
                            .unwrap();
                    *ix = PagedIndexerState::at(
                        ix.layer(),
                        TokTail::of_rows(&idx_g, Some(r), row).unwrap(),
                    );
                }
            }
        }
        cell.advance(&full[..row]).unwrap();
        cell.state().unwrap().unwrap()
    };

    // Session 1: persist 3 disk states along the same lineage (2048, 2080, 2112).
    {
        let mut disk = open_disk();
        let mut st = States::<PrefixState>::with_cap(8);
        for (seq, row) in [(1, row_fit), (2, row_big1), (3, row_big2)] {
            let blocks = row / BLOCK_ROWS;
            let hot = vec![0u64; blocks];
            disk.start(seq, &full[..row], &keys[..blocks], &hot, 0);
            let k = Key {
                tokens: &full[..row],
                keys: &keys[..blocks],
                salt: 0,
            };
            let id = st.publish(k, mk_state(row), Kind::Prompt).unwrap().unwrap();
            disk.published(st.store(), seq, row, None, Some(id));
        }
        let ddl = Instant::now() + Duration::from_secs(30);
        assert!(disk.flush_points(st.store_mut(), &mut ZeroBlocks(page), ddl));
    }

    // Session 2: attach persistent disk to live `Paged` (adopts all 3 as disk-only),
    // set a budget where 2112 and 2080 fail `reserve_for_promote` while 2048 fits,
    // and verify `resume_from_disk` skips the two oversized entries and restores 2048.
    let p = inner.paged_mut().unwrap();
    assert_eq!(p.attach_disk(open_disk()).unwrap(), 3);
    let prompt = Key {
        tokens: &full,
        keys: &[],
        salt: 0,
    };
    let found = p.states.disk_only_above(prompt, 0);
    assert_eq!(
        found.iter().map(|&(_, r)| r).collect::<Vec<_>>(),
        vec![row_big2, row_big1, row_fit]
    );
    let (big2_id, big1_id, fit_id) = (found[0].0, found[1].0, found[2].0);

    let need = p.disk.as_ref().unwrap().state_bytes_at(row_fit);
    let mtp_row = 1_000u64;
    let budget = RestoreBudget {
        total: p.owners.bytes() + need + mtp_row * row_fit as u64,
        owner_base: 0,
        mtp_row,
    };
    let resumed = resume_from_disk(p, 1, prompt, None, budget);
    assert_eq!(resumed, Some((fit_id, row_fit)));
    assert!(
        p.states.get(fit_id).is_some(),
        "fitting 3rd candidate restored"
    );
    assert!(
        p.states.get(big1_id).is_none() && p.states.get(big2_id).is_none(),
        "oversized candidates stayed disk-only"
    );
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
    let (mut inner, _) = Inner::open(
        &hy4_dir(),
        Some(16_384),
        None,
        Default::default(),
        false,
        None,
    )
    .unwrap();
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
