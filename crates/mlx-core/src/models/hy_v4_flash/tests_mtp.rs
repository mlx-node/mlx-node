//! MTP head: held sizes against `mtp-tensors.json`, the prefill lag
//! schedule, a lens model of the stepper's lengths over whole cycles at
//! depth 1..3, the K/V buffer and seeds; real (`HY4_DIR`): the held layer
//! size and a greedy MTP turn against the AR turn.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

use super::cache::WINDOW_SLACK;
use super::mtp::{KvRows, Lag, SEEDS, Seeds, lag};
use super::mtp_weights::{Held, held, held_bytes};
use super::spec::{LOOKAHEAD_MAX, commit_row, cycle_rows, unemitted_keep, verify_keep};
use crate::array::MxArray;
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::mtp_decode::MtpCommitAnchor;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash/mtp-tensors.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn info(t: &Value) -> GgufTensorInfo {
    let dims: Vec<u64> = t["ggml_dims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap())
        .collect();
    GgufTensorInfo {
        name: t["name"].as_str().unwrap().to_string(),
        n_dims: dims.len() as u32,
        dims,
        tensor_type: match t["type"].as_str().unwrap() {
            "BF16" => GgufTensorType::BF16,
            "F32" => GgufTensorType::F32,
            other => panic!("unexpected type {other}"),
        },
        offset: 0,
    }
}

#[test]
fn held_bytes_match_the_fixture() {
    let f = fixture();
    let mut total = 0u64;
    let mut experts = 0u64;
    for t in f["tensors"].as_array().unwrap() {
        let expert = t["category"] == "routed_expert";
        let i = info(t);
        let want = t["plan_form"]["bytes"].as_u64().unwrap();
        assert_eq!(held_bytes(&i, expert).unwrap(), want, "{}", i.name);
        let form = match held(&i, expert).unwrap() {
            Held::Qmm => "qmm_affine8_g32_f32_scale_bias",
            Held::F32 => "f32",
            Held::Affine4 => "affine4_g64_bf16_scale_bias",
        };
        assert_eq!(t["plan_form"]["form"], form, "{}", i.name);
        total += want;
        if expert {
            experts += want;
        }
    }
    assert_eq!(total, f["totals"]["plan_bytes"].as_u64().unwrap());
    assert_eq!(
        experts,
        f["totals"]["routed_expert_plan_bytes"].as_u64().unwrap()
    );
}

#[test]
fn held_refuses_types_the_layer_does_not_use() {
    let mut i = info(&fixture()["tensors"][0]);
    i.tensor_type = GgufTensorType::Q8_0;
    assert!(held(&i, false).is_err());
    i.tensor_type = GgufTensorType::F32;
    assert!(held(&i, true).is_err());
}

/// Every MTP row `p` pairs trunk output `p` with token `p + 1`, exactly
/// once and in order, however the prompt is cut into chunks.
#[test]
fn lag_rows_pair_each_trunk_row_with_the_next_token() {
    assert_eq!(
        lag(0, 5),
        Lag {
            at: 0,
            seed: false,
            rows: 4
        }
    );
    assert_eq!(lag(0, 1).rows, 0);
    assert_eq!(
        lag(8, 3),
        Lag {
            at: 7,
            seed: true,
            rows: 3
        }
    );
    for cuts in [
        vec![1, 1, 1, 1],
        vec![5],
        vec![1, 4, 2048, 3],
        vec![2048, 2048, 7],
        vec![32, 1, 31, 64],
    ] {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        for n in &cuts {
            let l = lag(start, *n);
            assert_eq!(l.at, pairs.len(), "cuts {cuts:?}");
            let mut hidden: Vec<usize> = Vec::new();
            if l.seed {
                hidden.push(start - 1);
            }
            hidden.extend(start..start + n - 1);
            let ids: Vec<usize> = (start..start + n).skip(n - l.rows).collect();
            assert_eq!(hidden.len(), l.rows);
            pairs.extend(hidden.into_iter().zip(ids));
            start += n;
        }
        let want: Vec<(usize, usize)> = (0..start - 1).map(|p| (p, p + 1)).collect();
        assert_eq!(pairs, want, "cuts {cuts:?}");
    }
}

/// A draft row: `Some((hidden position, token position))` for an exact
/// pair, `None` for a draft.
type Row = Option<(usize, usize)>;

/// The stepper's lengths through `run_mtp_cycle`'s calls.
struct Lens {
    trunk: usize,
    rows: Vec<Row>,
    committed: usize,
    snap: usize,
}

impl Lens {
    /// A settled sequence after a prompt of `t` tokens.
    fn new(t: usize) -> Self {
        Self {
            trunk: t,
            rows: (0..t - 1).map(|p| Some((p, p + 1))).collect(),
            committed: t - 1,
            snap: 0,
        }
    }

    fn step_a(&mut self) {
        assert_eq!(self.rows.len(), self.committed);
        if self.rows.len() + 1 == self.trunk {
            self.rows.push(Some((self.trunk - 1, self.trunk)));
        }
        assert_eq!(self.rows.len(), self.trunk);
        self.trunk += 1;
        self.committed = self.rows.len();
    }

    fn cut(&mut self, keep: usize) {
        assert!(keep <= self.rows.len());
        assert!(self.rows.len() - keep <= WINDOW_SLACK);
        self.rows.truncate(keep);
    }

    fn cycle(&mut self, depth: usize, accepted: usize, chained: bool) {
        assert!(accepted <= depth);
        self.cut(cycle_rows(self.committed, chained).unwrap());
        assert_eq!(
            self.rows.len() + 1,
            self.trunk,
            "first draft row is trunk - 1"
        );
        self.rows.push(Some((self.trunk - 1, self.trunk)));
        for _ in 1..depth {
            self.rows.push(None);
        }
        self.snap = self.trunk;
        assert!(depth < LOOKAHEAD_MAX);
        self.trunk += depth + 1;
        let anchor = if chained {
            MtpCommitAnchor::SkipAlreadyCommittedAnchor
        } else {
            MtpCommitAnchor::IncludeAnchor
        };
        let m = match anchor {
            MtpCommitAnchor::IncludeAnchor => accepted + 2,
            MtpCommitAnchor::SkipAlreadyCommittedAnchor => accepted + 1,
        };
        let at = commit_row(anchor, self.snap).unwrap();
        assert_eq!(at, self.committed);
        self.cut(at);
        // the hiddens of rows at.. are the seed (Step A) or verify rows
        for i in 0..m {
            self.rows.push(Some((at + i, at + i + 1)));
        }
        self.committed = self.rows.len();
        let keep = verify_keep(self.snap, accepted);
        assert!(self.trunk - keep <= WINDOW_SLACK);
        self.trunk = keep;
        self.check(self.trunk);
    }

    fn unemitted(&mut self, u: usize) {
        self.trunk = unemitted_keep(self.trunk, u);
        let keep = self.rows.len().min(self.trunk - 1);
        self.cut(keep);
        self.committed = keep;
        self.check(self.trunk - 1);
    }

    fn settle(&mut self) {
        let keep = self.rows.len().min(self.trunk - 1);
        self.rows.truncate(keep);
        self.check(self.trunk - 1);
    }

    /// `len` rows, each the exact pair of its own position.
    fn check(&self, len: usize) {
        assert_eq!(self.rows.len(), len);
        for (p, r) in self.rows.iter().enumerate() {
            assert_eq!(*r, Some((p, p + 1)), "row {p} of {:?}", self.rows);
        }
    }
}

/// Step-A and chained cycles at depth 1..3 with every accept count, a
/// near-tail skip (two Step A in a row) and a mid-cycle stop: the draft
/// rows stay the exact pairs of their positions and every cut stays
/// within the slack.
#[test]
fn lens_model_keeps_exact_rows_through_cycles() {
    for depth in 1..=3 {
        for chained in [false, true] {
            let mut l = Lens::new(37);
            l.step_a();
            for k in 0..=depth {
                l.cycle(depth, k, false);
                if chained {
                    l.cycle(depth, depth - k, true);
                } else {
                    l.step_a();
                }
                l.step_a();
            }
            for u in 1..=depth {
                let mut stop = Lens::new(20);
                stop.step_a();
                stop.cycle(depth, depth, false);
                stop.unemitted(u);
            }
            l.settle();
        }
    }
}

#[test]
fn trunk_cuts_fit_the_slack() {
    for depth in 1..=3 {
        for k in 0..=depth {
            let snap = 100;
            assert!(snap + depth + 1 - verify_keep(snap, k) <= WINDOW_SLACK);
        }
        for u in 1..=depth {
            assert!(100 - unemitted_keep(100, u) <= WINDOW_SLACK);
        }
    }
    assert_eq!(cycle_rows(0, true), None);
    assert_eq!(commit_row(MtpCommitAnchor::IncludeAnchor, 0), None);
}

fn rows(n: usize, from: usize) -> MxArray {
    let v: Vec<f32> = (from * 4..(from + n) * 4).map(|x| x as f32).collect();
    MxArray::from_float32(&v, &[n as i64, 4]).unwrap()
}

fn read(x: &MxArray, n: usize) -> Vec<f32> {
    x.slice(&[0, 0], &[n as i64, 4])
        .unwrap()
        .astype(crate::array::DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}

/// Writes past the buffer move the rows to one of twice the rows (a
/// multiple of 32, at most the cap) and read back in order; a draft read is the buffer itself; a
/// rollback rewrites in place; a deep cut shrinks it and a cut to 0 frees it.
#[test]
fn kv_rows_grow_cut_and_rewrite() {
    let mut b = KvRows::new(4, 300);
    let mut caps = Vec::new();
    let mut at = 0;
    for n in [1, 31, 40, 5, 200] {
        b.write(&rows(n, at), at).unwrap();
        at += n;
        caps.push(b.capacity());
    }
    assert_eq!(caps, vec![32, 32, 96, 96, 288]);
    assert_eq!(b.len(), at);
    assert_eq!(read(b.rows().unwrap(), at), read(&rows(at, 0), at));
    assert!(b.write(&rows(24, at), at).is_err());
    assert_eq!(b.len(), at);

    b.truncate(at - 3).unwrap();
    assert_eq!(b.capacity(), 288);
    b.write(&rows(3, at - 3), at - 3).unwrap();
    assert_eq!(read(b.rows().unwrap(), at), read(&rows(at, 0), at));

    b.truncate(70).unwrap();
    assert_eq!(b.capacity(), 288);
    b.truncate(40).unwrap();
    assert_eq!(b.capacity(), 128);
    assert_eq!(b.rows().unwrap().shape_at(0).unwrap(), 128);
    b.write(&rows(3, 40), 40).unwrap();
    assert_eq!(read(b.rows().unwrap(), 43), read(&rows(43, 0), 43));
    b.truncate(0).unwrap();
    assert_eq!((b.capacity(), b.nbytes()), (0, 0));
    assert!(b.rows().is_err());
    assert!(b.write(&rows(1, 0), 1).is_err());
}

#[test]
fn seeds_keep_the_last_rows_by_position() {
    let h = rows(12, 0);
    let mut s = Seeds::default();
    s.extend(Seeds::copies(&h, 100).unwrap());
    assert!(s.get(100 + 12 - SEEDS - 1).is_none());
    assert_eq!(read(s.get(111).unwrap(), 1), read(&rows(1, 11), 1));
    s.cut(110);
    assert!(s.get(110).is_none() && s.get(109).is_some());
    s.extend(Seeds::copies(&rows(1, 50), 110).unwrap());
    assert_eq!(read(s.get(110).unwrap(), 1), read(&rows(1, 50), 1));
}

#[test]
fn kv_rows_and_seeds_defer_in_scope_and_stay_bit_identical_after_rollback() {
    use super::stats::{self, DeferScope};

    let kv_bits =
        |x: &MxArray, n: usize| -> Vec<u32> { read(x, n).into_iter().map(f32::to_bits).collect() };

    let mut kv = KvRows::new(4, 128);
    // Initial write outside DeferScope performs the synchronous buffer resize.
    kv.write(&rows(4, 0), 0).unwrap();

    let mut seeds = Seeds::default();
    let want_kv = kv_bits(&rows(7, 0), 7);
    let want_seed_10 = kv_bits(&rows(1, 10), 1);
    let want_seed_11 = kv_bits(&rows(1, 11), 1);

    {
        let _scope = DeferScope::enter();
        assert_eq!(stats::deferred(), 0);
        let evals_before = stats::thread_evals();

        kv.write(&rows(3, 4), 4).unwrap();
        assert_eq!(stats::deferred(), 1);
        assert_eq!(stats::thread_evals(), evals_before);

        let copied = Seeds::copies(&rows(4, 8), 8).unwrap();
        assert_eq!(copied.len(), 4);
        assert_eq!(stats::deferred(), 1 + copied.len());
        assert_eq!(stats::thread_evals(), evals_before);
        seeds.extend(copied);

        stats::flush("test mtp defer flush").unwrap();
        assert_eq!(stats::deferred(), 0);
        assert_eq!(stats::thread_evals(), evals_before + 1);
    }

    assert_eq!(kv_bits(kv.rows().unwrap(), 7), want_kv);
    assert_eq!(kv_bits(seeds.get(10).unwrap(), 1), want_seed_10);
    assert_eq!(kv_bits(seeds.get(11).unwrap(), 1), want_seed_11);

    // An unflushed scope clears deferred arrays on drop; rollback + rewrite stays bit-identical.
    {
        let _scope = DeferScope::enter();
        kv.write(&rows(2, 7).mul_scalar(99.0).unwrap(), 7).unwrap();
        assert_eq!(stats::deferred(), 1);
        kv.truncate(4).unwrap();
    }
    assert_eq!(stats::deferred(), 0);
    seeds.cut(10);

    {
        let _scope = DeferScope::enter();
        let evals_before = stats::thread_evals();
        kv.write(&rows(3, 4), 4).unwrap();
        let copied = Seeds::copies(&rows(2, 10), 10).unwrap();
        assert_eq!(stats::deferred(), 1 + copied.len());
        assert_eq!(stats::thread_evals(), evals_before);
        seeds.extend(copied);
        stats::flush("test mtp rollback rewrite").unwrap();
        assert_eq!(stats::deferred(), 0);
        assert_eq!(stats::thread_evals(), evals_before + 1);
    }

    assert_eq!(kv_bits(kv.rows().unwrap(), 7), want_kv);
    assert_eq!(kv_bits(seeds.get(10).unwrap(), 1), want_seed_10);
    assert_eq!(kv_bits(seeds.get(11).unwrap(), 1), want_seed_11);
}

#[test]
fn mtp_draft_batch_matches_serial_multi_lane_multi_depth() {
    use super::config::{
        Attn, HcConfig, HyV4FlashConfig, IndexerConfig, LayerConfig, MlaConfig, MoeConfig,
    };
    use super::model::{Model, TrunkWeights};
    use super::moe::{AffineExperts, AffineProj};
    use super::mtp::{Draft, MtpLayer};
    use super::mtp_weights::{EXPERT_BITS, EXPERT_GROUP, MtpWeights};
    use super::rows::ResidentF8Table;
    use super::stats::{self, DeferScope};
    use super::weights::{
        AttnWeights, Dense, Ffn, HcWeights, IndexerWeights, LayerWeights, MoeWeights, NextnWeights,
        Residual,
    };
    use crate::array::DType;
    use crate::cache_limit::coordinator;
    use crate::models::deepseek_v41::EngramConfig;

    crate::models::hy_v4_flash::exact_f32();

    const V: usize = 96;
    const D: usize = 32;
    const HC: usize = 4;
    const Q_LORA: usize = 16;
    const KV_LORA: usize = 16;
    const NOPE: usize = 8;
    const ROPE: usize = 8;
    const HEAD_V: usize = 12;
    const N_HEADS: usize = 2;
    const IDX_HEADS: usize = 2;
    const IDX_DIM: usize = 16;
    const DENSE_FF: usize = 40;
    const SHARED_FF: usize = 24;
    const E: usize = 8;
    const MOE_FF: usize = 128;
    const MOE_LATENT: usize = 64;
    const CTX: usize = 64;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn step(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next(&mut self) -> f32 {
            ((self.step() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale).collect()
        }
        fn ids(&mut self, n: usize) -> Vec<u32> {
            (0..n).map(|_| (self.step() % V as u64) as u32).collect()
        }
    }
    fn arr(v: &[f32], shape: &[usize]) -> MxArray {
        let s: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        MxArray::from_float32(v, &s).unwrap()
    }
    fn mat(r: &mut Rng, rows: usize, cols: usize) -> Dense {
        Dense::Array(arr(
            &r.vec(rows * cols, 1.5 / (cols as f32).sqrt()),
            &[rows, cols],
        ))
    }
    fn norm(r: &mut Rng, n: usize) -> MxArray {
        let v: Vec<f32> = r.vec(n, 0.4).into_iter().map(|x| 1.0 + x).collect();
        arr(&v, &[n])
    }
    fn small(r: &mut Rng, shape: &[usize], scale: f32) -> MxArray {
        arr(&r.vec(shape.iter().product(), scale), shape)
    }
    fn hc_w(r: &mut Rng, rows: usize, scales: usize) -> HcWeights {
        HcWeights {
            weight: small(r, &[rows, HC * D], 0.5 / ((HC * D) as f32).sqrt()),
            base: small(r, &[rows], 0.5),
            scale: small(r, &[scales], 1.0),
        }
    }
    fn attn_w(r: &mut Rng) -> AttnWeights {
        AttnWeights {
            norm: norm(r, D),
            sinks: small(r, &[N_HEADS], 1.0),
            q_a: mat(r, Q_LORA, D),
            q_a_norm: norm(r, Q_LORA),
            q_b: mat(r, N_HEADS * (NOPE + ROPE), Q_LORA),
            kv_a: mat(r, KV_LORA + ROPE, D),
            kv_a_norm: norm(r, KV_LORA),
            k_b: mat(r, N_HEADS * KV_LORA, NOPE),
            v_b: mat(r, N_HEADS * HEAD_V, KV_LORA),
            output: mat(r, D, N_HEADS * HEAD_V),
            gate: mat(r, N_HEADS * HEAD_V, D),
            post_norm: norm(r, D),
            q_kv: None,
            q_kv_gate: None,
        }
    }
    fn idx_w(r: &mut Rng) -> IndexerWeights {
        IndexerWeights {
            attn_k: mat(r, IDX_DIM, D),
            k_norm: norm(r, IDX_DIM),
            q_b: mat(r, IDX_HEADS * IDX_DIM, Q_LORA),
            proj: mat(r, IDX_HEADS, D),
            compressor_value: small(r, &[IDX_DIM, IDX_DIM], 0.25),
            compressor_gate: small(r, &[IDX_DIM, IDX_DIM], 0.25),
            compressor_ape: small(r, &[4, IDX_DIM], 0.5),
            compressor_norm: norm(r, IDX_DIM),
        }
    }

    struct MiniTrunk {
        token_embd: Dense,
        output: Dense,
        output_norm: MxArray,
        output_hc: HcWeights,
        layers: Vec<LayerWeights>,
    }
    impl TrunkWeights for MiniTrunk {
        fn token_embd(&self) -> &Dense {
            &self.token_embd
        }
        fn output(&self) -> &Dense {
            &self.output
        }
        fn output_norm(&self) -> &MxArray {
            &self.output_norm
        }
        fn output_hc(&self) -> &HcWeights {
            &self.output_hc
        }
        fn layers(&self) -> &[LayerWeights] {
            &self.layers
        }
    }

    let token_map: Vec<u32> = (0..V as u32).map(|i| (i * 7) % 61).collect();
    let engram = EngramConfig {
        layers: vec![1],
        max_ngram: 3,
        n_heads: 2,
        head_dim: 8,
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15 | 1,
            0xBF58_476D_1CE4_E5B9 | 1,
            0x94D0_49BB_1331_11EB | 1,
        ],
        head_offsets: vec![0, 11, 24, 31],
        head_vocab_sizes: vec![11, 13, 7, 5],
        pad_token_id: V - 1,
        compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
        token_map,
        vocab_size: None,
    };
    let cfg = HyV4FlashConfig {
        n_layers: 1,
        layers: vec![
            LayerConfig {
                attn: Attn::Sliding,
                n_heads: N_HEADS,
                rope_theta: 1e4,
                swiglu_limit: 0.5,
                dense_ffn: true,
                engram: None,
                mtp: false,
            },
            LayerConfig {
                attn: Attn::Indexed,
                n_heads: N_HEADS,
                rope_theta: 1e7,
                swiglu_limit: 0.5,
                dense_ffn: false,
                engram: None,
                mtp: true,
            },
        ],
        vocab_size: V,
        context_length: CTX,
        dim: D,
        dense_ff: DENSE_FF,
        n_dense_lead: 1,
        norm_eps: 1e-5,
        window: 8,
        rope_theta: 1e7,
        rope_theta_swa: 1e4,
        mla: MlaConfig {
            q_lora_rank: Q_LORA,
            kv_lora_rank: KV_LORA,
            qk_nope_dim: NOPE,
            rope_dim: ROPE,
            head_dim_qk: NOPE + ROPE,
            head_dim_v: HEAD_V,
            scale: ((NOPE + ROPE) as f32).sqrt().recip(),
        },
        moe: MoeConfig {
            n_experts: E,
            n_used: 2,
            n_shared: 1,
            ff: MOE_FF,
            latent: MOE_LATENT,
            route_scale: 2.0,
            norm_topk: true,
        },
        hc: HcConfig {
            count: HC,
            eps: 1e-6,
            magnitude: 2.0,
        },
        indexer: IndexerConfig {
            n_heads: IDX_HEADS,
            head_dim: IDX_DIM,
            top_k: 16,
            kpool: 4,
            select_tail: true,
        },
        engram: engram.clone(),
    };

    let mut r = Rng::new(777);
    let trunk_layer = LayerWeights {
        attn: attn_w(&mut r),
        indexer: None,
        engram: None,
        residual: Residual::Hc {
            attn: hc_w(&mut r, 2 * HC, 2),
            ffn: hc_w(&mut r, 2 * HC, 2),
        },
        ffn_norm: norm(&mut r, D),
        post_ffw_norm: norm(&mut r, D),
        ffn: Ffn::Mlp {
            gate: mat(&mut r, DENSE_FF, D),
            up: mat(&mut r, DENSE_FF, D),
            down: mat(&mut r, D, DENSE_FF),
            gate_up: None,
        },
    };
    let output_bf16 = arr(&r.vec(V * D, 1.5 / (D as f32).sqrt()), &[V, D])
        .astype(DType::BFloat16)
        .unwrap();
    let trunk = MiniTrunk {
        token_embd: mat(&mut r, V, D),
        output: Dense::Array(output_bf16),
        output_norm: norm(&mut r, D),
        output_hc: hc_w(&mut r, HC, 1),
        layers: vec![trunk_layer],
    };
    let table_rows: u64 = engram.head_vocab_sizes.iter().sum();
    let table =
        ResidentF8Table::new(vec![0u8; table_rows as usize * engram.head_dim], &engram).unwrap();
    let model = Model::new(
        cfg.clone(),
        Box::new(trunk),
        |_| Err(napi::Error::from_reason("unexpected trunk MoE")),
        Box::new(table),
    )
    .unwrap();

    let mtp_lw = LayerWeights {
        attn: attn_w(&mut r),
        indexer: Some(idx_w(&mut r)),
        engram: None,
        residual: Residual::Mtp(NextnWeights {
            eh_proj: mat(&mut r, D, 2 * D),
            enorm: norm(&mut r, D),
            hnorm: norm(&mut r, D),
            shared_head_norm: norm(&mut r, D),
        }),
        ffn_norm: norm(&mut r, D),
        post_ffw_norm: norm(&mut r, D),
        ffn: Ffn::Moe(MoeWeights {
            router: mat(&mut r, E, D),
            router_bias: Some(small(&mut r, &[E], 0.1)),
            routed_down: mat(&mut r, MOE_LATENT, D),
            routed_up: mat(&mut r, D, MOE_LATENT),
            shared_gate: mat(&mut r, SHARED_FF, D),
            shared_up: mat(&mut r, SHARED_FF, D),
            shared_down: mat(&mut r, D, SHARED_FF),
            down_shared_gu: None,
        }),
    };
    let quant_exp = |r: &mut Rng, rows: usize, k: usize| -> AffineProj {
        let raw = arr(&r.vec(E * rows * k, 1.0 / (k as f32).sqrt()), &[E, rows, k])
            .astype(DType::BFloat16)
            .unwrap();
        let p = AffineProj::quantize(&raw, EXPERT_GROUP, EXPERT_BITS).unwrap();
        MxArray::eval_arrays_with_context(&[&p.w, &p.s, &p.b], "test mtp exp").unwrap();
        p
    };
    let experts = AffineExperts::new(
        [
            quant_exp(&mut r, MOE_FF, MOE_LATENT),
            quant_exp(&mut r, MOE_FF, MOE_LATENT),
            quant_exp(&mut r, MOE_LATENT, MOE_FF),
        ],
        EXPERT_GROUP,
        EXPERT_BITS,
    )
    .unwrap();
    let mw = MtpWeights {
        layer: 1,
        weights: mtp_lw,
        experts,
        read: 0,
        bytes: 0,
        guard: coordinator().register(0),
    };
    let mtp = MtpLayer::new(&cfg, mw, model.weights().token_embd()).unwrap();

    // Test N = 2 and N = 4 lanes at uniform depth 1, 2, 3 and heterogeneous depths [3, 2, 1, 3].
    let scenarios: &[(&str, &[usize], &[usize])] = &[
        ("n2_d1", &[6, 11], &[1, 1]),
        ("n2_d2", &[6, 11], &[2, 2]),
        ("n2_d3", &[6, 11], &[3, 3]),
        ("n4_d1", &[5, 9, 13, 7], &[1, 1, 1, 1]),
        ("n4_d2", &[5, 9, 13, 7], &[2, 2, 2, 2]),
        ("n4_d3", &[5, 9, 13, 7], &[3, 3, 3, 3]),
        ("n4_hetero", &[5, 9, 13, 7], &[3, 2, 1, 3]),
    ];

    for &(label, hist_lens, depths) in scenarios {
        let n = hist_lens.len();
        let mut serial_drafts: Vec<Draft> = (0..n).map(|_| mtp.new_draft(CTX)).collect();
        let mut batch_drafts: Vec<Draft> = (0..n).map(|_| mtp.new_draft(CTX)).collect();
        let mut seeds_h: Vec<MxArray> = Vec::with_capacity(n);
        let mut anchors: Vec<u32> = Vec::with_capacity(n);

        for i in 0..n {
            let hist_h = arr(&r.vec(hist_lens[i] * D, 1.0), &[hist_lens[i], D]);
            let hist_ids = r.ids(hist_lens[i]);
            mtp.commit(&model, &mut serial_drafts[i], &hist_h, &hist_ids)
                .unwrap();
            mtp.commit(&model, &mut batch_drafts[i], &hist_h, &hist_ids)
                .unwrap();
            seeds_h.push(arr(&r.vec(D, 1.0), &[1, D]));
            anchors.push(r.ids(1)[0]);
        }

        // Serial baseline via MtpLayer::draft.
        let mut serial_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n {
            let mut h = seeds_h[i].clone();
            let mut tok = anchors[i];
            for _ in 0..depths[i] {
                let idx = MxArray::from_uint32(&[tok], &[1]).unwrap();
                let emb = model.weights().token_embd().rows(&idx).unwrap();
                let (h_next, logits) = mtp.draft(&model, &mut serial_drafts[i], &h, &emb).unwrap();
                let tid = logits.argmax(-1, Some(false)).unwrap();
                tid.eval();
                tok = tid.to_int32().unwrap()[0] as u32;
                serial_tokens[i].push(tok);
                h = h_next.reshape(&[1, D as i64]).unwrap();
            }
        }

        // Batched path via MtpLayer::draft_batch with lazy GPU token chaining and 1 eval.
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        let mut batch_tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut h_states: Vec<MxArray> = seeds_h.clone();
        let mut tok_states: Vec<MxArray> = anchors
            .iter()
            .map(|&t| MxArray::from_uint32(&[t], &[1]).unwrap())
            .collect();
        let mut step_toks: Vec<(Vec<usize>, MxArray)> = Vec::with_capacity(max_depth);

        let evals_before = stats::thread_evals();
        {
            let _scope = DeferScope::enter();
            for step in 0..max_depth {
                let active: Vec<usize> = (0..n).filter(|&i| step < depths[i]).collect();
                let h_refs: Vec<&MxArray> = active.iter().map(|&i| &h_states[i]).collect();
                let t_refs: Vec<&MxArray> = active.iter().map(|&i| &tok_states[i]).collect();
                let h_batch = if h_refs.len() == 1 {
                    h_refs[0].clone()
                } else {
                    MxArray::concatenate_many(h_refs, Some(0)).unwrap()
                };
                let t_batch = if t_refs.len() == 1 {
                    t_refs[0].clone()
                } else {
                    MxArray::concatenate_many(t_refs, Some(0)).unwrap()
                };
                let e_batch = model.weights().token_embd().rows(&t_batch).unwrap();
                let mut lane_ptrs: Vec<&mut Draft> = batch_drafts
                    .iter_mut()
                    .enumerate()
                    .filter(|&(i, _)| step < depths[i])
                    .map(|(_, d)| d)
                    .collect();
                let (h_next, logits) = mtp
                    .draft_batch(&model, &mut lane_ptrs, &h_batch, &e_batch)
                    .unwrap();
                let next_ids = logits.argmax(-1, Some(false)).unwrap();
                for (k, &lane) in active.iter().enumerate() {
                    let lo = k as i64;
                    let hi = lo + 1;
                    h_states[lane] = h_next.slice(&[lo, 0], &[hi, D as i64]).unwrap();
                    tok_states[lane] = next_ids.slice(&[lo], &[hi]).unwrap();
                }
                step_toks.push((active, next_ids));
            }
            let eval_refs: Vec<&MxArray> = step_toks.iter().map(|(_, t)| t).collect();
            stats::eval(&eval_refs, "test mtp draft_batch").unwrap();
        }
        let evals_after = stats::thread_evals();
        assert_eq!(
            evals_after - evals_before,
            1,
            "{label}: expected 1 single eval across all {max_depth} draft steps, got {}",
            evals_after - evals_before
        );

        for (active, ids_arr) in &step_toks {
            let vals = ids_arr.to_int32().unwrap();
            for (k, &lane) in active.iter().enumerate() {
                batch_tokens[lane].push(vals[k] as u32);
            }
        }

        assert_eq!(
            batch_tokens, serial_tokens,
            "{label}: draft_batch tokens differ from serial draft"
        );
    }
}

fn hy4_dir() -> PathBuf {
    PathBuf::from(std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards"))
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_held_layer_matches_the_fixture() {
    let paths = crate::models::deepseek_v41::chat::shard_paths(&hy4_dir()).unwrap();
    let opts = super::store::Options::from_env(Some(16_384), true).unwrap();
    let store = super::store::Store::open(&paths, &opts).unwrap();
    let m = super::mtp::MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    assert_eq!(
        m.bytes(),
        fixture()["totals"]["plan_bytes"].as_u64().unwrap()
    );
}

fn chat_msg(role: &str, content: &str) -> crate::tokenizer::ChatMessage {
    serde_json::from_value(serde_json::json!({"role": role, "content": content})).unwrap()
}

/// Greedy MTP at depth 1..3 gives the AR tokens; cycles run.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_greedy_mtp_matches_ar() {
    use crate::engine::backend::{ChatBackend, ResetScope};
    let mut inner = super::chat::Inner::load(&hy4_dir(), Some(16_384), Default::default()).unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let msgs = || vec![chat_msg("user", "Count from 1 to 40, comma separated.")];
    let cfg = |mtp: bool, depth: i32| ChatConfig {
        max_new_tokens: Some(96),
        temperature: Some(0.0),
        enable_mtp: Some(mtp),
        mtp_depth: Some(depth),
        report_performance: Some(true),
        ..Default::default()
    };
    let ar =
        crate::engine::session::session_start(&mut inner, msgs(), cfg(false, 1), &flag).unwrap();
    for depth in 1..=3 {
        inner.reset_caches(ResetScope::Command).unwrap();
        let r = crate::engine::session::session_start(&mut inner, msgs(), cfg(true, depth), &flag)
            .unwrap();
        println!(
            "depth {depth}: {:?} perf {:?}",
            r.text,
            r.performance
                .as_ref()
                .map(|p| (p.mtp_cycles, p.mtp_mean_accepted_tokens))
        );
        assert_eq!(r.text, ar.text, "depth {depth}");
        assert!(
            r.performance
                .as_ref()
                .and_then(|p| p.mtp_cycles)
                .is_some_and(|c| c > 0)
        );
    }
}
