//! Manual benchmark of the K-quant MoE expert path on the Qwen3.6-35B-A3B
//! (`[E=256, N=512, K=2048]` gate/up, `[256, 2048, 512]` down, top-8) and
//! LFM2.5-8B-A1B (`[32, 1792, 2048]` / `[32, 2048, 1792]`) expert stacks,
//! q4k and q5k. Three routes for one MoE layer's expert matmuls of one
//! routing, each projection (gate, up, down) timed on its own:
//!
//! * A, today: row-major `mlx_gather_qmm` exactly as
//!   `QuantizedSwitchLinear::forward` calls it (`[B,1,K]` x, `[B]` rhs
//!   indices, `sorted` once `B >= 64` as `SwitchGLU` does): one dispatch
//!   (`gather_qmv` / `gather_qmm_rhs_nax` / `gather_qmm_rhs`).
//! * A0, route A pinned to the simdgroup fallback (test FFI): the simdgroup
//!   `gather_qmm_rhs` (bm 16, bn 32) the sorted route took before the
//!   tensor-op kernel; the same as A wherever `gather_qmv` is chosen.
//! * B, per-expert tiled: every expert's `[N,K]` slab in the `@t64` layout;
//!   one `mlx_quantized_matmul` per distinct expert over its gathered rows
//!   padded to 8 (`qmm_m8_nax_t64`), or `qmv_t64` when it has one row.
//!   D distinct experts -> D (or more, 8-row chunks) dispatches.
//! * C, floor: one dense `@t64` matmul over a `[D*N, K]` weight (what a
//!   grouped tile-descriptor kernel could reach), and the touched bytes at
//!   `FLOOR_GBPS`.
//!
//! Workloads: AR decode (8 routes, 8 experts, 1 row each), DFlash/MTP verify
//! (64 routes, ~45-55 distinct experts of 256; all 32 x 2 rows for LFM2),
//! prefill chunk (`PREFILL_TOKENS` x 8 routes). Samples interleave the routes
//! in one process; statistic = median over `REPS` samples of a `BATCH`-layer
//! eval. Each layer of a batch reads the next copy of a >= `RING_BYTES` ring
//! of expert stacks with its own routing, so the weights come from DRAM.
//! Routes A0 and B are also compared numerically with route A on one
//! routing. Run only on an idle GPU:
//!
//! ```text
//! MLX_KQUANT_MOE_BENCH=1 cargo test -p mlx-core --release \
//!   --test kquant_moe_bench -- --ignored --nocapture
//! ```
//!
//! `MLX_KQUANT_MOE_BENCH_MODES=q4k`, `_SHAPES=qwen|lfm2`, `_REPS=11`,
//! `_WORKLOADS=decode,verify,prefill`, `_PREFILL_TOKENS=128` narrow the run.

mod kquant_support;

use std::cell::Cell;
use std::ffi::CString;
use std::time::Instant;

use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::{KQUANT_TILE_ROWS, KQUANT_TILED_SUFFIX, kquant_tile_rows};

const WARMUP: usize = 3;
/// `MLX_KQUANT_MOE_BENCH_REPS` overrides (odd, >= 3).
const REPS: usize = 11;
/// Tokens of the prefill workload; `MLX_KQUANT_MOE_BENCH_PREFILL_TOKENS`
/// overrides (128+ puts the 256-expert stack on the sorted rhs route).
const PREFILL_TOKENS: usize = 64;
/// Layers (one projection each) per eval.
const BATCH: usize = 8;
const RING_BYTES: usize = 384 << 20;
/// Rows per route-B / Splash-style expert tile.
const TILE_ROWS: usize = 8;
/// Routes A, A0, B, C.
const ARMS: usize = 4;
const ARM_NAMES: [&str; ARMS] = ["A", "A0", "B", "C"];
const FLOOR_GBPS: f64 = 480.0;
const MODES: [&str; 2] = ["q4k", "q5k"];

struct MoeShape {
    name: &'static str,
    key: &'static str,
    experts: usize,
    hidden: i64,
    inter: i64,
}

const SHAPES: [MoeShape; 2] = [
    MoeShape {
        name: "Qwen3.6-35B-A3B",
        key: "qwen",
        experts: 256,
        hidden: 2048,
        inter: 512,
    },
    MoeShape {
        name: "LFM2.5-8B-A1B",
        key: "lfm2",
        experts: 32,
        hidden: 2048,
        inter: 1792,
    },
];

#[derive(Clone, Copy, PartialEq)]
enum Workload {
    Decode,
    Verify,
    Prefill,
}

impl Workload {
    fn name(self) -> String {
        match self {
            Self::Decode => "decode 1x8".to_string(),
            Self::Verify => "verify 8x8".to_string(),
            Self::Prefill => format!("prefill {}x8", self.tokens()),
        }
    }
    fn key(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::Prefill => "prefill",
        }
    }
    fn tokens(self) -> usize {
        match self {
            Self::Decode => 1,
            Self::Verify => 8,
            Self::Prefill => env_usize("MLX_KQUANT_MOE_BENCH_PREFILL_TOKENS", PREFILL_TOKENS),
        }
    }
    /// Routings per workload: enough that (routing, ring copy) pairs recur
    /// only after >= RING_BYTES of other expert traffic.
    fn routings(self) -> usize {
        match self {
            Self::Decode => 32,
            Self::Verify => 16,
            Self::Prefill => 8,
        }
    }
}

const TOP_K: usize = 8;

// ---------------------------------------------------------------- routing

fn lcg(st: &mut u32) -> u32 {
    *st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *st
}

/// One routing: `routes[b]` is the expert of route b (sorted ascending when
/// `sorted`, as `gather_sort` leaves them); `groups` lists each distinct
/// expert with the route indices that hit it.
struct Routing {
    routes: Vec<u32>,
    groups: Vec<(u32, Vec<usize>)>,
    sorted: bool,
}

impl Routing {
    fn new(mut routes: Vec<u32>) -> Self {
        let sorted = routes.len() >= 64;
        if sorted {
            routes.sort_unstable();
        }
        let mut groups: Vec<(u32, Vec<usize>)> = Vec::new();
        for (b, &e) in routes.iter().enumerate() {
            match groups.iter_mut().find(|(g, _)| *g == e) {
                Some((_, rows)) => rows.push(b),
                None => groups.push((e, vec![b])),
            }
        }
        groups.sort_by_key(|(e, _)| *e);
        Self {
            routes,
            groups,
            sorted,
        }
    }

    fn distinct(&self) -> usize {
        self.groups.len()
    }

    /// Route-B dispatches: 8-row chunks per expert.
    fn chunks(&self) -> usize {
        self.groups
            .iter()
            .map(|(_, rows)| rows.len().div_ceil(TILE_ROWS))
            .sum()
    }
}

fn pick_distinct(st: &mut u32, k: usize, below: usize) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(k);
    while out.len() < k {
        // High bits: the LCG's low bits cycle with a short period.
        let e = (lcg(st) >> 16) % below as u32;
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

fn routing(w: Workload, experts: usize, seed: u32) -> Routing {
    let tokens = w.tokens();
    if w == Workload::Verify && experts == 32 {
        // Every expert hit exactly twice: tokens t and t+4 share a block.
        let routes = (0..tokens)
            .flat_map(|t| (0..TOP_K).map(move |j| (8 * (t % 4) + j) as u32))
            .collect();
        return Routing::new(routes);
    }
    let mut st = seed;
    for attempt in 0..10_000 {
        let routes: Vec<u32> = (0..tokens)
            .flat_map(|_| pick_distinct(&mut st, TOP_K, experts))
            .collect();
        let r = Routing::new(routes);
        // Realistic verify spread for 256 experts: 45-55 distinct.
        if w != Workload::Verify || experts != 256 || (45..=55).contains(&r.distinct()) {
            return r;
        }
        assert!(
            attempt < 9_999,
            "no verify routing with 45-55 distinct experts"
        );
    }
    unreachable!()
}

// ---------------------------------------------------------------- weights

struct Proj {
    name: &'static str,
    n: i64,
    k: i64,
}

fn projections(s: &MoeShape) -> [Proj; 3] {
    [
        Proj {
            name: "gate",
            n: s.inter,
            k: s.hidden,
        },
        Proj {
            name: "up",
            n: s.inter,
            k: s.hidden,
        },
        Proj {
            name: "down",
            n: s.hidden,
            k: s.inter,
        },
    ]
}

/// Per-expert Tiled64 permutation of a `[E, N, cols]` stack: exactly
/// `kquant_tile_rows` applied to every `[N, cols]` slab.
fn tile_stack(a: &MxArray, unit: i64) -> MxArray {
    let shape = a.shape().expect("shape");
    let (e, n, cols) = (shape[0], shape[1], shape[2]);
    assert!(n % KQUANT_TILE_ROWS == 0 && cols % unit == 0);
    let t = a
        .reshape(&[e, n / KQUANT_TILE_ROWS, KQUANT_TILE_ROWS, cols / unit, unit])
        .and_then(|a| a.transpose(Some(&[0, 1, 3, 2, 4])))
        .and_then(|a| a.reshape(&[e, n, cols]))
        .expect("tile stack");
    t.eval();
    t
}

fn tiled_stack(w: &Weights, kq: &KQuant) -> Weights {
    Weights {
        w: tile_stack(&w.w, i64::from(kq.bits)),
        scales: tile_stack(&w.scales, kq.super_ratio() * kq.scale_bytes_per_group()),
        biases: tile_stack(&w.biases, kq.bias_entries_per_super_block()),
    }
}

/// `[N, cols]` view of expert `e` of a `[E, N, cols]` stack (no copy).
fn slab(a: &MxArray, e: i64) -> MxArray {
    let shape = a.shape().expect("shape");
    let v = a
        .slice(&[e, 0, 0], &[e + 1, shape[1], shape[2]])
        .and_then(|a| a.reshape(&[shape[1], shape[2]]))
        .expect("slab view");
    v.eval();
    v
}

/// One expert stack of a projection in both layouts.
struct Stack {
    /// Row-major `[E, N, cols]` (route A).
    row: Weights,
    /// Per-expert `[N, cols]` Tiled64 views (route B).
    slabs: Vec<Weights>,
}

struct StackRing {
    copies: Vec<Stack>,
    next: Cell<usize>,
    /// Bytes one expert's slab streams (`.weight` + `.scales` + `.biases`).
    slab_bytes: usize,
}

impl StackRing {
    fn new(kq: &KQuant, experts: usize, p: &Proj, seed: u32) -> Self {
        let e = experts as i64;
        let mut first = Some(weights(kq, &[e, p.n], p.k, seed));
        let stack_bytes = {
            let f = first.as_ref().unwrap();
            f.w.nbytes() + f.scales.nbytes() + f.biases.nbytes()
        };
        let count = RING_BYTES.div_ceil(stack_bytes).clamp(2, 64);
        let copies = (0..count)
            .map(|i| {
                let row = first
                    .take()
                    .unwrap_or_else(|| weights(kq, &[e, p.n], p.k, seed + i as u32));
                row.w.eval();
                row.scales.eval();
                row.biases.eval();
                let t = tiled_stack(&row, kq);
                let slabs = (0..e)
                    .map(|x| Weights {
                        w: slab(&t.w, x),
                        scales: slab(&t.scales, x),
                        biases: slab(&t.biases, x),
                    })
                    .collect();
                Stack { row, slabs }
            })
            .collect();
        Self {
            copies,
            next: Cell::new(0),
            slab_bytes: stack_bytes / experts,
        }
    }

    fn take(&self, routings: usize) -> &Stack {
        let c = self.next.get();
        self.next.set(c + 1);
        &self.copies[(c / routings) % self.copies.len()]
    }
}

/// Dense `[D*N, K]` Tiled64 ring (route C).
struct DenseRing {
    copies: Vec<Weights>,
    next: Cell<usize>,
}

impl DenseRing {
    fn new(kq: &KQuant, rows: i64, k: i64, seed: u32) -> Self {
        let first = weights(kq, &[rows], k, seed);
        let bytes = first.w.nbytes() + first.scales.nbytes() + first.biases.nbytes();
        let count = RING_BYTES.div_ceil(bytes).clamp(2, 64);
        let copies = (0..count)
            .map(|i| {
                let w = weights(kq, &[rows], k, seed + i as u32);
                let t = Weights {
                    w: kquant_tile_rows(&w.w, i64::from(kq.bits)).expect("tile"),
                    scales: kquant_tile_rows(
                        &w.scales,
                        kq.super_ratio() * kq.scale_bytes_per_group(),
                    )
                    .expect("tile"),
                    biases: kquant_tile_rows(&w.biases, kq.bias_entries_per_super_block())
                        .expect("tile"),
                };
                t.w.eval();
                t.scales.eval();
                t.biases.eval();
                t
            })
            .collect();
        Self {
            copies,
            next: Cell::new(0),
        }
    }

    fn take(&self) -> &Weights {
        let c = self.next.get();
        self.next.set(c + 1);
        &self.copies[c % self.copies.len()]
    }
}

// ---------------------------------------------------------------- inputs

/// The activations of one routing for one projection: route A's `[B,1,K]`
/// and `[B]` indices, route B's per-chunk `[1,K]` / `[8,K]` (zero padded)
/// rows of the same values.
struct Inputs {
    x_a: MxArray,
    rhs: MxArray,
    /// (expert, rows in this chunk, x of the chunk).
    chunks: Vec<(u32, usize, MxArray)>,
}

fn bf16_rows(rows: usize, k: i64, seed: u32) -> Vec<u16> {
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..rows * k as usize)
        .map(|_| {
            let v = f32::from((lcg(&mut st) >> 24) as i8) / 128.0;
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

fn inputs(r: &Routing, k: i64, seed: u32) -> Inputs {
    let b = r.routes.len();
    let kk = k as usize;
    let rows = bf16_rows(b, k, seed);
    let x_a = MxArray::from_bfloat16(&rows, &[b as i64, 1, k]).expect("x_a");
    let rhs = indices(&r.routes);
    let mut chunks = Vec::with_capacity(r.chunks());
    for (e, idx) in &r.groups {
        for chunk in idx.chunks(TILE_ROWS) {
            let m = if chunk.len() == 1 { 1 } else { TILE_ROWS };
            let mut buf = vec![0u16; m * kk];
            for (j, &row) in chunk.iter().enumerate() {
                buf[j * kk..(j + 1) * kk].copy_from_slice(&rows[row * kk..(row + 1) * kk]);
            }
            let x = MxArray::from_bfloat16(&buf, &[m as i64, k]).expect("x_b");
            chunks.push((*e, chunk.len(), x));
        }
    }
    x_a.eval();
    rhs.eval();
    for (_, _, x) in &chunks {
        x.eval();
    }
    Inputs { x_a, rhs, chunks }
}

// ---------------------------------------------------------------- matmuls

type H = *mut mlx_sys::mlx_array;

fn gather_a(inp: &Inputs, st: &Stack, kq: &KQuant, mode: &CString, sorted: bool) -> H {
    // SAFETY: every operand outlives the call; lhs null = derive from x.
    let h = unsafe {
        mlx_sys::mlx_gather_qmm(
            inp.x_a.as_raw_ptr(),
            st.row.w.as_raw_ptr(),
            st.row.scales.as_raw_ptr(),
            st.row.biases.as_raw_ptr(),
            std::ptr::null_mut(),
            inp.rhs.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            sorted,
        )
    };
    assert!(!h.is_null(), "{} gather_qmm construction failed", kq.mode);
    h
}

fn qmm(x: &MxArray, w: &Weights, kq: &KQuant, mode: &CString) -> H {
    // SAFETY: every operand outlives the call.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
        )
    };
    assert!(!h.is_null(), "{} qmm construction failed", kq.mode);
    h
}

fn eval_all(hs: &mut [H]) {
    let mut buf = [0i8; 512];
    // SAFETY: live handles; `buf` receives the error text.
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(hs.as_mut_ptr(), hs.len(), buf.as_mut_ptr(), buf.len())
    };
    assert!(ok, "eval failed");
    for &h in hs.iter() {
        // SAFETY: owned handle, not used afterwards.
        unsafe { mlx_sys::mlx_array_delete(h) };
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn env_list(var: &str) -> Option<Vec<String>> {
    std::env::var(var)
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect())
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn reps() -> usize {
    env_usize("MLX_KQUANT_MOE_BENCH_REPS", REPS)
}

/// Runs `f` with the simdgroup `gather_qmm_rhs` pinned (route A0); the pin
/// is per calling thread, which is the thread the eval `f` waits on encodes on.
fn with_rhs_nax_off<R>(f: impl FnOnce() -> R) -> R {
    // SAFETY: plain FFI setters on a thread-local flag.
    unsafe { mlx_sys::mlx_test_kquant_gather_rhs_fallback(true) };
    let r = f();
    unsafe { mlx_sys::mlx_test_kquant_gather_rhs_fallback(false) };
    r
}

/// One projection of one workload: the three routes and their inputs.
struct Case<'a> {
    kq: &'a KQuant,
    mode_row: CString,
    mode_t64: CString,
    stacks: &'a StackRing,
    dense: DenseRing,
    routings: &'a [Routing],
    inputs: Vec<Inputs>,
    /// Dense x `[m, K]`: 1 row for decode (1 row per expert), else 8.
    x_c: MxArray,
    /// Mean distinct experts over the routings.
    d_mean: f64,
}

impl Case<'_> {
    /// ms per layer-projection of a `BATCH`-layer eval.
    fn batch_a(&self) -> f64 {
        let n = self.routings.len();
        let mut hs: Vec<H> = (0..BATCH)
            .map(|_| {
                let c = self.stacks.next.get();
                let r = &self.routings[c % n];
                let inp = &self.inputs[c % n];
                gather_a(inp, self.stacks.take(n), self.kq, &self.mode_row, r.sorted)
            })
            .collect();
        let t = Instant::now();
        eval_all(&mut hs);
        t.elapsed().as_secs_f64() * 1e3 / BATCH as f64
    }

    fn batch_b(&self) -> f64 {
        let n = self.routings.len();
        let mut hs: Vec<H> = Vec::new();
        for _ in 0..BATCH {
            let c = self.stacks.next.get();
            let inp = &self.inputs[c % n];
            let st = self.stacks.take(n);
            for (e, _, x) in &inp.chunks {
                hs.push(qmm(x, &st.slabs[*e as usize], self.kq, &self.mode_t64));
            }
        }
        let t = Instant::now();
        eval_all(&mut hs);
        t.elapsed().as_secs_f64() * 1e3 / BATCH as f64
    }

    fn batch_c(&self) -> f64 {
        let mut hs: Vec<H> = (0..BATCH)
            .map(|_| qmm(&self.x_c, self.dense.take(), self.kq, &self.mode_t64))
            .collect();
        let t = Instant::now();
        eval_all(&mut hs);
        t.elapsed().as_secs_f64() * 1e3 / BATCH as f64
    }

    fn batch(&self, arm: usize) -> f64 {
        match arm {
            0 => self.batch_a(),
            1 => with_rhs_nax_off(|| self.batch_a()),
            2 => self.batch_b(),
            _ => self.batch_c(),
        }
    }

    /// Interleaved medians of A, A0, B, C.
    fn measure(&self) -> [f64; ARMS] {
        for _ in 0..WARMUP {
            for arm in 0..ARMS {
                let _ = self.batch(arm);
            }
        }
        let mut samples: [Vec<f64>; ARMS] = Default::default();
        for i in 0..reps() {
            let mut order: Vec<usize> = (0..ARMS).collect();
            if i % 2 == 1 {
                order.reverse();
            }
            for arm in order {
                samples[arm].push(self.batch(arm));
            }
        }
        std::array::from_fn(|arm| median(&mut samples[arm]))
    }

    /// Kernel families and dispatches per layer of each route.
    fn families(&self) -> [String; ARMS] {
        const NAMES: [&str; 11] = [
            "gather_qmv_fast",
            "gather_qmv",
            "gather_qmm_rhs_nax_nt",
            "gather_qmm_rhs_nt",
            "gather_qmm_t",
            "qmm_m8_nax_t64",
            "qmv_t64",
            "qmv_wide_t64",
            "qmm_t_splitk_t64",
            "qmm_t_nax_t64",
            "qmm_t_t64",
        ];
        std::array::from_fn(|arm| {
            start_counting();
            let _ = self.batch(arm);
            let parts: Vec<String> = NAMES
                .iter()
                .filter_map(|f| {
                    let c = family_count(f);
                    (c > 0).then(|| format!("{f} x{:.1}", c as f64 / BATCH as f64))
                })
                .collect();
            stop_counting();
            if parts.is_empty() {
                "(no K-quant family counted)".to_string()
            } else {
                parts.join(", ")
            }
        })
    }

    /// Route A against routes A0 and B on routing 0, copy 0: max |A - X| /
    /// max |A| for each.
    fn check(&self) -> [f64; 2] {
        let r = &self.routings[0];
        let inp = &self.inputs[0];
        let st = &self.stacks.copies[0];
        let (shape, _, a_bits) = read_output(
            "route A",
            gather_a(inp, st, self.kq, &self.mode_row, r.sorted),
        );
        let n = shape[2] as usize;
        let to_f32 = |b: u32| f32::from_bits(b << 16);
        let a: Vec<f32> = a_bits.into_iter().map(to_f32).collect();
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (_, _, a0_bits) = with_rhs_nax_off(|| {
            read_output(
                "route A0",
                gather_a(inp, st, self.kq, &self.mode_row, r.sorted),
            )
        });
        let worst_a0 = a0_bits
            .into_iter()
            .map(to_f32)
            .zip(&a)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let mut worst = 0f32;
        let mut row_of_chunk = 0usize;
        let mut last_expert = u32::MAX;
        for (e, rows, x) in &inp.chunks {
            if *e != last_expert {
                row_of_chunk = 0;
                last_expert = *e;
            }
            let (_, _, b_bits) = read_output(
                "route B",
                qmm(x, &st.slabs[*e as usize], self.kq, &self.mode_t64),
            );
            let b: Vec<f32> = b_bits.into_iter().map(to_f32).collect();
            let idx = &r.groups.iter().find(|(g, _)| g == e).expect("group").1;
            for j in 0..*rows {
                let route = idx[row_of_chunk + j];
                for c in 0..n {
                    let d = (a[route * n + c] - b[j * n + c]).abs();
                    worst = worst.max(d);
                }
            }
            row_of_chunk += rows;
        }
        [f64::from(worst_a0 / peak), f64::from(worst / peak)]
    }
}

/// ms per node of a `BATCH`-node eval of a tiny `[64, 256]` `qmv_t64`: the
/// fixed launch + eval cost every dispatch of every route pays.
fn launch_floor_ms(kq: &KQuant) -> f64 {
    let mode = CString::new(format!("{}{KQUANT_TILED_SUFFIX}", kq.mode)).unwrap();
    let w = weights(kq, &[64], 256, 0xF00D);
    let t = Weights {
        w: kquant_tile_rows(&w.w, i64::from(kq.bits)).expect("tile"),
        scales: kquant_tile_rows(&w.scales, kq.super_ratio() * kq.scale_bytes_per_group())
            .expect("tile"),
        biases: kquant_tile_rows(&w.biases, kq.bias_entries_per_super_block()).expect("tile"),
    };
    t.w.eval();
    t.scales.eval();
    t.biases.eval();
    let x = activation(&[1, 256], 0xF00E, DType::BFloat16);
    let batch = || {
        let mut hs: Vec<H> = (0..BATCH).map(|_| qmm(&x, &t, kq, &mode)).collect();
        let now = Instant::now();
        eval_all(&mut hs);
        now.elapsed().as_secs_f64() * 1e3 / BATCH as f64
    };
    for _ in 0..WARMUP {
        let _ = batch();
    }
    let mut s: Vec<f64> = (0..reps()).map(|_| batch()).collect();
    median(&mut s)
}

/// One table row: ms per route, GB/s of the touched expert bytes per route,
/// and the A0/A, A/B, A/C ratios.
fn print_row(name: &str, ms: &[f64], floor: f64, bytes: f64) {
    let gbps: Vec<String> = ms
        .iter()
        .map(|m| format!("{:>6.0}", bytes / m / 1e6))
        .collect();
    let times: Vec<String> = ms.iter().map(|m| format!("{m:>8.4}")).collect();
    println!(
        "  {name:<5} | {} {floor:>8.4} | {} | {:>5.2} {:>5.2} {:>5.2}",
        times.join(" "),
        gbps.join(" "),
        ms[1] / ms[0],
        ms[0] / ms[2],
        ms[0] / ms[3]
    );
}

fn selected<'a>(var: &str, all: &[&'a str]) -> Vec<&'a str> {
    match env_list(var) {
        Some(v) => all
            .iter()
            .copied()
            .filter(|m| v.iter().any(|s| s == m))
            .collect(),
        None => all.to_vec(),
    }
}

#[test]
#[ignore = "manual K-quant MoE expert-path microbenchmark"]
fn kquant_moe_expert_routes() {
    if std::env::var("MLX_KQUANT_MOE_BENCH").as_deref() != Ok("1") {
        eprintln!("set MLX_KQUANT_MOE_BENCH=1 to run this benchmark");
        return;
    }
    assert!(gpu_gen() > 0, "no Metal device");
    let nax = nax_available();
    println!(
        "\n  K-quant MoE expert routes, bfloat16, gpu gen {} (NAX {}), warmup={WARMUP}, reps={} \
         (medians, interleaved), per layer-projection of {BATCH} per eval, DRAM-cold ring >= {} MB; \
         floor = D x slab bytes / {FLOOR_GBPS} GB/s",
        gpu_gen(),
        if nax { "yes" } else { "no" },
        reps(),
        RING_BYTES >> 20
    );
    if !nax {
        println!("  WARNING: no NAX tensor-op kernels: route B's M=8 takes qmv_wide_t64");
    }
    let modes = selected("MLX_KQUANT_MOE_BENCH_MODES", &MODES);
    let shape_keys = selected("MLX_KQUANT_MOE_BENCH_SHAPES", &["qwen", "lfm2"]);
    let workload_keys = selected(
        "MLX_KQUANT_MOE_BENCH_WORKLOADS",
        &["decode", "verify", "prefill"],
    );
    let workloads: Vec<Workload> = [Workload::Decode, Workload::Verify, Workload::Prefill]
        .into_iter()
        .filter(|w| workload_keys.contains(&w.key()))
        .collect();

    for shape in SHAPES.iter().filter(|s| shape_keys.contains(&s.key)) {
        for (mi, mode) in modes.iter().enumerate() {
            let kq = kquant(mode);
            let mode_row = mode_cstr(kq);
            let mode_t64 = CString::new(format!("{mode}{KQUANT_TILED_SUFFIX}")).unwrap();
            println!(
                "\n  == {} E={} {mode}: gate/up [{},{},{}], down [{},{},{}]",
                shape.name,
                shape.experts,
                shape.experts,
                shape.inter,
                shape.hidden,
                shape.experts,
                shape.hidden,
                shape.inter
            );
            println!(
                "  launch floor: {:.4} ms per node of an {BATCH}-node eval (tiny [64,256] qmv_t64)",
                launch_floor_ms(kq)
            );
            let projs = projections(shape);
            let rings: Vec<StackRing> = projs
                .iter()
                .enumerate()
                .map(|(pi, p)| {
                    StackRing::new(kq, shape.experts, p, 0xA000 + (mi * 16 + pi) as u32 * 64)
                })
                .collect();
            println!(
                "  ring: {} copies per projection; slab bytes gate/up {:.3} MB, down {:.3} MB; \
                 layer (gate+up+down) per expert {:.3} MB",
                rings[0].copies.len(),
                rings[0].slab_bytes as f64 / 1e6,
                rings[2].slab_bytes as f64 / 1e6,
                rings.iter().map(|r| r.slab_bytes).sum::<usize>() as f64 / 1e6
            );

            for w in &workloads {
                let routings: Vec<Routing> = (0..w.routings())
                    .map(|i| routing(*w, shape.experts, 0xB000 + (mi * 64 + i) as u32))
                    .collect();
                let d_mean = routings.iter().map(|r| r.distinct() as f64).sum::<f64>()
                    / routings.len() as f64;
                let d_dense = d_mean.round() as i64;
                let chunks_mean =
                    routings.iter().map(|r| r.chunks() as f64).sum::<f64>() / routings.len() as f64;
                let rows_max = routings
                    .iter()
                    .flat_map(|r| r.groups.iter().map(|(_, idx)| idx.len()))
                    .max()
                    .unwrap_or(0);
                println!(
                    "\n  -- {} : B={} routes, distinct experts mean {:.1} (min {}, max {}), \
                     rows/expert max {}, route-B dispatches mean {:.1}, sorted={}",
                    w.name(),
                    routings[0].routes.len(),
                    d_mean,
                    routings.iter().map(|r| r.distinct()).min().unwrap(),
                    routings.iter().map(|r| r.distinct()).max().unwrap(),
                    rows_max,
                    chunks_mean,
                    routings[0].sorted
                );
                println!(
                    "  {:<5} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>6} {:>6} {:>6} {:>6} | {:>5} {:>5} {:>5}",
                    "proj",
                    "A ms",
                    "A0 ms",
                    "B ms",
                    "C ms",
                    "floor",
                    "A",
                    "A0",
                    "B",
                    "C GB/s",
                    "A0/A",
                    "A/B",
                    "A/C"
                );
                let mut sum = [0f64; ARMS + 1];
                let mut fams: Vec<(String, [String; ARMS])> = Vec::new();
                let mut checks: Vec<(String, [f64; 2])> = Vec::new();
                for (pi, p) in projs.iter().enumerate() {
                    let inputs: Vec<Inputs> = routings
                        .iter()
                        .enumerate()
                        .map(|(i, r)| inputs(r, p.k, 0xC000 + (pi * 64 + i) as u32))
                        .collect();
                    let m_c = if *w == Workload::Decode {
                        1
                    } else {
                        TILE_ROWS as i64
                    };
                    let case = Case {
                        kq,
                        mode_row: mode_row.clone(),
                        mode_t64: mode_t64.clone(),
                        stacks: &rings[pi],
                        dense: DenseRing::new(kq, d_dense * p.n, p.k, 0xD000 + pi as u32 * 64),
                        routings: &routings,
                        inputs,
                        x_c: activation(&[m_c, p.k], 0xE000 + pi as u32, DType::BFloat16),
                        d_mean,
                    };
                    let rel = case.check();
                    checks.push((p.name.to_string(), rel));
                    assert!(
                        rel.iter().all(|r| *r < 2e-2),
                        "{} {}: the routes disagree (max rel diff A0 {:.3e}, B {:.3e})",
                        shape.name,
                        p.name,
                        rel[0],
                        rel[1]
                    );
                    fams.push((p.name.to_string(), case.families()));
                    let ms = case.measure();
                    let bytes = case.d_mean * case.stacks.slab_bytes as f64;
                    let floor = bytes / FLOOR_GBPS / 1e6;
                    for (s, m) in sum.iter_mut().zip(ms.iter().chain([floor].iter())) {
                        *s += m;
                    }
                    print_row(p.name, &ms, floor, bytes);
                }
                let bytes: f64 = rings.iter().map(|r| d_mean * r.slab_bytes as f64).sum();
                print_row("sum", &sum[..ARMS], sum[ARMS], bytes);
                for (name, f) in &fams {
                    let parts: Vec<String> = ARM_NAMES
                        .iter()
                        .zip(f)
                        .map(|(arm, fam)| format!("{arm} {fam}"))
                        .collect();
                    println!("  dispatches/layer {name:<5}: {}", parts.join(" | "));
                }
                for (i, other) in ["A0", "B"].iter().enumerate() {
                    let worst = checks.iter().map(|(_, r)| r[i]).fold(0f64, f64::max);
                    println!(
                        "  A vs {other} max rel diff (routing 0): {} -> worst {worst:.2e}",
                        checks
                            .iter()
                            .map(|(n, r)| format!("{n} {:.2e}", r[i]))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        }
    }
}
