//! INT8 queries over INT8 pages (Splash `q8q`) by the position rule: the
//! tile's and the decode kernels' bits on fixed inputs, the INT8-query tile
//! against an fp64 reference over its own query codes and against the
//! bf16-query tile, and the kernel each launch takes.

use mlx_paged_attn::PagedAttentionConfig;
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::prefill_plan::Pass;

use super::paged_int8_tests::{adapter_for, quantize, shaped_rows};
use super::paged_kv_cache_adapter::{PagedKVCacheAdapter, PagedRaggedRow};
use crate::array::{DType, MxArray};
use crate::transformer::Activations;

/// An INT8 pool's stored rows as the store writes them (Splash
/// `splash_store_kv_row`): codes `[token][kv head][dim]`, one scale per
/// (token, kv head).
struct Stored {
    key_codes: Vec<i8>,
    key_scales: Vec<f32>,
    value_codes: Vec<i8>,
    value_scales: Vec<f32>,
}

impl Stored {
    fn push(codes: &mut Vec<i8>, scales: &mut Vec<f32>, rows: &[f32], d: usize) {
        for row in rows.chunks(d) {
            let (q, scale) = quantize(row);
            codes.extend(q);
            scales.push(scale);
        }
    }
}

/// One sequence of `tokens` stored rows in a one-layer pool, written 2,048
/// rows at a time: keys and values of `shaped_rows` at gain 1/256 (about
/// ±0.5 times 1, 2, 4, 8 or 16 by row). With `keep`, the INT8 rows as
/// stored.
fn filled(
    dtype: MetalDtype,
    kv: usize,
    d: usize,
    bs: u32,
    tokens: u32,
    keep: bool,
) -> Option<(PagedKVCacheAdapter, Option<Stored>)> {
    let config = PagedAttentionConfig {
        block_size: bs,
        gpu_memory_mb: 256,
        head_size: d as u32,
        num_kv_heads: kv as u32,
        num_layers: 1,
        max_seq_len: Some(tokens + 2048),
        max_batch_size: Some(1),
    };
    let blocks = tokens.div_ceil(bs) + 1;
    let (mut adapter, _) = adapter_for(config, blocks, blocks, dtype)?;
    adapter.begin_request(1).unwrap();
    adapter.allocate_suffix_blocks(tokens).unwrap();
    let mut stored = keep.then(|| Stored {
        key_codes: Vec::new(),
        key_scales: Vec::new(),
        value_codes: Vec::new(),
        value_scales: Vec::new(),
    });
    let mut first = 0u32;
    while first < tokens {
        let rows = (tokens - first).min(2048) as usize;
        adapter
            .record_tokens(
                &(first..first + rows as u32)
                    .map(|t| t % 1000)
                    .collect::<Vec<u32>>(),
            )
            .unwrap();
        let (keys, key_rows) = shaped_rows(rows, kv, d, 300 + first, Some(1.0 / 256.0));
        let (values, value_rows) = shaped_rows(rows, kv, d, 700 + first, Some(1.0 / 256.0));
        if let Some(stored) = stored.as_mut() {
            Stored::push(&mut stored.key_codes, &mut stored.key_scales, &key_rows, d);
            Stored::push(
                &mut stored.value_codes,
                &mut stored.value_scales,
                &value_rows,
                d,
            );
        }
        adapter
            .update_keys_values_native(0, &keys, &values, first)
            .unwrap();
        adapter.eval_pending_pool_writes().unwrap();
        first += rows as u32;
    }
    Some((adapter, stored))
}

/// Splitmix64 uniform in `[-amplitude, amplitude)`, rounded to bf16 (the
/// queries the kernel reads, which the reference quantizes).
fn uniform(count: usize, seed: u64, amplitude: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let x = ((z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amplitude;
            let bits = x.to_bits();
            f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
        })
        .collect()
}

fn bf16_array(values: &[f32], shape: &[i64]) -> MxArray {
    MxArray::from_float32(values, shape)
        .and_then(|array| array.astype(DType::BFloat16))
        .expect("bf16 array")
}

fn host(array: &MxArray) -> Vec<f32> {
    array
        .astype(DType::Float32)
        .and_then(|array| array.to_float32())
        .expect("read back")
        .to_vec()
}

/// The tile's output for `queries` `[rows, hq, d]` at `committed`.
fn attend(
    adapter: &mut PagedKVCacheAdapter,
    queries: &[f32],
    (rows, hq, d): (usize, usize, usize),
    committed: u32,
    pass: Pass,
) -> Vec<f32> {
    let q = bf16_array(queries, &[rows as i64, hq as i64, d as i64]);
    let out = adapter
        .paged_prefill_attention(0, &q, committed, pass, 1.0 / (d as f32).sqrt())
        .expect("paged prefill attention");
    host(&out)
}

fn fnv(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| {
        value
            .to_bits()
            .to_le_bytes()
            .iter()
            .fold(hash, |hash, &byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            })
    })
}

/// (label, pool format, kv heads, group, head dim, block size, pool tokens,
/// [(committed, rows, pass)]).
type GoldenPool = (
    &'static str,
    MetalDtype,
    usize,
    usize,
    usize,
    u32,
    u32,
    &'static [(u32, usize, Pass)],
);

const GOLDEN_POOLS: &[GoldenPool] = &[
    (
        "int8 kv4 g6 d256",
        MetalDtype::Int8,
        4,
        6,
        256,
        16,
        151_560,
        &[
            (0, 2048, Pass::Prefill),
            (32_768, 2048, Pass::Prefill),
            (129_024, 2048, Pass::Prefill),
            (16_384, 8, Pass::Verify),
            (150_000, 8, Pass::Verify),
        ],
    ),
    (
        "bf16 kv4 g6 d256",
        MetalDtype::BFloat16,
        4,
        6,
        256,
        16,
        34_816,
        &[
            (0, 2048, Pass::Prefill),
            (32_768, 2048, Pass::Prefill),
            (16_384, 8, Pass::Verify),
        ],
    ),
    (
        "int8 kv2 g16 d256",
        MetalDtype::Int8,
        2,
        16,
        256,
        16,
        4_643,
        &[(4_096, 547, Pass::Prefill)],
    ),
    (
        "int8 kv4 g4 d128 bs32",
        MetalDtype::Int8,
        4,
        4,
        128,
        32,
        5_150,
        &[(4_093, 1_057, Pass::Prefill)],
    ),
];

/// FNV-1a 64 of every output of `GOLDEN_POOLS`, in order, taken on the tree
/// before INT8 queries existed.
const GOLDEN: &[u64] = &[
    0xee7f_f28f_55de_9021,
    0x6a50_d203_263f_cd00,
    0xb4e2_54bc_cc63_7f3e,
    0x3f70_13ea_c7e2_f999,
    0xb803_1256_7257_5baa,
    0x5694_6b6f_9b68_5741,
    0x2e1a_9dc3_a4da_2b37,
    0x7072_9b79_d6ed_af37,
    0x81e5_8496_f320_fd08,
    0xf74d_f101_fede_72ac,
];

/// With activations bf16 the tile is bit-identical to the tree before INT8
/// queries: the production shapes of the 27B (2,048 rows after 0, 32K and
/// 127K tokens, verify after 16K and 150K), a bf16 pool, a group of 16 in
/// two parts and a D128 / bs32 pool.
/// Mutation off-ignored: the adapter ignores activations bf16 → red.
#[test]
fn bf16_query_tile_keeps_its_bits() {
    let mut hashes = Vec::new();
    for &(label, dtype, kv, group, d, bs, tokens, cases) in GOLDEN_POOLS {
        let Some((mut adapter, _)) = filled(dtype, kv, d, bs, tokens, false) else {
            return;
        };
        adapter.set_activations(Activations::Bf16);
        let hq = kv * group;
        for &(committed, rows, pass) in cases {
            let q = uniform(rows * hq * d, u64::from(committed) * 31 + rows as u64, 2.0);
            let out = attend(&mut adapter, &q, (rows, hq, d), committed, pass);
            let hash = fnv(&out);
            eprintln!("GOLDEN {label} {pass:?} (P {committed}, T {rows}): {hash:#018x}");
            hashes.push(hash);
        }
    }
    assert_eq!(hashes, GOLDEN, "the bf16-query tile changed its bits");
}

/// FNV-1a 64 of every INT8-pool output of `GOLDEN_POOLS` with activations
/// auto, in order, and the INT8-query launches among them, taken on the tree
/// that staged each page's keys behind their own barrier.
const GOLDEN_INT8_QUERIES: &[u64] = &[
    0x01e3_eccb_b8ea_3587,
    0x7c66_20c4_1655_a3da,
    0xe1da_7f8e_c944_a86f,
    0x71e5_4b03_efa0_1076,
    0x992c_3d12_1563_befe,
    0x0d33_f2ee_78ae_8691,
    0x9e12_4f96_3ecb_609d,
];
const GOLDEN_INT8_QUERY_LAUNCHES: u64 = 7;

/// With activations auto and the threshold at 0 every launch of the INT8
/// pools of `GOLDEN_POOLS` takes the INT8-query tile, bit-identical to the
/// tile that first scored INT8 queries.
/// Mutations same-page (a page's keys staged again for the next page) and
/// no-transpose (keys staged in pool order) → red, here and in
/// `bf16_query_tile_keeps_its_bits`.
#[test]
fn int8_query_tile_keeps_its_bits() {
    if !scores_int8_queries("int8_query_tile_keeps_its_bits") {
        return;
    }
    let mut hashes = Vec::new();
    let mut launches = 0;
    for &(label, dtype, kv, group, d, bs, tokens, cases) in GOLDEN_POOLS {
        if dtype != MetalDtype::Int8 {
            continue;
        }
        let Some((mut adapter, _)) = filled(dtype, kv, d, bs, tokens, false) else {
            return;
        };
        adapter.set_activations(Activations::Auto);
        adapter.set_int8_query_threshold(0);
        let hq = kv * group;
        for &(committed, rows, pass) in cases {
            let q = uniform(rows * hq * d, u64::from(committed) * 31 + rows as u64, 2.0);
            let before = adapter.tile_query_launches()[1];
            let out = attend(&mut adapter, &q, (rows, hq, d), committed, pass);
            launches += adapter.tile_query_launches()[1] - before;
            let hash = fnv(&out);
            eprintln!("GOLDEN q8q {label} {pass:?} (P {committed}, T {rows}): {hash:#018x}");
            hashes.push(hash);
        }
    }
    eprintln!("GOLDEN q8q launches {launches}");
    assert_eq!(
        hashes, GOLDEN_INT8_QUERIES,
        "the INT8-query tile changed its bits"
    );
    assert_eq!(launches, GOLDEN_INT8_QUERY_LAUNCHES);
}

/// Whether this GPU scores INT8 queries; prints a skip when it does not.
fn scores_int8_queries(test: &str) -> bool {
    // SAFETY: nullary FFI query.
    let gpu_gen = unsafe { mlx_sys::mlx_gpu_architecture_gen() };
    let scores = mlx_paged_attn::prefill_plan::int8_query_from(gpu_gen).is_some();
    if !scores {
        eprintln!("skipping {test}: GPU generation {gpu_gen} never scores INT8 queries");
    }
    scores
}

/// Queries `[rows][hq][d]` of about ±1 with one outlier of ±24 per row and
/// head (an outlier channel), so a row's INT8 scale is 24 / 127 and its
/// other dimensions take codes of at most ±6: bf16 and INT8 queries then
/// score visibly apart. Row 1's first head is all zeros (scale 0).
fn outlier_queries(rows: usize, hq: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut q = uniform(rows * hq * d, seed, 1.0);
    for row in 0..rows {
        for head in 0..hq {
            let base = (row * hq + head) * d;
            if row == 1 && head == 0 {
                q[base..base + d].fill(0.0);
                continue;
            }
            let at = (row * 7 + head * 13) % d;
            q[base + at] = if (row + head) % 2 == 0 { 24.0 } else { -24.0 };
        }
    }
    q
}

/// Splash `int8Block` / `int8Code` (int8_activations.h): `m` the row's
/// largest magnitude, scale `m / 127`, codes `clamp(rint(x * (127 / m)),
/// ±127)` ties to even; a row of `m <= 2^-100` codes as zeros, scale 0.
fn query_codes(row: &[f32]) -> (Vec<i32>, f32) {
    let largest = row.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if largest <= f32::from_bits(0x0d80_0000) {
        return (vec![0; row.len()], 0.0);
    }
    let inverse = 127.0f32 / largest;
    let codes = row
        .iter()
        .map(|x| (x * inverse).round_ties_even().clamp(-127.0, 127.0) as i32)
        .collect();
    (codes, largest / 127.0f32)
}

/// The fp64 attention of chunk row `r` (position `committed + r`) and query
/// head `head` over the stored INT8 rows, scoring the row's INT8 query codes
/// times their scale (Splash `attention_variants_metal_test.mm`
/// `reference` with INT8 queries): the output and the mass
/// `sum_t p_t |v_t[d]|`.
fn reference(
    stored: &Stored,
    (kv, group, d): (usize, usize, usize),
    q: &[f32],
    (committed, r, head): (usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let hq = kv * group;
    let kv_head = head / group;
    let (codes, query_scale) = query_codes(&q[(r * hq + head) * d..(r * hq + head + 1) * d]);
    let visible = committed + r + 1;
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores: Vec<f64> = (0..visible)
        .map(|t| {
            let row = t * kv + kv_head;
            let keys = &stored.key_codes[row * d..(row + 1) * d];
            let dot: i64 = codes
                .iter()
                .zip(keys)
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum();
            dot as f64 * f64::from(query_scale) * f64::from(stored.key_scales[row]) * scale
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for x in scores.iter_mut() {
        *x = (*x - maximum).exp();
        sum += *x;
    }
    let mut out = vec![0.0; d];
    let mut mass = vec![0.0; d];
    for (t, p) in scores.iter().enumerate() {
        let row = t * kv + kv_head;
        let value_scale = f64::from(stored.value_scales[row]);
        let p = p / sum;
        for ((o, m), &c) in out
            .iter_mut()
            .zip(mass.iter_mut())
            .zip(&stored.value_codes[row * d..(row + 1) * d])
        {
            let v = f64::from(c) * value_scale;
            *o += p * v;
            *m += p * v.abs();
        }
    }
    (out, mass)
}

fn bf16_ulp(x: f32) -> f64 {
    let exponent = x.abs().max(f32::from_bits(0x0080_0000)).log2().floor();
    2f64.powf(f64::from(exponent) - 7.0)
}

/// The rows a case checks: all of a short chunk's, else the first and last
/// tiles' and the middle one's.
fn checked_rows(rows: usize) -> Vec<usize> {
    if rows <= 64 {
        return (0..rows).collect();
    }
    let mut checked: Vec<usize> = (0..8)
        .chain(rows / 2..rows / 2 + 1)
        .chain(rows - 8..rows)
        .collect();
    checked.dedup();
    checked
}

/// Splash's INT8-query gate (`attention_variants_metal_test.mm`
/// `checkOutput`): every checked output of the q8q kernel within
/// `mass / 256 + ulp_bf16(ref)` of the fp64 attention over the kernel's own
/// query codes, on the 27B geometry (a 2,048-row chunk, a 64-row chunk after
/// 2K, a verify after 16K), a group of 16 in two parts and a D128 / bs32
/// pool, the threshold at 0 so each launch takes the q8q entry. Under the
/// device's threshold a 64-row chunk with no history keeps bf16 queries, bit
/// for bit.
/// Mutation q8q-as-q8: a selected launch runs the bf16-query entry → red.
#[test]
fn int8_query_tile_tracks_the_fp64_reference_over_its_codes() {
    if !scores_int8_queries("int8_query_tile_tracks_the_fp64_reference_over_its_codes") {
        return;
    }
    type Pool = (usize, usize, usize, u32, u32, &'static [(u32, usize, Pass)]);
    const POOLS: &[Pool] = &[
        (
            4,
            6,
            256,
            16,
            16_400,
            &[
                (0, 2048, Pass::Prefill),
                (2048, 64, Pass::Prefill),
                (16_384, 8, Pass::Verify),
            ],
        ),
        (2, 16, 256, 16, 4_643, &[(4_096, 547, Pass::Prefill)]),
        (4, 4, 128, 32, 5_150, &[(4_093, 1_057, Pass::Prefill)]),
    ];
    let mut failures = Vec::new();
    for &(kv, group, d, bs, tokens, cases) in POOLS {
        let Some((mut adapter, Some(stored))) = filled(MetalDtype::Int8, kv, d, bs, tokens, true)
        else {
            return;
        };
        adapter.set_activations(Activations::Auto);
        assert!(
            adapter.int8_queries_active(),
            "kv{kv} g{group} d{d}: q8q inactive"
        );
        adapter.set_int8_query_threshold(0);
        let hq = kv * group;
        for &(committed, rows, pass) in cases {
            let q = outlier_queries(rows, hq, d, u64::from(committed) + rows as u64);
            let before = adapter.tile_query_launches();
            let got = attend(&mut adapter, &q, (rows, hq, d), committed, pass);
            let after = adapter.tile_query_launches();
            assert_eq!(
                (after[0] - before[0], after[1] - before[1]),
                (0, 1),
                "kv{kv} g{group} d{d} {pass:?} (P {committed}, T {rows}) did not take q8q"
            );
            let mut worst = 0f64;
            for r in checked_rows(rows) {
                for head in 0..hq {
                    let (want, mass) =
                        reference(&stored, (kv, group, d), &q, (committed as usize, r, head));
                    for dim in 0..d {
                        let actual = f64::from(got[(r * hq + head) * d + dim]);
                        let bound = mass[dim] / 256.0 + bf16_ulp(want[dim] as f32);
                        let error = (actual - want[dim]).abs();
                        if !actual.is_finite() || error > bound {
                            failures.push(format!(
                                "kv{kv} g{group} d{d} {pass:?} (P {committed}, T {rows}) row {r} \
                                 head {head} dim {dim}: {actual} vs {} (bound {bound})",
                                want[dim]
                            ));
                        }
                        worst = worst.max(error / bound);
                    }
                }
            }
            eprintln!(
                "kv{kv} g{group} d{d} {pass:?} (P {committed}, T {rows}): worst error / bound {worst:.3}"
            );
        }
        if (kv, group) == (4, 6) {
            let Some((mut adapter, _)) = filled(MetalDtype::Int8, kv, d, bs, 64, false) else {
                return;
            };
            adapter.set_activations(Activations::Auto);
            let q = outlier_queries(64, hq, d, 99);
            let before = adapter.tile_query_launches();
            let auto = attend(&mut adapter, &q, (64, hq, d), 0, Pass::Prefill);
            let after = adapter.tile_query_launches();
            assert_eq!(after[1], before[1], "rows below the threshold took q8q");
            adapter.set_activations(Activations::Bf16);
            let bf16 = attend(&mut adapter, &q, (64, hq, d), 0, Pass::Prefill);
            assert!(
                auto.iter()
                    .zip(&bf16)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "below the threshold auto differs from bf16 queries"
            );
        }
    }
    failures.truncate(12);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Each tile launch takes the kernel its rows' positions select, whatever
/// its rows: on the 27B geometry a 2,048-row chunk after 0 and a 64-row
/// piece ending at 7,168 keep bf16 queries, a 64-row piece from 7,168 scores
/// INT8 codes, a piece or a verify across 7,168 runs as one launch per side,
/// and activations bf16 never takes q8q. The split launch equals its two
/// halves launched alone, bit for bit.
/// Mutations rows16 (q8q iff 16 rows or more), no-split (a straddling launch
/// runs whole on bf16 queries) and gt (`>` for `>=`) → red.
#[test]
fn each_launch_takes_the_kernel_its_position_selects() {
    if !scores_int8_queries("each_launch_takes_the_kernel_its_position_selects") {
        return;
    }
    const KV: usize = 4;
    const GROUP: usize = 6;
    const D: usize = 256;
    let hq = KV * GROUP;
    let Some((mut adapter, _)) = filled(MetalDtype::Int8, KV, D, 16, 7_232, false) else {
        return;
    };
    // (committed, rows, pass, [bf16, q8q] launches under auto)
    let cases: &[(u32, usize, Pass, [u64; 2])] = &[
        (0, 2048, Pass::Prefill, [1, 0]),
        (7_104, 64, Pass::Prefill, [1, 0]),
        (7_168, 64, Pass::Prefill, [0, 1]),
        (7_136, 64, Pass::Prefill, [1, 1]),
        (7_164, 8, Pass::Verify, [1, 1]),
        (7_168, 8, Pass::Verify, [0, 1]),
        (7_167, 1, Pass::Verify, [1, 0]),
    ];
    let mut wrong = Vec::new();
    for activations in [Activations::Auto, Activations::Bf16] {
        adapter.set_activations(activations);
        for &(committed, rows, pass, auto) in cases {
            let q = uniform(rows * hq * D, u64::from(committed) + rows as u64, 2.0);
            let before = adapter.tile_query_launches();
            attend(&mut adapter, &q, (rows, hq, D), committed, pass);
            let after = adapter.tile_query_launches();
            let want = if activations == Activations::Auto {
                auto
            } else {
                [1, 0]
            };
            let took = [after[0] - before[0], after[1] - before[1]];
            if took != want {
                wrong.push(format!(
                    "{} {pass:?} (P {committed}, T {rows}): launches [bf16, q8q] {took:?}, \
                     want {want:?}",
                    activations.as_str()
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));

    adapter.set_activations(Activations::Auto);
    let q = outlier_queries(64, hq, D, 7);
    let whole = attend(&mut adapter, &q, (64, hq, D), 7_136, Pass::Prefill);
    let half = 32 * hq * D;
    let mut halves = attend(&mut adapter, &q[..half], (32, hq, D), 7_136, Pass::Prefill);
    halves.extend(attend(
        &mut adapter,
        &q[half..],
        (32, hq, D),
        7_168,
        Pass::Prefill,
    ));
    assert!(
        whole
            .iter()
            .zip(&halves)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a launch split at the threshold differs from its halves"
    );
    adapter.set_activations(Activations::Bf16);
    let mut bf16 = attend(&mut adapter, &q[..half], (32, hq, D), 7_136, Pass::Prefill);
    bf16.extend(attend(
        &mut adapter,
        &q[half..],
        (32, hq, D),
        7_168,
        Pass::Prefill,
    ));
    assert!(
        whole[..half]
            .iter()
            .zip(&bf16[..half])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "rows below the threshold differ from bf16 queries"
    );
    assert!(
        whole[half..] != bf16[half..],
        "rows past the threshold kept bf16 queries"
    );
}

/// The q8q kernel against the bf16-query kernel on the same pool and queries
/// (±2 uniform, no outliers): a 2,048-row chunk of the 27B geometry after 0,
/// 32K and 128K tokens. Per (row, query head), the relative L2 error
/// `|a - b| / |b|` of the attention output: printed (max, mean, p99), held
/// under a "clearly broken" band of 2^-3 (max) and 2^-5 (mean), and nonzero
/// (q8q ran).
/// Mutation q8q-as-q8: a selected launch runs the bf16-query entry → red.
#[test]
fn int8_query_tile_error_against_bf16_queries() {
    if !scores_int8_queries("int8_query_tile_error_against_bf16_queries") {
        return;
    }
    const KV: usize = 4;
    const GROUP: usize = 6;
    const D: usize = 256;
    const ROWS: usize = 2048;
    let hq = KV * GROUP;
    let Some((mut adapter, _)) = filled(MetalDtype::Int8, KV, D, 16, 131_072 + ROWS as u32, false)
    else {
        return;
    };
    adapter.set_int8_query_threshold(0);
    let mut failures = Vec::new();
    for committed in [0u32, 32_768, 131_072] {
        let q = uniform(ROWS * hq * D, u64::from(committed) + 5, 2.0);
        adapter.set_activations(Activations::Bf16);
        let bf16 = attend(&mut adapter, &q, (ROWS, hq, D), committed, Pass::Prefill);
        adapter.set_activations(Activations::Auto);
        let before = adapter.tile_query_launches()[1];
        let int8 = attend(&mut adapter, &q, (ROWS, hq, D), committed, Pass::Prefill);
        assert_eq!(
            adapter.tile_query_launches()[1],
            before + 1,
            "P {committed}: no q8q launch"
        );
        let mut errors: Vec<f64> = int8
            .chunks(D)
            .zip(bf16.chunks(D))
            .map(|(a, b)| {
                let diff: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                    .sum();
                let norm: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
                (diff / norm.max(f64::MIN_POSITIVE)).sqrt()
            })
            .collect();
        errors.sort_by(f64::total_cmp);
        let max = errors[errors.len() - 1];
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let p99 = errors[errors.len() * 99 / 100];
        let max_abs = int8
            .iter()
            .zip(&bf16)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "ERROR P {committed}: relative L2 per (row, head) max {max:.3e} mean {mean:.3e} \
             p99 {p99:.3e}; max |a - b| {max_abs:.3e}"
        );
        if !(max > 0.0 && max <= 1.0 / 8.0 && mean <= 1.0 / 32.0) {
            failures.push(format!("P {committed}: max {max:e} mean {mean:e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// One pool of sequences `1..=n` holding `lengths[i]` stored rows each
/// (keys and values as [`filled`]), every write evaluated.
fn filled_sequences(
    dtype: MetalDtype,
    kv: usize,
    d: usize,
    lengths: &[u32],
) -> Option<PagedKVCacheAdapter> {
    let sequences: Vec<(u32, u32)> = lengths
        .iter()
        .enumerate()
        .map(|(index, &tokens)| (tokens, index as u32 + 1))
        .collect();
    filled_contents(dtype, kv, d, &sequences)
}

/// [`filled_sequences`] of `(tokens, content)`: sequences of one content
/// hold the same rows.
fn filled_contents(
    dtype: MetalDtype,
    kv: usize,
    d: usize,
    sequences: &[(u32, u32)],
) -> Option<PagedKVCacheAdapter> {
    let lengths: Vec<u32> = sequences.iter().map(|&(tokens, _)| tokens).collect();
    let blocks: u32 = lengths.iter().map(|&n| n.div_ceil(16) + 1).sum();
    let config = PagedAttentionConfig {
        block_size: 16,
        gpu_memory_mb: 512,
        head_size: d as u32,
        num_kv_heads: kv as u32,
        num_layers: 1,
        max_seq_len: lengths.iter().max().map(|&n| n + 2048),
        max_batch_size: Some(lengths.len() as u32),
    };
    let (mut adapter, _) = adapter_for(config, blocks, blocks, dtype)?;
    for (index, &(tokens, content)) in sequences.iter().enumerate() {
        let seq = index as u32 + 1;
        adapter.begin_request(seq).unwrap();
        adapter.allocate_suffix_blocks(tokens).unwrap();
        let mut first = 0u32;
        while first < tokens {
            let rows = (tokens - first).min(2048) as usize;
            adapter
                .record_tokens(
                    &(first..first + rows as u32)
                        .map(|t| t % 1000)
                        .collect::<Vec<u32>>(),
                )
                .unwrap();
            let salt = content * 100_000 + first;
            let (keys, _) = shaped_rows(rows, kv, d, 300 + salt, Some(1.0 / 256.0));
            let (values, _) = shaped_rows(rows, kv, d, 700 + salt, Some(1.0 / 256.0));
            adapter
                .update_keys_values_native(0, &keys, &values, first)
                .unwrap();
            adapter.eval_pending_pool_writes().unwrap();
            first += rows as u32;
        }
    }
    Some(adapter)
}

/// A decode-family read of [`decode_fixtures`]: a 1-row decode of a
/// sequence, a batched decode of sequences, a ragged read of
/// (sequence, last rows) or a varlen prefill piece of `rows` after
/// `committed` of a sequence.
#[derive(Clone, Copy, Debug)]
enum DecodeRead {
    Single(u32),
    Batched(&'static [u32]),
    Ragged(&'static [(u32, u32)]),
    Varlen(u32, u32, u32),
}

impl DecodeRead {
    fn rows(self) -> usize {
        match self {
            DecodeRead::Single(_) => 1,
            DecodeRead::Batched(seqs) => seqs.len(),
            DecodeRead::Ragged(rows) => rows.iter().map(|&(_, n)| n as usize).sum(),
            DecodeRead::Varlen(_, _, rows) => rows as usize,
        }
    }
}

/// (label, kv heads, group, sequence lengths, reads): the 27B geometry's
/// generic v1 (300) and v2 (4,000) kernels, its grouped int8 kernel (16,500,
/// one row; two ragged rows), a batched and a ragged v2 launch and a varlen
/// prefill piece; the 35B geometry's v2 and grouped kernels.
type DecodeFixture = (
    &'static str,
    usize,
    usize,
    &'static [u32],
    &'static [(&'static str, DecodeRead)],
);

const DECODE_FIXTURES: &[DecodeFixture] = &[
    (
        "kv4 g6",
        4,
        6,
        &[300, 4_000, 16_500],
        &[
            ("v1", DecodeRead::Single(1)),
            ("v2", DecodeRead::Single(2)),
            ("grouped", DecodeRead::Single(3)),
            ("batched", DecodeRead::Batched(&[1, 2, 3])),
            ("ragged", DecodeRead::Ragged(&[(2, 3), (1, 2)])),
            ("ragged grouped", DecodeRead::Ragged(&[(3, 2)])),
            ("varlen", DecodeRead::Varlen(2, 3_936, 64)),
        ],
    ),
    (
        "kv2 g8",
        2,
        8,
        &[600, 33_000],
        &[
            ("v2", DecodeRead::Single(1)),
            ("grouped", DecodeRead::Single(2)),
            ("ragged grouped", DecodeRead::Ragged(&[(2, 2)])),
        ],
    ),
];

/// The output of `read` for queries `q` `[rows, hq, d]` (bf16 values).
fn decode_read(
    adapter: &mut PagedKVCacheAdapter,
    read: DecodeRead,
    q: &[f32],
    hq: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let queries = bf16_array(q, &[read.rows() as i64, hq as i64, d as i64]);
    let out = match read {
        DecodeRead::Single(seq) => {
            adapter.activate_request(seq).unwrap();
            adapter.gather_kv_for_decode_graph(0, &queries, scale, 1.0)
        }
        DecodeRead::Batched(seqs) => {
            adapter.gather_kv_for_decode_graph_batched(0, &queries, seqs, scale, 1.0)
        }
        DecodeRead::Ragged(rows) => {
            let rows: Vec<PagedRaggedRow> = rows
                .iter()
                .map(|&(seq_id, query_len)| PagedRaggedRow {
                    seq_id,
                    first_logical_position: adapter.current_token_count_for(seq_id).unwrap()
                        - query_len,
                    query_len,
                })
                .collect();
            adapter.gather_kv_for_ragged_graph(0, &queries, &rows, scale, 1.0)
        }
        DecodeRead::Varlen(seq, committed, _) => {
            adapter.activate_request(seq).unwrap();
            adapter.gather_kv_for_prefill_chunk_varlen(0, &queries, committed, scale)
        }
    }
    .unwrap_or_else(|error| panic!("{read:?}: {error}"));
    host(&out)
}

/// FNV-1a 64 of every read of `DECODE_FIXTURES` over INT8 pools with
/// activations bf16, in order, taken on the tree before the decode kernels
/// took score scales.
const GOLDEN_DECODE: &[u64] = &[
    0xf0a8_aac4_0929_6902,
    0x9f1c_db83_4021_01f0,
    0xb774_78d3_253d_bb44,
    0x837d_a55c_82af_7df6,
    0x3ccd_f380_5ca2_2eb2,
    0x3ecb_b0f3_9b78_b697,
    0x6903_3914_87aa_9544,
    0xaf58_d37a_81bd_262f,
    0x2abd_59fc_20fe_7fff,
    0x822e_38cf_6986_3a8a,
];

/// With activations bf16 (no score scales) every decode-family kernel over
/// an INT8 pool is bit-identical to the tree before it took score scales:
/// generic v1 and v2, grouped int8, batched, ragged (v2 and grouped) and a
/// varlen prefill piece, on the 27B and 35B geometries.
#[test]
fn decode_kernels_keep_their_bits_without_score_scales() {
    let mut hashes = Vec::new();
    for &(label, kv, group, lengths, reads) in DECODE_FIXTURES {
        let Some(mut adapter) = filled_sequences(MetalDtype::Int8, kv, 256, lengths) else {
            return;
        };
        adapter.set_activations(Activations::Bf16);
        let hq = kv * group;
        for (index, &(name, read)) in reads.iter().enumerate() {
            let q = uniform(read.rows() * hq * 256, 41 + index as u64, 2.0);
            let hash = fnv(&decode_read(&mut adapter, read, &q, hq, 256));
            eprintln!("GOLDEN decode {label} {name}: {hash:#018x}");
            hashes.push(hash);
        }
    }
    assert_eq!(hashes, GOLDEN_DECODE, "a decode kernel changed its bits");
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Root mean square of `a - b`.
fn rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum();
    (sum / a.len() as f64).sqrt()
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The query of KV index `position`: [`outlier_queries`] of one row seeded by
/// the position, so a row has the same query in every launch.
fn row_query(position: u32, hq: usize, d: usize) -> Vec<f32> {
    outlier_queries(1, hq, d, 1_000_003 + u64::from(position))
}

fn rows_query(first: u32, rows: usize, hq: usize, d: usize) -> Vec<f32> {
    (first..first + rows as u32)
        .flat_map(|p| row_query(p, hq, d))
        .collect()
}

/// `paged_query_codes` on `q` `[rows, hq, d]` (bf16 values) at `positions`:
/// (codes as f32, score scales).
fn query_codes_op(
    q: &[f32],
    (rows, hq, d): (usize, usize, usize),
    positions: &[i32],
    from: u32,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q = bf16_array(q, &[rows as i64, hq as i64, d as i64]);
    let positions = MxArray::from_int32(positions, &[rows as i64]).unwrap();
    let mut codes = std::ptr::null_mut();
    let mut score_scales = std::ptr::null_mut();
    // SAFETY: live inputs; the out pointers are written on success.
    let ok = unsafe {
        mlx_sys::mlx_paged_query_codes_forward(
            q.as_raw_ptr(),
            positions.as_raw_ptr(),
            from,
            scale,
            &mut codes,
            &mut score_scales,
        )
    };
    assert!(ok, "paged_query_codes refused");
    let codes = MxArray::from_handle(codes, "codes").unwrap();
    let score_scales = MxArray::from_handle(score_scales, "score_scales").unwrap();
    (host(&codes), host(&score_scales))
}

/// The decode kernels' query codes equal the int8-activation rule (Splash
/// `int8Block` / `int8Code`, host IEEE quotients, ties to even, ±127) bit
/// for bit, and each coded (row, query head) carries the tile's score scale
/// `1/sqrt(D) literal × row scale` as one IEEE product, the literal (read
/// off a row of scale 1) equal to the caller's f32 `1/sqrt(D)`; a zero row and a row of magnitudes at
/// most 2^-100 code as zeros with scale 0, a row holding an infinity or a
/// NaN takes a NaN scale; rows below `from` keep their queries and the
/// caller's scale bit for bit. D 128 and 256.
/// Mutation codes-trunc (the quantizer truncates for rint; Metal source
/// edit) → red.
#[test]
fn query_codes_match_the_host_rule() {
    if !scores_int8_queries("query_codes_match_the_host_rule") {
        return;
    }
    const ROWS: usize = 6;
    const HQ: usize = 3;
    let positions = [10, 20, 30, 40, 5, 7_000];
    let from = 20;
    let caller_scale = 0.123_f32;
    let mut failures = Vec::new();
    for d in [128usize, 256] {
        let mut q = outlier_queries(ROWS, HQ, d, d as u64);
        let at = |row: usize, head: usize| (row * HQ + head) * d;
        let tiny = at(2, 1);
        for (i, x) in q[tiny..tiny + d].iter_mut().enumerate() {
            *x = f32::from_bits(0x0880_0000) * (i % 7) as f32;
        }
        q[at(3, 2) + 5] = f32::INFINITY;
        q[at(5, 0) + 9] = f32::NAN;
        // Row scale exactly 1: its score scale is the kernel's literal.
        q[at(1, 2) + 3] = 127.0;
        let (codes, scales) = query_codes_op(&q, (ROWS, HQ, d), &positions, from, caller_scale);
        let literal = scales[HQ + 2];
        let host = 1.0 / (d as f32).sqrt();
        eprintln!(
            "D{d}: kernel literal {:#010x}, host 1/sqrt(D) {:#010x}, decimal literal {:#010x}",
            literal.to_bits(),
            host.to_bits(),
            if d == 256 {
                0.0625f32
            } else {
                0.088_388_35_f32
            }
            .to_bits()
        );
        if literal.to_bits() != host.to_bits() {
            failures.push(format!(
                "D{d}: the literal {literal:e} is not the caller's 1/sqrt(D) {host:e}"
            ));
        }
        for row in 0..ROWS {
            for head in 0..HQ {
                let base = at(row, head);
                let input = &q[base..base + d];
                let got_codes = &codes[base..base + d];
                let got_scale = scales[row * HQ + head];
                let case = format!("D{d} row {row} head {head}");
                if positions[row] < from as i32 {
                    if !same_bits(got_codes, input) || got_scale.to_bits() != caller_scale.to_bits()
                    {
                        failures.push(format!("{case}: a passed-through row changed"));
                    }
                    continue;
                }
                if input.iter().any(|x| !x.is_finite()) {
                    if !got_scale.is_nan() {
                        failures.push(format!("{case}: non-finite row scale {got_scale}"));
                    }
                    continue;
                }
                let (want, row_scale) = query_codes(input);
                let want: Vec<f32> = want.iter().map(|&c| c as f32).collect();
                if !same_bits(got_codes, &want) {
                    failures.push(format!("{case}: codes differ"));
                }
                let want_scale = literal * row_scale;
                if got_scale.to_bits() != want_scale.to_bits() {
                    failures.push(format!(
                        "{case}: score scale {got_scale:e} want {want_scale:e}"
                    ));
                }
            }
        }
    }
    failures.truncate(12);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Queries in the exact domain of the INT8 codes: per (row, query head)
/// integers times a power of two of its own (2^-3 to 2^-9 by row and head)
/// with largest magnitude exactly 127 integers, so the codes are the
/// integers and each (row, head) has its own score scale. Over INT8 pages
/// the INT8-query tile then equals the bf16-query tile, and every
/// decode-family kernel with codes equals itself with bf16 queries, bit for
/// bit (QK is the same integer, the scales powers of two), varlen pieces of
/// one and of two dispatch tiles included.
/// Mutations scale-dropped (a coded read keeps the caller's scale), ss-head
/// (every head reads head 0's score scale), ss-row (every row reads row 0's)
/// and varlen-offset (each varlen tile binds the score scales at row 0;
/// C++ source edit) → red.
#[test]
fn exact_codes_score_like_bf16_queries() {
    if !scores_int8_queries("exact_codes_score_like_bf16_queries") {
        return;
    }
    let exact = |rows: usize, hq: usize, d: usize, seed: u64| -> Vec<f32> {
        let mut q: Vec<f32> = uniform(rows * hq * d, seed, 126.0)
            .iter()
            .map(|x| x.round())
            .collect();
        for (index, row) in q.chunks_mut(d).enumerate() {
            row[(index * 5) % d] = if index % 2 == 0 { 127.0 } else { -127.0 };
            let (r, head) = (index / hq, index % hq);
            let factor = 2f32.powi(-3 - ((r * 3 + head * 5) % 7) as i32);
            for x in row.iter_mut() {
                *x *= factor;
            }
        }
        q
    };
    let mut failures = Vec::new();
    let Some((mut adapter, _)) = filled(MetalDtype::Int8, 4, 256, 16, 4_200, false) else {
        return;
    };
    adapter.set_int8_query_threshold(0);
    for (committed, rows, pass) in [(0u32, 2048usize, Pass::Prefill), (4_000, 8, Pass::Verify)] {
        let q = exact(rows, 24, 256, u64::from(committed) + 3);
        adapter.set_activations(Activations::Bf16);
        let bf16 = attend(&mut adapter, &q, (rows, 24, 256), committed, pass);
        adapter.set_activations(Activations::Auto);
        let before = adapter.tile_query_launches()[1];
        let int8 = attend(&mut adapter, &q, (rows, 24, 256), committed, pass);
        assert_eq!(
            adapter.tile_query_launches()[1],
            before + 1,
            "no q8q launch"
        );
        if !same_bits(&bf16, &int8) {
            failures.push(format!(
                "tile {pass:?} (P {committed}, T {rows}): {}",
                max_abs(&bf16, &int8)
            ));
        }
    }
    for &(label, kv, group, lengths, reads) in DECODE_FIXTURES {
        let Some(mut adapter) = filled_sequences(MetalDtype::Int8, kv, 256, lengths) else {
            return;
        };
        adapter.set_int8_query_threshold(0);
        let hq = kv * group;
        let mut reads = reads.to_vec();
        if (kv, group) == (4, 6) {
            // At 16.4K a 64-row varlen piece runs in two dispatch tiles.
            reads.push(("varlen two tiles", DecodeRead::Varlen(3, 16_436, 64)));
            reads.push(("varlen deep 8 rows", DecodeRead::Varlen(3, 16_492, 8)));
        }
        for (index, &(name, read)) in reads.iter().enumerate() {
            let q = exact(read.rows(), hq, 256, 77 + index as u64);
            adapter.set_activations(Activations::Bf16);
            let bf16 = decode_read(&mut adapter, read, &q, hq, 256);
            adapter.set_activations(Activations::Auto);
            let before = adapter.decode_query_rows()[1];
            let int8 = decode_read(&mut adapter, read, &q, hq, 256);
            assert_eq!(
                adapter.decode_query_rows()[1],
                before + read.rows() as u64,
                "{label} {name}: rows not coded"
            );
            if !same_bits(&bf16, &int8) {
                failures.push(format!("{label} {name}: {}", max_abs(&bf16, &int8)));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A decode-family row scores like the same row in a verify block: the last
/// row of an 8-row verify through the tile against the 1-row decode of that
/// row, at histories that take generic v1 (300), v2 (4,000) and the grouped
/// int8 kernel (16,500; 33,000 at the 35B geometry), the threshold at 0. With
/// INT8 queries their RMS distance stays within twice the same distance with
/// bf16 queries (QK is one integer in both kernels; softmax, PV and the
/// partition differ as they do with bf16 queries).
/// Mutations ar-bf16 (the decode keeps bf16 queries) and scale-dropped →
/// red.
#[test]
fn a_decode_row_scores_like_its_verify_row() {
    if !scores_int8_queries("a_decode_row_scores_like_its_verify_row") {
        return;
    }
    let mut failures = Vec::new();
    for &(label, kv, group, lengths, _) in DECODE_FIXTURES {
        let Some(mut adapter) = filled_sequences(MetalDtype::Int8, kv, 256, lengths) else {
            return;
        };
        adapter.set_int8_query_threshold(0);
        let hq = kv * group;
        for (index, &tokens) in lengths.iter().enumerate() {
            let seq = index as u32 + 1;
            let q = rows_query(tokens - 8, 8, hq, 256);
            let last = &q[7 * hq * 256..];
            let mut distance = [0f64; 2];
            let mut largest = [0f32; 2];
            for (slot, activations) in [Activations::Bf16, Activations::Auto]
                .into_iter()
                .enumerate()
            {
                adapter.set_activations(activations);
                adapter.activate_request(seq).unwrap();
                let verify = attend(&mut adapter, &q, (8, hq, 256), tokens - 8, Pass::Verify);
                let decode = decode_read(&mut adapter, DecodeRead::Single(seq), last, hq, 256);
                distance[slot] = rms(&verify[7 * hq * 256..], &decode);
                largest[slot] = max_abs(&verify[7 * hq * 256..], &decode);
            }
            eprintln!(
                "K1 {label} history {tokens}: verify vs decode rms bf16 {:.3e}, int8 {:.3e} \
                 (ratio {:.2}); max |a - b| {:.3e}, {:.3e}",
                distance[0],
                distance[1],
                distance[1] / distance[0].max(f64::MIN_POSITIVE),
                largest[0],
                largest[1]
            );
            if distance[1] > 2.0 * distance[0] {
                failures.push(format!(
                    "{label} history {tokens}: int8 {:e} > 2 x bf16 {:e}",
                    distance[1], distance[0]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The 27B's hidden width.
const HIDDEN: usize = 5_120;

/// A query projection as the model runs one: hidden rows `[rows, 5120]`
/// times a `[hq * d, 5120]` weight, bf16 or 4-bit affine (group 64), then the
/// per-head RMS norm (eps 1e-6) with channel 5 of every head lifted `lift`
/// times (an outlier channel; 1 for none).
struct Projection {
    linear: crate::nn::linear::Linear,
    norm: MxArray,
    hq: usize,
    d: usize,
}

impl Projection {
    fn new(hq: usize, d: usize, quantized: bool, lift: f32, seed: u64) -> Projection {
        // SAFETY: seeds MLX's global key; the weights are random.
        unsafe { mlx_sys::mlx_seed(seed) };
        let out = (hq * d) as i64;
        let weight =
            MxArray::random_normal(&[out, HIDDEN as i64], 0.0, 0.014, Some(DType::BFloat16))
                .expect("projection weight");
        let linear = if quantized {
            let (mut packed, mut scales, mut biases) = (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            // SAFETY: a live bf16 weight; the out pointers are written on success.
            let ok = unsafe {
                mlx_sys::mlx_quantize(
                    weight.as_raw_ptr(),
                    64,
                    4,
                    c"affine".as_ptr(),
                    &mut packed,
                    &mut scales,
                    &mut biases,
                )
            };
            assert!(ok, "quantize");
            let packed = MxArray::from_handle(packed, "packed").unwrap();
            let scales = MxArray::from_handle(scales, "scales").unwrap();
            let biases = MxArray::from_handle(biases, "biases").unwrap();
            let mut linear =
                crate::nn::linear::Linear::new(HIDDEN as u32, out as u32, Some(false)).unwrap();
            linear
                .load_quantized_mode(&packed, &scales, Some(&biases), 64, 4, "affine")
                .unwrap();
            linear
        } else {
            crate::nn::linear::Linear::from_weights(&weight, None).unwrap()
        };
        let mut norm = vec![1.0f32; d];
        norm[5] = lift;
        Projection {
            linear,
            norm: bf16_array(&norm, &[d as i64]),
            hq,
            d,
        }
    }

    /// The queries `[rows, hq, d]` of hidden rows `x` `[rows, 5120]`, all
    /// rows in one projection.
    fn queries(&self, x: &[f32]) -> Vec<f32> {
        let rows = (x.len() / HIDDEN) as i64;
        let q = self
            .linear
            .forward(&bf16_array(x, &[rows, HIDDEN as i64]))
            .and_then(|q| q.reshape(&[rows, self.hq as i64, self.d as i64]))
            .and_then(|q| crate::nn::normalization::rms_norm_scaled(&q, &self.norm, 1e-6))
            .expect("projection");
        host(&q)
    }
}

/// Elements whose bits differ.
fn differing(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// A decode row against its verify row when each gets its query from its
/// own projection, as in the model: an 8-row verify projects 8 hidden rows
/// at once, the 1-row decode projects the last one alone (bf16 and 4-bit
/// weights; a lifted channel), on the histories and geometries of
/// [`a_decode_row_scores_like_its_verify_row`], the threshold at 0. The two
/// projections round apart on a few elements at most, and with INT8 queries
/// the RMS distance per case is at most `PER_CASE` times the bf16 one, over
/// every case of a projection at most `OVERALL` times (measured 0.87-1.30
/// and 0.98-1.11). The N-lane version: a batched decode of
/// three lanes whose queries come from one 3-row projection against each
/// lane decoded alone from its 1-row projection: bit for bit in both modes
/// when the projections agree bit for bit, else within `PER_CASE` times
/// the bf16 distance.
/// Mutations ar-bf16 (the decode keeps bf16 queries), verify-bf16 (the
/// verify keeps bf16 queries), rows16 and ss-row (every row reads row 0's
/// score scale) → red.
#[test]
fn projected_decode_rows_score_like_their_verify_rows() {
    const PER_CASE: f64 = 2.0;
    const OVERALL: f64 = 1.5;
    if !scores_int8_queries("projected_decode_rows_score_like_their_verify_rows") {
        return;
    }
    let mut failures = Vec::new();
    for &(label, kv, group, lengths, _) in DECODE_FIXTURES {
        let Some(mut adapter) = filled_sequences(MetalDtype::Int8, kv, 256, lengths) else {
            return;
        };
        adapter.set_int8_query_threshold(0);
        let hq = kv * group;
        let width = hq * 256;
        for (name, quantized, lift) in [
            ("bf16", false, 1.0),
            ("q4", true, 1.0),
            ("q4 lift 8", true, 8.0),
        ] {
            let projection = Projection::new(hq, 256, quantized, lift, 0x51_0000 + kv as u64);
            let mut sums = [0f64; 2];
            for (index, &tokens) in lengths.iter().enumerate() {
                let seq = index as u32 + 1;
                let x = uniform(8 * HIDDEN, 0x6b_0000 + u64::from(tokens), 1.0);
                let block = projection.queries(&x);
                let alone = projection.queries(&x[7 * HIDDEN..]);
                let apart = differing(&block[7 * width..], &alone);
                let mut distance = [0f64; 2];
                for (slot, activations) in [Activations::Bf16, Activations::Auto]
                    .into_iter()
                    .enumerate()
                {
                    adapter.set_activations(activations);
                    adapter.activate_request(seq).unwrap();
                    let verify =
                        attend(&mut adapter, &block, (8, hq, 256), tokens - 8, Pass::Verify);
                    let decode =
                        decode_read(&mut adapter, DecodeRead::Single(seq), &alone, hq, 256);
                    distance[slot] = rms(&verify[7 * width..], &decode);
                    sums[slot] += distance[slot].powi(2);
                }
                let ratio = distance[1] / distance[0].max(f64::MIN_POSITIVE);
                eprintln!(
                    "K1b {label} {name} history {tokens}: {apart} of {width} query elements \
                     apart; verify vs decode rms bf16 {:.3e}, int8 {:.3e} (ratio {ratio:.2})",
                    distance[0], distance[1]
                );
                if ratio > PER_CASE {
                    failures.push(format!("{label} {name} history {tokens}: ratio {ratio:.2}"));
                }
            }
            let overall = (sums[1] / sums[0].max(f64::MIN_POSITIVE)).sqrt();
            eprintln!("K1b {label} {name}: ratio over the histories {overall:.2}");
            if overall > OVERALL {
                failures.push(format!(
                    "{label} {name}: ratio over the histories {overall:.2}"
                ));
            }
        }
    }

    let (label, kv, group, lengths, _) = DECODE_FIXTURES[0];
    let Some(mut adapter) = filled_sequences(MetalDtype::Int8, kv, 256, lengths) else {
        return;
    };
    adapter.set_int8_query_threshold(0);
    let hq = kv * group;
    let width = hq * 256;
    let lanes: &'static [u32] = &[1, 2, 3];
    for (name, quantized) in [("bf16", false), ("q4", true)] {
        let projection = Projection::new(hq, 256, quantized, 1.0, 0x52_0000);
        let x = uniform(3 * HIDDEN, 0x6c_0001, 1.0);
        let batch = projection.queries(&x);
        let alone: Vec<Vec<f32>> = (0..3)
            .map(|lane| projection.queries(&x[lane * HIDDEN..(lane + 1) * HIDDEN]))
            .collect();
        let apart: usize = (0..3)
            .map(|lane| differing(&batch[lane * width..(lane + 1) * width], &alone[lane]))
            .sum();
        let mut distance = [0f64; 2];
        for (slot, activations) in [Activations::Bf16, Activations::Auto]
            .into_iter()
            .enumerate()
        {
            adapter.set_activations(activations);
            let batched = decode_read(&mut adapter, DecodeRead::Batched(lanes), &batch, hq, 256);
            let lone: Vec<f32> = (0..3)
                .flat_map(|lane| {
                    decode_read(
                        &mut adapter,
                        DecodeRead::Single(lanes[lane]),
                        &alone[lane],
                        hq,
                        256,
                    )
                })
                .collect();
            distance[slot] = rms(&batched, &lone);
            if apart == 0 && !same_bits(&batched, &lone) {
                failures.push(format!(
                    "{label} {name} lanes, {activations:?}: equal queries, outputs apart {:e}",
                    max_abs(&batched, &lone)
                ));
            }
        }
        let ratio = distance[1] / distance[0].max(f64::MIN_POSITIVE);
        eprintln!(
            "K2N {label} {name}: {apart} of {} query elements apart; lanes vs alone rms bf16 \
             {:.3e}, int8 {:.3e}",
            3 * width,
            distance[0],
            distance[1]
        );
        if apart > 0 && ratio > PER_CASE {
            failures.push(format!("{label} {name} lanes: ratio {ratio:.2}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A row past the threshold scores the same whatever launch holds it: row
/// 2,060 (threshold 2,048) in a 64-row piece, a 2,048-row chunk, a verify
/// block and a launch across the threshold stays within twice the RMS
/// spread the same launches show with bf16 queries; decode lanes equal their lone
/// decodes (a lane at 899 keeps bf16 queries beside a coded lane at 2,060).
/// Mutations rows16 (q8q iff 16 rows or more) and v-shared (a batched
/// launch codes every lane once one lane is coded) → red.
#[test]
fn a_row_does_not_follow_its_launch() {
    if !scores_int8_queries("a_row_does_not_follow_its_launch") {
        return;
    }
    const HQ: usize = 24;
    const D: usize = 256;
    const P: u32 = 2_060;
    let Some(mut adapter) = filled_contents(
        MetalDtype::Int8,
        4,
        D,
        &[(4_100, 1), (P + 1, 2), (P + 1, 2), (900, 3), (900, 3)],
    ) else {
        return;
    };
    adapter.set_int8_query_threshold(2_048);
    let row = |out: &[f32], index: u32| {
        out[index as usize * HQ * D..(index as usize + 1) * HQ * D].to_vec()
    };
    let launches: [(u32, usize, Pass); 4] = [
        (2_048, 64, Pass::Prefill),
        (2_048, 2_048, Pass::Prefill),
        (P - 7, 8, Pass::Verify),
        (2_000, 64, Pass::Prefill),
    ];
    let mut spread = [0f64; 2];
    for (slot, activations) in [Activations::Bf16, Activations::Auto]
        .into_iter()
        .enumerate()
    {
        adapter.set_activations(activations);
        adapter.activate_request(1).unwrap();
        let rows: Vec<Vec<f32>> = launches
            .iter()
            .map(|&(first, rows, pass)| {
                let out = attend(
                    &mut adapter,
                    &rows_query(first, rows, HQ, D),
                    (rows, HQ, D),
                    first,
                    pass,
                );
                row(&out, P - first)
            })
            .collect();
        spread[slot] = rows[1..]
            .iter()
            .map(|r| rms(&rows[0], r))
            .fold(0.0, f64::max);
    }
    eprintln!(
        "K2 tile spread (rms) of row {P}: bf16 {:.3e}, int8 {:.3e}",
        spread[0], spread[1]
    );
    assert!(
        spread[1] <= 2.0 * spread[0],
        "row {P} follows its launch: int8 spread {:e} > 2 x bf16 {:e}",
        spread[1],
        spread[0]
    );

    adapter.set_activations(Activations::Auto);
    let lanes = [row_query(899, HQ, D), row_query(P, HQ, D)].concat();
    let lone_short = decode_read(&mut adapter, DecodeRead::Single(4), &lanes[..HQ * D], HQ, D);
    let lone_long = decode_read(&mut adapter, DecodeRead::Single(2), &lanes[HQ * D..], HQ, D);
    let before = adapter.decode_query_rows();
    let batched = decode_read(&mut adapter, DecodeRead::Batched(&[5, 3]), &lanes, HQ, D);
    let after = adapter.decode_query_rows();
    assert_eq!(
        [after[0] - before[0], after[1] - before[1]],
        [1, 1],
        "lanes [bf16, int8]"
    );
    assert!(
        same_bits(&batched[..HQ * D], &lone_short),
        "the bf16 lane follows its batch: {:e}",
        max_abs(&batched[..HQ * D], &lone_short)
    );
    assert!(
        same_bits(&batched[HQ * D..], &lone_long),
        "the coded lane follows its batch: {:e}",
        max_abs(&batched[HQ * D..], &lone_long)
    );
}

/// The decode rule turns exactly at the threshold, on the row's KV index:
/// with the threshold at 1,000 a 1-row decode of the row at 999 keeps bf16
/// queries (its bits are the bf16 read's), the row at 1,000 is coded; a
/// batched launch of both and a ragged launch of rows 999 and 1,000 code one
/// row each and keep the bits of their bf16 row. The batched launch of the
/// 27B decode fixture with the threshold at 4,000 keeps the no-scale bits of
/// its two rows below it.
/// Mutations gt (`>` for `>=`) and off-by-one (the context length for the
/// KV index) → red.
#[test]
fn the_decode_rule_turns_at_the_threshold() {
    if !scores_int8_queries("the_decode_rule_turns_at_the_threshold") {
        return;
    }
    const HQ: usize = 24;
    const D: usize = 256;
    let Some(mut adapter) = filled_contents(MetalDtype::Int8, 4, D, &[(1_000, 1), (1_001, 1)])
    else {
        return;
    };
    adapter.set_int8_query_threshold(1_000);
    let q999 = row_query(999, HQ, D);
    let q1000 = row_query(1_000, HQ, D);
    let both = [q999.clone(), q1000.clone()].concat();
    let reads: [(&str, DecodeRead, Vec<f32>, [u64; 2]); 4] = [
        ("decode 999", DecodeRead::Single(1), q999, [1, 0]),
        ("decode 1000", DecodeRead::Single(2), q1000, [0, 1]),
        (
            "batched",
            DecodeRead::Batched(&[1, 2]),
            both.clone(),
            [1, 1],
        ),
        ("ragged", DecodeRead::Ragged(&[(2, 2)]), both, [1, 1]),
    ];
    let mut failures = Vec::new();
    for (name, read, q, want) in reads {
        adapter.set_activations(Activations::Bf16);
        let bf16 = decode_read(&mut adapter, read, &q, HQ, D);
        adapter.set_activations(Activations::Auto);
        let before = adapter.decode_query_rows();
        let auto = decode_read(&mut adapter, read, &q, HQ, D);
        let after = adapter.decode_query_rows();
        let took = [after[0] - before[0], after[1] - before[1]];
        if took != want {
            failures.push(format!("{name}: rows [bf16, int8] {took:?}, want {want:?}"));
        }
        let below = want[0] as usize * HQ * D;
        if !same_bits(&auto[..below], &bf16[..below]) {
            failures.push(format!("{name}: a bf16 row changed its bits"));
        }
        if want[1] > 0 && auto[below..] == bf16[below..] {
            failures.push(format!("{name}: the coded row kept bf16 queries"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    let (_, kv, group, lengths, _) = DECODE_FIXTURES[0];
    let Some(mut adapter) = filled_sequences(MetalDtype::Int8, kv, D, lengths) else {
        return;
    };
    adapter.set_int8_query_threshold(4_000);
    let read = DecodeRead::Batched(&[1, 2, 3]);
    let q = uniform(3 * kv * group * D, 91, 2.0);
    adapter.set_activations(Activations::Bf16);
    let bf16 = decode_read(&mut adapter, read, &q, kv * group, D);
    adapter.set_activations(Activations::Auto);
    let auto = decode_read(&mut adapter, read, &q, kv * group, D);
    let below = 2 * kv * group * D;
    assert!(
        same_bits(&auto[..below], &bf16[..below]),
        "a mixed batched launch changed its bf16 rows"
    );
    assert!(
        auto[below..] != bf16[below..],
        "the coded lane kept bf16 queries"
    );
}

#[path = "host_block_table_tests.rs"]
mod host_block_table_tests;

/// A dense draft load that names no activations takes the position rule at
/// the device's threshold (7,168 on Apple10 and later): over the 27B geometry
/// a 64-row piece ending at 7,168 and a verify ending at it launch bf16
/// queries, the same rows from 7,168 launch INT8 queries, which the
/// attention trace marks `queries=bf16` / `queries=int8` from these
/// counters.
/// Mutation rule-off (the rule never fires) → red.
#[test]
fn a_default_dense_load_scores_int8_queries_from_the_threshold() {
    if !scores_int8_queries("a_default_dense_load_scores_int8_queries_from_the_threshold") {
        return;
    }
    const KV: usize = 4;
    const GROUP: usize = 6;
    const D: usize = 256;
    let hq = KV * GROUP;
    let Some((mut adapter, _)) = filled(MetalDtype::Int8, KV, D, 16, 7_232, false) else {
        return;
    };
    adapter.set_activations(
        Activations::parse(
            None,
            crate::models::qwen3_5::load_plan::default_activations(false, true),
        )
        .expect("no option"),
    );
    assert_eq!(
        adapter.int8_queries_from(),
        Some(7_168),
        "the rule's threshold"
    );
    let mut wrong = Vec::new();
    for (committed, rows, pass, want) in [
        (7_104u32, 64usize, Pass::Prefill, [1u64, 0]),
        (7_160, 8, Pass::Verify, [1, 0]),
        (7_168, 64, Pass::Prefill, [0, 1]),
        (7_168, 8, Pass::Verify, [0, 1]),
    ] {
        let q = uniform(rows * hq * D, u64::from(committed) + rows as u64, 2.0);
        let before = adapter.tile_query_launches();
        attend(&mut adapter, &q, (rows, hq, D), committed, pass);
        let after = adapter.tile_query_launches();
        let took = [after[0] - before[0], after[1] - before[1]];
        println!("DEFAULTRULE {pass:?} P {committed} T {rows}: launches [bf16, int8] {took:?}");
        if took != want {
            wrong.push(format!(
                "{pass:?} (P {committed}, T {rows}): launches [bf16, q8q] {took:?}, want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A dense load with no draft that names no activations scores bf16 queries
/// at every row: over the 27B geometry a 64-row piece and a verify from
/// 7,168 launch bf16 queries only, and its decode rows are bf16 rows.
/// Mutation draft-ignored (the default reads no draft flag) → red.
#[test]
fn a_dense_load_with_no_draft_scores_bf16_queries() {
    if !scores_int8_queries("a_dense_load_with_no_draft_scores_bf16_queries") {
        return;
    }
    const KV: usize = 4;
    const GROUP: usize = 6;
    const D: usize = 256;
    let hq = KV * GROUP;
    let Some((mut adapter, _)) = filled(MetalDtype::Int8, KV, D, 16, 7_232, false) else {
        return;
    };
    let unset = Activations::parse(
        None,
        crate::models::qwen3_5::load_plan::default_activations(false, false),
    )
    .expect("no option");
    adapter.set_activations(unset);
    let mut wrong = Vec::new();
    for (committed, rows, pass) in [
        (7_104u32, 64usize, Pass::Prefill),
        (7_168, 64, Pass::Prefill),
        (7_168, 8, Pass::Verify),
    ] {
        let q = uniform(rows * hq * D, u64::from(committed) + rows as u64, 2.0);
        let before = adapter.tile_query_launches();
        attend(&mut adapter, &q, (rows, hq, D), committed, pass);
        let after = adapter.tile_query_launches();
        let took = [after[0] - before[0], after[1] - before[1]];
        println!("NODRAFT {pass:?} P {committed} T {rows}: launches [bf16, int8] {took:?}");
        if took != [1, 0] {
            wrong.push(format!(
                "{pass:?} (P {committed}, T {rows}): launches [bf16, q8q] {took:?}, want [1, 0]"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(unset, Activations::Bf16, "the no-draft default");
}
