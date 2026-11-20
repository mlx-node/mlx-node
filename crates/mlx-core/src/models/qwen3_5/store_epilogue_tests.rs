//! The SwiGLU and the residual add in the Tiled64 K-quant GEMM store against
//! the separate ops, bit for bit: every K-quant format of Qwen3.8-27B
//! UD-Q4_K_XL at its MLP and mixer-output shapes, the row counts a prefill
//! hands them, hostile values and every bfloat16 gate and residual pattern;
//! the route (the epilogue only where the plain op runs the tensor-op GEMM);
//! and a whole prefill of an 8-layer stack at the 27B geometry.

use std::ffi::CStr;

use napi::{Error, Result};

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::{LinearProj, MLPVariant, QuantizedLinear, StoreEpilogue};
use crate::nn::{Activations, Embedding, Linear, RMSNorm};
use crate::stream::{Stream, StreamContext};
use crate::transformer::paged_int8_tests::adapter_for;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

use super::decoder_layer::{Qwen3_5LayerKind, compute_layer_kinds};
use super::gdn_sidecar;
use super::layer_cache::Qwen3_5LayerCache;
use super::paged_forward::{
    PREFILL_SEPARATE_OPS, PrefillParts, PrefillTapSink, run_paged_prefill_chunk_with_size,
};
use super::prefill_kernel_timing::{
    BLOCKS, Bank, Fmt, IQ3S, IQ4NL, IQ4XS, Mixer, Q2K, Q3K, Q4K, Q5K, Q6K, Q8, bf16_input,
    build_linear, config, layer_of, packed_rows, share,
};
use super::prefill_tail_tests::{RecordingSink, bits, gdn_bits};

const HIDDEN: i64 = 5120;
const FFN: i64 = 17408;
const MIXER_K: i64 = 6144;
const VOCAB: i64 = 16;
/// Every K-quant format of the 27B's MLPs and mixer outputs.
const FORMATS: [Fmt; 7] = [Q3K, Q4K, Q5K, Q6K, IQ4NL, IQ4XS, IQ3S];
const EPILOGUE_FAMILIES: [&CStr; 2] = [c"qmm_t_nax_t64_res", c"qmm_t_nax_t64_swiglu"];

fn unsupported(test: &str) -> bool {
    crate::test_support::paged_prefill_unsupported(test)
}

/// `f` with the K-quant family counters on: `(result, [residual, swiglu])`
/// epilogue dispatches and the plain tensor-op GEMM's.
fn counted<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, [u64; 2], u64)> {
    // SAFETY: test-only thread-local counter hooks.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = f();
    // SAFETY: NUL-terminated names.
    let count = |name: &CStr| unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) };
    let eps = [count(EPILOGUE_FAMILIES[0]), count(EPILOGUE_FAMILIES[1])];
    let plain = count(c"qmm_t_nax_t64");
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    Ok((out?, eps, plain))
}

/// Raw bfloat16 bits (NaN payloads and signed zeros included).
fn raw(a: &MxArray) -> Result<Vec<u16>> {
    a.eval();
    a.to_uint16_native()
}

fn assert_same(got: &MxArray, want: &MxArray, what: &str) -> Result<()> {
    assert_eq!(got.dtype()?, DType::BFloat16, "{what}: dtype");
    assert_eq!(
        got.shape()?.as_ref(),
        want.shape()?.as_ref(),
        "{what}: shape"
    );
    let (g, w) = (raw(got)?, raw(want)?);
    let differing = g.iter().zip(&w).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{what}: {differing} of {} elements differ",
        g.len()
    );
    Ok(())
}

/// `[1, rows, k]` bfloat16 rows by row class: normal, x 1e18 (the sums
/// overflow), zeros, x 1e-36 (subnormal products), x 64; plus an inf, a NaN
/// and a -inf lane.
fn hostile_x(rows: i64, k: i64, seed: u64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let base = MxArray::random_normal(&[rows, k], 0.0, 1.0, Some(DType::Float32))?;
    base.eval();
    let mut v = base.to_float32()?.to_vec();
    for r in 0..rows as usize {
        let scale = match r % 5 {
            0 => 1.0f32,
            1 => 1e18,
            2 => 0.0,
            3 => 1e-36,
            _ => 64.0,
        };
        for x in &mut v[r * k as usize..(r + 1) * k as usize] {
            *x *= scale;
        }
    }
    for (row, col, value) in [
        (5, 7, f32::INFINITY),
        (9, 3, f32::NAN),
        (13, 0, f32::NEG_INFINITY),
    ] {
        if row < rows as usize {
            v[row * k as usize + col] = value;
        }
    }
    let x = MxArray::from_float32(&v, &[1, rows, k])?.astype(DType::BFloat16)?;
    x.eval();
    Ok(x)
}

/// `[1, rows, n]` cycling through all 65,536 bfloat16 patterns from `start`.
fn every_pattern(rows: i64, n: i64, start: u32) -> Result<MxArray> {
    let v: Vec<u16> = (0..(rows * n) as u32)
        .map(|i| (i.wrapping_mul(40_503).wrapping_add(start) & 0xffff) as u16)
        .collect();
    let a = MxArray::from_bfloat16(&v, &[1, rows, n])?;
    a.eval();
    Ok(a)
}

fn linear(fmt: Fmt, n: i64, k: i64, seed: u64) -> Result<QuantizedLinear> {
    build_linear(fmt, packed_rows(fmt, n, k, seed)?)
}

/// gate, up, down and a mixer output (GDN out and attention o share
/// `[5120, 6144]`) of one format, each from its own bytes.
struct Weights {
    gate: QuantizedLinear,
    up: QuantizedLinear,
    down: QuantizedLinear,
    out: QuantizedLinear,
}

fn weights(fmt: Fmt, seed: u64) -> Result<Weights> {
    Ok(Weights {
        gate: linear(fmt, FFN, HIDDEN, seed)?,
        up: linear(fmt, FFN, HIDDEN, seed + 1)?,
        down: linear(fmt, HIDDEN, FFN, seed + 2)?,
        out: linear(fmt, HIDDEN, MIXER_K, seed + 3)?,
    })
}

const ROWS: [i64; 3] = [2048, 33, 1];

/// The epilogue where the route takes it (2048 and 33 rows), the separate
/// ops' bits; `None` at one row.
fn check_epilogue(
    proj: &QuantizedLinear,
    x: &MxArray,
    aux: &MxArray,
    epilogue: StoreEpilogue,
    want: &MxArray,
    what: &str,
) -> Result<()> {
    let rows = x.shape_at(1)?;
    let k = x.shape_at(2)?;
    let route = proj.store_epilogue_route(rows, k)?;
    assert_eq!(route, rows > 1, "{what}: route at {rows} rows");
    let (got, eps, _) = counted(|| {
        let got = proj.try_store_epilogue(x, aux, epilogue)?;
        if let Some(got) = &got {
            got.eval();
        }
        Ok(got)
    })?;
    match got {
        Some(got) => {
            let slot = (epilogue as usize) - 1;
            assert_eq!(eps[slot], 1, "{what}: epilogue dispatches {eps:?}");
            assert_same(&got, want, what)
        }
        None => {
            assert!(
                !route,
                "{what}: the route took the epilogue but the op refused it"
            );
            assert_eq!(eps, [0, 0], "{what}: an epilogue ran");
            Ok(())
        }
    }
}

/// SwiGLU in the up GEMM's store against the compiled `swiglu(gate, up)`:
/// with every bfloat16 gate pattern, and with the gate GEMM's own output;
/// every format, 2,048 / 33 / 1 rows, hostile x.
/// The mutation that swaps the gate and the up is caught by the MLP test below.
#[test]
fn swiglu_store_keeps_the_compiled_swiglu_bits() -> Result<()> {
    if unsupported("swiglu_store_keeps_the_compiled_swiglu_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    for (i, &fmt) in FORMATS.iter().enumerate() {
        let w = weights(fmt, 7000 + 10 * i as u64)?;
        for rows in ROWS {
            let x = hostile_x(rows, HIDDEN, 31 + rows as u64)?;
            let up = w.up.forward(&x)?;
            for (gate, label) in [
                (every_pattern(rows, FFN, i as u32)?, "every gate pattern"),
                (w.gate.forward(&x)?, "gate GEMM"),
            ] {
                gate.eval();
                let want = Activations::swiglu_compiled(&gate, &up)?;
                check_epilogue(
                    &w.up,
                    &x,
                    &gate,
                    StoreEpilogue::Swiglu,
                    &want,
                    &format!("swiglu {} rows {rows} ({label})", fmt.name()),
                )?;
            }
        }
    }
    Ok(())
}

/// `residual + y` in the store of the down projection and of the mixer
/// output projection against the separate add: every bfloat16 residual
/// pattern, every format, 2,048 / 33 / 1 rows, hostile x.
/// Mutation: the residual is read from another row → red.
#[test]
fn residual_store_keeps_the_add_bits() -> Result<()> {
    if unsupported("residual_store_keeps_the_add_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    for (i, &fmt) in FORMATS.iter().enumerate() {
        let w = weights(fmt, 8000 + 10 * i as u64)?;
        for rows in ROWS {
            for (proj, k, label) in [(&w.down, FFN, "down"), (&w.out, MIXER_K, "mixer out")] {
                let x = hostile_x(rows, k, 77 + rows as u64)?;
                let residual = every_pattern(rows, HIDDEN, 3 * i as u32)?;
                let want = residual.add(&proj.forward(&x)?)?;
                check_epilogue(
                    proj,
                    &x,
                    &residual,
                    StoreEpilogue::Residual,
                    &want,
                    &format!("residual {label} {} rows {rows}", fmt.name()),
                )?;
            }
        }
    }
    Ok(())
}

fn mlp(gate: &QuantizedLinear, up: &QuantizedLinear, down: &QuantizedLinear) -> Result<MLPVariant> {
    let mut mlp = MLPVariant::Quantized {
        gate_proj: share(gate),
        up_proj: share(up),
        down_proj: share(down),
        gate_up: None,
    };
    mlp.finalize_gate_up()?;
    Ok(mlp)
}

/// The MLP with both epilogues against `residual + mlp(x)`: a merged gate|up
/// (one format) and an unmerged pair (gate IQ4_XS, up of the format, as
/// blk.0's IQ4_XS / Q3_K), at 2,048 / 33 / 16 / 1 rows (16 rows: the SwiGLU
/// store and the down projection's split-K route). Which epilogues ran is
/// the route's answer for each projection.
/// Mutation: the gate GEMM's store takes the up as the gate → red.
#[test]
fn mlp_store_keeps_the_separate_mlp_bits() -> Result<()> {
    if unsupported("mlp_store_keeps_the_separate_mlp_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let iq4xs_gate = linear(IQ4XS, FFN, HIDDEN, 9500)?;
    for (i, &fmt) in FORMATS.iter().enumerate() {
        let w = weights(fmt, 9000 + 10 * i as u64)?;
        let merged = mlp(&w.gate, &w.up, &w.down)?;
        let split = mlp(&iq4xs_gate, &w.up, &w.down)?;
        assert!(
            matches!(
                &merged,
                MLPVariant::Quantized {
                    gate_up: Some(_),
                    ..
                }
            ),
            "{}: gate|up did not merge",
            fmt.name()
        );
        let unmerged = matches!(&split, MLPVariant::Quantized { gate_up: None, .. });
        assert!(
            unmerged || fmt == IQ4XS,
            "{}: an IQ4_XS gate merged with another format",
            fmt.name()
        );
        for rows in [2048, 33, 16, 1] {
            let x = hostile_x(rows, HIDDEN, 101 + rows as u64)?;
            let residual = every_pattern(rows, HIDDEN, 11 + i as u32)?;
            for (block, label) in [(&merged, "merged"), (&split, "gate IQ4_XS")] {
                let want = residual.add(&block.forward(&x)?)?;
                want.eval();
                let (got, eps, _) = counted(|| {
                    let got = block.forward_store_residual(&x, &residual)?;
                    got.eval();
                    Ok(got)
                })?;
                let MLPVariant::Quantized {
                    up_proj, down_proj, ..
                } = block
                else {
                    unreachable!("quantized fixture");
                };
                let swiglu = up_proj.store_epilogue_route(rows, HIDDEN)?;
                let resid = down_proj.store_epilogue_route(rows, FFN)?;
                assert_eq!(
                    eps,
                    [u64::from(resid), u64::from(swiglu)],
                    "{} {label} rows {rows}: epilogues ran",
                    fmt.name()
                );
                if rows >= 33 {
                    assert!(
                        swiglu && resid,
                        "{} {label} rows {rows}: no epilogue",
                        fmt.name()
                    );
                }
                if rows == 16 {
                    assert!(swiglu && !resid, "{} {label} rows 16: route", fmt.name());
                }
                if rows == 1 {
                    assert!(!swiglu && !resid, "{} {label} rows 1: route", fmt.name());
                }
                assert_same(
                    &got,
                    &want,
                    &format!("mlp {} {label} rows {rows}", fmt.name()),
                )?;
            }
        }
    }
    Ok(())
}

/// The route says yes only where the plain op runs the Tiled64 tensor-op
/// GEMM (the epilogue kernel's sums are that kernel's): every 27B shape at
/// rows around the matvec limit and the split-K rule. A format without
/// epilogue kernels (Q2_K) and Q8_0 on MLX's affine route never take it and
/// keep the separate ops' bits.
/// Mutation: any K-quant mode takes the epilogue → red.
#[test]
fn the_epilogue_takes_only_the_tensor_op_gemm() -> Result<()> {
    if unsupported("the_epilogue_takes_only_the_tensor_op_gemm") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let shapes = [
        (Q4K, FFN, HIDDEN, "gate|up"),
        (Q5K, HIDDEN, FFN, "down"),
        (Q6K, HIDDEN, MIXER_K, "mixer out"),
        (Q4K, 2 * FFN, HIDDEN, "merged gate|up"),
    ];
    for (fmt, n, k, label) in shapes {
        let proj = linear(fmt, n, k, 4000 + n as u64)?;
        let mut taken = Vec::new();
        for rows in [
            1, 2, 8, 9, 10, 11, 12, 13, 16, 17, 18, 24, 32, 33, 48, 64, 65, 2048,
        ] {
            let x = bf16_input(&[1, rows, k], 5 + rows as u64)?;
            let route = proj.store_epilogue_route(rows, k)?;
            let (_, _, plain) = counted(|| {
                let y = proj.forward(&x)?;
                y.eval();
                Ok(())
            })?;
            if route {
                assert_eq!(
                    plain, 1,
                    "{label} rows {rows}: the route took a shape off the GEMM"
                );
                taken.push(rows);
            }
        }
        assert!(taken.contains(&2048), "{label}: 2,048 rows not taken");
        println!("ROW store_route {label} rows_taken={taken:?}");
    }
    for (fmt, label) in [(Q2K, "Q2_K"), (Q8, "Q8_0")] {
        let proj = linear(fmt, HIDDEN, MIXER_K, 4500)?;
        let x = bf16_input(&[1, 2048, MIXER_K], 6)?;
        let residual = every_pattern(2048, HIDDEN, 5)?;
        assert!(
            !proj.store_epilogue_route(2048, MIXER_K)?,
            "{label}: a format without epilogue kernels took the route"
        );
        let want = residual.add(&proj.forward(&x)?)?;
        let lp = LinearProj::Quantized(share(&proj));
        let (got, eps, _) = counted(|| {
            let got = lp.forward_mixer_output(
                &x,
                None,
                Some(crate::models::quantized_linear::MixerResidual::Store(
                    &residual,
                )),
            )?;
            got.eval();
            Ok(got)
        })?;
        assert_eq!(eps, [0, 0], "{label}: an epilogue ran");
        assert_same(&got, &want, &format!("{label} fallback"))?;
    }
    Ok(())
}

/// Blocks of the 27B table as an 8-layer stack (full attention at layers 3
/// and 7): merged and unmerged gate|up, every MLP format (Q3_K, Q4_K, Q5_K,
/// Q6_K, IQ4_NL, IQ4_XS, IQ3_S), GDN outputs in Q8_0 (the separate add),
/// Q6_K and Q5_K, attention outputs in Q6_K and Q5_K.
const STACK: [(Mixer, Fmt, Fmt, Fmt); 8] = [
    BLOCKS[0], BLOCKS[1], BLOCKS[14], BLOCKS[3], BLOCKS[13], BLOCKS[56], BLOCKS[58], BLOCKS[59],
];

/// What a prefill leaves behind.
#[derive(Debug, PartialEq)]
struct Left {
    logits: Vec<u32>,
    gdn: Vec<Vec<u32>>,
    pool: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    taps: Vec<Vec<u32>>,
    captured: Vec<(usize, Vec<Vec<u32>>)>,
    checkpoints: Vec<(u32, Vec<Vec<u32>>)>,
    /// The GDN cells as the persistent tier encodes them: the final state,
    /// each checkpoint, each captured state.
    encoded: Vec<Vec<u8>>,
}

fn pool_bytes(adapter: &mut PagedKVCacheAdapter) -> Result<Vec<Vec<(Vec<u8>, Vec<u8>)>>> {
    adapter
        .eval_pending_pool_writes()
        .map_err(Error::from_reason)?;
    crate::array::synchronize_and_clear_cache();
    let ids = adapter
        .block_table()
        .map(|table| table.block_ids())
        .unwrap_or_default();
    let pool = adapter.layer_kv_pool();
    ids.into_iter()
        .map(|id| pool.read_block_all_layers(id).map_err(Error::from_reason))
        .collect()
}

fn encode(cfg: &super::config::Qwen3_5Config, caches: &[Qwen3_5LayerCache]) -> Result<Vec<u8>> {
    let geo = gdn_sidecar::geometry(cfg, DType::BFloat16)
        .ok_or_else(|| Error::from_reason("no GDN sidecar geometry"))?;
    let bytes = geo
        .cell_bytes()
        .ok_or_else(|| Error::from_reason("no GDN cell size"))?;
    let mut cell = vec![0u8; bytes as usize];
    assert!(
        gdn_sidecar::encode_into(cfg, &geo, caches, &mut cell)?,
        "the GDN state does not match the sidecar geometry"
    );
    Ok(cell)
}

struct Stack {
    cfg: super::config::Qwen3_5Config,
    bank: Bank,
    embed: Embedding,
    final_norm: RMSNorm,
    lm_head: Option<LinearProj>,
}

fn stack() -> Result<Stack> {
    let mut cfg = config();
    cfg.num_layers = STACK.len() as i32;
    let bank = Bank::new(&STACK)?;
    let embed = Embedding::from_weight(&bf16_input(&[VOCAB, HIDDEN], 61)?)?;
    let final_norm = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
    let mut head = Linear::new(HIDDEN as u32, VOCAB as u32, Some(false))?;
    head.set_weight(&bf16_input(&[VOCAB, HIDDEN], 62)?.mul_scalar(0.02)?)?;
    Ok(Stack {
        cfg,
        bank,
        embed,
        final_norm,
        lm_head: Some(LinearProj::Standard(head)),
    })
}

/// One whole prefill of `len` tokens in `rows`-row chunks, with draft taps
/// and capture rows; returns what it leaves and the epilogue dispatches.
fn prefill(st: &Stack, len: usize, rows: i32, separate: bool) -> Result<(Left, [u64; 2])> {
    let err = |e: String| Error::from_reason(e);
    let cfg = &st.cfg;
    let mut layers = STACK
        .iter()
        .enumerate()
        .map(|(i, &block)| layer_of(cfg, &st.bank, block, i))
        .collect::<Result<Vec<_>>>()?;
    let kinds = compute_layer_kinds(STACK.len(), |i| cfg.is_linear_layer(i));
    let mut caches: Vec<Qwen3_5LayerCache> = kinds
        .iter()
        .map(|k| match k {
            Qwen3_5LayerKind::Linear => Qwen3_5LayerCache::new_linear(),
            Qwen3_5LayerKind::FullAttentionPaged { .. } => Qwen3_5LayerCache::new_full_attention(),
        })
        .collect();
    let block = cfg.paged_block_size.unwrap_or(16);
    let attention_layers = kinds
        .iter()
        .filter(|k| matches!(k, Qwen3_5LayerKind::FullAttentionPaged { .. }))
        .count() as u32;
    let pool_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: block,
        gpu_memory_mb: 2048,
        head_size: cfg.head_dim as u32,
        num_kv_heads: cfg.num_kv_heads as u32,
        num_layers: attention_layers,
        max_seq_len: Some(262_144),
        max_batch_size: Some(1),
    };
    let blocks = len as u32 / block + 8;
    let (mut adapter, _pool) = adapter_for(
        pool_config,
        blocks,
        blocks,
        mlx_paged_attn::metal::MetalDtype::Int8,
    )
    .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    adapter.reset_for_new_request(0).map_err(err)?;
    adapter.allocate_suffix_blocks(len as u32).map_err(err)?;
    let tokens: Vec<u32> = (0..len as u32)
        .map(|i| (i * 7 + 3) % VOCAB as u32)
        .collect();
    let captures = vec![len / 3, len / 2];
    let mut sink = RecordingSink {
        taps: vec![1, 4, 6],
        captures,
        seen: Vec::new(),
        linear: Vec::new(),
    };
    PREFILL_SEPARATE_OPS.with(|flag| flag.set(separate));
    let ran = counted(|| {
        let mut target = PrefillParts {
            layers: &mut layers,
            caches: &mut caches,
            final_norm: &st.final_norm,
            lm_head: &st.lm_head,
            adapter: &mut adapter,
        };
        let (logits, checkpoints) = run_paged_prefill_chunk_with_size(
            &tokens,
            &tokens,
            0,
            false,
            &st.embed,
            &mut target,
            &kinds,
            rows,
            0,
            None,
            Some(&mut sink as &mut dyn PrefillTapSink),
        )?;
        logits.eval();
        Ok((logits, checkpoints))
    });
    PREFILL_SEPARATE_OPS.with(|flag| flag.set(false));
    let ((logits, checkpoints), eps, _) = ran?;
    let mut encoded = vec![encode(cfg, &caches)?];
    for point in &checkpoints {
        encoded.push(encode(cfg, &point.caches)?);
    }
    for (_, linear) in &sink.linear {
        encoded.push(encode(cfg, linear)?);
    }
    let left = Left {
        logits: bits(&logits)?,
        gdn: gdn_bits(&caches)?,
        pool: pool_bytes(&mut adapter)?,
        taps: sink.seen.iter().map(bits).collect::<Result<Vec<_>>>()?,
        captured: sink
            .linear
            .iter()
            .map(|(row, linear)| Ok((*row, gdn_bits(linear)?)))
            .collect::<Result<Vec<_>>>()?,
        checkpoints: checkpoints
            .iter()
            .map(|point| Ok((point.prefix_len, gdn_bits(&point.caches)?)))
            .collect::<Result<Vec<_>>>()?,
        encoded,
    };
    Ok((left, eps))
}

/// A whole prefill of the 8-layer stack at the 27B geometry (hidden 5120,
/// FFN 17408, 24 / 4 attention heads of 256, 16 / 48 GDN heads of 128, int8
/// pool) with the GEMM store epilogues against the separate ops: 4,129
/// tokens in 2,048-row chunks (2,048, 2,048 and 33 rows, split at the
/// capture rows) and 49 tokens in 16-row chunks (16-row chunks and a 1-row
/// one; only the SwiGLU store applies there). The logits, every pool byte, the GDN state, the draft taps, the
/// captured states, the checkpoints and the GDN cells as the persistent tier
/// encodes them keep their bits; the separate arm runs no epilogue.
/// The gate/up swap and the wrong residual row mutations turn it red.
#[test]
fn a_27b_geometry_prefill_keeps_the_separate_ops_bits() -> Result<()> {
    if unsupported("a_27b_geometry_prefill_keeps_the_separate_ops_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let st = stack()?;
    for (len, rows) in [(4129usize, 2048i32), (49, 16)] {
        let (separate, separate_eps) = prefill(&st, len, rows, true)?;
        let (store, store_eps) = prefill(&st, len, rows, false)?;
        assert_eq!(
            separate_eps,
            [0, 0],
            "{len}/{rows}: the separate arm ran an epilogue"
        );
        // 16-row chunks: the SwiGLU store only (the 5120-wide outputs take
        // split-K there).
        assert!(
            store_eps[1] > 0 && (store_eps[0] > 0) == (rows > 16),
            "{len}/{rows}: the store arm's epilogues {store_eps:?}"
        );
        assert!(!separate.taps.is_empty(), "{len}/{rows}: no taps");
        assert!(
            separate.encoded.len() > 1,
            "{len}/{rows}: no captured state"
        );
        assert_eq!(separate.logits, store.logits, "{len}/{rows}: logits");
        assert!(separate.pool == store.pool, "{len}/{rows}: pool bytes");
        assert!(separate.gdn == store.gdn, "{len}/{rows}: GDN state");
        assert!(separate.taps == store.taps, "{len}/{rows}: taps");
        assert!(
            separate.captured == store.captured,
            "{len}/{rows}: captured states"
        );
        assert!(
            separate.checkpoints == store.checkpoints,
            "{len}/{rows}: checkpoints"
        );
        assert!(
            separate.encoded == store.encoded,
            "{len}/{rows}: encoded GDN cells"
        );
        assert!(
            separate == store,
            "{len}/{rows}: the prefill's bits changed"
        );
        println!(
            "ROW store_prefill len={len} rows={rows} epilogues_res={} epilogues_swiglu={} \
             taps={} encoded={}",
            store_eps[0],
            store_eps[1],
            store.taps.len(),
            store.encoded.len()
        );
    }
    Ok(())
}

#[path = "store_epilogue_edge_tests.rs"]
mod store_epilogue_edge_tests;
