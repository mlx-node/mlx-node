//! Checks of the GEMM store epilogues against the separate
//! ops: row counts at every phase of the 64-row tile, the route at every row
//! count from 1 to 70 for every format and shape, operands bound as views,
//! and a prefill of all 64 blocks of the 27B table that is resumed across
//! the reference order (reshaped block table, head-kernel GDN scan,
//! separate ops) and the new order.

use super::*;
use crate::models::quantized_linear::MixerResidual;
use crate::models::qwen3_5::gated_delta::HEAD_SCAN_PREFILL;
use crate::models::qwen3_5::prefill_kernel_timing::BLOCKS;
use crate::transformer::paged_metadata_cache::RESHAPED_BLOCK_TABLE;

/// Row counts on both sides of each tile edge the store can meet: the
/// matvec limit, one partial tile, full tiles, a full tile plus a partial
/// one, and one row short of a 2,048-row chunk.
const TILE_ROWS: [i64; 17] = [
    10, 11, 13, 17, 31, 32, 33, 47, 63, 64, 65, 97, 127, 128, 129, 191, 2047,
];

fn merged_of(block: &MLPVariant) -> Option<&QuantizedLinear> {
    match block {
        MLPVariant::Quantized {
            gate_up: Some(pair),
            ..
        } => Some(&pair.0),
        _ => None,
    }
}

/// The whole MLP and the mixer output projection with their epilogues at
/// row counts around every tile edge, for all seven formats: the bits of
/// `residual + mlp(x)` and `residual + out(x)`, and the epilogues that ran
/// are the ones the route names.
#[test]
fn rows_at_every_tile_phase_keep_the_separate_bits() -> Result<()> {
    if unsupported("rows_at_every_tile_phase_keep_the_separate_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let mut stored = 0u64;
    for (i, &fmt) in FORMATS.iter().enumerate() {
        let w = weights(fmt, 21_000 + 10 * i as u64)?;
        let block = mlp(&w.gate, &w.up, &w.down)?;
        assert!(
            merged_of(&block).is_some(),
            "{}: gate|up did not merge",
            fmt.name()
        );
        let out = LinearProj::Quantized(share(&w.out));
        for rows in TILE_ROWS {
            let tag = format!("{} rows {rows}", fmt.name());
            let x = hostile_x(rows, HIDDEN, 300 + rows as u64)?;
            let residual = every_pattern(rows, HIDDEN, 17 * rows as u32 + i as u32)?;
            let want = residual.add(&block.forward(&x)?)?;
            want.eval();
            let (got, eps, _) = counted(|| {
                let got = block.forward_store_residual(&x, &residual)?;
                got.eval();
                Ok(got)
            })?;
            let swiglu = w.up.store_epilogue_route(rows, HIDDEN)?;
            let resid = w.down.store_epilogue_route(rows, FFN)?;
            assert_eq!(
                eps,
                [u64::from(resid), u64::from(swiglu)],
                "{tag}: mlp epilogues ran"
            );
            assert!(swiglu, "{tag}: the SwiGLU store did not apply");
            assert_same(&got, &want, &format!("mlp {tag}"))?;
            stored += eps[0] + eps[1];

            let xm = hostile_x(rows, MIXER_K, 500 + rows as u64)?;
            let want = residual.add(&w.out.forward(&xm)?)?;
            want.eval();
            let (got, eps, _) = counted(|| {
                let got =
                    out.forward_mixer_output(&xm, None, Some(MixerResidual::Store(&residual)))?;
                got.eval();
                Ok(got)
            })?;
            let route = w.out.store_epilogue_route(rows, MIXER_K)?;
            assert_eq!(eps, [u64::from(route), 0], "{tag}: mixer epilogues ran");
            assert_eq!(route, rows >= 33, "{tag}: mixer output route");
            assert_same(&got, &want, &format!("mixer out {tag}"))?;
            stored += eps[0];
        }
    }
    println!("V2_E3 tile rows: {stored} epilogue stores, 0 differing elements");
    Ok(())
}

/// For all seven formats, at the gate/up, merged gate|up, down and mixer
/// output shapes and every row count from 1 to 70 (and 96 to 129, 2,048):
/// wherever the route says yes the plain projection runs the Tiled64
/// tensor-op GEMM once, so the epilogue GEMM's sums are the plain op's.
/// Mutation v-e3-splitk (the route ignores the split-K rule) turns it red.
#[test]
fn the_route_is_on_the_gemm_at_every_row_count() -> Result<()> {
    if unsupported("the_route_is_on_the_gemm_at_every_row_count") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let rows_list: Vec<i64> = (1..=70).chain([96, 97, 127, 128, 129, 2048]).collect();
    for (i, &fmt) in FORMATS.iter().enumerate() {
        let w = weights(fmt, 22_000 + 10 * i as u64)?;
        let block = mlp(&w.gate, &w.up, &w.down)?;
        let merged = merged_of(&block).expect("gate|up merges");
        for (proj, k, label) in [
            (&w.up, HIDDEN, "up"),
            (merged, HIDDEN, "merged gate|up"),
            (&w.down, FFN, "down"),
            (&w.out, MIXER_K, "mixer out"),
        ] {
            let mut taken = Vec::new();
            let mut gemm_not_taken = Vec::new();
            for &rows in &rows_list {
                let x = bf16_input(&[1, rows, k], 900 + rows as u64)?;
                let route = proj.store_epilogue_route(rows, k)?;
                let (_, _, plain) = counted(|| {
                    let y = proj.forward(&x)?;
                    y.eval();
                    Ok(())
                })?;
                if route {
                    assert_eq!(
                        plain,
                        1,
                        "{} {label} rows {rows}: the route took a shape off the GEMM",
                        fmt.name()
                    );
                    taken.push(rows);
                } else if plain == 1 {
                    gemm_not_taken.push(rows);
                }
            }
            assert!(
                taken.contains(&2048),
                "{} {label}: 2,048 rows not taken",
                fmt.name()
            );
            println!(
                "V2_E3 route {} {label}: first row taken {:?}, taken {} of {}, on the GEMM \
                 but not taken {:?}",
                fmt.name(),
                taken.first(),
                taken.len(),
                rows_list.len(),
                gemm_not_taken
            );
        }
    }
    Ok(())
}

/// `x` as rows of a longer buffer at an element offset that is not a
/// multiple of four (a reshape of a contiguous slice).
fn at_odd_offset(x: &MxArray) -> Result<MxArray> {
    let shape = x.shape()?.to_vec();
    let n: i64 = shape.iter().product();
    let flat = x.reshape(&[n])?;
    let pad = bf16_input(&[2], 77)?;
    let joined = MxArray::concatenate(&MxArray::concatenate(&pad, &flat, 0)?, &pad, 0)?;
    joined.eval();
    joined.slice_axis(0, 2, 2 + n)?.reshape(&shape)
}

/// `x` as every second row of a buffer with twice the rows (not row
/// contiguous).
fn as_strided_rows(x: &MxArray) -> Result<MxArray> {
    let shape = x.shape()?.to_vec();
    let (rows, n) = (shape[1], shape[2]);
    let lanes = x.reshape(&[rows, 1, n])?;
    let filler = bf16_input(&[rows, 1, n], 78)?;
    let woven = MxArray::concatenate(&lanes, &filler, 1)?;
    woven.eval();
    woven
        .slice_axis(1, 0, 1)?
        .reshape(&[rows, n])?
        .reshape(&[1, rows, n])
}

/// The epilogue operand and the GEMM input handed over as views (an odd
/// element offset inside a longer buffer; every second row of a wider
/// buffer) give the bits of the separate ops on plain copies: the residual
/// store of the down projection and the SwiGLU store of the up projection,
/// 2,048 rows (vector store) and 97 rows (scalar store on the partial tile).
/// Mutation v-e3-aux-view (the op binds its operand without making it row
/// contiguous) turns it red.
#[test]
fn operands_as_views_keep_the_bits() -> Result<()> {
    if unsupported("operands_as_views_keep_the_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let w = weights(Q5K, 23_000)?;
    for rows in [2048i64, 97] {
        let x_up = hostile_x(rows, HIDDEN, 610 + rows as u64)?;
        let gate = every_pattern(rows, FFN, 5)?;
        let x_down = hostile_x(rows, FFN, 620 + rows as u64)?;
        let residual = every_pattern(rows, HIDDEN, 9)?;
        let want_swiglu = Activations::swiglu_compiled(&gate, &w.up.forward(&x_up)?)?;
        let want_resid = residual.add(&w.down.forward(&x_down)?)?;
        want_swiglu.eval();
        want_resid.eval();
        type View = fn(&MxArray) -> Result<MxArray>;
        let views: [(&str, View); 2] = [
            ("odd offset", at_odd_offset),
            ("every second row", as_strided_rows),
        ];
        for (label, view) in views {
            for (aux_view, x_view) in [(true, false), (false, true), (true, true)] {
                let pick = |a: &MxArray, on: bool| if on { view(a) } else { Ok(a.clone()) };
                let tag = format!("rows {rows} {label} aux={aux_view} x={x_view}");
                let (got, eps, _) = counted(|| {
                    let got = w.up.try_store_epilogue(
                        &pick(&x_up, x_view)?,
                        &pick(&gate, aux_view)?,
                        StoreEpilogue::Swiglu,
                    )?;
                    if let Some(got) = &got {
                        got.eval();
                    }
                    Ok(got)
                })?;
                let got =
                    got.ok_or_else(|| Error::from_reason(format!("{tag}: no swiglu store")))?;
                assert_eq!(eps, [0, 1], "{tag}: swiglu dispatches");
                assert_same(&got, &want_swiglu, &format!("swiglu {tag}"))?;
                let (got, eps, _) = counted(|| {
                    let got = w.down.try_store_epilogue(
                        &pick(&x_down, x_view)?,
                        &pick(&residual, aux_view)?,
                        StoreEpilogue::Residual,
                    )?;
                    if let Some(got) = &got {
                        got.eval();
                    }
                    Ok(got)
                })?;
                let got =
                    got.ok_or_else(|| Error::from_reason(format!("{tag}: no residual store")))?;
                assert_eq!(eps, [1, 0], "{tag}: residual dispatches");
                assert_same(&got, &want_resid, &format!("residual {tag}"))?;
            }
        }
    }
    Ok(())
}

/// `true`: the reference order on this thread (the varlen
/// block table reshaped and evaluated, the head-kernel GDN scan, the
/// separate ops).
fn set_order(before: bool) {
    RESHAPED_BLOCK_TABLE.with(|flag| flag.set(before));
    HEAD_SCAN_PREFILL.with(|flag| flag.set(before));
    PREFILL_SEPARATE_OPS.with(|flag| flag.set(before));
}

/// What a prefill of all 64 blocks leaves after its first part and after
/// the continuation on the live request.
#[derive(Debug, PartialEq)]
struct Resumed {
    first: Left,
    last: Left,
}

struct Counts {
    epilogues: [u64; 2],
    row_scans: u64,
    head_scans: u64,
}

fn family(name: &CStr) -> u64 {
    // SAFETY: a NUL-terminated name; test-only counter.
    unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) }
}

/// `first` tokens prefilled in `rows`-row chunks in one order (`write`),
/// then `more` tokens on the live request in the other (`read`): the state
/// one order leaves is the state the other continues from.
#[allow(clippy::too_many_arguments)]
fn resumed(
    cfg: &super::super::config::Qwen3_5Config,
    bank: &Bank,
    embed: &Embedding,
    final_norm: &RMSNorm,
    lm_head: &Option<LinearProj>,
    first: usize,
    more: usize,
    rows: i32,
    write: bool,
    read: bool,
) -> Result<(Resumed, Counts, Counts)> {
    let err = |e: String| Error::from_reason(e);
    let mut layers = BLOCKS
        .iter()
        .enumerate()
        .map(|(i, &block)| layer_of(cfg, bank, block, i))
        .collect::<Result<Vec<_>>>()?;
    let kinds = compute_layer_kinds(BLOCKS.len(), |i| cfg.is_linear_layer(i));
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
    let len = first + more;
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

    let part = |cached: usize,
                end: usize,
                before: bool,
                captures: Vec<usize>,
                layers: &mut Vec<_>,
                caches: &mut Vec<Qwen3_5LayerCache>,
                adapter: &mut PagedKVCacheAdapter|
     -> Result<(Left, Counts)> {
        let mut sink = RecordingSink {
            taps: vec![1, 30, 62],
            captures,
            seen: Vec::new(),
            linear: Vec::new(),
        };
        set_order(before);
        // SAFETY: test-only thread-local counter hooks.
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let ran = (|| {
            let mut target = PrefillParts {
                layers,
                caches,
                final_norm,
                lm_head,
                adapter,
            };
            let (logits, checkpoints) = run_paged_prefill_chunk_with_size(
                &tokens[..end],
                &tokens[cached..end],
                cached as u32,
                cached > 0,
                embed,
                &mut target,
                &kinds,
                rows,
                0,
                None,
                Some(&mut sink as &mut dyn PrefillTapSink),
            )?;
            logits.eval();
            Ok::<_, Error>((logits, checkpoints))
        })();
        let counts = Counts {
            epilogues: [family(EPILOGUE_FAMILIES[0]), family(EPILOGUE_FAMILIES[1])],
            row_scans: family(c"gdn_prefill_row_scan"),
            head_scans: family(c"gdn_prefill_head_scan"),
        };
        // SAFETY: as above.
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        set_order(false);
        let (logits, checkpoints) = ran?;
        let mut encoded = vec![encode(cfg, caches)?];
        for point in &checkpoints {
            encoded.push(encode(cfg, &point.caches)?);
        }
        for (_, linear) in &sink.linear {
            encoded.push(encode(cfg, linear)?);
        }
        let left = Left {
            logits: bits(&logits)?,
            gdn: gdn_bits(caches)?,
            pool: pool_bytes(adapter)?,
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
        Ok((left, counts))
    };
    let (first_left, first_counts) = part(
        0,
        first,
        write,
        vec![first / 3, first - 45],
        &mut layers,
        &mut caches,
        &mut adapter,
    )?;
    let (last_left, last_counts) = part(
        first,
        len,
        read,
        Vec::new(),
        &mut layers,
        &mut caches,
        &mut adapter,
    )?;
    Ok((
        Resumed {
            first: first_left,
            last: last_left,
        },
        first_counts,
        last_counts,
    ))
}

/// All 64 blocks of the 27B table (every real gate / up / down / mixer
/// output format pair) at the 27B geometry with an int8 pool: 2,145 tokens
/// prefilled in 2,048-row chunks (2,048 and 97 rows, split again at the
/// capture rows), then 130 more tokens on the live request. The state is
/// written in one order and continued in the other, for the four pairs of
/// the reference order and the new order: the logits,
/// every pool byte, the GDN state, the taps, the captured states, the
/// checkpoints and the GDN cells as the persistent tier encodes them are
/// equal after the first part and after the continuation. The reference order
/// runs the head-kernel scan and no epilogue; the new order runs the row
/// scan and both epilogues.
#[test]
fn a_64_block_prefill_resumes_across_the_orders() -> Result<()> {
    if unsupported("a_64_block_prefill_resumes_across_the_orders") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let cfg = config();
    assert_eq!(cfg.num_layers as usize, BLOCKS.len());
    let bank = Bank::new(&BLOCKS)?;
    let embed = Embedding::from_weight(&bf16_input(&[VOCAB, HIDDEN], 61)?)?;
    let final_norm = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
    let mut head = Linear::new(HIDDEN as u32, VOCAB as u32, Some(false))?;
    head.set_weight(&bf16_input(&[VOCAB, HIDDEN], 62)?.mul_scalar(0.02)?)?;
    let lm_head = Some(LinearProj::Standard(head));
    let run = |write: bool, read: bool| {
        resumed(
            &cfg,
            &bank,
            &embed,
            &final_norm,
            &lm_head,
            2145,
            130,
            2048,
            write,
            read,
        )
    };
    let (reference, before_first, before_last) = run(true, true)?;
    assert_eq!(
        (before_first.epilogues, before_last.epilogues),
        ([0, 0], [0, 0]),
        "the reference order ran an epilogue"
    );
    assert!(
        before_first.head_scans > 0 && before_first.row_scans == 0 && before_last.head_scans > 0,
        "the reference order did not run the head-kernel scan"
    );
    assert!(
        reference.first.encoded.len() > 1 && !reference.first.taps.is_empty(),
        "the first part captured nothing"
    );
    assert!(!reference.first.pool.is_empty(), "the pool holds no block");
    for (write, read) in [(true, false), (false, true), (false, false)] {
        let (pair, first, last) = run(write, read)?;
        let name = format!(
            "written in the {} order, continued in the {} order",
            if write { "reference" } else { "new" },
            if read { "reference" } else { "new" }
        );
        for (counts, before, part) in [(&first, write, "first"), (&last, read, "last")] {
            if before {
                assert_eq!(counts.epilogues, [0, 0], "{name}: {part} part epilogues");
                assert!(
                    counts.head_scans > 0 && counts.row_scans == 0,
                    "{name}: {part} scan"
                );
            } else {
                assert!(
                    counts.epilogues[0] > 0 && counts.epilogues[1] > 0,
                    "{name}: {part} part ran epilogues {:?}",
                    counts.epilogues
                );
                assert!(
                    counts.row_scans > 0 && counts.head_scans == 0,
                    "{name}: {part} scan"
                );
            }
        }
        for (got, want, part) in [
            (&pair.first, &reference.first, "first part"),
            (&pair.last, &reference.last, "continuation"),
        ] {
            assert_eq!(got.logits, want.logits, "{name}: {part} logits");
            assert!(got.pool == want.pool, "{name}: {part} pool bytes");
            assert!(got.gdn == want.gdn, "{name}: {part} GDN state");
            assert!(got.taps == want.taps, "{name}: {part} taps");
            assert!(
                got.captured == want.captured,
                "{name}: {part} captured states"
            );
            assert!(
                got.checkpoints == want.checkpoints,
                "{name}: {part} checkpoints"
            );
            assert!(
                got.encoded == want.encoded,
                "{name}: {part} encoded GDN cells"
            );
        }
        println!(
            "V2_E3 resume {name}: first part res={} swiglu={} row_scans={} head_scans={}; \
             continuation res={} swiglu={} row_scans={} head_scans={}; encoded cells {} + {}",
            first.epilogues[0],
            first.epilogues[1],
            first.row_scans,
            first.head_scans,
            last.epilogues[0],
            last.epilogues[1],
            last.row_scans,
            last.head_scans,
            pair.first.encoded.len(),
            pair.last.encoded.len()
        );
    }
    Ok(())
}

/// Where the route holds but the op declines its operands, the layer adds
/// the residual and runs the post norm as two ops in place of the fused
/// add + norm. The two ops give the fused kernel's hidden state and norm in
/// bfloat16, float16 and float32: 2,048 / 33 / 1 rows of hostile values.
#[test]
fn the_separate_add_and_norm_keep_the_fused_bits() -> Result<()> {
    if unsupported("the_separate_add_and_norm_keep_the_fused_bits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let mut norm = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
    norm.set_weight(&bf16_input(&[HIDDEN], 91)?.add_scalar(1.0)?)?;
    let raw_bits = |a: &MxArray| -> Result<Vec<u32>> {
        a.eval();
        Ok(match a.dtype()? {
            DType::Float32 => a.to_float32()?.iter().map(|x| x.to_bits()).collect(),
            _ => a.to_uint16_native()?.into_iter().map(u32::from).collect(),
        })
    };
    for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
        let mut typed = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
        typed.set_weight(&norm.get_weight().astype(dtype)?)?;
        for rows in [2048i64, 33, 1] {
            let x = hostile_x(rows, HIDDEN, 700 + rows as u64)?.astype(dtype)?;
            let delta = hostile_x(rows, HIDDEN, 800 + rows as u64)?.astype(dtype)?;
            let Some((h, normed)) = typed.forward_residual_add(&x, &delta) else {
                println!("V2_E3 fused add+norm declines {dtype:?} rows {rows}");
                continue;
            };
            let h2 = x.add(&delta)?;
            let normed2 = typed.forward(&h2)?;
            let differing = |a: &MxArray, b: &MxArray| -> Result<usize> {
                let (a, b) = (raw_bits(a)?, raw_bits(b)?);
                Ok(a.iter().zip(&b).filter(|(x, y)| x != y).count())
            };
            let (dh, dn) = (differing(&h, &h2)?, differing(&normed, &normed2)?);
            println!("V2_E3 add+norm {dtype:?} rows {rows}: hidden differ {dh}, norm differ {dn}");
            assert_eq!(
                (dh, dn),
                (0, 0),
                "{dtype:?} rows {rows}: fused add + norm bits"
            );
        }
    }
    Ok(())
}
