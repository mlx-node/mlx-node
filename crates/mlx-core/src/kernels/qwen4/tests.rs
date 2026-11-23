use super::*;
use crate::accelerator::{ActiveBackend, active_backend};
use crate::array::{DType, MxArray};
use crate::kernels::{attention, gdn, linear};
use crate::models::quant_dispatch::kquant_tile_rows;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_gpu::{
    GpuRepackKind, OutLayout, RepackDst, band_offset_rule, gpu_kind_for, gpu_repack_available,
    gpu_repack_band, gpu_repack_supports_kind,
};
use crate::utils::gguf_kquant::{KQuantScales, repack_kquant};
use half::{bf16, f16};

fn cuda_available() -> bool {
    active_backend() == ActiveBackend::CudaGpu
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u32() & 0x00FF_FFFF) as f32 / 16_777_216.0)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u32() >> 24) as u8).collect()
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}
fn f32_round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}
fn sigmoid_f32(x: f32) -> f32 {
    let y = 1.0 / (1.0 + x.abs().exp());
    if x < 0.0 { y } else { 1.0 - y }
}
fn silu_bf16(x: f32) -> f32 {
    let xb = f32_round_bf16(x);
    let s = f32_round_bf16(sigmoid_f32(xb));
    f32_round_bf16(xb * s)
}

fn bf16_array(vals: &[f32], shape: &[i64]) -> MxArray {
    let bits: Vec<u16> = vals.iter().copied().map(f32_to_bf16_bits).collect();
    MxArray::from_bfloat16(&bits, shape).unwrap()
}

fn read_bf16_f32(arr: &MxArray) -> Vec<f32> {
    arr.eval();
    arr.to_uint16_native()
        .unwrap()
        .into_iter()
        .map(bf16_bits_to_f32)
        .collect()
}

fn assert_close(label: &str, got: &[f32], want: &[f32], max_abs_tol: f32, max_rel_tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: len mismatch");
    let mut max_abs = 0.0f32;
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "{label}: non-finite output {g}");
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff_sq += (d as f64) * (d as f64);
        ref_sq += (w as f64) * (w as f64);
    }
    let rel = (diff_sq / (ref_sq + 1e-12)).sqrt() as f32;
    assert!(
        max_abs <= max_abs_tol || rel <= max_rel_tol,
        "{label}: max_abs={max_abs:.6} (tol {max_abs_tol}), rel={rel:.6} (tol {max_rel_tol})"
    );
}

fn repack_staging(blocks: &[u8], pad: usize) -> MxArray {
    let total = pad + blocks.len() + 16;
    let (arr, ptr, bytes) = MxArray::alloc_uninit(&[total as i64], DType::Uint8).unwrap();
    assert_eq!(bytes, total);
    unsafe {
        std::ptr::write_bytes(ptr.0, 0xA5, total);
        std::ptr::copy_nonoverlapping(blocks.as_ptr(), ptr.0.add(pad), blocks.len());
    }
    arr
}

fn gguf_row_bytes(kind: GpuRepackKind, k: usize) -> usize {
    let t = kind.gguf_type();
    k / t.block_size() * t.type_size()
}

fn gguf_tensor_info(kind: GpuRepackKind, rows: usize, k: usize) -> GgufTensorInfo {
    GgufTensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![k as u64, rows as u64],
        tensor_type: kind.gguf_type(),
        offset: 0,
    }
}

fn cpu_repack_oracle(
    kind: GpuRepackKind,
    layout: OutLayout,
    blocks: &[u8],
    rows: usize,
    k: usize,
) -> (Vec<u32>, Vec<u8>, Vec<u16>) {
    if let Some(format) = kind.gguf_type().k_quant_format() {
        let arrays = repack_kquant(format, blocks, rows, k).unwrap();
        let scales = match arrays.scales {
            KQuantScales::Unsigned(v) => v,
            KQuantScales::Signed(v) => v.into_iter().map(|x| x as u8).collect(),
        };
        if layout == OutLayout::RowMajor {
            return (arrays.weight, scales, arrays.biases);
        }
        let r = rows as i64;
        let per_group = format.scale_bytes_per_group() as i64;
        let super_ratio = format.super_ratio() as i64;
        let per_super = format.bias_entries_per_super_block() as i64;
        let w = MxArray::from_uint32(&arrays.weight, &[r, format.weight_cols(k) as i64]).unwrap();
        let s = MxArray::from_uint8(&scales, &[r, format.scales_cols(k) as i64]).unwrap();
        let b = MxArray::from_float16(&arrays.biases, &[r, format.biases_cols(k) as i64]).unwrap();
        let tw = kquant_tile_rows(&w, format.bits() as i64).unwrap();
        let ts = kquant_tile_rows(&s, super_ratio * per_group).unwrap();
        let tb = kquant_tile_rows(&b, per_super).unwrap();
        MxArray::eval_arrays_with_context(&[&tw, &ts, &tb], "cpu_repack_oracle").unwrap();
        return (
            tw.to_uint32().unwrap().to_vec(),
            ts.to_uint8().unwrap(),
            tb.to_uint16_native().unwrap(),
        );
    }
    let n_blocks = rows * (k / 32);
    match kind {
        GpuRepackKind::Q5_1 => {
            let mut weight = vec![0u32; n_blocks * 5];
            let mut scales = Vec::with_capacity(n_blocks * 2);
            let mut biases = Vec::with_capacity(n_blocks);
            for (i, blk) in blocks.as_chunks::<24>().0.iter().enumerate() {
                scales.extend_from_slice(&blk[0..2]);
                biases.push(u16::from_le_bytes([blk[2], blk[3]]));
                let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                let qs = &blk[8..24];
                let mut acc = 0u64;
                let mut pending = 0u32;
                let mut word = i * 5;
                for val in 0..32 {
                    let (nib, bit) = if val < 16 {
                        (qs[val] & 0x0F, (qh >> val) & 1)
                    } else {
                        let j = val - 16;
                        (qs[j] >> 4, (qh >> (j + 16)) & 1)
                    };
                    acc |= u64::from(u32::from(nib) | (bit << 4)) << pending;
                    pending += 5;
                    while pending >= 32 {
                        weight[word] = acc as u32;
                        acc >>= 32;
                        pending -= 32;
                        word += 1;
                    }
                }
            }
            (weight, scales, biases)
        }
        GpuRepackKind::Q8_0 | GpuRepackKind::Q8_0Bias => {
            let mut weight = vec![0u32; n_blocks * 8];
            let mut scale_u16 = Vec::with_capacity(n_blocks);
            let mut scales = Vec::with_capacity(n_blocks * 2);
            for (i, blk) in blocks.as_chunks::<34>().0.iter().enumerate() {
                scale_u16.push(u16::from_le_bytes([blk[0], blk[1]]));
                scales.extend_from_slice(&blk[0..2]);
                for w in 0..8 {
                    let mut packed = 0u32;
                    for b in 0..4 {
                        let u = blk[2 + w * 4 + b] ^ 0x80;
                        packed |= u32::from(u) << (b * 8);
                    }
                    weight[i * 8 + w] = packed;
                }
            }
            let biases = if kind == GpuRepackKind::Q8_0Bias {
                // Matches Metal's `q8_0_derived_bias` and CPU IEEE f16 (-128 * d,
                // canonicalizing any NaN input to +qNaN 0x7E00).
                scale_u16
                    .iter()
                    .map(|&bits| {
                        if (bits & 0x7FFF) > 0x7C00 {
                            0x7E00
                        } else {
                            f16::from_f32(-128.0 * f16::from_bits(bits).to_f32()).to_bits()
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            (weight, scales, biases)
        }
        _ => unreachable!("unexpected affine kind {kind:?}"),
    }
}

fn read_repack_dst(dst: &RepackDst) -> (Vec<u32>, Vec<u8>, Vec<u16>) {
    let s = dst.scales.as_ref().unwrap();
    let mut all = vec![&dst.weight, s];
    all.extend(dst.biases.as_ref());
    MxArray::eval_arrays_with_context(&all, "read_repack_dst").unwrap();
    let scales = match s.dtype().unwrap() {
        DType::Float16 => s
            .to_uint16_native()
            .unwrap()
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
        _ => s.to_uint8().unwrap(),
    };
    let biases = dst
        .biases
        .as_ref()
        .map(|b| b.to_uint16_native().unwrap())
        .unwrap_or_default();
    (dst.weight.to_uint32().unwrap().to_vec(), scales, biases)
}

// =========================================================================
// 1. CUDA GGUF Repack (7 kernels: Q4_K, Q4_K_t64, Q5_K, Q5_K_t64, Q5_1, Q8_0, Q8_0Bias)
// =========================================================================
#[test]
fn cuda_gguf_repack_bit_exact() {
    if !cuda_available() {
        return;
    }
    assert!(
        gpu_repack_available(),
        "CUDA GGUF repack must report available"
    );
    for k in [
        GpuRepackKind::Q4K,
        GpuRepackKind::Q5K,
        GpuRepackKind::Q5_1,
        GpuRepackKind::Q8_0,
        GpuRepackKind::Q8_0Bias,
    ] {
        assert!(gpu_repack_supports_kind(k), "{k:?} must be supported");
    }
    assert!(
        !gpu_repack_supports_kind(GpuRepackKind::Q6K),
        "Q6K must fall back to CPU on CUDA"
    );
    assert!(
        gpu_kind_for(GgufTensorType::Q6K).is_none(),
        "gpu_kind_for(Q6K) must return None on CUDA"
    );

    let mut rng = Rng::new(0x1122_3344);
    let cases: &[(GpuRepackKind, OutLayout, &[usize], usize)] = &[
        (GpuRepackKind::Q4K, OutLayout::RowMajor, &[1, 64, 65], 2560),
        (GpuRepackKind::Q4K, OutLayout::Tiled64, &[64, 128], 2560),
        (GpuRepackKind::Q5K, OutLayout::RowMajor, &[1, 64, 65], 2560),
        (GpuRepackKind::Q5K, OutLayout::Tiled64, &[64, 128], 2560),
        (GpuRepackKind::Q5_1, OutLayout::RowMajor, &[3, 64], 640),
        (GpuRepackKind::Q8_0, OutLayout::RowMajor, &[3, 64], 640),
        (GpuRepackKind::Q8_0Bias, OutLayout::RowMajor, &[3, 64], 640),
    ];
    for &(kind, layout, splits, k) in cases {
        let total_rows: usize = splits.iter().sum();
        let rb = gguf_row_bytes(kind, k);
        let blocks = rng.bytes(total_rows * rb);
        let pad = 32 + band_offset_rule(kind);
        let staging = repack_staging(&blocks, pad);
        let info = gguf_tensor_info(kind, total_rows, k);
        let (mut dst, _) = RepackDst::alloc_for_kind(&info, kind, layout).unwrap();
        let mut row0 = 0usize;
        for &band in splits {
            let off = pad + row0 * rb;
            dst = gpu_repack_band(
                &staging,
                off as u64,
                kind,
                layout,
                k as u32,
                row0 as u32,
                band as u32,
                dst,
            )
            .unwrap();
            row0 += band;
        }
        let got = read_repack_dst(&dst);
        let want = cpu_repack_oracle(kind, layout, &blocks, total_rows, k);
        assert_eq!(got.0, want.0, "{kind:?} {layout:?}: weight mismatch");
        assert_eq!(got.1, want.1, "{kind:?} {layout:?}: scales mismatch");
        assert_eq!(got.2, want.2, "{kind:?} {layout:?}: biases mismatch");
    }

    // Exhaustive 65,536 f16 scale bit patterns for Q8_0Bias `-128 * d` derivation.
    let kind = GpuRepackKind::Q8_0Bias;
    let k = 32 * (usize::from(u16::MAX) + 1);
    let mut blocks = rng.bytes(gguf_row_bytes(kind, k));
    for (blk, d) in blocks.as_chunks_mut::<34>().0.iter_mut().zip(0..=u16::MAX) {
        blk[..2].copy_from_slice(&d.to_le_bytes());
    }
    let staging = repack_staging(&blocks, 2);
    let (dst, _) =
        RepackDst::alloc_for_kind(&gguf_tensor_info(kind, 1, k), kind, OutLayout::RowMajor)
            .unwrap();
    let out = gpu_repack_band(&staging, 2, kind, OutLayout::RowMajor, k as u32, 0, 1, dst).unwrap();
    let got = read_repack_dst(&out);
    let want = cpu_repack_oracle(kind, OutLayout::RowMajor, &blocks, 1, k);
    assert_eq!(got, want, "Q8_0Bias all 65,536 f16 scale bit patterns");

    // Unsupported Q6K rejected cleanly.
    let q6_blocks = rng.bytes(gguf_row_bytes(GpuRepackKind::Q6K, 256));
    let q6_staging = repack_staging(&q6_blocks, 16);
    let (q6_dst, _) = RepackDst::alloc_for_kind(
        &gguf_tensor_info(GpuRepackKind::Q6K, 1, 256),
        GpuRepackKind::Q6K,
        OutLayout::RowMajor,
    )
    .unwrap();
    assert!(
        gpu_repack_band(
            &q6_staging,
            16,
            GpuRepackKind::Q6K,
            OutLayout::RowMajor,
            256,
            0,
            1,
            q6_dst,
        )
        .is_err(),
        "Q6K band must return Err on CUDA"
    );
}

// =========================================================================
// 2. Qwen4 MoE Decode/Verify (`M=1,4,8,16`) & Prefill (`M=64`)
// =========================================================================
fn build_kquant_bank(
    rng: &mut Rng,
    mode_gu: i32,
    experts: usize,
    rows_per_e: usize,
    k: usize,
) -> (MxArray, MxArray, MxArray, Vec<f32>) {
    // Pack well-scaled synthetic Q4_K (mode=0) or Q5_K (mode=1) blocks, then
    // repack on GPU and dequantize on CPU to build the exact reference weight.
    let kind = if mode_gu == 0 {
        GpuRepackKind::Q4K
    } else {
        GpuRepackKind::Q5K
    };
    let sb_bytes = kind.gguf_type().type_size();
    let sbs = k / 256;
    let total_rows = experts * rows_per_e;
    let mut raw = vec![0u8; total_rows * sbs * sb_bytes];
    for blk in raw.chunks_exact_mut(sb_bytes) {
        let d = f16::from_f32(rng.uniform(0.002, 0.006)).to_bits();
        let dmin = f16::from_f32(rng.uniform(0.001, 0.004)).to_bits();
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&dmin.to_le_bytes());
        for byte in &mut blk[4..] {
            *byte = (rng.next_u32() >> 24) as u8;
        }
    }
    let staging = repack_staging(&raw, 16);
    let info = GgufTensorInfo {
        name: "moe.gu".to_string(),
        n_dims: 3,
        dims: vec![k as u64, rows_per_e as u64, experts as u64],
        tensor_type: kind.gguf_type(),
        offset: 0,
    };
    let (dst, _) = RepackDst::alloc_for_kind(&info, kind, OutLayout::RowMajor).unwrap();
    let repacked = gpu_repack_band(
        &staging,
        16,
        kind,
        OutLayout::RowMajor,
        k as u32,
        0,
        total_rows as u32,
        dst,
    )
    .unwrap();
    let (w_u32, s_u8, b_u16) = read_repack_dst(&repacked);
    // Dequantize from the repacked format directly so reference math matches K-quant decode.
    let mut deq = vec![0.0f32; total_rows * k];
    let w_cols = if mode_gu == 0 { k * 4 / 32 } else { k * 5 / 32 };
    let s_cols = sbs * 16;
    let b_cols = sbs * 2;
    for row in 0..total_rows {
        let w_row = &w_u32[row * w_cols..(row + 1) * w_cols];
        let s_row = &s_u8[row * s_cols..(row + 1) * s_cols];
        let b_row = &b_u16[row * b_cols..(row + 1) * b_cols];
        let out_row = &mut deq[row * k..(row + 1) * k];
        for sb in 0..sbs {
            let d = f16::from_bits(b_row[sb * 2]).to_f32();
            let dmin = f16::from_bits(b_row[sb * 2 + 1]).to_f32();
            for sub in 0..8 {
                let sc = d * (s_row[sb * 16 + sub * 2] as f32);
                let mn = dmin * (s_row[sb * 16 + sub * 2 + 1] as f32);
                for j in 0..32 {
                    let col = sb * 256 + sub * 32 + j;
                    let q = if mode_gu == 0 {
                        (w_row[col >> 3] >> ((col & 7) * 4)) & 0xF
                    } else {
                        let bit = col * 5;
                        let wi = bit >> 5;
                        let sh = bit & 31;
                        if sh <= 27 {
                            (w_row[wi] >> sh) & 0x1F
                        } else {
                            ((w_row[wi] >> sh) | (w_row[wi + 1] << (32 - sh))) & 0x1F
                        }
                    };
                    out_row[col] = sc * (q as f32) - mn;
                }
            }
        }
    }
    (
        repacked.weight,
        repacked.scales.unwrap(),
        repacked.biases.unwrap(),
        deq,
    )
}

fn build_affine_bank(
    rng: &mut Rng,
    bits: i32,
    experts: usize,
    rows_per_e: usize,
    k: usize,
    is_3d: bool,
) -> (MxArray, MxArray, MxArray, Vec<f32>) {
    let kind = if bits == 5 {
        GpuRepackKind::Q5_1
    } else {
        GpuRepackKind::Q8_0Bias
    };
    let blk_bytes = kind.gguf_type().type_size();
    let groups = k / 32;
    let total_rows = experts * rows_per_e;
    let mut raw = vec![0u8; total_rows * groups * blk_bytes];
    for blk in raw.chunks_exact_mut(blk_bytes) {
        if bits == 5 {
            let d = f16::from_f32(rng.uniform(0.002, 0.006)).to_bits();
            let m = f16::from_f32(rng.uniform(-0.08, 0.08)).to_bits();
            blk[0..2].copy_from_slice(&d.to_le_bytes());
            blk[2..4].copy_from_slice(&m.to_le_bytes());
            for byte in &mut blk[4..] {
                *byte = (rng.next_u32() >> 24) as u8;
            }
        } else {
            let d = f16::from_f32(rng.uniform(0.001, 0.003)).to_bits();
            blk[0..2].copy_from_slice(&d.to_le_bytes());
            for byte in &mut blk[2..] {
                *byte = (rng.next_u32() >> 24) as u8;
            }
        }
    }
    let staging = repack_staging(&raw, 4);
    let info = if is_3d {
        GgufTensorInfo {
            name: "moe.dn".to_string(),
            n_dims: 3,
            dims: vec![k as u64, rows_per_e as u64, experts as u64],
            tensor_type: kind.gguf_type(),
            offset: 0,
        }
    } else {
        gguf_tensor_info(kind, total_rows, k)
    };
    let (dst, _) = RepackDst::alloc_for_kind(&info, kind, OutLayout::RowMajor).unwrap();
    let repacked = gpu_repack_band(
        &staging,
        4,
        kind,
        OutLayout::RowMajor,
        k as u32,
        0,
        total_rows as u32,
        dst,
    )
    .unwrap();
    let (w_u32, s_bytes, b_u16) = read_repack_dst(&repacked);
    let w_cols = k * (bits as usize) / 32;
    let mut deq = vec![0.0f32; total_rows * k];
    for row in 0..total_rows {
        let w_row = &w_u32[row * w_cols..(row + 1) * w_cols];
        let out_row = &mut deq[row * k..(row + 1) * k];
        for g in 0..groups {
            let idx = row * groups + g;
            let sc = f16::from_bits(u16::from_le_bytes([s_bytes[idx * 2], s_bytes[idx * 2 + 1]]))
                .to_f32();
            let bi = f16::from_bits(b_u16[idx]).to_f32();
            for j in 0..32 {
                let col = g * 32 + j;
                let q = if bits == 5 {
                    let bit = col * 5;
                    let wi = bit >> 5;
                    let sh = bit & 31;
                    if sh <= 27 {
                        (w_row[wi] >> sh) & 0x1F
                    } else {
                        ((w_row[wi] >> sh) | (w_row[wi + 1] << (32 - sh))) & 0x1F
                    }
                } else {
                    (w_row[col >> 2] >> ((col & 3) * 8)) & 0xFF
                };
                out_row[col] = sc * (q as f32) + bi;
            }
        }
    }
    (
        repacked.weight,
        repacked.scales.unwrap(),
        repacked.biases.unwrap(),
        deq,
    )
}

fn cpu_moe_ref(
    x: &[f32],
    ids: &[i32],
    scores: &[f32],
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    shared: Option<(&[f32], &[f32], &[f32], &[f32])>,
    m: usize,
    top_k: usize,
) -> Vec<f32> {
    const H: usize = 2560;
    const DFF: usize = 640;
    let mut out = vec![0.0f32; m * H];
    let mut mid = vec![0.0f32; DFF];
    for t in 0..m {
        let xt = &x[t * H..(t + 1) * H];
        for k_idx in 0..top_k {
            let e = ids[t * top_k + k_idx] as usize;
            let w_score = scores[t * top_k + k_idx];
            let e_gu = e * DFF * H;
            for n in 0..DFF {
                let mut g_acc = 0.0f64;
                let mut u_acc = 0.0f64;
                let g_row = &wg[e_gu + n * H..e_gu + (n + 1) * H];
                let u_row = &wu[e_gu + n * H..e_gu + (n + 1) * H];
                for c in 0..H {
                    let xv = xt[c] as f64;
                    g_acc += xv * (g_row[c] as f64);
                    u_acc += xv * (u_row[c] as f64);
                }
                let gf = g_acc as f32;
                let silu = gf * sigmoid_f32(gf);
                mid[n] = f32_round_bf16(silu * (u_acc as f32));
            }
            let e_dn = e * H * DFF;
            for h in 0..H {
                let mut d_acc = 0.0f64;
                let d_row = &wd[e_dn + h * DFF..e_dn + (h + 1) * DFF];
                for n in 0..DFF {
                    d_acc += (mid[n] as f64) * (d_row[n] as f64);
                }
                out[t * H + h] += w_score * f32_round_bf16(d_acc as f32);
            }
        }
        if let Some((swg, swu, swd, sgate)) = shared {
            for n in 0..DFF {
                let mut g_acc = 0.0f64;
                let mut u_acc = 0.0f64;
                let g_row = &swg[n * H..(n + 1) * H];
                let u_row = &swu[n * H..(n + 1) * H];
                for c in 0..H {
                    let xv = xt[c] as f64;
                    g_acc += xv * (g_row[c] as f64);
                    u_acc += xv * (u_row[c] as f64);
                }
                let gf = g_acc as f32;
                mid[n] = f32_round_bf16(gf * sigmoid_f32(gf) * (u_acc as f32));
            }
            let sg = sigmoid_f32(sgate[t]);
            for h in 0..H {
                let mut d_acc = 0.0f64;
                let d_row = &swd[h * DFF..(h + 1) * DFF];
                for n in 0..DFF {
                    d_acc += (mid[n] as f64) * (d_row[n] as f64);
                }
                out[t * H + h] += sg * f32_round_bf16(d_acc as f32);
            }
        }
    }
    out.into_iter().map(f32_round_bf16).collect()
}

#[test]
fn qwen4_moe_few_and_prefill_parity() {
    if !cuda_available() {
        return;
    }
    const H: usize = 2560;
    const DFF: usize = 640;
    const E: usize = 16;
    const TOP_K: usize = 10;
    let mut rng = Rng::new(0x5566_7788);

    // Shared Q8_0Bias weights reused across both configs.
    let (swg, ssg, sbg, swg_f) = build_affine_bank(&mut rng, 8, 1, DFF, H, false);
    let (swu, ssu, sbu, swu_f) = build_affine_bank(&mut rng, 8, 1, DFF, H, false);
    let (swd, ssd, sbd, swd_f) = build_affine_bank(&mut rng, 8, 1, H, DFF, false);

    for &(mode_gu, bits_dn, tag) in &[(0, 5, "Q4K+Q5_1"), (1, 8, "Q5K+Q8_0")] {
        let (wg, sg, bg, wg_f) = build_kquant_bank(&mut rng, mode_gu, E, DFF, H);
        let (wu, su, bu, wu_f) = build_kquant_bank(&mut rng, mode_gu, E, DFF, H);
        let (wd, sd, bd, wd_f) = build_affine_bank(&mut rng, bits_dn, E, H, DFF, true);
        let routed = Qwen4MoeRoutedWeights {
            gate_w: &wg,
            gate_s: &sg,
            gate_b: &bg,
            up_w: &wu,
            up_s: &su,
            up_b: &bu,
            down_w: &wd,
            down_s: &sd,
            down_b: &bd,
            mode_gu,
            bits_dn,
        };

        // Few path: M in {1, 4, 8, 16}, both routed-only and routed+shared.
        for m in [1usize, 4, 8, 16] {
            let x_vals: Vec<f32> = (0..m * H)
                .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
                .collect();
            let mut ids = Vec::with_capacity(m * TOP_K);
            let mut scores = Vec::with_capacity(m * TOP_K);
            for t in 0..m {
                let mut sum = 0.0f32;
                let base = (t * 3) % E;
                for k_i in 0..TOP_K {
                    ids.push(((base + k_i) % E) as i32);
                    let s = rng.uniform(0.05, 0.35);
                    scores.push(s);
                    sum += s;
                }
                for s in &mut scores[t * TOP_K..(t + 1) * TOP_K] {
                    *s /= sum;
                }
            }
            let sg_vals: Vec<f32> = (0..m)
                .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
                .collect();

            let x_arr = bf16_array(&x_vals, &[1, m as i64, H as i64]);
            let ids_arr = MxArray::from_int32(&ids, &[1, m as i64, TOP_K as i64]).unwrap();
            let sc_arr = MxArray::from_float32(&scores, &[1, m as i64, TOP_K as i64]).unwrap();
            let sg_arr = bf16_array(&sg_vals, &[1, m as i64, 1]);

            // Routed-only
            let got_routed = moe_routed_shared_few_cuda(&x_arr, &ids_arr, &sc_arr, &routed, None)
                .unwrap()
                .expect("moe_few routed-only admitted");
            let want_routed =
                cpu_moe_ref(&x_vals, &ids, &scores, &wg_f, &wu_f, &wd_f, None, m, TOP_K);
            assert_close(
                &format!("moe_few({tag}, M={m}, routed-only)"),
                &read_bf16_f32(&got_routed),
                &want_routed,
                0.15,
                0.03,
            );

            // Routed + fused shared
            let shared = Qwen4MoeSharedWeights {
                gate_w: &swg,
                gate_s: &ssg,
                gate_b: &sbg,
                up_w: &swu,
                up_s: &ssu,
                up_b: &sbu,
                down_w: &swd,
                down_s: &ssd,
                down_b: &sbd,
                shared_gate: &sg_arr,
            };
            let got_both =
                moe_routed_shared_few_cuda(&x_arr, &ids_arr, &sc_arr, &routed, Some(&shared))
                    .unwrap()
                    .expect("moe_few routed+shared admitted");
            let want_both = cpu_moe_ref(
                &x_vals,
                &ids,
                &scores,
                &wg_f,
                &wu_f,
                &wd_f,
                Some((&swg_f, &swu_f, &swd_f, &sg_vals)),
                m,
                TOP_K,
            );
            assert_close(
                &format!("moe_few({tag}, M={m}, routed+shared)"),
                &read_bf16_f32(&got_both),
                &want_both,
                0.15,
                0.03,
            );
        }

        // Prefill path: M=64 (with hot expert 0 selected by all 64 tokens -> cnt=64 > BM=32!).
        let m = 64usize;
        let x_vals: Vec<f32> = (0..m * H)
            .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
            .collect();
        let mut ids = Vec::with_capacity(m * TOP_K);
        let mut scores = Vec::with_capacity(m * TOP_K);
        for t in 0..m {
            let mut sum = 0.0f32;
            ids.push(0); // Force multi-tile hot expert (64 tokens -> 2 tiles of 32)
            let s0 = rng.uniform(0.1, 0.3);
            scores.push(s0);
            sum += s0;
            for k_i in 1..TOP_K {
                let e = 1 + ((t + k_i) % (E - 1));
                ids.push(e as i32);
                let s = rng.uniform(0.05, 0.25);
                scores.push(s);
                sum += s;
            }
            for s in &mut scores[t * TOP_K..(t + 1) * TOP_K] {
                *s /= sum;
            }
        }
        let x_arr = bf16_array(&x_vals, &[1, m as i64, H as i64]);
        let ids_arr = MxArray::from_int32(&ids, &[1, m as i64, TOP_K as i64]).unwrap();
        let sc_arr = MxArray::from_float32(&scores, &[1, m as i64, TOP_K as i64]).unwrap();

        let got_pf = moe_prefill_cuda(&x_arr, &ids_arr, &sc_arr, &routed, None)
            .unwrap()
            .expect("moe_prefill routed-only admitted");
        let want_pf = cpu_moe_ref(&x_vals, &ids, &scores, &wg_f, &wu_f, &wd_f, None, m, TOP_K);
        assert_close(
            &format!("moe_prefill({tag}, M={m}, routed-only)"),
            &read_bf16_f32(&got_pf),
            &want_pf,
            0.20,
            0.035,
        );

        // Prefill + shared_out/shared_gate epilogue
        let sout_vals: Vec<f32> = (0..m * H)
            .map(|_| f32_round_bf16(rng.uniform(-0.5, 0.5)))
            .collect();
        let sg_vals: Vec<f32> = (0..m)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();
        let sout_arr = bf16_array(&sout_vals, &[1, m as i64, H as i64]);
        let sg_arr = bf16_array(&sg_vals, &[1, m as i64, 1]);
        let got_pf_sh = moe_prefill_cuda(
            &x_arr,
            &ids_arr,
            &sc_arr,
            &routed,
            Some((&sout_arr, &sg_arr)),
        )
        .unwrap()
        .expect("moe_prefill +shared admitted");
        let mut want_pf_sh = want_pf;
        for t in 0..m {
            let sg = sigmoid_f32(sg_vals[t]);
            for h in 0..H {
                want_pf_sh[t * H + h] =
                    f32_round_bf16(want_pf_sh[t * H + h] + sg * sout_vals[t * H + h]);
            }
        }
        assert_close(
            &format!("moe_prefill({tag}, M={m}, +shared)"),
            &read_bf16_f32(&got_pf_sh),
            &want_pf_sh,
            0.20,
            0.035,
        );
    }
}

// =========================================================================
// 3. mHC v2: inject_norm, hc_down_act, hc_up_mix (`M=1,4,8,16`)
// =========================================================================
#[test]
fn qwen4_mhc_ops_parity() {
    if !cuda_available() {
        return;
    }
    const H: usize = 2560;
    const HC: usize = 4;
    const W: usize = H * HC;
    const R: usize = 320;
    const EPS: f32 = 1e-6;
    let mut rng = Rng::new(0x99AA_BBCC);

    let nw_f32: Vec<f32> = (0..W).map(|_| rng.uniform(0.7, 1.3)).collect();
    let nw_bf16: Vec<f32> = nw_f32.iter().copied().map(f32_round_bf16).collect();
    let nw_f32_arr = MxArray::from_float32(&nw_f32, &[W as i64]).unwrap();
    let nw_bf16_arr = bf16_array(&nw_bf16, &[W as i64]);

    let (wd, sd, bd, wd_f) = build_affine_bank(&mut rng, 8, 1, R, W, false);
    let (wi_q8, si_q8, bi_q8, wi_q8_f) = build_affine_bank(&mut rng, 8, 1, HC, W, false);
    let (wu, su, bu, wu_f) = build_affine_bank(&mut rng, 8, 1, W, R, false);

    let wi_dense_vals: Vec<f32> = (0..HC * W)
        .map(|_| f32_round_bf16(rng.uniform(-0.05, 0.05)))
        .collect();
    let wi_dense_bf16 = bf16_array(&wi_dense_vals, &[HC as i64, W as i64]);
    let wi_dense_f32 = MxArray::from_float32(&wi_dense_vals, &[HC as i64, W as i64]).unwrap();

    for m in [1usize, 4, 8, 16] {
        let x_vals: Vec<f32> = (0..m * W)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();
        let br_vals: Vec<f32> = (0..m * H)
            .map(|_| f32_round_bf16(rng.uniform(-1.2, 1.2)))
            .collect();
        let g_vals: Vec<f32> = (0..m * HC)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();

        let x_arr = bf16_array(&x_vals, &[1, m as i64, W as i64]);
        let br_arr = bf16_array(&br_vals, &[1, m as i64, H as i64]);
        let g_arr = bf16_array(&g_vals, &[1, m as i64, HC as i64]);

        // inject_norm: (a) with inject + F32 norm_w + w_off=0.0
        let (got_x, got_norm) =
            inject_norm_cuda(&x_arr, Some((&br_arr, &g_arr)), &nw_f32_arr, EPS, 0.0)
                .unwrap()
                .expect("inject_norm admitted");
        let mut ref_x = vec![0.0f32; m * W];
        let mut ref_norm = vec![0.0f32; m * W];
        for t in 0..m {
            for s in 0..HC {
                let mut sq = 0.0f64;
                for h in 0..H {
                    let sp = f32_round_bf16(br_vals[t * H + h] * g_vals[t * HC + s]);
                    let r = f32_round_bf16(x_vals[t * W + s * H + h] + sp);
                    ref_x[t * W + s * H + h] = r;
                    sq += (r as f64) * (r as f64);
                }
                let denom = ((sq / H as f64) as f32 + EPS).sqrt();
                for h in 0..H {
                    let idx = t * W + s * H + h;
                    ref_norm[idx] = f32_round_bf16((ref_x[idx] / denom) * nw_f32[s * H + h]);
                }
            }
        }
        assert_close(
            &format!("inject_norm(M={m}) x_out"),
            &read_bf16_f32(&got_x),
            &ref_x,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("inject_norm(M={m}) normed"),
            &read_bf16_f32(&got_norm),
            &ref_norm,
            2e-2,
            1e-3,
        );

        // inject_norm: (b) norm-only + BF16 norm_w + w_off=1.0
        let (got_x_noinj, got_norm_bf) = inject_norm_cuda(&x_arr, None, &nw_bf16_arr, EPS, 1.0)
            .unwrap()
            .expect("inject_norm norm-only admitted");
        let mut ref_norm_bf = vec![0.0f32; m * W];
        for t in 0..m {
            for s in 0..HC {
                let mut sq = 0.0f64;
                for h in 0..H {
                    let r = x_vals[t * W + s * H + h];
                    sq += (r as f64) * (r as f64);
                }
                let denom = ((sq / H as f64) as f32 + EPS).sqrt();
                for h in 0..H {
                    let idx = t * W + s * H + h;
                    ref_norm_bf[idx] =
                        f32_round_bf16((x_vals[idx] / denom) * (nw_bf16[s * H + h] + 1.0));
                }
            }
        }
        assert_close(
            &format!("inject_norm(M={m}, no-inject) x"),
            &read_bf16_f32(&got_x_noinj),
            &x_vals,
            0.0,
            0.0,
        );
        assert_close(
            &format!("inject_norm(M={m}, bf16+off1) normed"),
            &read_bf16_f32(&got_norm_bf),
            &ref_norm_bf,
            3e-2,
            1e-3,
        );

        // hc_down_act: inj_mode=1 (Q8_0Bias), inj_mode=0 (None), inj_mode=2 (DenseBf16), inj_mode=4 (DenseFp32)
        let norm_in = bf16_array(&ref_norm, &[1, m as i64, W as i64]);
        let (got_down, got_inj_opt) = hc_down_act_cuda(
            &norm_in,
            &wd,
            &sd,
            &bd,
            Qwen4HcInjectWeight::Q8_0Bias {
                weight: &wi_q8,
                scales: &si_q8,
                biases: &bi_q8,
            },
        )
        .unwrap()
        .expect("hc_down_act(Q8_0Bias) admitted");
        let got_inj = got_inj_opt.expect("inj_mode=1 must return inject_gate");

        let mut ref_down = vec![0.0f32; m * R];
        let mut ref_inj_q8 = vec![0.0f32; m * HC];
        let mut ref_inj_dense = vec![0.0f32; m * HC];
        for t in 0..m {
            let row = &ref_norm[t * W..(t + 1) * W];
            for n in 0..R {
                let mut acc = 0.0f64;
                let wr = &wd_f[n * W..(n + 1) * W];
                for c in 0..W {
                    acc += (row[c] as f64) * (wr[c] as f64);
                }
                let scv = f32_round_bf16(f32_round_bf16(acc as f32) * 0.25);
                ref_down[t * R + n] = silu_bf16(scv);
            }
            for s in 0..HC {
                let mut acc_q8 = 0.0f64;
                let mut acc_d = 0.0f64;
                let wr_q8 = &wi_q8_f[s * W..(s + 1) * W];
                let wr_d = &wi_dense_vals[s * W..(s + 1) * W];
                for c in 0..W {
                    acc_q8 += (row[c] as f64) * (wr_q8[c] as f64);
                    acc_d += (row[c] as f64) * (wr_d[c] as f64);
                }
                let sc_q8 = f32_round_bf16(f32_round_bf16(acc_q8 as f32) * 0.25);
                ref_inj_q8[t * HC + s] = f32_round_bf16(f32_round_bf16(sigmoid_f32(sc_q8)) * 2.0);
                let sc_d = f32_round_bf16(f32_round_bf16(acc_d as f32) * 0.25);
                ref_inj_dense[t * HC + s] = f32_round_bf16(f32_round_bf16(sigmoid_f32(sc_d)) * 2.0);
            }
        }
        assert_close(
            &format!("hc_down_act(M={m}, Q8_0Bias) down"),
            &read_bf16_f32(&got_down),
            &ref_down,
            5e-2,
            5e-3,
        );
        assert_close(
            &format!("hc_down_act(M={m}, Q8_0Bias) inj"),
            &read_bf16_f32(&got_inj),
            &ref_inj_q8,
            5e-2,
            5e-3,
        );

        // inj_mode = 0 (None)
        let (got_down0, got_inj0) =
            hc_down_act_cuda(&norm_in, &wd, &sd, &bd, Qwen4HcInjectWeight::None)
                .unwrap()
                .expect("hc_down_act(None) admitted");
        assert!(got_inj0.is_none());
        assert_close(
            &format!("hc_down_act(M={m}, None) down"),
            &read_bf16_f32(&got_down0),
            &ref_down,
            5e-2,
            5e-3,
        );

        // inj_mode = 2 (DenseBf16) & inj_mode = 4 (DenseFp32)
        for (tag_i, inj_w) in [
            ("DenseBf16", Qwen4HcInjectWeight::DenseBf16(&wi_dense_bf16)),
            ("DenseFp32", Qwen4HcInjectWeight::DenseFp32(&wi_dense_f32)),
        ] {
            let (_, got_inj_d) = hc_down_act_cuda(&norm_in, &wd, &sd, &bd, inj_w)
                .unwrap()
                .expect("hc_down_act(Dense) admitted");
            assert_close(
                &format!("hc_down_act(M={m}, {tag_i}) inj"),
                &read_bf16_f32(got_inj_d.as_ref().unwrap()),
                &ref_inj_dense,
                5e-2,
                5e-3,
            );
        }

        // hc_up_mix
        let down_in = bf16_array(&ref_down, &[1, m as i64, R as i64]);
        let got_mix = hc_up_mix_cuda(&down_in, &norm_in, &wu, &su, &bu)
            .unwrap()
            .expect("hc_up_mix admitted");
        let mut ref_mix = vec![0.0f32; m * H];
        for t in 0..m {
            let drow = &ref_down[t * R..(t + 1) * R];
            for h in 0..H {
                let mut sum4 = 0.0f32;
                for s in 0..HC {
                    let row_idx = s * H + h;
                    let mut acc = 0.0f64;
                    let wr = &wu_f[row_idx * R..(row_idx + 1) * R];
                    for c in 0..R {
                        acc += (drow[c] as f64) * (wr[c] as f64);
                    }
                    let sig = f32_round_bf16(sigmoid_f32(f32_round_bf16(acc as f32)));
                    let p = f32_round_bf16(sig * ref_norm[t * W + row_idx]);
                    sum4 += p;
                }
                ref_mix[t * H + h] = f32_round_bf16(0.25 * sum4);
            }
        }
        assert_close(
            &format!("hc_up_mix(M={m})"),
            &read_bf16_f32(&got_mix),
            &ref_mix,
            5e-2,
            5e-3,
        );
    }
}

// =========================================================================
// 4. QSA Sparse SDPA (`K=4095, Q=1,8,16` causal & tree, `Q=64` prefill)
//    + Dense `sdpa_tc_few` (`GQA=12, Q=1,8,16`)
// =========================================================================
fn cpu_qsa_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    iq: &[f32],
    pooled_k: &[f32],
    tree_mask: Option<&[u32]>,
    hq: usize,
    hkv: usize,
    q_len: usize,
    kv_len: usize,
    num_blocks: usize,
    scale: f32,
) -> Vec<f32> {
    const D: usize = 256;
    const ID: usize = 128;
    let gqa = hq / hkv;
    let base_pos = kv_len - q_len;
    let inv_sqrt_id = 1.0f32 / (ID as f32).sqrt();
    let tree_mode = tree_mask.is_some();
    let mut out = vec![0.0f32; hq * q_len * D];

    for qi in 0..q_len {
        let complete = if tree_mode {
            (base_pos / 4).min(num_blocks)
        } else {
            ((base_pos + qi + 1) / 4).min(num_blocks)
        };
        let selected_blocks: Vec<usize> = if complete <= 512 {
            (0..complete).collect()
        } else {
            let mut scored: Vec<(f32, usize)> = (0..complete)
                .map(|b| {
                    let pk = &pooled_k[b * ID..(b + 1) * ID];
                    let mut s = 0.0f32;
                    for ih in 0..4 {
                        let iqr = &iq[(ih * q_len + qi) * ID..(ih * q_len + qi + 1) * ID];
                        let mut dot = 0.0f32;
                        for d in 0..ID {
                            dot += iqr[d] * pk[d];
                        }
                        s += dot.max(0.0);
                    }
                    (s * inv_sqrt_id, b)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut top: Vec<usize> = scored.into_iter().take(512).map(|(_, b)| b).collect();
            top.sort_unstable();
            top
        };

        // Build exact allowed token list: selected complete blocks + tail tokens.
        let mut toks: Vec<usize> = Vec::with_capacity(selected_blocks.len() * 4 + 16);
        for &b in &selected_blocks {
            for j in 0..4 {
                toks.push(b * 4 + j);
            }
        }
        if let Some(tm) = tree_mask {
            let tail_start = (base_pos / 4) * 4;
            for t in tail_start..base_pos {
                toks.push(t);
            }
            let mask = tm[qi];
            for node in 0..q_len {
                if ((mask >> node) & 1) != 0 {
                    toks.push(base_pos + node);
                }
            }
        } else {
            let end = base_pos + qi + 1;
            let tail_start = complete * 4;
            for t in tail_start..end {
                toks.push(t);
            }
        }

        for h in 0..hq {
            let kh = h / gqa;
            let qrow = &q[(h * q_len + qi) * D..(h * q_len + qi + 1) * D];
            let mut logits = Vec::with_capacity(toks.len());
            let mut max_l = f32::NEG_INFINITY;
            for &ti in &toks {
                let krow = &k[(kh * kv_len + ti) * D..(kh * kv_len + ti + 1) * D];
                let mut dot = 0.0f32;
                for d in 0..D {
                    dot += qrow[d] * krow[d];
                }
                let l = dot * scale;
                if l > max_l {
                    max_l = l;
                }
                logits.push(l);
            }
            let mut sum_e = 0.0f64;
            let mut acc = [0.0f64; D];
            for (&l, &ti) in logits.iter().zip(&toks) {
                let p = ((l - max_l) as f64).exp();
                sum_e += p;
                let vrow = &v[(kh * kv_len + ti) * D..(kh * kv_len + ti + 1) * D];
                for d in 0..D {
                    acc[d] += p * (vrow[d] as f64);
                }
            }
            let inv = 1.0 / sum_e;
            let orow = &mut out[(h * q_len + qi) * D..(h * q_len + qi + 1) * D];
            for d in 0..D {
                orow[d] = f32_round_bf16((acc[d] * inv) as f32);
            }
        }
    }
    out
}

#[test]
fn qwen4_qsa_sparse_sdpa_and_dense_gqa12_parity() {
    if !cuda_available() {
        return;
    }
    const HQ: usize = 24;
    const HKV: usize = 2;
    const GQA: usize = HQ / HKV; // 12
    const D: usize = 256;
    const ID: usize = 128;
    const KV_LEN: usize = 4095;
    const NUM_BLOCKS: usize = KV_LEN / 4; // 1023
    let scale = 1.0f32 / (D as f32).sqrt();
    let idx_scale = 1.0f32 / (ID as f32).sqrt();
    let mut rng = Rng::new(0xCAFE_BABE);

    let k_vals: Vec<f32> = (0..HKV * KV_LEN * D)
        .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
        .collect();
    let v_vals: Vec<f32> = (0..HKV * KV_LEN * D)
        .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
        .collect();
    let pk_vals: Vec<f32> = (0..NUM_BLOCKS * ID)
        .map(|_| f32_round_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let k_arr = bf16_array(&k_vals, &[1, HKV as i64, KV_LEN as i64, D as i64]);
    let v_arr = bf16_array(&v_vals, &[1, HKV as i64, KV_LEN as i64, D as i64]);
    let pk_arr = bf16_array(&pk_vals, &[NUM_BLOCKS as i64, ID as i64]);

    // 4A. QSA Sparse SDPA causal: Q in {1, 8, 16} (split-K) and Q=64 (prefill fused)
    for q_len in [1usize, 8, 16, 64] {
        let q_vals: Vec<f32> = (0..HQ * q_len * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
            .collect();
        let iq_vals: Vec<f32> = (0..4 * q_len * ID)
            .map(|_| f32_round_bf16(rng.uniform(-1.0, 1.0)))
            .collect();
        let q_arr = bf16_array(&q_vals, &[1, HQ as i64, q_len as i64, D as i64]);
        let iq_arr = bf16_array(&iq_vals, &[1, 4, q_len as i64, ID as i64]);

        let got = qsa_sparse_sdpa_cuda(
            &q_arr,
            &k_arr,
            &v_arr,
            &iq_arr,
            &pk_arr,
            None,
            KV_LEN as i32,
            NUM_BLOCKS as i32,
            scale,
            idx_scale,
            0,
        )
        .unwrap()
        .expect("qsa_sparse_sdpa causal admitted");
        let want = cpu_qsa_ref(
            &q_vals, &k_vals, &v_vals, &iq_vals, &pk_vals, None, HQ, HKV, q_len, KV_LEN,
            NUM_BLOCKS, scale,
        );
        assert_close(
            &format!("qsa_sparse_sdpa(K={KV_LEN}, Q={q_len}, causal)"),
            &read_bf16_f32(&got),
            &want,
            2.5e-2,
            5e-3,
        );
    }

    // 4B. QSA Sparse SDPA tree mode: Q=8 and Q=16 with branching tree_mask
    for q_len in [8usize, 16] {
        let q_vals: Vec<f32> = (0..HQ * q_len * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
            .collect();
        let iq_vals: Vec<f32> = (0..4 * q_len * ID)
            .map(|_| f32_round_bf16(rng.uniform(-1.0, 1.0)))
            .collect();
        // Build binary-like tree: node 0 is root, node i parent is (i-1)/2.
        let mut tm = vec![0u32; q_len];
        for (i, slot) in tm.iter_mut().enumerate() {
            let mut cur = i;
            let mut mask = 1u32 << i;
            while cur > 0 {
                cur = (cur - 1) / 2;
                mask |= 1u32 << cur;
            }
            *slot = mask;
        }
        let q_arr = bf16_array(&q_vals, &[1, HQ as i64, q_len as i64, D as i64]);
        let iq_arr = bf16_array(&iq_vals, &[1, 4, q_len as i64, ID as i64]);
        let tm_arr = MxArray::from_uint32(&tm, &[q_len as i64]).unwrap();

        let got = qsa_sparse_sdpa_cuda(
            &q_arr,
            &k_arr,
            &v_arr,
            &iq_arr,
            &pk_arr,
            Some(&tm_arr),
            KV_LEN as i32,
            NUM_BLOCKS as i32,
            scale,
            idx_scale,
            1,
        )
        .unwrap()
        .expect("qsa_sparse_sdpa tree admitted");
        let want = cpu_qsa_ref(
            &q_vals,
            &k_vals,
            &v_vals,
            &iq_vals,
            &pk_vals,
            Some(&tm),
            HQ,
            HKV,
            q_len,
            KV_LEN,
            NUM_BLOCKS,
            scale,
        );
        assert_close(
            &format!("qsa_sparse_sdpa(K={KV_LEN}, Q={q_len}, tree)"),
            &read_bf16_f32(&got),
            &want,
            2.5e-2,
            5e-3,
        );
    }

    // 4C. Dense SDPA `sdpa_tc_few` GQA=12 (Q=1, 8, 16 causal & Q=16 tree)
    let k_small_len = 512usize;
    let ks_vals: Vec<f32> = (0..HKV * k_small_len * D)
        .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
        .collect();
    let vs_vals: Vec<f32> = (0..HKV * k_small_len * D)
        .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
        .collect();
    let ks_arr = bf16_array(&ks_vals, &[1, HKV as i64, k_small_len as i64, D as i64]);
    let vs_arr = bf16_array(&vs_vals, &[1, HKV as i64, k_small_len as i64, D as i64]);
    let dummy_pk = bf16_array(&vec![0.0; ID], &[1, ID as i64]);

    for q_len in [1usize, 8, 16] {
        let q_vals: Vec<f32> = (0..HQ * q_len * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
            .collect();
        let q_arr = bf16_array(&q_vals, &[1, HQ as i64, q_len as i64, D as i64]);
        let got = attention::sdpa_causal_few_q1_to_16(
            &q_arr,
            &ks_arr,
            &vs_arr,
            D as i64,
            GQA as i64,
            q_len as i64,
            scale,
        )
        .unwrap()
        .expect("sdpa_tc_few GQA=12 admitted");
        // With num_blocks = 128 <= 512, cpu_qsa_ref attends to ALL causal tokens (exact dense SDPA!).
        let dummy_iq = vec![0.0f32; 4 * q_len * ID];
        let want = cpu_qsa_ref(
            &q_vals,
            &ks_vals,
            &vs_vals,
            &dummy_iq,
            &vec![0.0; 128 * ID],
            None,
            HQ,
            HKV,
            q_len,
            k_small_len,
            128,
            scale,
        );
        assert_close(
            &format!("sdpa_tc_few(GQA=12, Q={q_len}, causal)"),
            &read_bf16_f32(&got),
            &want,
            2.5e-2,
            5e-3,
        );
        let _ = &dummy_pk;
    }

    // Dense SDPA tree with GQA=12, Q=16 (exercises 2-tile M_tile=96!)
    {
        let q_len = 16usize;
        let cache_len = (k_small_len - q_len) as i32;
        let q_vals: Vec<f32> = (0..HQ * q_len * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.8, 0.8)))
            .collect();
        let mut tm = vec![0u32; q_len];
        for (i, slot) in tm.iter_mut().enumerate() {
            let mut cur = i;
            let mut mask = 1u32 << i;
            while cur > 0 {
                cur = (cur - 1) / 2;
                mask |= 1u32 << cur;
            }
            *slot = mask;
        }
        let q_arr = bf16_array(&q_vals, &[1, HQ as i64, q_len as i64, D as i64]);
        let tm_arr = MxArray::from_uint32(&tm, &[q_len as i64]).unwrap();
        let got = attention::sdpa_tree_few(
            &q_arr,
            &ks_arr,
            &vs_arr,
            &tm_arr,
            cache_len,
            D as i64,
            GQA as i64,
            q_len as i64,
            scale,
        )
        .unwrap()
        .expect("sdpa_tree_few GQA=12 Q=16 admitted");
        let dummy_iq = vec![0.0f32; 4 * q_len * ID];
        let want = cpu_qsa_ref(
            &q_vals,
            &ks_vals,
            &vs_vals,
            &dummy_iq,
            &vec![0.0; 128 * ID],
            Some(&tm),
            HQ,
            HKV,
            q_len,
            k_small_len,
            128,
            scale,
        );
        assert_close(
            "sdpa_tree_few(GQA=12, Q=16)",
            &read_bf16_f32(&got),
            &want,
            2.5e-2,
            5e-3,
        );
    }
}

// =========================================================================
// 5. PLE Layer-1 Fused Gate + Dilated Causal Conv1d (`M=1, 16, 64`)
// =========================================================================
#[test]
fn qwen4_ple_gate_conv_parity() {
    if !cuda_available() {
        return;
    }
    const H: usize = 2560;
    const HC: usize = 4;
    const W: usize = H * HC;
    const KEEP: usize = 9;
    const EPS: f32 = 1e-6;
    let mut rng = Rng::new(0xDEAD_BEEF);

    let wk: Vec<f32> = (0..W).map(|_| rng.uniform(0.5, 1.5)).collect();
    let wq: Vec<f32> = (0..W).map(|_| rng.uniform(0.5, 1.5)).collect();
    let wc: Vec<f32> = (0..W).map(|_| rng.uniform(0.5, 1.5)).collect();
    let cw: Vec<f32> = (0..W * 4).map(|_| rng.uniform(-0.4, 0.4)).collect();

    let wk_f32 = MxArray::from_float32(&wk, &[W as i64]).unwrap();
    let wq_f32 = MxArray::from_float32(&wq, &[W as i64]).unwrap();
    let wc_f32 = MxArray::from_float32(&wc, &[W as i64]).unwrap();
    let cw_f32 = MxArray::from_float32(&cw, &[W as i64, 1, 4]).unwrap();

    let wk_bf_v: Vec<f32> = wk.iter().copied().map(f32_round_bf16).collect();
    let wq_bf_v: Vec<f32> = wq.iter().copied().map(f32_round_bf16).collect();
    let wc_bf_v: Vec<f32> = wc.iter().copied().map(f32_round_bf16).collect();
    let cw_bf_v: Vec<f32> = cw.iter().copied().map(f32_round_bf16).collect();
    let wk_bf = bf16_array(&wk_bf_v, &[W as i64]);
    let wq_bf = bf16_array(&wq_bf_v, &[W as i64]);
    let wc_bf = bf16_array(&wc_bf_v, &[W as i64]);
    let cw_bf = bf16_array(&cw_bf_v, &[W as i64, 4]);

    for &(t_len, fp32_w, fuse_res, has_hist) in &[
        (1usize, true, true, true),
        (16, true, false, true),
        (16, false, true, true),
        (64, true, true, false),
    ] {
        let w_off = if fp32_w { 0.0f32 } else { 1.0f32 };
        let x_v: Vec<f32> = (0..t_len * W)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();
        let k_v: Vec<f32> = (0..t_len * W)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();
        let v_v: Vec<f32> = (0..t_len * H)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();
        let h_v: Vec<f32> = (0..KEEP * W)
            .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
            .collect();

        let x_arr = bf16_array(&x_v, &[1, t_len as i64, W as i64]);
        let k_arr = bf16_array(&k_v, &[1, t_len as i64, W as i64]);
        let v_arr = bf16_array(&v_v, &[1, t_len as i64, H as i64]);
        let h_arr = bf16_array(&h_v, &[KEEP as i64, W as i64]);

        let (got_x, got_h) = ple_gate_conv_cuda(
            &x_arr,
            &k_arr,
            &v_arr,
            if fp32_w { &wk_f32 } else { &wk_bf },
            if fp32_w { &wq_f32 } else { &wq_bf },
            if fp32_w { &wc_f32 } else { &wc_bf },
            if fp32_w { &cw_f32 } else { &cw_bf },
            has_hist.then_some(&h_arr),
            EPS,
            w_off,
            fuse_res,
        )
        .unwrap()
        .expect("ple_gate_conv_cuda admitted");

        // CPU reference
        let wk_r = if fp32_w { &wk } else { &wk_bf_v };
        let wq_r = if fp32_w { &wq } else { &wq_bf_v };
        let wc_r = if fp32_w { &wc } else { &wc_bf_v };
        let cw_r = if fp32_w { &cw } else { &cw_bf_v };

        let mut gated_val = vec![0.0f32; t_len * W];
        let mut conv_in = vec![0.0f32; t_len * W];
        for t in 0..t_len {
            for s in 0..HC {
                let soff = (t * HC + s) * H;
                let woff = s * H;
                let mut sq_k = 0.0f64;
                let mut sq_q = 0.0f64;
                for h in 0..H {
                    sq_k += (k_v[soff + h] as f64).powi(2);
                    sq_q += (x_v[soff + h] as f64).powi(2);
                }
                let dk = ((sq_k / H as f64) as f32 + EPS).sqrt();
                let dq = ((sq_q / H as f64) as f32 + EPS).sqrt();
                let mut dot = 0.0f64;
                for h in 0..H {
                    let kn = f32_round_bf16((k_v[soff + h] / dk) * (wk_r[woff + h] + w_off));
                    let qn = f32_round_bf16((x_v[soff + h] / dq) * (wq_r[woff + h] + w_off));
                    dot += (kn as f64) * (qn as f64);
                }
                let sc = (dot as f32) / (H as f32).sqrt();
                let sgn = if sc > 0.0 {
                    1.0
                } else if sc < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let ssqrt = sgn * sc.abs().max(1e-6).sqrt();
                let gate = f32_round_bf16(sigmoid_f32(ssqrt));

                let mut sq_gv = 0.0f64;
                for h in 0..H {
                    let gv = f32_round_bf16(gate * v_v[t * H + h]);
                    gated_val[soff + h] = gv;
                    sq_gv += (gv as f64).powi(2);
                }
                let dc = ((sq_gv / H as f64) as f32 + EPS).sqrt();
                for h in 0..H {
                    conv_in[soff + h] =
                        f32_round_bf16((gated_val[soff + h] / dc) * (wc_r[woff + h] + w_off));
                }
            }
        }

        let load_full = |row: usize, c: usize| -> f32 {
            if row < KEEP {
                if has_hist { h_v[row * W + c] } else { 0.0 }
            } else {
                conv_in[(row - KEEP) * W + c]
            }
        };
        let mut ref_out = vec![0.0f32; t_len * W];
        let mut ref_hist = vec![0.0f32; KEEP * W];
        for t in 0..t_len {
            for c in 0..W {
                let acc = load_full(t, c) * cw_r[c * 4]
                    + load_full(t + 3, c) * cw_r[c * 4 + 1]
                    + load_full(t + 6, c) * cw_r[c * 4 + 2]
                    + load_full(t + 9, c) * cw_r[c * 4 + 3];
                let silu = silu_bf16(acc);
                let off = t * W + c;
                let ple = f32_round_bf16(gated_val[off] + silu);
                ref_out[off] = if fuse_res {
                    f32_round_bf16(x_v[off] + ple)
                } else {
                    ple
                };
            }
        }
        for r in 0..KEEP {
            for c in 0..W {
                ref_hist[r * W + c] = load_full(t_len + r, c);
            }
        }
        let tag = format!("ple(T={t_len},fp32={fp32_w},res={fuse_res},hist={has_hist})");
        assert_close(
            &format!("{tag} out"),
            &read_bf16_f32(&got_x),
            &ref_out,
            2e-2,
            1e-3,
        );
        assert_close(
            &format!("{tag} hist"),
            &read_bf16_f32(&got_h),
            &ref_hist,
            2e-2,
            1e-3,
        );
    }
}

#[test]
fn qwen4_ple_gate_conv_tree_parity() {
    if !cuda_available() {
        return;
    }
    const H: usize = 2560;
    const HC: usize = 4;
    const W: usize = H * HC;
    const KEEP: usize = 9;
    const EPS: f32 = 1e-6;
    let mut rng = Rng::new(0xBEEF_CAFE);

    let wk: Vec<f32> = (0..W).map(|_| rng.uniform(0.5, 1.5)).collect();
    let wq: Vec<f32> = (0..W).map(|_| rng.uniform(0.5, 1.5)).collect();
    let wc: Vec<f32> = (0..W).map(|_| rng.uniform(0.5, 1.5)).collect();
    let cw: Vec<f32> = (0..W * 4).map(|_| rng.uniform(-0.4, 0.4)).collect();

    let wk_f32 = MxArray::from_float32(&wk, &[W as i64]).unwrap();
    let wq_f32 = MxArray::from_float32(&wq, &[W as i64]).unwrap();
    let wc_f32 = MxArray::from_float32(&wc, &[W as i64]).unwrap();
    let cw_f32 = MxArray::from_float32(&cw, &[W as i64, 1, 4]).unwrap();

    let wk_bf = bf16_array(&wk, &[W as i64]);
    let wq_bf = bf16_array(&wq, &[W as i64]);
    let wc_bf = bf16_array(&wc, &[W as i64]);
    let cw_bf = bf16_array(&cw, &[W as i64, 4]);

    let t_len = 16usize;
    let x_v: Vec<f32> = (0..t_len * W)
        .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
        .collect();
    let k_v: Vec<f32> = (0..t_len * W)
        .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
        .collect();
    let v_v: Vec<f32> = (0..t_len * H)
        .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
        .collect();
    let h_v: Vec<f32> = (0..KEEP * W)
        .map(|_| f32_round_bf16(rng.uniform(-1.5, 1.5)))
        .collect();
    let x_arr = bf16_array(&x_v, &[1, t_len as i64, W as i64]);
    let k_arr = bf16_array(&k_v, &[1, t_len as i64, W as i64]);
    let v_arr = bf16_array(&v_v, &[1, t_len as i64, H as i64]);
    let h_arr = bf16_array(&h_v, &[KEEP as i64, W as i64]);

    let build_taps = |parent: &[i32]| -> MxArray {
        let step_back = |idx: i32| -> i32 {
            if idx >= 0 {
                parent[idx as usize]
            } else {
                idx - 1
            }
        };
        let back3 = |idx: i32| -> i32 { step_back(step_back(step_back(idx))) };
        let mut flat = Vec::with_capacity(parent.len() * 4);
        for i in 0..parent.len() {
            let t3 = i as i32;
            let t2 = back3(t3);
            let t1 = back3(t2);
            let t0 = back3(t1);
            flat.extend_from_slice(&[t0, t1, t2, t3]);
        }
        MxArray::from_int32(&flat, &[parent.len() as i64, 4]).unwrap()
    };

    // 1. Linear chain: taps-mode must be 0-ULP bit-identical to linear ple_gate_conv_cuda.
    let chain_parent: Vec<i32> = (0..t_len as i32).map(|i| i - 1).collect();
    let chain_taps = build_taps(&chain_parent);
    for &(fp32_w, fuse_res, has_hist) in &[
        (true, true, true),
        (false, true, true),
        (true, false, false),
    ] {
        let w_off = if fp32_w { 0.0 } else { 1.0 };
        let (lin_x, lin_h) = ple_gate_conv_cuda(
            &x_arr,
            &k_arr,
            &v_arr,
            if fp32_w { &wk_f32 } else { &wk_bf },
            if fp32_w { &wq_f32 } else { &wq_bf },
            if fp32_w { &wc_f32 } else { &wc_bf },
            if fp32_w { &cw_f32 } else { &cw_bf },
            has_hist.then_some(&h_arr),
            EPS,
            w_off,
            fuse_res,
        )
        .unwrap()
        .expect("linear ple admitted");
        let (tree_x, conv_in) = ple_gate_conv_tree_cuda(
            &x_arr,
            &k_arr,
            &v_arr,
            if fp32_w { &wk_f32 } else { &wk_bf },
            if fp32_w { &wq_f32 } else { &wq_bf },
            if fp32_w { &wc_f32 } else { &wc_bf },
            if fp32_w { &cw_f32 } else { &cw_bf },
            has_hist.then_some(&h_arr),
            Some(&chain_taps),
            EPS,
            w_off,
            fuse_res,
        )
        .unwrap()
        .expect("tree ple admitted");
        assert_eq!(
            tree_x.to_uint16_native().unwrap(),
            lin_x.to_uint16_native().unwrap(),
            "linear-chain tree PLE must be 0-ULP identical (fp32={fp32_w}, res={fuse_res}, hist={has_hist})"
        );
        // Rebuilt history from last 9 rows of conv_in (since T=16 >= 9) must match lin_h bit-for-bit.
        let rebuilt = conv_in
            .slice_axis(0, (t_len - KEEP) as i64, t_len as i64)
            .unwrap();
        assert_eq!(
            rebuilt.to_uint16_native().unwrap(),
            lin_h.to_uint16_native().unwrap(),
            "rebuilt linear PLE history must be 0-ULP identical"
        );
    }

    // 2. Arbitrary branching tree (depth up to 11 -> exercises all 4 dilated taps across
    //    both committed history and tree ancestors!) vs sequential AR step-by-step.
    // Tree topology (16 nodes, DFS preorder):
    //   0(-1) -> 1(0) -> 2(1) -> 3(2) -> 4(3) -> 5(4) -> 6(5) -> 7(6) -> 8(7) -> 9(8) -> 10(9) -> 11(10)
    //         -> 12(0) -> 13(12) -> 14(13) -> 15(14)
    let tree_parent = [-1i32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 12, 13, 14];
    let tree_taps = build_taps(&tree_parent);
    let (got_tree_x, got_conv_in) = ple_gate_conv_tree_cuda(
        &x_arr,
        &k_arr,
        &v_arr,
        &wk_f32,
        &wq_f32,
        &wc_f32,
        &cw_f32,
        Some(&h_arr),
        Some(&tree_taps),
        EPS,
        0.0,
        true,
    )
    .unwrap()
    .expect("branching tree ple admitted");
    let got_bits = got_tree_x.to_uint16_native().unwrap();

    // Check every node against running linear ple_gate_conv_cuda along its root-to-node path!
    for node in 0..t_len {
        let mut path = Vec::new();
        let mut cur = node as i32;
        while cur >= 0 {
            path.push(cur as usize);
            cur = tree_parent[cur as usize];
        }
        path.reverse();
        let idx_i32: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        let idx_arr = MxArray::from_int32(&idx_i32, &[idx_i32.len() as i64]).unwrap();
        let px = x_arr.take(&idx_arr, 1).unwrap();
        let pk = k_arr.take(&idx_arr, 1).unwrap();
        let pv = v_arr.take(&idx_arr, 1).unwrap();
        let (ref_x, ref_h) = ple_gate_conv_cuda(
            &px,
            &pk,
            &pv,
            &wk_f32,
            &wq_f32,
            &wc_f32,
            &cw_f32,
            Some(&h_arr),
            EPS,
            0.0,
            true,
        )
        .unwrap()
        .unwrap();
        let ref_bits = ref_x.to_uint16_native().unwrap();
        let last_start = (path.len() - 1) * W;
        assert_eq!(
            &got_bits[node * W..(node + 1) * W],
            &ref_bits[last_start..last_start + W],
            "branching tree PLE node {node} (path {path:?}) must be 0-ULP identical to AR path"
        );

        // Verify commit history reconstruction along this branch (both < 9 and >= 9 lengths).
        let gathered_ci = got_conv_in.take(&idx_arr, 0).unwrap();
        let combined = MxArray::concatenate(&h_arr, &gathered_ci, 0).unwrap();
        let total_rows = (KEEP + path.len()) as i64;
        let rebuilt_h = combined
            .slice_axis(0, total_rows - KEEP as i64, total_rows)
            .unwrap();
        assert_eq!(
            rebuilt_h.to_uint16_native().unwrap(),
            ref_h.to_uint16_native().unwrap(),
            "rebuilt PLE history for path {path:?} must be 0-ULP identical to AR path"
        );
    }
}

// =========================================================================
// 6. GDN with `sigmoid_gate=true` + FP32 `norm_w` + FP32 `beta`
//    (fused_step, deferred, tree, prefill_chain_gated, replay)
// =========================================================================
fn cpu_gdn_step_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    z: &[f32],
    w: &[f32],
    t_len: usize,
    hk: usize,
    hv: usize,
    tiled_heads: bool,
    eps: f32,
    sigmoid_gate: bool,
) -> (Vec<f32>, Vec<f32>) {
    const D: usize = 128;
    let mut state = state_in.to_vec();
    let mut out = vec![0.0f32; t_len * hv * D];
    for h in 0..hv {
        let kh = if tiled_heads { h % hk } else { h / (hv / hk) };
        let s_off = h * D * D;
        for t in 0..t_len {
            let q_row = &q[(t * hk + kh) * D..(t * hk + kh + 1) * D];
            let k_row = &k[(t * hk + kh) * D..(t * hk + kh + 1) * D];
            let v_row = &v[(t * hv + h) * D..(t * hv + h + 1) * D];
            let decay = g[t * hv + h];
            let bt = beta[t * hv + h];
            let mut y_head = [0.0f32; D];
            for dv in 0..D {
                let s_row = &mut state[s_off + dv * D..s_off + (dv + 1) * D];
                let mut kv = 0.0f32;
                for dk in 0..D {
                    kv += s_row[dk] * k_row[dk];
                }
                let delta = (v_row[dv] - decay * kv) * bt;
                let mut y = 0.0f32;
                for dk in 0..D {
                    s_row[dk] = decay * s_row[dk] + k_row[dk] * delta;
                    y += s_row[dk] * q_row[dk];
                }
                y_head[dv] = f32_round_bf16(y);
            }
            let sq: f32 = y_head.iter().map(|&v| v * v).sum();
            let denom = (sq / D as f32 + eps).sqrt();
            let z_row = &z[(t * hv + h) * D..(t * hv + h + 1) * D];
            let o_row = &mut out[(t * hv + h) * D..(t * hv + h + 1) * D];
            for d in 0..D {
                let normed = f32_round_bf16((y_head[d] / denom) * w[d]);
                let sig = sigmoid_f32(z_row[d]);
                o_row[d] = if sigmoid_gate {
                    f32_round_bf16(sig * normed)
                } else {
                    let silu = f32_round_bf16(f32_round_bf16(sig) * z_row[d]);
                    f32_round_bf16(silu * normed)
                };
            }
        }
    }
    (out, state)
}

#[test]
fn qwen4_gdn_sigmoid_gate_and_f32_parity() {
    if !cuda_available() {
        return;
    }
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    const EPS: f32 = 1e-6;
    let mut rng = Rng::new(0x1357_2468);

    let state0: Vec<f32> = (0..HV * D * D).map(|_| rng.uniform(-0.2, 0.2)).collect();
    let w_f32: Vec<f32> = (0..D).map(|_| rng.uniform(0.5, 1.5)).collect();
    let state0_arr = MxArray::from_float32(&state0, &[1, HV as i64, D as i64, D as i64]).unwrap();
    let w_arr = MxArray::from_float32(&w_f32, &[D as i64]).unwrap();

    let gen_window = |rng: &mut Rng, t: usize| {
        let q: Vec<f32> = (0..t * HK * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.3, 0.3)))
            .collect();
        let k: Vec<f32> = (0..t * HK * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.3, 0.3)))
            .collect();
        let v: Vec<f32> = (0..t * HV * D)
            .map(|_| f32_round_bf16(rng.uniform(-0.5, 0.5)))
            .collect();
        let g: Vec<f32> = (0..t * HV).map(|_| rng.uniform(0.85, 0.99)).collect();
        let beta: Vec<f32> = (0..t * HV).map(|_| rng.uniform(0.05, 0.95)).collect();
        let z: Vec<f32> = (0..t * HV * D)
            .map(|_| f32_round_bf16(rng.uniform(-2.0, 2.0)))
            .collect();
        (q, k, v, g, beta, z)
    };

    // 6A. fused_step_with_gate(sigmoid_gate=true) for T in {1, 4, 8}
    for t in [1usize, 4, 8] {
        let (q, k, v, g, beta, z) = gen_window(&mut rng, t);
        let qa = bf16_array(&q, &[1, t as i64, HK as i64, D as i64]);
        let ka = bf16_array(&k, &[1, t as i64, HK as i64, D as i64]);
        let va = bf16_array(&v, &[1, t as i64, HV as i64, D as i64]);
        let ga = MxArray::from_float32(&g, &[1, t as i64, HV as i64]).unwrap();
        let ba = MxArray::from_float32(&beta, &[1, t as i64, HV as i64]).unwrap();
        let za = bf16_array(&z, &[1, t as i64, (HV * D) as i64]);

        let (got_y, got_s) = gdn::fused_step_with_gate(
            &qa,
            &ka,
            &va,
            &ga,
            &ba,
            &state0_arr,
            &za,
            &w_arr,
            EPS,
            true,
            true,
            None,
        )
        .unwrap()
        .expect("gdn::fused_step_with_gate admitted");
        let (want_y, want_s) = cpu_gdn_step_ref(
            &q, &k, &v, &g, &beta, &state0, &z, &w_f32, t, HK, HV, true, EPS, true,
        );
        assert_close(
            &format!("gdn_fused_step(T={t}, sigmoid, f32_w, f32_beta) y"),
            &read_bf16_f32(&got_y),
            &want_y,
            2e-2,
            2e-3,
        );
        got_s.eval();
        assert_close(
            &format!("gdn_fused_step(T={t}, sigmoid) state"),
            &got_s.to_float32().unwrap(),
            &want_s,
            2e-3,
            2e-4,
        );

        // Also verify FP32-beta gdn::replay matches!
        let got_rep = gdn::replay(&ka, &va, &ga, &ba, &state0_arr, t as i32, true)
            .unwrap()
            .expect("gdn::replay FP32 beta admitted");
        got_rep.eval();
        assert_close(
            &format!("gdn_replay(T={t}, f32_beta)"),
            &got_rep.to_float32().unwrap(),
            &want_s,
            2e-3,
            2e-4,
        );
    }

    // 6B. fused_step_deferred_with_gate(sigmoid_gate=true) with prev_keep=4
    {
        let (pq, pk, pv, pg, pbeta, pz) = gen_window(&mut rng, 6);
        let (q, k, v, g, beta, z) = gen_window(&mut rng, 8);
        let pka = bf16_array(&pk, &[1, 6, HK as i64, D as i64]);
        let pva = bf16_array(&pv, &[1, 6, HV as i64, D as i64]);
        let pga = MxArray::from_float32(&pg, &[1, 6, HV as i64]).unwrap();
        let pba = MxArray::from_float32(&pbeta, &[1, 6, HV as i64]).unwrap();
        let qa = bf16_array(&q, &[1, 8, HK as i64, D as i64]);
        let ka = bf16_array(&k, &[1, 8, HK as i64, D as i64]);
        let va = bf16_array(&v, &[1, 8, HV as i64, D as i64]);
        let ga = MxArray::from_float32(&g, &[1, 8, HV as i64]).unwrap();
        let ba = MxArray::from_float32(&beta, &[1, 8, HV as i64]).unwrap();
        let za = bf16_array(&z, &[1, 8, (HV * D) as i64]);

        let (got_y, got_cs) = gdn::fused_step_deferred_with_gate(
            &qa,
            &ka,
            &va,
            &ga,
            &ba,
            &state0_arr,
            Some((&pka, &pva, &pga, &pba, 4)),
            &za,
            &w_arr,
            EPS,
            true,
            true,
        )
        .unwrap()
        .expect("gdn_deferred(sigmoid) admitted");
        let (_, committed) = cpu_gdn_step_ref(
            &pq[..4 * HK * D],
            &pk[..4 * HK * D],
            &pv[..4 * HV * D],
            &pg[..4 * HV],
            &pbeta[..4 * HV],
            &state0,
            &pz[..4 * HV * D],
            &w_f32,
            4,
            HK,
            HV,
            true,
            EPS,
            true,
        );
        let (want_y, _) = cpu_gdn_step_ref(
            &q, &k, &v, &g, &beta, &committed, &z, &w_f32, 8, HK, HV, true, EPS, true,
        );
        assert_close(
            "gdn_deferred(prev=4, T=8, sigmoid) y",
            &read_bf16_f32(&got_y),
            &want_y,
            2e-2,
            2e-3,
        );
        got_cs.eval();
        assert_close(
            "gdn_deferred(prev=4, T=8, sigmoid) committed_state",
            &got_cs.to_float32().unwrap(),
            &committed,
            2e-3,
            2e-4,
        );
    }

    // 6C. tree_step_deferred_with_gate(sigmoid_gate=true) with branching tree + non-contiguous prev_indices
    {
        let (_, pk, pv, pg, pbeta, _) = gen_window(&mut rng, 6);
        let prev_idx = [0i32, 2, 5];
        let (q, k, v, g, beta, z) = gen_window(&mut rng, 8);
        // Tree: 0(-1) -> 1(0), 2(0); 1 -> 3(1), 4(1); 2 -> 5(2); 3 -> 6(3); 5 -> 7(5)
        let parent = [-1i32, 0, 0, 1, 1, 2, 3, 5];

        let pka = bf16_array(&pk, &[1, 6, HK as i64, D as i64]);
        let pva = bf16_array(&pv, &[1, 6, HV as i64, D as i64]);
        let pga = MxArray::from_float32(&pg, &[1, 6, HV as i64]).unwrap();
        let pba = MxArray::from_float32(&pbeta, &[1, 6, HV as i64]).unwrap();
        let pidx_arr = MxArray::from_int32(&prev_idx, &[3]).unwrap();
        let par_arr = MxArray::from_int32(&parent, &[8]).unwrap();
        let qa = bf16_array(&q, &[1, 8, HK as i64, D as i64]);
        let ka = bf16_array(&k, &[1, 8, HK as i64, D as i64]);
        let va = bf16_array(&v, &[1, 8, HV as i64, D as i64]);
        let ga = MxArray::from_float32(&g, &[1, 8, HV as i64]).unwrap();
        let ba = MxArray::from_float32(&beta, &[1, 8, HV as i64]).unwrap();
        let za = bf16_array(&z, &[1, 8, (HV * D) as i64]);

        let (got_y, got_cs) = gdn::tree_step_deferred_with_gate(
            &qa,
            &ka,
            &va,
            &ga,
            &ba,
            &state0_arr,
            Some((&pka, &pva, &pga, &pba, 3, Some(&pidx_arr))),
            &par_arr,
            &za,
            &w_arr,
            EPS,
            true,
            true,
        )
        .unwrap()
        .expect("gdn_tree(sigmoid) admitted");

        // Reference: fold prev_idx [0, 2, 5] into state0, then evaluate each node along its path.
        let mut gpk = Vec::new();
        let mut gpv = Vec::new();
        let mut gpg = Vec::new();
        let mut gpb = Vec::new();
        for &pi in &prev_idx {
            let idx = pi as usize;
            gpk.extend_from_slice(&pk[idx * HK * D..(idx + 1) * HK * D]);
            gpv.extend_from_slice(&pv[idx * HV * D..(idx + 1) * HV * D]);
            gpg.extend_from_slice(&pg[idx * HV..(idx + 1) * HV]);
            gpb.extend_from_slice(&pbeta[idx * HV..(idx + 1) * HV]);
        }
        let dummy_q = vec![0.0f32; 3 * HK * D];
        let dummy_z = vec![0.0f32; 3 * HV * D];
        let (_, committed) = cpu_gdn_step_ref(
            &dummy_q, &gpk, &gpv, &gpg, &gpb, &state0, &dummy_z, &w_f32, 3, HK, HV, true, EPS, true,
        );
        let mut want_y = vec![0.0f32; 8 * HV * D];
        for node in 0..8usize {
            let mut path = Vec::new();
            let mut cur = node as i32;
            while cur >= 0 {
                path.push(cur as usize);
                cur = parent[cur as usize];
            }
            path.reverse();
            let plen = path.len();
            let mut pq_p = Vec::new();
            let mut pk_p = Vec::new();
            let mut pv_p = Vec::new();
            let mut pg_p = Vec::new();
            let mut pb_p = Vec::new();
            let mut pz_p = Vec::new();
            for &ti in &path {
                pq_p.extend_from_slice(&q[ti * HK * D..(ti + 1) * HK * D]);
                pk_p.extend_from_slice(&k[ti * HK * D..(ti + 1) * HK * D]);
                pv_p.extend_from_slice(&v[ti * HV * D..(ti + 1) * HV * D]);
                pg_p.extend_from_slice(&g[ti * HV..(ti + 1) * HV]);
                pb_p.extend_from_slice(&beta[ti * HV..(ti + 1) * HV]);
                pz_p.extend_from_slice(&z[ti * HV * D..(ti + 1) * HV * D]);
            }
            let (py, _) = cpu_gdn_step_ref(
                &pq_p, &pk_p, &pv_p, &pg_p, &pb_p, &committed, &pz_p, &w_f32, plen, HK, HV, true,
                EPS, true,
            );
            let last_off = (plen - 1) * HV * D;
            want_y[node * HV * D..(node + 1) * HV * D]
                .copy_from_slice(&py[last_off..last_off + HV * D]);
        }
        assert_close(
            "gdn_tree(prev_idx=[0,2,5], T=8, sigmoid) y",
            &read_bf16_f32(&got_y),
            &want_y,
            2.5e-2,
            3e-3,
        );
        got_cs.eval();
        assert_close(
            "gdn_tree(prev_idx=[0,2,5], T=8, sigmoid) committed_state",
            &got_cs.to_float32().unwrap(),
            &committed,
            2e-3,
            2e-4,
        );
    }

    // 6D. prefill_chain_gated_with_gate(sigmoid_gate=true) for T=48
    {
        let t = 48usize;
        let (q, k, v, g, beta, z) = gen_window(&mut rng, t);
        let qa = bf16_array(&q, &[1, t as i64, HK as i64, D as i64]);
        let ka = bf16_array(&k, &[1, t as i64, HK as i64, D as i64]);
        let va = bf16_array(&v, &[1, t as i64, HV as i64, D as i64]);
        let ga = MxArray::from_float32(&g, &[1, t as i64, HV as i64]).unwrap();
        let ba = MxArray::from_float32(&beta, &[1, t as i64, HV as i64]).unwrap();
        let za = bf16_array(&z, &[1, t as i64, (HV * D) as i64]);

        let (got_y, got_s) = gdn::prefill_chain_gated_with_gate(
            &qa,
            &ka,
            &va,
            &ga,
            &ba,
            &state0_arr,
            &za,
            &w_arr,
            EPS,
            true,
            true,
        )
        .unwrap()
        .expect("gdn_prefill_chain_gated(sigmoid) admitted");
        let (want_y, want_s) = cpu_gdn_step_ref(
            &q, &k, &v, &g, &beta, &state0, &z, &w_f32, t, HK, HV, true, EPS, true,
        );
        assert_close(
            "gdn_prefill_chain_gated(T=48, sigmoid, f32_w, f32_beta) y",
            &read_bf16_f32(&got_y),
            &want_y,
            2e-2,
            2e-3,
        );
        got_s.eval();
        assert_close(
            "gdn_prefill_chain_gated(T=48, sigmoid) state",
            &got_s.to_float32().unwrap(),
            &want_s,
            2e-3,
            2e-4,
        );
    }
}

#[test]
fn cuda_router_topk_matches_ref() {
    if active_backend() != ActiveBackend::CudaGpu {
        return;
    }
    let mut rng = Rng::new(0x709_C0DE);
    for &(m, e, top_k, fp32) in &[
        (1usize, 512usize, 10usize, false),
        (4, 512, 10, false),
        (16, 512, 10, true),
        (64, 512, 10, false),
        (3, 16, 10, false),
    ] {
        let raw: Vec<f32> = (0..m * e).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let logits = if fp32 {
            MxArray::from_float32(&raw, &[1, m as i64, e as i64]).unwrap()
        } else {
            bf16_array(&raw, &[1, m as i64, e as i64])
        };
        let logits_f32: Vec<f32> = if fp32 {
            raw.clone()
        } else {
            raw.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
        };
        let (got_ids, got_scores) = router_topk_cuda(&logits, top_k)
            .unwrap()
            .expect("router_topk_cuda admitted");
        got_ids.eval();
        got_scores.eval();
        assert_eq!(got_ids.dtype().unwrap(), DType::Uint32);
        assert_eq!(got_scores.dtype().unwrap(), DType::Float32);
        let ids = got_ids.to_uint32().unwrap();
        let scores = got_scores.to_float32().unwrap();
        assert_eq!(ids.len(), m * top_k);
        assert_eq!(scores.len(), m * top_k);

        for row in 0..m {
            let slice = &logits_f32[row * e..(row + 1) * e];
            let rmax = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|&v| (v - rmax).exp()).collect();
            let rsum: f32 = exps.iter().sum();
            let mut pairs: Vec<(usize, f32)> = exps
                .iter()
                .enumerate()
                .map(|(idx, &v)| (idx, v / rsum))
                .collect();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let top = &pairs[..top_k];
            let top_sum: f32 = top.iter().map(|p| p.1).sum();
            for k in 0..top_k {
                assert_eq!(
                    ids[row * top_k + k] as usize,
                    top[k].0,
                    "router_topk id mismatch (m={m}, e={e}, fp32={fp32}) row={row} k={k}"
                );
                let want_s = top[k].1 / top_sum;
                let got_s = scores[row * top_k + k];
                assert!(
                    (got_s - want_s).abs() < 1e-5,
                    "router_topk score mismatch (m={m}, e={e}, fp32={fp32}) row={row} k={k}: got={got_s} want={want_s}"
                );
            }
        }
    }
}

#[test]
fn qwen4_q8_prefill_parity() {
    if !cuda_available() {
        return;
    }
    let mut rng = Rng::new(0x51A8_4B1B);
    // Covers: 128x64 (K=640, N=320 / N=48 / N=4) AND 128x128 (M=512, N=2560, K=2560: 4*20=80 -> 128x64; M=1024, N=1536, K=512: 8*12=96 -> 128x128!)
    let cases: &[(usize, usize, usize)] = &[
        (197, 320, 640),
        (197, 48, 256),
        (197, 4, 256),
        (1024, 1536, 512),
    ];
    for &(m, n, k) in cases {
        let x_vals: Vec<f32> = (0..m * k)
            .map(|_| f32_round_bf16(rng.uniform(-0.5, 0.5)))
            .collect();
        let x_arr = bf16_array(&x_vals, &[1, m as i64, k as i64]);
        let (w_arr, sc_arr, bi_sym_arr, _) = build_affine_bank(&mut rng, 8, 1, n, k, false);

        // Build non-symmetric biases to test general_bias = true too.
        let kg = k / 32;
        let bi_sym_f32: Vec<f32> = bi_sym_arr
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let bi_gen_f32: Vec<f32> = bi_sym_f32
            .iter()
            .map(|&v| v + rng.uniform(-0.02, 0.02))
            .collect();
        let bi_gen_arr = MxArray::from_float32(&bi_gen_f32, &[n as i64, kg as i64])
            .unwrap()
            .astype(DType::Float16)
            .unwrap();

        for (gen_bias, bi_arr) in [(false, &bi_sym_arr), (true, &bi_gen_arr)] {
            let w_bf16 = linear::dequant_q8_bf16_cuda(&w_arr, &sc_arr, bi_arr)
                .unwrap()
                .expect("dequant_q8_bf16_cuda admitted");
            let want_arr = x_arr
                .matmul(&w_bf16.transpose(Some(&[1, 0])).unwrap())
                .unwrap();
            let want = read_bf16_f32(&want_arr);
            let mut modes = vec![0, 7, -1, -2];
            if k % 256 == 0 {
                modes.push(8);
            }
            for gs_shift in modes {
                let got_arr =
                    linear::q8_prefill_cuda(&x_arr, &w_arr, &sc_arr, bi_arr, gs_shift, gen_bias)
                        .unwrap()
                        .unwrap_or_else(|| {
                            panic!("q8_prefill_cuda rejected (m={m}, n={n}, k={k}, gs={gs_shift}, gen={gen_bias})")
                        });
                let got = read_bf16_f32(&got_arr);
                assert_eq!(got.len(), want.len());
                let mut se = 0.0f64;
                let mut sr = 0.0f64;
                for (&g, &w) in got.iter().zip(want.iter()) {
                    let d = (g - w) as f64;
                    se += d * d;
                    sr += (w as f64) * (w as f64);
                }
                let nrmse = (se / sr.max(1e-12)).sqrt();
                let lim = if gs_shift == -2 { 3.5e-3 } else { 1.2e-2 };
                assert!(
                    nrmse <= lim,
                    "q8_prefill_cuda mismatch (m={m}, n={n}, k={k}, gs={gs_shift}, gen={gen_bias}): nrmse={nrmse:.3e} > {lim:.1e}"
                );
            }
        }
    }

    // Odd N=1 must fall back cleanly to None.
    let x_odd = bf16_array(&vec![0.1f32; 32 * 256], &[1, 32, 256]);
    let (w_odd, s_odd, b_odd, _) = build_affine_bank(&mut rng, 8, 1, 1, 256, false);
    assert!(
        linear::q8_prefill_cuda(&x_odd, &w_odd, &s_odd, &b_odd, 0, false)
            .unwrap()
            .is_none(),
        "N=1 must return None cleanly"
    );
}

#[test]
fn qwen4_hc_up_mix_prefill_and_inject_only_parity() {
    if !cuda_available() {
        return;
    }
    let mut rng = Rng::new(0x7C4B_1B00);
    // 1. inject_only_cuda bit-exact parity for M=1, 17, 197
    for &m in &[1usize, 17, 197] {
        let x_vals: Vec<f32> = (0..m * 10240)
            .map(|_| f32_round_bf16(rng.uniform(-1.0, 1.0)))
            .collect();
        let b_vals: Vec<f32> = (0..m * 2560)
            .map(|_| f32_round_bf16(rng.uniform(-1.0, 1.0)))
            .collect();
        let g_vals: Vec<f32> = (0..m * 4)
            .map(|_| f32_round_bf16(rng.uniform(0.0, 2.0)))
            .collect();
        let x = bf16_array(&x_vals, &[1, m as i64, 10240]);
        let branch = bf16_array(&b_vals, &[1, m as i64, 2560]);
        let gate = bf16_array(&g_vals, &[1, m as i64, 4]);
        let got = inject_only_cuda(&x, &branch, &gate)
            .unwrap()
            .expect("inject_only_cuda admitted");
        let want = x
            .add(
                &branch
                    .expand_dims(-2)
                    .unwrap()
                    .mul(&gate.expand_dims(-1).unwrap())
                    .unwrap()
                    .reshape(&[1, m as i64, 10240])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            read_bf16_f32(&got),
            read_bf16_f32(&want),
            "inject_only_cuda must be bit-exact for M={m}"
        );
    }

    // 2. hc_up_mix_prefill_cuda parity vs dequant + matmul + prefill_hc_mix_cuda
    //    Tests both M <= 512 (exact BF16) and M > 512 (v2 INT8 MMA).
    let m = 197usize;
    let da_vals: Vec<f32> = (0..m * 320)
        .map(|_| f32_round_bf16(rng.uniform(-0.5, 0.5)))
        .collect();
    let nm_vals: Vec<f32> = (0..m * 10240)
        .map(|_| f32_round_bf16(rng.uniform(-0.5, 0.5)))
        .collect();
    let da = bf16_array(&da_vals, &[1, m as i64, 320]);
    let nm = bf16_array(&nm_vals, &[1, m as i64, 10240]);
    let (w_up, s_up, b_up, _) = build_affine_bank(&mut rng, 8, 1, 10240, 320, false);

    let w_bf16 = linear::dequant_q8_bf16_cuda(&w_up, &s_up, &b_up)
        .unwrap()
        .expect("dequant admitted");
    let w_bf16_t = w_bf16.transpose(Some(&[1, 0])).unwrap();
    let da_520 = MxArray::concatenate_many(vec![&da, &da, &da], Some(1))
        .unwrap()
        .slice_axis(1, 0, 520)
        .unwrap();
    let nm_520 = MxArray::concatenate_many(vec![&nm, &nm, &nm], Some(1))
        .unwrap()
        .slice_axis(1, 0, 520)
        .unwrap();
    for (m_tag, da_cur, nm_cur) in [("M=197", &da, &nm), ("M=520", &da_520, &nm_520)] {
        let up = da_cur.matmul(&w_bf16_t).unwrap();
        let want_arr = prefill_hc_mix_cuda(&up, nm_cur)
            .unwrap()
            .expect("prefill_hc_mix admitted");
        let got_arr = hc_up_mix_prefill_cuda(da_cur, nm_cur, &w_up, &s_up, &b_up)
            .unwrap()
            .expect("hc_up_mix_prefill admitted");

        let want = read_bf16_f32(&want_arr);
        let got = read_bf16_f32(&got_arr);
        let mut se = 0.0f64;
        let mut sr = 0.0f64;
        for (&g, &w) in got.iter().zip(want.iter()) {
            let d = (g - w) as f64;
            se += d * d;
            sr += (w as f64) * (w as f64);
        }
        let nrmse = (se / sr.max(1e-12)).sqrt();
        assert!(
            nrmse <= 3.5e-3,
            "hc_up_mix_prefill_cuda({m_tag}) mismatch: nrmse={nrmse:.3e}"
        );
    }

    // 3. hc_down_act_cuda prefill (M = 197 > 16) parity across all 5 inj_modes
    let (wd, sd, bd, wd_f) = build_affine_bank(&mut rng, 8, 1, 320, 10240, false);
    let (wi_q8, si_q8, bi_q8, wi_q8_f) = build_affine_bank(&mut rng, 8, 1, 4, 10240, false);
    let wi_dense_vals: Vec<f32> = (0..4 * 10240)
        .map(|_| f32_round_bf16(rng.uniform(-0.15, 0.15)))
        .collect();
    let wi_bf16 = bf16_array(&wi_dense_vals, &[4, 10240]);
    let wi_f32 = MxArray::from_float32(&wi_dense_vals, &[4, 10240]).unwrap();
    let wi_f16 = wi_f32.astype(DType::Float16).unwrap();
    let wi_f16_vals = wi_f16.astype(DType::Float32).unwrap().to_float32().unwrap();

    let mut ref_down = vec![0.0f32; m * 320];
    let mut ref_inj_q8 = vec![0.0f32; m * 4];
    let mut ref_inj_bf = vec![0.0f32; m * 4];
    let mut ref_inj_f16 = vec![0.0f32; m * 4];
    for t in 0..m {
        let row = &nm_vals[t * 10240..(t + 1) * 10240];
        for n in 0..320 {
            let mut acc = 0.0f64;
            let wr = &wd_f[n * 10240..(n + 1) * 10240];
            for c in 0..10240 {
                acc += (row[c] as f64) * (wr[c] as f64);
            }
            let scv = f32_round_bf16(f32_round_bf16(acc as f32) * 0.25);
            ref_down[t * 320 + n] = silu_bf16(scv);
        }
        for s in 0..4 {
            let mut acc_q8 = 0.0f64;
            let mut acc_bf = 0.0f64;
            let mut acc_f16 = 0.0f64;
            let wr_q8 = &wi_q8_f[s * 10240..(s + 1) * 10240];
            let wr_bf = &wi_dense_vals[s * 10240..(s + 1) * 10240];
            let wr_f16 = &wi_f16_vals[s * 10240..(s + 1) * 10240];
            for c in 0..10240 {
                let x = row[c] as f64;
                acc_q8 += x * (wr_q8[c] as f64);
                acc_bf += x * (wr_bf[c] as f64);
                acc_f16 += x * (wr_f16[c] as f64);
            }
            let sc_q8 = f32_round_bf16(f32_round_bf16(acc_q8 as f32) * 0.25);
            ref_inj_q8[t * 4 + s] = f32_round_bf16(f32_round_bf16(sigmoid_f32(sc_q8)) * 2.0);
            let sc_bf = f32_round_bf16(f32_round_bf16(acc_bf as f32) * 0.25);
            ref_inj_bf[t * 4 + s] = f32_round_bf16(f32_round_bf16(sigmoid_f32(sc_bf)) * 2.0);
            let sc_f16 = f32_round_bf16(f32_round_bf16(acc_f16 as f32) * 0.25);
            ref_inj_f16[t * 4 + s] = f32_round_bf16(f32_round_bf16(sigmoid_f32(sc_f16)) * 2.0);
        }
    }

    let (got_d0, got_i0) = hc_down_act_cuda(&nm, &wd, &sd, &bd, Qwen4HcInjectWeight::None)
        .unwrap()
        .expect("hc_down_act prefill None admitted");
    assert!(got_i0.is_none());
    let got_d0_f = read_bf16_f32(&got_d0);
    let mut d_se = 0.0f64;
    let mut d_sr = 0.0f64;
    for (&g, &w) in got_d0_f.iter().zip(ref_down.iter()) {
        let d = (g - w) as f64;
        d_se += d * d;
        d_sr += (w as f64) * (w as f64);
    }
    let d_nrmse = (d_se / d_sr.max(1e-12)).sqrt();
    assert!(
        d_nrmse <= 8e-3,
        "hc_down_act_prefill down mismatch: nrmse={d_nrmse:.3e}"
    );

    for (tag_i, inj_w, want_inj) in [
        (
            "Q8_0Bias",
            Qwen4HcInjectWeight::Q8_0Bias {
                weight: &wi_q8,
                scales: &si_q8,
                biases: &bi_q8,
            },
            &ref_inj_q8,
        ),
        (
            "DenseBf16",
            Qwen4HcInjectWeight::DenseBf16(&wi_bf16),
            &ref_inj_bf,
        ),
        (
            "DenseFp16",
            Qwen4HcInjectWeight::DenseFp16(&wi_f16),
            &ref_inj_f16,
        ),
        (
            "DenseFp32",
            Qwen4HcInjectWeight::DenseFp32(&wi_f32),
            &ref_inj_bf,
        ),
    ] {
        let (got_d, got_i) = hc_down_act_cuda(&nm, &wd, &sd, &bd, inj_w)
            .unwrap()
            .expect("hc_down_act prefill inj admitted");
        assert_eq!(read_bf16_f32(&got_d), got_d0_f, "down changed with {tag_i}");
        assert_close(
            &format!("hc_down_act_prefill({tag_i}) inj"),
            &read_bf16_f32(got_i.as_ref().unwrap()),
            want_inj,
            3e-2,
            3e-3,
        );
    }
}

#[test]
fn qwen4_moe_prefill_v2_hot_cold_parity() {
    if !cuda_available() {
        return;
    }
    const H: usize = 2560;
    const DFF: usize = 640;
    const E: usize = 16;
    const TOP_K: usize = 4;
    const M: usize = 160;
    let mut rng = Rng::new(0x40E2_7788);
    let (wg, sg, bg, wg_f) = build_kquant_bank(&mut rng, 0, E, DFF, H);
    let (wu, su, bu, wu_f) = build_kquant_bank(&mut rng, 0, E, DFF, H);
    let (wd, sd, bd, wd_f) = build_affine_bank(&mut rng, 5, E, H, DFF, true);
    let routed = Qwen4MoeRoutedWeights {
        gate_w: &wg,
        gate_s: &sg,
        gate_b: &bg,
        up_w: &wu,
        up_s: &su,
        up_b: &bu,
        down_w: &wd,
        down_s: &sd,
        down_b: &bd,
        mode_gu: 0,
        bits_dn: 5,
    };
    // Skew routing so experts 0..4 are HOT (>64 tokens, hitting BM=64) and
    // experts 5..15 are COLD (1..25 tokens, hitting BM=32)!
    let x_vals: Vec<f32> = (0..M * H)
        .map(|_| f32_round_bf16(rng.uniform(-0.4, 0.4)))
        .collect();
    let mut ids = Vec::with_capacity(M * TOP_K);
    let mut scores = Vec::with_capacity(M * TOP_K);
    for t in 0..M {
        let hot_a = (t % 4) as i32;
        let hot_b = ((t + 1) % 4) as i32;
        let hot_c = ((t + 2) % 4) as i32;
        let cold = (4 + (t % (E - 4))) as i32;
        ids.extend_from_slice(&[hot_a, hot_b, hot_c, cold]);
        scores.extend_from_slice(&[0.4f32, 0.3, 0.2, 0.1]);
    }
    let x_arr = bf16_array(&x_vals, &[1, M as i64, H as i64]);
    let ids_arr = MxArray::from_int32(&ids, &[1, M as i64, TOP_K as i64]).unwrap();
    let sc_arr = MxArray::from_float32(&scores, &[1, M as i64, TOP_K as i64]).unwrap();
    let got = moe_prefill_cuda(&x_arr, &ids_arr, &sc_arr, &routed, None)
        .unwrap()
        .expect("moe_prefill_v2 hot+cold admitted");
    let want = cpu_moe_ref(&x_vals, &ids, &scores, &wg_f, &wu_f, &wd_f, None, M, TOP_K);
    assert_close(
        "moe_prefill_v2(hot64+cold32)",
        &read_bf16_f32(&got),
        &want,
        0.20,
        0.035,
    );
}

#[test]
fn qwen4_qsa_v2_premerge_and_radix_race_regression() {
    if !cuda_available() {
        return;
    }
    const HQ: usize = 24;
    const HKV: usize = 2;
    const D: usize = 256;
    const ID: usize = 128;
    let scale = (D as f32).powf(-0.5);
    let idx_scale = (ID as f32).powf(-0.5);

    for &(q_len, kv_len, kv_cap, num_blocks, pk_cap) in &[
        (1035usize, 2070usize, 2560usize, 517usize, 640usize),
        (4096usize, 8192usize, 8192usize, 2048usize, 2048usize),
    ] {
        let q = MxArray::full(
            &[1, HQ as i64, q_len as i64, D as i64],
            napi::Either::A(0.1),
            Some(DType::BFloat16),
        )
        .unwrap();
        let k = MxArray::full(
            &[1, HKV as i64, kv_cap as i64, D as i64],
            napi::Either::A(0.1),
            Some(DType::BFloat16),
        )
        .unwrap();
        let v = MxArray::full(
            &[1, HKV as i64, kv_cap as i64, D as i64],
            napi::Either::A(0.1),
            Some(DType::BFloat16),
        )
        .unwrap();
        let pk = MxArray::full(
            &[pk_cap as i64, ID as i64],
            napi::Either::A(0.25),
            Some(DType::BFloat16),
        )
        .unwrap();
        let iq_rand = MxArray::random_normal(
            &[1, 4, q_len as i64, ID as i64],
            0.0,
            1.0,
            Some(DType::BFloat16),
        )
        .unwrap();
        let pk_rand =
            MxArray::random_normal(&[pk_cap as i64, ID as i64], 0.0, 1.0, Some(DType::BFloat16))
                .unwrap();
        let iq_base =
            MxArray::random_normal(&[1, 4, 1, ID as i64], 0.0, 1.0, Some(DType::BFloat16)).unwrap();
        let iq_corr = iq_base
            .add(&iq_rand.mul_scalar(0.25).unwrap())
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let iq_zero =
            MxArray::zeros(&[1, 4, q_len as i64, ID as i64], Some(DType::BFloat16)).unwrap();
        MxArray::eval_arrays(&[&q, &k, &v, &pk, &iq_rand, &pk_rand, &iq_corr, &iq_zero]).unwrap();

        let mut outs = Vec::with_capacity(6);
        for rep in 0..6 {
            let (iq_r, pk_r) = match rep % 3 {
                0 => (&iq_zero, &pk),
                1 => (&iq_rand, &pk_rand),
                _ => (&iq_corr, &pk_rand),
            };
            let o = qsa_sparse_sdpa_cuda(
                &q,
                &k,
                &v,
                iq_r,
                pk_r,
                None,
                kv_len as i32,
                num_blocks as i32,
                scale,
                idx_scale,
                0,
            )
            .unwrap()
            .expect("admitted");
            outs.push(o);
        }
        let refs: Vec<&MxArray> = outs.iter().collect();
        MxArray::eval_arrays(&refs)
            .unwrap_or_else(|e| panic!("qsa v2 regression failed (q={q_len}, kv={kv_len}): {e}"));
    }
}
