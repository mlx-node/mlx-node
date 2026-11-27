//! The MTP layer `blk.{n_layers}` (all BF16 + F32 in the file), read once at
//! load and held in a smaller form:
//!
//! | file | held as | bytes of n values |
//! |---|---|---|
//! | BF16 routed experts `[E, rows, k]` | affine-4 g64, bf16 scales and biases, 32 experts per batch | n/2 + n/16 |
//! | BF16 indexer compressor matrices | f32 array | 4n |
//! | other BF16 | symmetric Q8_0 g32, f16 scales (`Dense::SymQ8`) | n + n/16 |
//! | F32 | f32 array | 4n |
//!
//! A drafted token is verified by the trunk, so the held precision moves
//! the acceptance rate, never the output.

use napi::{Error, Result};

use super::moe::{AffineExperts, AffineProj};
use super::scan::{ExpertClass, Located, Role, Scan};
use super::weights::{
    AttnWeights, CUDA_CLEAR_CACHE_BYTES, CUDA_REPACK_BAND_BYTES, CudaStaging, Dense, Ffn, Form,
    IndexerWeights, LayerWeights, MoeWeights, NextnWeights, Reader, Residual,
    cuda_repack_from_shards, dense_from_raw, held_shape, quantize_affine4_from_bf16_raw,
    quantize_sym_q8_from_bf16_raw,
};
use crate::array::alloc::OutLayout;
use crate::array::{DType, MxArray};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::{
    CudaDeviceBudget, GpuRepackKind, RepackDst, gpu_repack_available, gpu_repack_supports_kind,
};

/// Experts requantized per batch.
pub const EXPERT_BATCH: usize = 32;
pub const EXPERT_GROUP: i32 = 64;
pub const EXPERT_BITS: i32 = 4;
pub const DENSE_GROUP: i32 = 32;
pub const DENSE_BITS: i32 = 8;
/// MTP non-expert BF16 dense weight format override (`symq8` default, `q4` opt-in).
pub const MTP_DENSE_ENV: &str = "MLX_HY4_MTP_DENSE";
const COMPRESSOR_MATS: [&str; 2] = [
    ".indexer_compressor_value.weight",
    ".indexer_compressor_gate.weight",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpDenseMode {
    Q4,
    SymQ8,
}

#[cfg(test)]
thread_local! {
    static TEST_MTP_DENSE: std::cell::Cell<Option<MtpDenseMode>> = const { std::cell::Cell::new(None) };
}

pub fn parse_mtp_dense_mode(raw: Option<&str>) -> Result<MtpDenseMode> {
    let Some(v) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(MtpDenseMode::SymQ8);
    };
    if v.eq_ignore_ascii_case("q4") {
        Ok(MtpDenseMode::Q4)
    } else if v.eq_ignore_ascii_case("symq8") {
        Ok(MtpDenseMode::SymQ8)
    } else {
        Err(bad(format!(
            "{MTP_DENSE_ENV}={v:?}: expected 'q4' or 'symq8'"
        )))
    }
}

pub fn mtp_dense_mode() -> Result<MtpDenseMode> {
    #[cfg(test)]
    if let Some(m) = TEST_MTP_DENSE.with(std::cell::Cell::get) {
        return Ok(m);
    }
    parse_mtp_dense_mode(std::env::var(MTP_DENSE_ENV).ok().as_deref())
}

#[cfg(test)]
pub(crate) fn with_mtp_dense<T>(mode: MtpDenseMode, f: impl FnOnce() -> T) -> T {
    let prev = TEST_MTP_DENSE.with(|c| c.replace(Some(mode)));
    struct Reset(Option<MtpDenseMode>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_MTP_DENSE.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(prev);
    f()
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 MTP weights: {msg}"))
}

/// What one MTP tensor is held as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    /// Symmetric Q8_0 g32 codes, f16 scales (derived `-128 * scale` bias).
    SymQ8,
    /// An f32 array.
    F32,
    /// Affine-4 g64 codes, bf16 scales and biases.
    Affine4,
    /// Affine-8 g32 codes, f16 scales and derived `-128 * scale` biases (Q8_0 MTP experts).
    Affine8,
}

fn is_compressor_mat(name: &str) -> bool {
    COMPRESSOR_MATS.iter().any(|s| name.ends_with(s))
}

/// The held form of an MTP tensor; `expert` for a routed expert tensor.
pub fn held(info: &GgufTensorInfo, expert: bool) -> Result<Held> {
    use GgufTensorType::{BF16, F32, Q8_0};
    Ok(match (info.tensor_type, expert) {
        (BF16, true) => Held::Affine4,
        (Q8_0, true) => Held::Affine8,
        (BF16, false) if is_compressor_mat(&info.name) => Held::F32,
        (BF16, false) => match mtp_dense_mode()? {
            MtpDenseMode::Q4
                if held_shape(info)?
                    .last()
                    .is_some_and(|&k| k > 0 && k % i64::from(EXPERT_GROUP) == 0) =>
            {
                Held::Affine4
            }
            _ => Held::SymQ8,
        },
        (Q8_0, false) if is_compressor_mat(&info.name) => Held::F32,
        (Q8_0, false) => Held::SymQ8,
        (F32, false) => Held::F32,
        (t, _) => {
            return Err(bad(format!(
                "'{}': type {} is not supported for an MTP tensor{}",
                info.name,
                t.name(),
                if expert { " (expert)" } else { "" }
            )));
        }
    })
}

/// Bytes of the arrays this tensor is held as.
pub fn held_bytes(info: &GgufTensorInfo, expert: bool) -> Result<u64> {
    let n = info.num_elements()?;
    let b = match held(info, expert)? {
        Held::SymQ8 => n.checked_add(n / 16),
        Held::F32 => n.checked_mul(4),
        Held::Affine4 => (n / 2).checked_add(n / 16),
        Held::Affine8 => n.checked_add(n / 8),
    };
    b.ok_or_else(|| bad(format!("'{}': held size overflows u64", info.name)))
}

/// The MTP layer as held, before the modules are built from it.
pub struct MtpWeights {
    pub layer: usize,
    pub weights: LayerWeights,
    pub experts: AffineExperts,
    /// GGUF bytes read.
    pub read: u64,
    /// Bytes of the held arrays.
    pub bytes: u64,
    pub guard: CacheLimitGuard,
}

struct Loader<'a> {
    scan: &'a Scan,
    shards: &'a ShardSet,
    il: usize,
    buf: AlignedBuf,
    cuda_budget: Option<CudaDeviceBudget>,
    cuda_staging: CudaStaging,
    cuda_bytes_since_clear: u64,
    read: u64,
    bytes: u64,
    count: usize,
}

impl Loader<'_> {
    fn maybe_clear_cuda_cache(&mut self, added_bytes: u64) {
        if self.cuda_budget.is_some() {
            self.cuda_bytes_since_clear += added_bytes;
            if self.cuda_bytes_since_clear >= CUDA_CLEAR_CACHE_BYTES {
                self.cuda_bytes_since_clear = 0;
                if crate::array::get_cache_memory() >= CUDA_CLEAR_CACHE_BYTES as f64 {
                    crate::array::clear_cache();
                }
            }
        }
    }

    fn located(&self, name: &str) -> Result<&Located> {
        let e = self
            .scan
            .tensors
            .get(name)
            .ok_or_else(|| bad(format!("tensor '{name}' missing")))?;
        if e.role != Role::Mtp {
            return Err(bad(format!("'{name}' is a {:?} tensor", e.role)));
        }
        Ok(&e.loc)
    }

    fn tensor(&mut self, name: &str) -> Result<Dense> {
        let loc = self.located(name)?.clone();
        let info = &loc.info;
        let held_form = held(info, false)?;
        let (mut d, read_len) = if held_form == Held::SymQ8
            && info.tensor_type == GgufTensorType::Q8_0
            && let Some(budget) = &self.cuda_budget
            && gpu_repack_available()
            && gpu_repack_supports_kind(GpuRepackKind::Q8_0)
        {
            let (dst, rows, read_len) = cuda_repack_from_shards(
                self.shards,
                &loc,
                GpuRepackKind::Q8_0,
                budget,
                &mut self.buf,
                &mut self.cuda_staging,
            )?;
            let scales = dst
                .scales
                .ok_or_else(|| bad(format!("'{name}': Q8_0 CUDA repack gave no scales")))?;
            let rows_i64 = i64::try_from(rows).map_err(|_| bad("row count exceeds i64"))?;
            let fold = |a: &MxArray| -> Result<MxArray> {
                let nd = a.ndim()?;
                a.reshape(&[rows_i64, a.shape_at(nd.saturating_sub(1))?])
            };
            let (w, scales) = (fold(&dst.weight)?, fold(&scales)?);
            MxArray::eval_arrays_with_context(&[&w, &scales], "hy_v4_flash MTP cuda q8 fold")?;
            (Dense::SymQ8 { w, scales }, read_len)
        } else {
            let e = Extent {
                shard: loc.shard,
                offset: loc.abs_offset()?,
                len: info.data_size()?,
            };
            let raw = self.shards.read_aligned(e, &mut self.buf)?;
            let d = match held_form {
                Held::Affine4 => {
                    let shape = held_shape(info)?;
                    let &[rows, k] = shape.as_slice() else {
                        return Err(bad(format!(
                            "'{}': Affine4 requires 2-D held shape, got {shape:?}",
                            info.name
                        )));
                    };
                    quantize_affine4_from_bf16_raw(raw, rows as usize, k as usize)?
                }
                Held::SymQ8 if info.tensor_type == GgufTensorType::Q8_0 => {
                    dense_from_raw(info, raw, Form::Qmm)?
                }
                Held::SymQ8 => {
                    let shape = held_shape(info)?;
                    let &[rows, k] = shape.as_slice() else {
                        return Err(bad(format!(
                            "'{}': SymQ8 requires 2-D held shape, got {shape:?}",
                            info.name
                        )));
                    };
                    quantize_sym_q8_from_bf16_raw(raw, rows as usize, k as usize)?
                }
                Held::F32 => dense_from_raw(info, raw, Form::F32)?,
                Held::Affine8 => {
                    return Err(bad(format!(
                        "'{}': Affine8 is only supported for MTP routed experts",
                        info.name
                    )));
                }
            };
            (d, e.len)
        };
        if self.cuda_budget.is_some() {
            d.ensure_cuda_device()?;
        }
        let nbytes = d.nbytes() as u64;
        self.maybe_clear_cuda_cache(nbytes);
        self.read += read_len;
        self.bytes += nbytes;
        self.count += 1;
        Ok(d)
    }

    fn name(&self, s: &str) -> String {
        format!("blk.{}.{s}.weight", self.il)
    }

    fn dense(&mut self, s: &str) -> Result<Dense> {
        let n = self.name(s);
        self.tensor(&n)
    }

    fn plain(&mut self, s: &str) -> Result<MxArray> {
        self.dense(s)?.array()
    }

    fn attn(&mut self) -> Result<AttnWeights> {
        let mut w = AttnWeights {
            norm: self.plain("attn_norm")?,
            sinks: self.plain("attn_sinks")?,
            q_a: self.dense("attn_q_a")?,
            q_a_norm: self.plain("attn_q_a_norm")?,
            q_b: self.dense("attn_q_b")?,
            kv_a: self.dense("attn_kv_a_mqa")?,
            kv_a_norm: self.plain("attn_kv_a_norm")?,
            k_b: self.dense("attn_k_b")?,
            v_b: self.dense("attn_v_b")?,
            output: self.dense("attn_output")?,
            gate: self.dense("attn_gate")?,
            post_norm: self.plain("post_attention_norm")?,
            q_kv: None,
            q_kv_gate: None,
        };
        w.fuse()?;
        Ok(w)
    }

    fn indexer(&mut self) -> Result<IndexerWeights> {
        Ok(IndexerWeights {
            attn_k: self.dense("indexer.attn_k")?,
            k_norm: self.plain("indexer.k_norm")?,
            q_b: self.dense("indexer.attn_q_b")?,
            proj: self.dense("indexer.proj")?,
            compressor_value: self.plain("indexer_compressor_value")?,
            compressor_gate: self.plain("indexer_compressor_gate")?,
            compressor_ape: self.plain("indexer_compressor_ape")?,
            compressor_norm: self.plain("indexer_compressor_norm")?,
        })
    }

    fn nextn(&mut self) -> Result<NextnWeights> {
        Ok(NextnWeights {
            enorm: self.plain("nextn.enorm")?,
            hnorm: self.plain("nextn.hnorm")?,
            eh_proj: self.dense("nextn.eh_proj")?,
            shared_head_norm: self.plain("nextn.shared_head_norm")?,
        })
    }

    fn moe(&mut self) -> Result<MoeWeights> {
        let bias = format!("blk.{}.exp_probs_b.bias", self.il);
        let router_bias = if self.scan.opt(&bias).is_some() {
            Some(self.tensor(&bias)?.array()?)
        } else {
            None
        };
        let mut w = MoeWeights {
            router: self.dense("ffn_gate_inp")?,
            router_bias,
            routed_down: self.dense("ffn_routed_down")?,
            routed_up: self.dense("ffn_routed_up")?,
            shared_gate: self.dense("ffn_gate_shexp")?,
            shared_up: self.dense("ffn_up_shexp")?,
            shared_down: self.dense("ffn_down_shexp")?,
            down_shared_gu: None,
        };
        w.fuse()?;
        Ok(w)
    }

    fn layer(&mut self) -> Result<LayerWeights> {
        let attn = self.attn()?;
        let indexer = Some(self.indexer()?);
        let residual = Residual::Mtp(self.nextn()?);
        let ffn = Ffn::Moe(self.moe()?);
        let w = LayerWeights {
            attn,
            indexer,
            engram: None,
            residual,
            ffn_norm: self.plain("ffn_norm")?,
            post_ffw_norm: self.plain("post_ffw_norm")?,
            ffn,
        };
        let want = self.scan.with_role(Role::Mtp).count();
        if self.count != want {
            return Err(bad(format!(
                "loaded {} of the {want} MTP tensors",
                self.count
            )));
        }
        Ok(w)
    }

    /// One routed expert tensor `[E, rows, k]` as affine-4 g64 codes,
    /// requantized `EXPERT_BATCH` experts at a time into arrays made once.
    fn experts(&mut self, loc: &Located) -> Result<AffineProj> {
        let info = &loc.info;
        if held(info, true)? != Held::Affine4 {
            return Err(bad(format!("'{}' is not a BF16 expert tensor", info.name)));
        }
        let &[k, rows, n] = info.dims.as_slice() else {
            return Err(bad(format!("'{}' is not a 3-D expert tensor", info.name)));
        };
        let g = EXPERT_GROUP as u64;
        if !k.is_multiple_of(g) || !(k * EXPERT_BITS as u64).is_multiple_of(32) {
            return Err(bad(format!(
                "'{}': {k} inputs do not fit group {g}",
                info.name
            )));
        }
        let dim = |x: u64| i64::try_from(x).map_err(|_| bad("expert dim exceeds i64"));
        let (n_i, rows_i) = (dim(n)?, dim(rows)?);
        let mut w = MxArray::zeros(
            &[n_i, rows_i, dim(k * EXPERT_BITS as u64 / 32)?],
            Some(DType::Uint32),
        )?;
        let mut s = MxArray::zeros(&[n_i, rows_i, dim(k / g)?], Some(DType::BFloat16))?;
        let mut b = MxArray::zeros(&[n_i, rows_i, dim(k / g)?], Some(DType::BFloat16))?;
        MxArray::eval_arrays_with_context(&[&w, &s, &b], "hy_v4_flash MTP expert alloc")?;
        let mut reader = Reader::new(self.scan, self.shards);
        let total = n as usize;
        let mut lo = 0;
        while lo < total {
            let hi = (lo + EXPERT_BATCH).min(total);
            let values = reader.bf16_experts(loc, lo, hi)?;
            let p = AffineProj::quantize(&values, EXPERT_GROUP, EXPERT_BITS)?;
            let (a, z) = (lo as i64, hi as i64);
            w.slice_assign_axis_inplace(0, a, z, &p.w)?;
            s.slice_assign_axis_inplace(0, a, z, &p.s)?;
            b.slice_assign_axis_inplace(0, a, z, &p.b)?;
            MxArray::eval_arrays_with_context(&[&w, &s, &b], "hy_v4_flash MTP expert requant")?;
            lo = hi;
        }
        self.read += reader.read;
        if self.cuda_budget.is_some() {
            w.promote_cuda_device()?;
            s.promote_cuda_device()?;
            b.promote_cuda_device()?;
        }
        let out = AffineProj { w, s, b };
        let nbytes = out.nbytes() as u64;
        self.maybe_clear_cuda_cache(nbytes);
        self.bytes += nbytes;
        Ok(out)
    }

    /// One Q8_0 routed expert tensor `[E, rows, k]` as affine-8 g32 codes +
    /// f16 scales + derived `-128 * scale` f16 biases. On CUDA, repacks in
    /// bands directly into GPU VRAM (`device == 0`) via `GpuRepackKind::Q8_0Bias`.
    fn experts_q8_0(&mut self, loc: &Located) -> Result<AffineProj> {
        let info = &loc.info;
        if held(info, true)? != Held::Affine8 {
            return Err(bad(format!("'{}' is not a Q8_0 expert tensor", info.name)));
        }
        let &[k, _rows, _n] = info.dims.as_slice() else {
            return Err(bad(format!("'{}' is not a 3-D expert tensor", info.name)));
        };
        if !k.is_multiple_of(DENSE_GROUP as u64) {
            return Err(bad(format!(
                "'{}': {k} inputs do not fit group {DENSE_GROUP}",
                info.name
            )));
        }
        let (mut w, mut s, mut b, read_len) = if let Some(budget) = &self.cuda_budget
            && gpu_repack_available()
            && gpu_repack_supports_kind(GpuRepackKind::Q8_0Bias)
        {
            let (dst, _, read_len) = cuda_repack_from_shards(
                self.shards,
                loc,
                GpuRepackKind::Q8_0Bias,
                budget,
                &mut self.buf,
                &mut self.cuda_staging,
            )?;
            let s = dst
                .scales
                .ok_or_else(|| bad(format!("'{}': Q8_0Bias repack gave no scales", info.name)))?;
            let b = dst
                .biases
                .ok_or_else(|| bad(format!("'{}': Q8_0Bias repack gave no biases", info.name)))?;
            (dst.weight, s, b, read_len)
        } else {
            let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
            let ty = info.tensor_type;
            let row_bytes = (ptrs.k / ty.block_size()) * ty.type_size();
            let band_rows = (CUDA_REPACK_BAND_BYTES / row_bytes).clamp(1, ptrs.rows);
            let base_offset = loc.abs_offset()?;
            let mut row0 = 0usize;
            let mut read_len = 0u64;
            while row0 < ptrs.rows {
                let count = band_rows.min(ptrs.rows - row0);
                let band_len = (count as u64) * (row_bytes as u64);
                let raw = self.shards.read_aligned(
                    Extent {
                        shard: loc.shard,
                        offset: base_offset + (row0 as u64) * (row_bytes as u64),
                        len: band_len,
                    },
                    &mut self.buf,
                )?;
                decode_affine_band_into(ty, raw, row0, count, ptrs.k, &ptrs)?;
                read_len += band_len;
                row0 += count;
            }
            let s = dst
                .scales
                .ok_or_else(|| bad(format!("'{}': Q8_0 repack gave no scales", info.name)))?;
            let b = s
                .astype(DType::Float32)?
                .mul_scalar(-128.0)?
                .astype(DType::Float16)?;
            MxArray::eval_arrays_with_context(
                &[&dst.weight, &s, &b],
                "hy_v4_flash MTP q8 experts",
            )?;
            (dst.weight, s, b, read_len)
        };
        if self.cuda_budget.is_some() {
            w.promote_cuda_device()?;
            s.promote_cuda_device()?;
            b.promote_cuda_device()?;
        }
        self.read += read_len;
        let out = AffineProj { w, s, b };
        let nbytes = out.nbytes() as u64;
        self.maybe_clear_cuda_cache(nbytes);
        self.bytes += nbytes;
        Ok(out)
    }
}

/// Reads and requantizes the MTP layer of `scan`. Model thread only.
pub fn load(scan: &Scan, shards: &ShardSet) -> Result<MtpWeights> {
    let out = read(scan, shards);
    crate::array::clear_cache();
    out
}

fn read(scan: &Scan, shards: &ShardSet) -> Result<MtpWeights> {
    let mtp = scan
        .mtp
        .as_ref()
        .ok_or_else(|| bad("the file has no MTP layer"))?;
    let ex = scan
        .mtp_experts()
        .ok_or_else(|| bad("the MTP layer has no routed experts"))?;
    if !matches!(ex.class, ExpertClass::Bf16 | ExpertClass::Q8_0) {
        return Err(bad(format!(
            "MTP experts are {:?}, expected BF16 or Q8_0",
            ex.class
        )));
    }
    let t0 = std::time::Instant::now();
    let cuda_budget =
        crate::accelerator::cuda_ops_available().then(CudaDeviceBudget::for_current_device);
    let mut l = Loader {
        scan,
        shards,
        il: mtp.layer,
        buf: AlignedBuf::new(),
        cuda_budget,
        cuda_staging: CudaStaging::new(),
        cuda_bytes_since_clear: 0,
        read: 0,
        bytes: 0,
        count: 0,
    };
    let weights = l.layer()?;
    let (projs, group_size, bits) = match ex.class {
        ExpertClass::Bf16 => (
            [
                l.experts(&ex.gate)?,
                l.experts(&ex.up)?,
                l.experts(&ex.down)?,
            ],
            EXPERT_GROUP,
            EXPERT_BITS,
        ),
        ExpertClass::Q8_0 => (
            [
                l.experts_q8_0(&ex.gate)?,
                l.experts_q8_0(&ex.up)?,
                l.experts_q8_0(&ex.down)?,
            ],
            DENSE_GROUP,
            DENSE_BITS,
        ),
        _ => unreachable!(),
    };
    let experts = AffineExperts::new(projs, group_size, bits)?;
    tracing::info!(
        "hy_v4_flash MTP layer {}: {} GGUF bytes read, {} bytes held, {:.1} s",
        mtp.layer,
        l.read,
        l.bytes,
        t0.elapsed().as_secs_f64()
    );
    Ok(MtpWeights {
        layer: mtp.layer,
        weights,
        experts,
        read: l.read,
        bytes: l.bytes,
        guard: coordinator().register(l.bytes),
    })
}
