//! The MTP layer `blk.{n_layers}` (all BF16 + F32 in the file), read once at
//! load and held in a smaller form:
//!
//! | file | held as | bytes of n values |
//! |---|---|---|
//! | BF16 routed experts `[E, rows, k]` | affine-4 g64, bf16 scales and biases, 32 experts per batch | n/2 + n/16 |
//! | BF16 indexer compressor matrices | f32 array | 4n |
//! | other BF16 | affine-8 g32, f32 scales and biases (`Dense::Qmm`) | n + n/4 |
//! | F32 | f32 array | 4n |
//!
//! A drafted token is verified by the trunk, so the held precision moves
//! the acceptance rate, never the output.

use napi::{Error, Result};

use super::moe::{AffineExperts, AffineProj};
use super::scan::{ExpertClass, Located, Role, Scan};
use super::weights::{
    AttnWeights, Dense, Ffn, Form, IndexerWeights, LayerWeights, MoeWeights, NextnWeights, Reader,
    Residual, dense_from_raw,
};
use crate::array::{DType, MxArray};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

/// Experts requantized per batch.
pub const EXPERT_BATCH: usize = 32;
pub const EXPERT_GROUP: i32 = 64;
pub const EXPERT_BITS: i32 = 4;
pub const DENSE_GROUP: i32 = 32;
pub const DENSE_BITS: i32 = 8;
const COMPRESSOR_MATS: [&str; 2] = [
    ".indexer_compressor_value.weight",
    ".indexer_compressor_gate.weight",
];

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 MTP weights: {msg}"))
}

/// What one MTP tensor is held as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    /// Affine-8 g32 codes, f32 scales and biases.
    Qmm,
    /// An f32 array.
    F32,
    /// Affine-4 g64 codes, bf16 scales and biases (routed experts).
    Affine4,
}

fn is_compressor_mat(name: &str) -> bool {
    COMPRESSOR_MATS.iter().any(|s| name.ends_with(s))
}

/// The held form of an MTP tensor; `expert` for a routed expert tensor.
pub fn held(info: &GgufTensorInfo, expert: bool) -> Result<Held> {
    use GgufTensorType::{BF16, F32};
    Ok(match (info.tensor_type, expert) {
        (BF16, true) => Held::Affine4,
        (BF16, false) if is_compressor_mat(&info.name) => Held::F32,
        (BF16, false) => Held::Qmm,
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
        Held::Qmm => n.checked_add(n / 4),
        Held::F32 => n.checked_mul(4),
        Held::Affine4 => (n / 2).checked_add(n / 16),
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
    read: u64,
    bytes: u64,
    count: usize,
}

impl Loader<'_> {
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
        let e = Extent {
            shard: loc.shard,
            offset: loc.abs_offset()?,
            len: info.data_size()?,
        };
        let raw = self.shards.read_aligned(e, &mut self.buf)?;
        let values = dense_from_raw(info, raw, Form::F32)?;
        let d = match held(info, false)? {
            Held::Qmm => {
                let p = AffineProj::quantize(&values.array()?, DENSE_GROUP, DENSE_BITS)?;
                MxArray::eval_arrays_with_context(&[&p.w, &p.s, &p.b], "hy_v4_flash MTP requant")?;
                Dense::Qmm {
                    w: p.w,
                    scales: p.s,
                    biases: p.b,
                }
            }
            _ => values,
        };
        self.read += e.len;
        self.bytes += d.nbytes() as u64;
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
        let out = AffineProj { w, s, b };
        self.bytes += out.nbytes() as u64;
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
    if ex.class != ExpertClass::Bf16 {
        return Err(bad(format!("MTP experts are {:?}, not BF16", ex.class)));
    }
    let t0 = std::time::Instant::now();
    let mut l = Loader {
        scan,
        shards,
        il: mtp.layer,
        buf: AlignedBuf::new(),
        read: 0,
        bytes: 0,
        count: 0,
    };
    let weights = l.layer()?;
    let projs = [
        l.experts(&ex.gate)?,
        l.experts(&ex.up)?,
        l.experts(&ex.down)?,
    ];
    let experts = AffineExperts::new(projs, EXPERT_GROUP, EXPERT_BITS)?;
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
