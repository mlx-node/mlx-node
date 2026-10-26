//! Store v1: the model straight from the GGUF shards. Dense tensors are read
//! once by aligned `F_NOCACHE` `pread` and kept resident (Q8_0 as MLX affine
//! 8-bit codes + f16 scales, dequantized to f32 at each use; F32 as is);
//! routed experts stream per layer into the shared staging bank; engram rows
//! come from `PreadTable`.

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::{Error, Result};

use super::attention::FakeQuant;
use super::config::DeepseekV41Config;
use super::model::Model;
use super::moe::ExpertBank;
use super::rows::PreadTable;
use crate::array::alloc::OutLayout;
use crate::array::{DType, MxArray};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::moe_stream::bank::{ExpertTensor, StagingBank};
use crate::moe_stream::io::{AlignedBuf, Extent, IoPool, ShardSet, open_nocache};
use crate::utils::gguf::{GgufFile, GgufTensorInfo, GgufTensorType, parse_gguf_reader};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::RepackDst;
use crate::utils::host::host_memory;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("deepseek_v41 store: {msg}"))
}

/// Something `linear` can use as an f32 `[out, in]` weight.
pub trait Weight {
    fn f32(&self) -> Result<MxArray>;
}

impl Weight for MxArray {
    fn f32(&self) -> Result<MxArray> {
        self.astype(DType::Float32)
    }
}

/// A dense weight: an array, or Q8_0 as MLX affine 8-bit (`w` `[rows, k /
/// 4]` uint32 codes q + 128, `scales` `[rows, k / 32]` f16). The f32 form
/// (`scale * code - 128 * scale` = `d * q`, exact) is built at each use
/// and never kept.
pub enum Dense {
    Array(MxArray),
    Q8 { w: MxArray, scales: MxArray },
}

impl From<MxArray> for Dense {
    fn from(a: MxArray) -> Self {
        Self::Array(a)
    }
}

fn dequant_q8(w: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let s = scales.astype(DType::Float32)?;
    let b = s.mul_scalar(-128.0)?;
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            32,
            8,
            -1,
            mode.as_ptr(),
        )
    };
    MxArray::from_handle(h, "deepseek_v41 q8_0 dequantize")
}

impl Weight for Dense {
    fn f32(&self) -> Result<MxArray> {
        match self {
            Self::Array(a) => a.astype(DType::Float32),
            Self::Q8 { w, scales } => dequant_q8(w, scales),
        }
    }
}

impl Dense {
    /// The array of a plain tensor (norms, biases, hc, router).
    pub fn array(self) -> Result<MxArray> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Q8 { .. } => Err(bad("Q8_0 tensor where an f32 array is expected")),
        }
    }

    /// Logical size of `axis`.
    pub fn shape_at(&self, axis: u32) -> Result<i64> {
        match self {
            Self::Array(a) => a.shape_at(axis),
            Self::Q8 { w, scales } if axis + 1 == w.ndim()? => Ok(scales.shape_at(axis)? * 32),
            Self::Q8 { w, .. } => w.shape_at(axis),
        }
    }

    /// Rows `idx` of a 2-D weight: f32 for Q8_0, the array's dtype otherwise.
    pub fn rows(&self, idx: &MxArray) -> Result<MxArray> {
        match self {
            Self::Array(a) => a.take(idx, 0),
            Self::Q8 { w, scales } => dequant_q8(&w.take(idx, 0)?, &scales.take(idx, 0)?),
        }
    }

    /// Rows `lo..hi` of a 2-D weight.
    pub fn row_range(&self, lo: i64, hi: i64) -> Result<Dense> {
        let cut = |a: &MxArray| a.slice(&[lo, 0], &[hi, a.shape_at(1)?]);
        Ok(match self {
            Self::Array(a) => Self::Array(cut(a)?),
            Self::Q8 { w, scales } => Self::Q8 {
                w: cut(w)?,
                scales: cut(scales)?,
            },
        })
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Array(a) => a.nbytes(),
            Self::Q8 { w, scales } => w.nbytes() + scales.nbytes(),
        }
    }
}

/// Where a tensor lives: shard index, that shard's data start, header.
type Located = (usize, u64, GgufTensorInfo);

/// Expert projections in bank order.
const PROJS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
const ENGRAM_TABLE: &str = "per_layer_token_embd.weight";
/// Free host memory left after the load (D-15 reserve).
const RESERVE: u64 = 16 << 30;
/// MLX free-buffer pool ceiling (D-33).
pub const POOL_LIMIT: usize = 1 << 30;

pub struct Store {
    pub model: Model,
    pub bank: Arc<StagingBank>,
    /// GGUF bytes of the dense tensors read at load.
    pub dense_read: u64,
    /// Bytes of the resident dense arrays.
    pub dense_bytes: u64,
    _guard: CacheLimitGuard,
}

fn header(path: &Path) -> Result<GgufFile> {
    parse_gguf_reader(std::io::BufReader::new(open_nocache(path)?))
        .map_err(|e| bad(format!("'{}': {e}", path.display())))
}

impl Store {
    /// `paths` are every shard in order (shard 0 holds the metadata).
    pub fn open(paths: &[PathBuf], act: DType, fq: FakeQuant) -> Result<Self> {
        let mut where_: HashMap<String, Located> = HashMap::new();
        let mut cfg = None;
        for (i, p) in paths.iter().enumerate() {
            let g = header(p)?;
            for t in &g.tensors {
                where_.insert(t.name.clone(), (i, g.data_offset, t.clone()));
            }
            if i == 0 {
                cfg = Some(DeepseekV41Config::from_gguf(&g)?);
            }
        }
        let cfg = cfg.ok_or_else(|| bad("no shards"))?;
        let find = |n: &str| {
            where_
                .get(n)
                .ok_or_else(|| bad(format!("tensor '{n}' missing")))
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let mut v = Vec::with_capacity(PROJS.len());
            for p in PROJS {
                let (shard, data_offset, info) = find(&format!("blk.{l}.{p}.weight"))?.clone();
                v.push(ExpertTensor {
                    shard,
                    data_offset,
                    info,
                });
            }
            layers.push(v);
        }
        let bank_need: u64 = layers[0]
            .iter()
            .map(|t| t.info.data_size())
            .sum::<Result<u64>>()?;
        let dense_need: u64 = where_
            .values()
            .filter(|(_, _, t)| is_dense(&t.name))
            .map(|(_, _, t)| t.data_size())
            .sum::<Result<u64>>()?;
        if let Some(h) = host_memory() {
            let free = h.physical.saturating_sub(h.non_reclaimable());
            if bank_need + dense_need + RESERVE > free {
                return Err(bad(format!(
                    "load needs {} GiB + {} GiB reserve, host has {} GiB free",
                    (bank_need + dense_need) >> 30,
                    RESERVE >> 30,
                    free >> 30
                )));
            }
        }
        let shards = Arc::new(ShardSet::open(paths)?);
        let bank = Arc::new(StagingBank::new(
            IoPool::from_env(Arc::clone(&shards))?,
            layers,
        )?);
        let (mut dense_read, mut dense_bytes) = (0u64, 0u64);
        let mut buf = AlignedBuf::new();
        let get = |n: &str| -> Result<Dense> {
            let (shard, data_offset, info) = find(n)?;
            let e = Extent {
                shard: *shard,
                offset: data_offset + info.offset,
                len: info.data_size()?,
            };
            let raw = shards.read_aligned(e, &mut buf)?;
            dense_read += e.len;
            let d = match info.tensor_type {
                GgufTensorType::Q8_0 => {
                    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
                    decode_affine_band_into(info.tensor_type, raw, 0, ptrs.rows, ptrs.k, &ptrs)?;
                    Dense::Q8 {
                        w: dst.weight,
                        scales: dst
                            .scales
                            .ok_or_else(|| bad("Q8_0 repack gave no scales"))?,
                    }
                }
                GgufTensorType::F32 => {
                    let v: Vec<f32> = raw
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&b| f32::from_le_bytes(b))
                        .collect();
                    Dense::Array(MxArray::from_float32(&v, &info.mlx_shape())?)
                }
                t => return Err(bad(format!("{n}: dense type {} not handled", t.name()))),
            };
            dense_bytes += d.nbytes() as u64;
            Ok(d)
        };
        let bank_of = |n: &str| -> Result<ExpertBank> {
            let p = PROJS
                .iter()
                .position(|p| n.ends_with(&format!(".{p}.weight")))
                .ok_or_else(|| bad(format!("{n} is not an expert tensor")))?;
            let (weight, scales) = bank.arrays(p)?;
            Ok(ExpertBank { weight, scales })
        };
        let table = match &cfg.engram {
            Some(ec) => {
                let (shard, data_offset, info) = find(ENGRAM_TABLE)?;
                let t = PreadTable::open(&paths[*shard], *data_offset, info, ec)?;
                Some(Box::new(t) as Box<dyn super::engram::EngramRows>)
            }
            None => None,
        };
        let model = Model::from_gguf(cfg, act, fq, get, bank_of, table, Some(Arc::clone(&bank)))?;
        let guard =
            coordinator().register_with_cache_limit(dense_bytes + bank.nbytes(), Some(POOL_LIMIT));
        Ok(Self {
            model,
            bank,
            dense_read,
            dense_bytes,
            _guard: guard,
        })
    }
}

/// Read at load: everything but routed experts, the engram table and the
/// vision-only router bias.
fn is_dense(name: &str) -> bool {
    name != ENGRAM_TABLE
        && !name.ends_with("exp_probs_b_vl.bias")
        && !PROJS
            .iter()
            .any(|p| name.ends_with(&format!(".{p}.weight")))
}

#[cfg(test)]
mod tests;
