//! Store v1: the model straight from the GGUF shards. Dense tensors are read
//! once by aligned `F_NOCACHE` `pread` and kept resident (Q8_0 as MLX affine
//! 8-bit codes multiplied by `quantized_matmul`; F32 as is);
//! routed experts stream per layer into the shared bank (staging + global
//! cache, sized by `CachePlan`); engram rows come from `PreadTable`.
//!
//! A turn is `begin_turn` .. `end_turn`: with `CachePlan::Admit` the start
//! re-admits and may drop the bank to staging-only or rebuild it, then
//! holds a wired-limit lease of fixed + cache bytes until the end, when the
//! bank becomes volatile.

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

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
use crate::moe_stream::budget::{
    Admission, Limits, ModelSize, Own, Turn, admit, readmit, working_set,
};
use crate::moe_stream::io::{AlignedBuf, DEFAULT_THREADS, Extent, IoPool, ShardSet, open_nocache};
use crate::stream::WiredLimitContext;
use crate::utils::gguf::{GgufFile, GgufTensorInfo, GgufTensorType, parse_gguf_reader};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::RepackDst;
use crate::utils::host::{HostMemory, host_memory};

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("deepseek_v41 store: {msg}"))
}

/// Something `linear` can use as an f32 `[out, in]` weight.
pub trait Weight {
    fn f32(&self) -> Result<MxArray>;

    /// `x @ W^T` for an f32 `x`.
    fn mm_t(&self, x: &MxArray) -> Result<MxArray> {
        x.matmul(&self.f32()?.transpose(Some(&[1, 0]))?)
    }
}

impl Weight for MxArray {
    fn f32(&self) -> Result<MxArray> {
        self.astype(DType::Float32)
    }
}

/// A dense weight: an array, or Q8_0 as MLX affine 8-bit (`w` `[rows, k /
/// 4]` uint32 codes q + 128, `scales` `[rows, k / 32]` f16). The f32 form
/// (`scale * code - 128 * scale` = `d * q`, exact) is built at each use
/// and never kept. `Qmm` holds the same codes with f32 scales and biases
/// (`-128 * scale`, exact) and multiplies by `quantized_matmul` (other
/// rounding than the f32 path); its f32 form is the exact one.
pub enum Dense {
    Array(MxArray),
    Q8 {
        w: MxArray,
        scales: MxArray,
    },
    Qmm {
        w: MxArray,
        scales: MxArray,
        biases: MxArray,
    },
}

/// How Q8_0 matrices are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseMode {
    /// `quantized_matmul` on the codes (embedding: codes, rows only).
    Q8,
    /// Codes + f16 scales, f32 built at each use: the reference's f32 math,
    /// for the bitwise tests.
    #[cfg(test)]
    Exact,
}

/// What one Q8_0 tensor becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    Codes,
    Qmm,
}

impl DenseMode {
    /// Hash-file suffix of runs in this mode ("" for the exact one).
    pub fn tag(self) -> &'static str {
        match self {
            Self::Q8 => ".q8",
            #[cfg(test)]
            Self::Exact => "",
        }
    }

    fn form(self, name: &str) -> Form {
        match self {
            _ if name == "token_embd.weight" => Form::Codes,
            Self::Q8 => Form::Qmm,
            #[cfg(test)]
            Self::Exact => Form::Codes,
        }
    }

    /// Resident bytes of one dense tensor in this mode.
    fn resident(self, t: &GgufTensorInfo) -> Result<u64> {
        if t.tensor_type != GgufTensorType::Q8_0 {
            return t.data_size();
        }
        let n = t.num_elements()?;
        let r = match self.form(&t.name) {
            Form::Codes => n.checked_add(n / 16),
            Form::Qmm => n.checked_add(n / 4),
        };
        r.ok_or_else(|| bad(format!("{}: resident size overflows u64", t.name)))
    }
}

impl From<MxArray> for Dense {
    fn from(a: MxArray) -> Self {
        Self::Array(a)
    }
}

fn dequant_q8(w: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let s = scales.astype(DType::Float32)?;
    let b = s.mul_scalar(-128.0)?;
    dequant_affine8(w, &s, &b)
}

fn dequant_affine8(w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
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

/// `x @ W^T` on affine 8-bit codes (group 32); `W` may carry batch axes.
fn qmm_t(x: &MxArray, w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            true,
            32,
            8,
            mode.as_ptr(),
        )
    };
    MxArray::from_handle(h, "deepseek_v41 q8_0 quantized_matmul")
}

impl Weight for Dense {
    fn f32(&self) -> Result<MxArray> {
        match self {
            Self::Array(a) => a.astype(DType::Float32),
            Self::Q8 { w, scales } => dequant_q8(w, scales),
            Self::Qmm { w, scales, biases } => dequant_affine8(w, scales, biases),
        }
    }

    fn mm_t(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Self::Qmm { w, scales, biases } => qmm_t(x, w, scales, biases),
            _ => x.matmul(&self.f32()?.transpose(Some(&[1, 0]))?),
        }
    }
}

impl Dense {
    /// The array of a plain tensor (norms, biases, hc, router).
    pub fn array(self) -> Result<MxArray> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Q8 { .. } | Self::Qmm { .. } => {
                Err(bad("Q8_0 tensor where an f32 array is expected"))
            }
        }
    }

    /// Logical size of `axis`.
    pub fn shape_at(&self, axis: u32) -> Result<i64> {
        match self {
            Self::Array(a) => a.shape_at(axis),
            Self::Q8 { w, scales } | Self::Qmm { w, scales, .. } if axis + 1 == w.ndim()? => {
                Ok(scales.shape_at(axis)? * 32)
            }
            Self::Q8 { w, .. } | Self::Qmm { w, .. } => w.shape_at(axis),
        }
    }

    /// Rows `idx` of a 2-D weight: f32 for Q8_0, the array's dtype otherwise.
    pub fn rows(&self, idx: &MxArray) -> Result<MxArray> {
        match self {
            Self::Array(a) => a.take(idx, 0),
            Self::Q8 { w, scales } => dequant_q8(&w.take(idx, 0)?, &scales.take(idx, 0)?),
            Self::Qmm { w, scales, biases } => dequant_affine8(
                &w.take(idx, 0)?,
                &scales.take(idx, 0)?,
                &biases.take(idx, 0)?,
            ),
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
            Self::Qmm { w, scales, biases } => Self::Qmm {
                w: cut(w)?,
                scales: cut(scales)?,
                biases: cut(biases)?,
            },
        })
    }

    /// `x [g, s, k] @ W_i^T` for `g` equal row groups `W_i` of a `Qmm`
    /// weight (`[g, s, rows / g]`); `None` for the other forms.
    pub fn grouped_qmm_t(&self, x: &MxArray, g: i64) -> Result<Option<MxArray>> {
        let Self::Qmm { w, scales, biases } = self else {
            return Ok(None);
        };
        let r = |a: &MxArray| a.reshape(&[g, a.shape_at(0)? / g, a.shape_at(1)?]);
        Ok(Some(qmm_t(x, &r(w)?, &r(scales)?, &r(biases)?)?))
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Array(a) => a.nbytes(),
            Self::Q8 { w, scales } => w.nbytes() + scales.nbytes(),
            Self::Qmm { w, scales, biases } => w.nbytes() + scales.nbytes() + biases.nbytes(),
        }
    }
}

/// Where a tensor lives: shard index, that shard's data start, header.
type Located = (usize, u64, GgufTensorInfo);

/// Expert projections in bank order.
const PROJS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
const ENGRAM_TABLE: &str = "per_layer_token_embd.weight";
/// MLX free-buffer pool ceiling.
pub const POOL_LIMIT: usize = 1 << 30;
/// v1 context limit: the admission's KV / compute terms use it.
pub const CTX_LIMIT: usize = 16_384;

/// How many cache slots the bank gets.
#[derive(Clone, Debug, PartialEq)]
pub enum CachePlan {
    /// Staging only, no admission (tests: the admission is the one memory
    /// gate of a product load).
    #[cfg(test)]
    Off,
    /// Exactly these cache slots and staging layers, no admission (tests).
    #[cfg(test)]
    Slots { cache: usize, staging: usize },
    /// Admission at load and at every turn start.
    Admit(Limits),
}

struct TurnState {
    admission: Option<Admission>,
    wired: Option<WiredLimitContext>,
    guard: CacheLimitGuard,
    /// Bank switches: (cache slots, seconds to free the old bank and allocate).
    switches: Vec<(usize, f64)>,
}

pub struct Store {
    pub model: Model,
    pub bank: Arc<StagingBank>,
    /// GGUF bytes of the dense tensors read at load.
    pub dense_read: u64,
    /// Bytes of the resident dense arrays.
    pub dense_bytes: u64,
    pub size: ModelSize,
    /// Seconds to allocate the bank at load (no page touched yet).
    pub bank_alloc_s: f64,
    pub dense_mode: DenseMode,
    plan: CachePlan,
    turn: Mutex<TurnState>,
}

#[cfg(test)]
thread_local! {
    /// The host reading the admission sees on this thread instead of the
    /// real one, so tiny tests do not depend on the test host's free memory.
    pub(super) static TEST_HOST: std::cell::Cell<Option<(HostMemory, u64)>> =
        const { std::cell::Cell::new(None) };
}

/// Host memory and the Metal working set for an admission.
fn host_reading() -> Result<(HostMemory, u64)> {
    #[cfg(test)]
    if let Some(r) = TEST_HOST.with(std::cell::Cell::get) {
        return Ok(r);
    }
    let h = host_memory().ok_or_else(|| bad("no host memory reading for admission"))?;
    Ok((h, working_set()))
}

fn header(path: &Path) -> Result<GgufFile> {
    parse_gguf_reader(std::io::BufReader::new(open_nocache(path)?))
        .map_err(|e| bad(format!("'{}': {e}", path.display())))
}

impl Store {
    /// Staging only (`CachePlan::Off`).
    #[cfg(test)]
    pub fn open(paths: &[PathBuf], act: DType, fq: FakeQuant) -> Result<Self> {
        Self::open_with(paths, act, fq, CachePlan::Off)
    }

    /// `paths` are every shard in order (shard 0 holds the metadata); dense
    /// Q8_0 as `DenseMode::Q8`.
    pub fn open_with(
        paths: &[PathBuf],
        act: DType,
        fq: FakeQuant,
        plan: CachePlan,
    ) -> Result<Self> {
        Self::open_dense(paths, act, fq, plan, DenseMode::Q8)
    }

    pub fn open_dense(
        paths: &[PathBuf],
        act: DType,
        fq: FakeQuant,
        plan: CachePlan,
        mode: DenseMode,
    ) -> Result<Self> {
        let Scan {
            where_,
            cfg,
            layers,
            size,
        } = scan(paths, mode)?;
        let find = |n: &str| {
            where_
                .get(n)
                .ok_or_else(|| bad(format!("tensor '{n}' missing")))
        };
        let (cache_slots, staging, admission) = match &plan {
            CachePlan::Admit(limits) => {
                let (h, ws) = host_reading()?;
                // nothing of this model exists yet: every byte in use is others'
                let a = admit(&size, limits, &h, &Own::default(), ws)?;
                tracing::info!("deepseek_v41 admission at load: {a}");
                (a.cache_slots as usize, a.staging_layers as usize, Some(a))
            }
            #[cfg(test)]
            CachePlan::Off => (0, 1, None),
            #[cfg(test)]
            CachePlan::Slots { cache, staging } => (*cache, *staging, None),
        };
        let shards = Arc::new(ShardSet::open(paths)?);
        let t_bank = Instant::now();
        let bank = Arc::new(StagingBank::with_staging(
            IoPool::new(Arc::clone(&shards), DEFAULT_THREADS)?,
            layers,
            cache_slots,
            staging,
        )?);
        bank.set_split(true);
        bank.set_prefetch(true);
        let bank_alloc_s = t_bank.elapsed().as_secs_f64();
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
                    let (w, scales) = (
                        dst.weight,
                        dst.scales
                            .ok_or_else(|| bad("Q8_0 repack gave no scales"))?,
                    );
                    match mode.form(n) {
                        Form::Codes => Dense::Q8 { w, scales },
                        Form::Qmm => {
                            let s = scales.astype(DType::Float32)?;
                            let b = s.mul_scalar(-128.0)?;
                            MxArray::eval_arrays_with_context(&[&s, &b], "deepseek_v41 q8 scales")?;
                            Dense::Qmm {
                                w,
                                scales: s,
                                biases: b,
                            }
                        }
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
            Err(bad(format!("{n}: routed experts stream through the bank")))
        };
        let table = match &cfg.engram {
            Some(ec) => {
                let (shard, data_offset, info) = find(ENGRAM_TABLE)?;
                let t = PreadTable::open(&paths[*shard], *data_offset, info, ec)?;
                Some(Box::new(t) as Box<dyn super::engram::EngramRows>)
            }
            None => None,
        };
        let mut model =
            match Model::from_gguf(cfg, act, fq, get, bank_of, table, Some(Arc::clone(&bank))) {
                Ok(m) => m,
                Err(e) => {
                    // the partial model is gone; keep its buffers out of MLX's free pool
                    drop(bank);
                    crate::array::clear_cache();
                    return Err(e);
                }
            };
        // the admission's KV / compute terms hold up to this context only
        model.ctx_limit = Some(size.ctx as usize);
        let guard =
            coordinator().register_with_cache_limit(dense_bytes + bank.nbytes(), Some(POOL_LIMIT));
        Ok(Self {
            model,
            bank,
            dense_read,
            dense_bytes,
            size,
            bank_alloc_s,
            dense_mode: mode,
            plan,
            turn: Mutex::new(TurnState {
                admission,
                wired: None,
                guard,
                switches: Vec::new(),
            }),
        })
    }

    fn turn(&self) -> MutexGuard<'_, TurnState> {
        self.turn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn plan(&self) -> &CachePlan {
        &self.plan
    }

    pub fn cache_slots(&self) -> usize {
        self.bank.cache_slots()
    }

    /// The admission in force (`CachePlan::Admit` only).
    pub fn admission(&self) -> Option<Admission> {
        self.turn().admission
    }

    /// Every bank switch so far: (cache slots, seconds).
    pub fn switches(&self) -> Vec<(usize, f64)> {
        self.turn().switches.clone()
    }

    /// This model's bytes inside `host_used`: dense arrays + the bank's
    /// resident pages, read while the bank is non-volatile (`begin_turn`
    /// re-admits after `NonVolatile`). A volatile bank the GPU has used stays
    /// wired, so `host_used` keeps it; between turns the resident
    /// pages are reported as volatile. Process-wide MLX counters would also
    /// hold other loaded models, which must stay `others`; KV, scratch and
    /// the shared pool are left in `others` (conservative).
    pub fn own(&self) -> Result<Own> {
        let resident = self.bank.resident_bytes()?;
        Ok(Own {
            active: self.dense_bytes + resident,
            pool: 0,
            volatile: if self.bank.in_turn() { 0 } else { resident },
        })
    }

    fn switch(&self, t: &mut TurnState, slots: usize, staging: usize) -> Result<()> {
        let t0 = Instant::now();
        self.bank.rebuild_staging(slots, staging)?;
        t.guard = coordinator()
            .register_with_cache_limit(self.dense_bytes + self.bank.nbytes(), Some(POOL_LIMIT));
        let s = t0.elapsed().as_secs_f64();
        tracing::info!("deepseek_v41 bank switch to {slots} cache slots in {s:.2} s");
        t.switches.push((slots, s));
        Ok(())
    }

    /// Drops the bank and allocates one with `slots` cache slots (between
    /// turns; the old bank's buffers must be gone first, `StagingBank::rebuild`).
    pub fn set_cache_slots(&self, slots: usize) -> Result<()> {
        let mut t = self.turn();
        if t.wired.is_some() {
            return Err(bad("bank switch inside a turn"));
        }
        let staging = self.bank.staging_layers();
        self.switch(&mut t, slots, staging)
    }

    /// Turn start: re-admit (`CachePlan::Admit`), switch the bank if the
    /// admission says so, take the wired-limit lease, make the bank
    /// non-volatile (forgetting every slot if Metal emptied it). Errs when
    /// the fixed working set alone no longer fits.
    pub fn begin_turn(&self) -> Result<()> {
        let mut t = self.turn();
        if let CachePlan::Admit(limits) = &self.plan
            && t.wired.is_none()
        {
            let cur = t.admission.ok_or_else(|| bad("no admission"))?;
            // non-volatile first: its resident pages are then in `host_used`
            // for certain and counted as own (allocates nothing)
            self.bank.begin_turn()?;
            let next = (|| {
                let (h, ws) = host_reading()?;
                let next = match readmit(&cur, &self.size, limits, &h, &self.own()?, ws)? {
                    Turn::Keep => cur,
                    Turn::StagingOnly(a) | Turn::Rebuild(a) => {
                        self.switch(&mut t, a.cache_slots as usize, a.staging_layers as usize)?;
                        tracing::info!("deepseek_v41 re-admission: {a}");
                        a
                    }
                };
                Ok(next)
            })();
            let next = match next {
                Ok(n) => n,
                Err(e) => {
                    let _ = self.bank.end_turn();
                    return Err(e);
                }
            };
            t.admission = Some(next);
            t.wired = Some(WiredLimitContext::bounded(
                usize::try_from(next.wired_limit()).unwrap_or(usize::MAX),
                Vec::new(),
            ));
        }
        self.bank.begin_turn()?;
        Ok(())
    }

    /// Turn end (after the turn's last eval): release the wired lease, make
    /// the bank volatile.
    pub fn end_turn(&self) -> Result<()> {
        let mut t = self.turn();
        crate::array::synchronize();
        t.wired = None;
        self.bank.end_turn()
    }
}

/// The shard headers: where each tensor is, the config, the expert tensors
/// per layer and the sizes the admission needs.
struct Scan {
    where_: HashMap<String, Located>,
    cfg: DeepseekV41Config,
    layers: Vec<Vec<ExpertTensor>>,
    size: ModelSize,
}

fn scan(paths: &[PathBuf], mode: DenseMode) -> Result<Scan> {
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
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let mut v = Vec::with_capacity(PROJS.len());
        for p in PROJS {
            let n = format!("blk.{l}.{p}.weight");
            let (shard, data_offset, info) = where_
                .get(&n)
                .ok_or_else(|| bad(format!("tensor '{n}' missing")))?
                .clone();
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
    let dense_resident = where_
        .values()
        .filter(|(_, _, t)| is_dense(&t.name))
        .try_fold(0u64, |acc, (_, _, t)| {
            acc.checked_add(mode.resident(t)?)
                .ok_or_else(|| bad("dense resident bytes overflow u64"))
        })?;
    let n_experts = cfg.n_routed_experts as u64;
    let size = ModelSize {
        dense: dense_resident,
        slot: bank_need / n_experts.max(1),
        n_experts,
        ctx: cfg.context_length.min(CTX_LIMIT) as u64,
    };
    Ok(Scan {
        where_,
        cfg,
        layers,
        size,
    })
}

/// The admission's model sizes from the shard headers alone (no tensor
/// data read, nothing allocated); `dense` = resident bytes in `mode`.
pub fn model_size(paths: &[PathBuf], mode: DenseMode) -> Result<ModelSize> {
    Ok(scan(paths, mode)?.size)
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
