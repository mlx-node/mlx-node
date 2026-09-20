//! Bounded, positional SSD reads. Descriptors never create MLX arrays. Only
//! selected expert matrices and embedding rows are read on cache misses.
//! Admitted hot banks can be prepared once; PLE stays row-addressable. Matrix chunks
//! may reuse a bounded persistent cache of their losslessly packed arrays.
use crate::models::qwen4_exp::runtime_flags;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::{Error, Result};
use serde::Deserialize;

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::QuantizedLinear;
use crate::utils::gguf::{
    GgufFile, GgufLoadOptions, GgufMetaValue, GgufTensorInfo, GgufTensorType, load_gguf_tensors,
    parse_gguf, symmetric_zero_point,
};

#[path = "expert_slots.rs"]
mod expert_slots;
#[path = "residency.rs"]
mod residency;

pub const CACHE_BYTES: u64 = 8 << 30;
pub(super) const MAX_READ_BYTES: u64 = 64 << 20;
const MAX_HEADER_BYTES: u64 = 16 << 20;

fn err(message: impl ToString) -> Error {
    Error::from_reason(message.to_string())
}

#[derive(Clone)]
enum Encoding {
    Safe(String),
    Gguf(GgufTensorType),
}
#[derive(Clone)]
pub struct Tensor {
    path: PathBuf,
    pub shape: Vec<usize>,
    offset: u64,
    bytes: u64,
    encoding: Encoding,
}
impl Tensor {
    pub fn width(&self) -> Result<usize> {
        self.shape
            .last()
            .copied()
            .filter(|&width| width > 0)
            .ok_or_else(|| err("SSD tensor requires a positive row width"))
    }
    pub fn rows(&self) -> Result<usize> {
        let width = self.width()?;
        let elements = self
            .shape
            .iter()
            .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
            .ok_or_else(|| err("Invalid or overflowing SSD tensor dimensions"))?;
        Ok(elements / width)
    }
    fn row_bytes(&self) -> Result<u64> {
        Ok(self.bytes / self.rows()? as u64)
    }
    fn range(&self, start: usize, rows: usize) -> Result<(u64, u64)> {
        let total_rows = self.rows()?;
        if rows == 0 || start.checked_add(rows).is_none_or(|end| end > total_rows) {
            return Err(err("SSD tensor row range is outside the descriptor"));
        }
        let bytes = self
            .row_bytes()?
            .checked_mul(rows as u64)
            .ok_or_else(|| err("SSD read size overflow"))?;
        if bytes > MAX_READ_BYTES
            || (rows as u64)
                .saturating_mul(self.width()? as u64)
                .saturating_mul(4)
                > MAX_READ_BYTES
        {
            return Err(err(
                "SSD tensor read exceeds the 64 MiB staging budget; use row chunks",
            ));
        }
        Ok((self.offset + self.row_bytes()? * start as u64, bytes))
    }
    fn packed_key(&self, name: &str, start: usize, rows: usize) -> Result<Option<String>> {
        let (offset, bytes) = self.range(start, rows)?;
        Ok(match self.encoding {
            Encoding::Gguf(ty)
                if rows > 1
                    && bytes >= 4096
                    && (ty.k_quant_format().is_some()
                        || matches!(
                            ty,
                            GgufTensorType::Q4_0
                                | GgufTensorType::Q4_1
                                | GgufTensorType::Q5_1
                                | GgufTensorType::Q8_0
                        ))
                    && name != "per_layer_token_embd.weight"
                    && name != "token_embd.weight" =>
            {
                Some(super::packed_cache::chunk_key(
                    &self.path,
                    offset,
                    bytes,
                    rows,
                    self.width()?,
                    ty as u32,
                ))
            }
            _ => None,
        })
    }
}

#[derive(Clone)]
pub struct Weight {
    pub values: MxArray,
    // Evaluated once for admitted dense projection banks. Keep the source for
    // other activation dtypes and include both allocations in residency bytes.
    pub(super) dense_bf16: Option<MxArray>,
    pub(super) scales: Option<MxArray>,
    pub(super) biases: Option<MxArray>,
    pub(super) group: i32,
    pub(super) bits: i32,
    pub(super) mode: String,
}
impl Weight {
    pub(super) fn mixer_down_inject(
        &self,
        x: &MxArray,
        injection: &Self,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if !(runtime_flags::is_one(c"MLX_QWEN4_MIXER_DOWN_INJECT")
            || runtime_flags::is_one(c"MLX_QWEN4_MIXER_SPLIT_K"))
            || runtime_flags::is_zero(c"MLX_QWEN4_DECODE_MIXER_ACT")
            || !crate::engine::persistence::compiled_forward_backend_available()
            || [self, injection]
                .iter()
                .any(|w| w.mode != "affine" || w.bits != 8 || w.group != 32)
        {
            return Ok(None);
        }
        let (Some(sd), Some(bd), Some(si), Some(bi)) = (
            &self.scales,
            &self.biases,
            &injection.scales,
            &injection.biases,
        ) else {
            return Ok(None);
        };
        let mut out = std::ptr::null_mut();
        let mut gate = std::ptr::null_mut();
        if !unsafe {
            mlx_sys::mlx_qwen4_mixer_down_inject(
                x.as_raw_ptr(),
                self.values.as_raw_ptr(),
                sd.as_raw_ptr(),
                bd.as_raw_ptr(),
                injection.values.as_raw_ptr(),
                si.as_raw_ptr(),
                bi.as_raw_ptr(),
                &mut out,
                &mut gate,
            )
        } {
            return Ok(None);
        }
        Ok(Some((
            MxArray::from_handle(out, "Qwen4 mixer activation with injection")?,
            MxArray::from_handle(gate, "Qwen4 mixer injection projection")?,
        )))
    }

    pub(super) fn mixer_act(&self, x: &MxArray) -> Result<Option<MxArray>> {
        let decode = x.shape_at(1)? == 1;
        let setting = if decode {
            c"MLX_QWEN4_DECODE_MIXER_ACT"
        } else {
            c"MLX_QWEN4_PREFILL_MIXER_BM32"
        };
        if self.mode != "affine"
            || self.bits != 8
            || self.group != 32
            || !crate::engine::persistence::compiled_forward_backend_available()
            || runtime_flags::is_zero(setting)
        {
            return Ok(None);
        }
        let (Some(scales), Some(biases)) = (&self.scales, &self.biases) else {
            return Ok(None);
        };
        let raw = unsafe {
            let kernel = if decode {
                mlx_sys::mlx_qwen4_decode_mixer_act
            } else {
                mlx_sys::mlx_qwen4_prefill_mixer_act
            };
            kernel(
                x.as_raw_ptr(),
                self.values.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
            )
        };
        if raw.is_null() {
            Ok(None)
        } else {
            MxArray::from_handle(raw, "Qwen4 reference mixer activation").map(Some)
        }
    }

    pub(super) fn hyper_up_inject(
        &self,
        x: &MxArray,
        normed: &MxArray,
        injection: &MxArray,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if self.mode != "affine"
            || self.bits != 8
            || self.group != 32
            || !runtime_flags::is_one(c"MLX_QWEN4_MIXER_INJECT")
            || runtime_flags::is_zero(c"MLX_QWEN4_HYPER_UP")
            || runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
        {
            return Ok(None);
        }
        let (Some(scales), Some(biases)) = (&self.scales, &self.biases) else {
            return Ok(None);
        };
        let mut mixed = std::ptr::null_mut();
        let mut gate = std::ptr::null_mut();
        if !unsafe {
            mlx_sys::mlx_qwen4_hyper_up_inject(
                x.as_raw_ptr(),
                self.values.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                normed.as_raw_ptr(),
                injection.as_raw_ptr(),
                &mut mixed,
                &mut gate,
            )
        } {
            return Ok(None);
        }
        Ok(Some((
            MxArray::from_handle(mixed, "Qwen4 mixer with injection")?,
            MxArray::from_handle(gate, "Qwen4 mixer injection gate")?,
        )))
    }

    pub(super) fn hyper_up(&self, x: &MxArray, normed: &MxArray) -> Result<Option<MxArray>> {
        if self.mode != "affine"
            || self.bits != 8
            || self.group != 32
            || runtime_flags::is_zero(c"MLX_QWEN4_HYPER_UP")
            || runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
        {
            return Ok(None);
        }
        let (Some(scales), Some(biases)) = (&self.scales, &self.biases) else {
            return Ok(None);
        };
        let raw = unsafe {
            mlx_sys::mlx_qwen4_hyper_up(
                x.as_raw_ptr(),
                self.values.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                normed.as_raw_ptr(),
            )
        };
        if raw.is_null() {
            Ok(None)
        } else {
            MxArray::from_handle(raw, "Qwen4 hyper up").map(Some)
        }
    }

    pub(super) fn stack(weights: &[Arc<Self>]) -> Result<Option<Self>> {
        let Some(first) = weights.first() else {
            return Ok(None);
        };
        if first.scales.is_none() {
            return Ok(None);
        }
        for w in weights {
            if w.bits != first.bits
                || w.group != first.group
                || w.mode != first.mode
                || w.biases.is_some() != first.biases.is_some()
                || w.scales.is_none()
            {
                return Ok(None);
            }
            for (a, b) in [
                (Some(&w.values), Some(&first.values)),
                (w.scales.as_ref(), first.scales.as_ref()),
                (w.biases.as_ref(), first.biases.as_ref()),
            ] {
                if let (Some(a), Some(b)) = (a, b)
                    && (a.dtype()? != b.dtype()? || *a.shape()? != *b.shape()?)
                {
                    return Ok(None);
                }
            }
        }
        let Some(scales) = weights
            .iter()
            .map(|w| w.scales.as_ref())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let biases = if first.biases.is_some() {
            let Some(biases) = weights
                .iter()
                .map(|w| w.biases.as_ref())
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            Some(MxArray::stack(biases, Some(0))?)
        } else {
            None
        };
        Ok(Some(Self {
            values: MxArray::stack(weights.iter().map(|w| &w.values).collect(), Some(0))?,
            dense_bf16: None,
            scales: Some(MxArray::stack(scales, Some(0))?),
            biases,
            bits: first.bits,
            group: first.group,
            mode: first.mode.clone(),
        }))
    }

    pub fn linear(&self, x: &MxArray) -> Result<MxArray> {
        if self.mode == "affine"
            && self.bits == 8
            && self.group == 32
            && !runtime_flags::is_zero(c"MLX_QWEN4_DENSE_DECODE_STORAGE")
            && let (Some(scales), Some(biases)) = (&self.scales, &self.biases)
        {
            let compact = unsafe {
                mlx_sys::mlx_qwen4_dense_decode(
                    x.as_raw_ptr(),
                    self.values.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases.as_raw_ptr(),
                )
            };
            if !compact.is_null() {
                let y = MxArray::from_handle(compact, "Qwen4 compact dense decode")?;
                if runtime_flags::is_one(c"MLX_QWEN4_SYNC_PROJECTIONS") {
                    MxArray::eval_arrays_with_context(&[&y], "qwen4::weights::y")?;
                }
                return Ok(y);
            }
        }
        let compact = if self.mode == "affine"
            && self.bits == 8
            && self.group == 32
            && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_DENSE_STORAGE")
        {
            match (&self.scales, &self.biases) {
                (Some(scales), Some(biases)) => unsafe {
                    mlx_sys::mlx_qwen4_dense_prefill(
                        x.as_raw_ptr(),
                        self.values.as_raw_ptr(),
                        scales.as_raw_ptr(),
                        biases.as_raw_ptr(),
                    )
                },
                _ => std::ptr::null_mut(),
            }
        } else {
            std::ptr::null_mut()
        };
        let y = if !compact.is_null() {
            MxArray::from_handle(compact, "Qwen4 compact dense prefill")?
        } else if let Some(scales) = &self.scales {
            QuantizedLinear::new(
                self.values.clone(),
                scales.clone(),
                self.biases.clone(),
                None,
                self.group,
                self.bits,
                self.mode.clone(),
            )
            .forward(x)?
        } else {
            let values = if x.dtype()? == DType::BFloat16
                && let Some(dense) = &self.dense_bf16
                && !runtime_flags::is_zero(c"MLX_QWEN4_DENSE_BF16_CACHE")
            {
                dense
            } else {
                &self.values
            };
            if runtime_flags::is_one(c"MLX_QWEN4_ROUTER_ROW_SCHEDULE") {
                let compact = unsafe {
                    mlx_sys::mlx_qwen4_router_decode(x.as_raw_ptr(), values.as_raw_ptr())
                };
                if !compact.is_null() {
                    let y = MxArray::from_handle(compact, "Qwen4 router row schedule")?;
                    if runtime_flags::is_one(c"MLX_QWEN4_SYNC_PROJECTIONS") {
                        MxArray::eval_arrays_with_context(&[&y], "qwen4::weights::y")?;
                    }
                    return Ok(y);
                }
            }
            let weight = values.transpose(None)?.astype(x.dtype()?)?;
            let shape = x.shape()?;
            let (&width, outer_shape) = shape
                .split_last()
                .filter(|(width, _)| **width > 0)
                .ok_or_else(|| err("Qwen4 projection input requires a positive final dimension"))?;
            if x.dtype()? == DType::Float32 && x.size()? / width as u64 > 1 {
                // MLX enables TF32 GEMMs by default on NAX hardware. Preserve
                // the F32 source's GEMV arithmetic without changing global
                // process settings or lowering the fixture's accuracy gate.
                let rows = x.reshape(&[-1, width])?;
                let mut outputs = Vec::new();
                for row in 0..rows.shape()?[0] {
                    outputs.push(rows.slice_axis(0, row, row + 1)?.matmul(&weight)?);
                }
                let mut shape = outer_shape.to_vec();
                shape.push(self.values.shape_at(0)?);
                MxArray::concatenate_many(outputs.iter().collect(), Some(0))?.reshape(&shape)?
            } else {
                x.matmul(&weight)?
            }
        };
        // MLX owns the input buffers until this output is evaluated, including
        // when an LRU entry is evicted. The decoder fences each expert output,
        // recurrent update and layer; these bounded groups can submit their
        // dependent projections together instead of blocking after each one.
        if runtime_flags::is_one(c"MLX_QWEN4_SYNC_PROJECTIONS") {
            MxArray::eval_arrays_with_context(&[&y], "qwen4::weights::y")?;
        }
        Ok(y)
    }
    pub fn dense(&self) -> Result<MxArray> {
        if self.scales.is_none() {
            return Ok(self.values.clone());
        }
        let a = self.dense_pending()?;
        MxArray::eval_arrays_with_context(&[&a], "qwen4::weights::dense")?;
        Ok(a)
    }
    fn dense_pending(&self) -> Result<MxArray> {
        let Some(scales) = &self.scales else {
            return Ok(self.values.clone());
        };
        let mode = std::ffi::CString::new(self.mode.as_str()).map_err(err)?;
        let a = unsafe {
            mlx_sys::mlx_dequantize(
                self.values.as_raw_ptr(),
                scales.as_raw_ptr(),
                self.biases
                    .as_ref()
                    .map_or(std::ptr::null_mut(), MxArray::as_raw_ptr),
                self.group,
                self.bits,
                DType::BFloat16 as i32,
                mode.as_ptr(),
            )
        };
        MxArray::from_handle(a, "SSD row dequantize")
    }
    pub(super) fn bytes(&self) -> Result<u64> {
        let mut bytes = 0;
        for a in [
            Some(&self.values),
            self.dense_bf16.as_ref(),
            self.scales.as_ref(),
            self.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            bytes += a.size()?
                * match a.dtype()? {
                    DType::Uint8 | DType::Int8 => 1,
                    DType::Float16 | DType::BFloat16 => 2,
                    _ => 4,
                };
        }
        Ok(bytes)
    }
}

pub struct Store {
    #[cfg(test)]
    fixture: bool,
    pub tensors: HashMap<String, Tensor>,
    pub gguf: bool,
    pub metadata: HashMap<String, GgufMetaValue>,
    cache: HashMap<(String, usize, usize), Arc<Weight>>,
    order: VecDeque<((String, usize, usize), u64)>,
    recency: HashMap<(String, usize, usize), u64>,
    clock: u64,
    protected: HashSet<(String, usize, usize)>,
    protected_bytes: u64,
    cache_bytes: u64,
    cache_limit: u64,
    pub(super) plan: super::memory::Plan,
    banks: HashMap<String, Arc<Weight>>,
    bank_bytes: u64,
    paired_banks: HashMap<(String, String), Arc<Weight>>,
    slots: HashMap<usize, expert_slots::ExpertSlots>,
    wide_routes: HashMap<usize, Vec<u32>>,
    deferred_reduction_layer: Option<usize>,
    slot_capacity: Option<usize>,
    slot_bytes: u64,
    pub slot_upload_bytes: u64,
    pub slot_hits: u64,
    pub slot_misses: u64,
    growth_bytes: u64,
    pub bytes_read: u64,
    pub peak_cache_bytes: u64,
    packed: Option<super::packed_cache::PackedCache>,
    pub packed_hits: u64,
    pub packed_bytes_read: u64,
    pub lookup_batches: u64,
    pub lookup_misses: u64,
}

#[derive(Deserialize)]
struct SafeDescriptor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl Store {
    pub(super) fn check_forward_headroom(&self) -> Result<()> {
        self.check_forward_headroom_with(super::memory::check_growth_headroom)
    }

    fn check_forward_headroom_with(&self, check: impl FnOnce() -> Result<()>) -> Result<()> {
        // Only explicitly opened unit fixtures skip the production forward
        // reserve. Payload admission and the growth check in read() remain live.
        #[cfg(test)]
        if self.fixture {
            return Ok(());
        }
        check()
    }

    pub fn open(path: &Path) -> Result<Self> {
        let mut s = Self::open_metadata(path, None)?;
        s.plan = super::memory::plan(s.hot_bytes()?)?;
        s.cache_limit = s.plan.budget;
        Ok(s)
    }

    /// Tiny synthetic fixtures exercise IO and kernels without reserving the
    /// multi-GiB working allowance required by a production checkpoint. Actual
    /// payload reads still pass the ordinary live admission checks.
    #[cfg(test)]
    pub(in super::super) fn open_fixture(path: &Path, root: Option<&Path>) -> Result<Self> {
        let mut store = Self::open_metadata(path, root)?;
        store.fixture = true;
        // Logical ceiling includes the importer staging allowance; the
        // synthetic tensors allocate only their actual payload sizes.
        store.cache_limit = 1 << 30;
        store.plan.budget = store.cache_limit;
        store.plan.hot_bytes = store.hot_bytes()?;
        Ok(store)
    }

    /// Auxiliary source checkpoints contribute only descriptors, so their
    /// unrelated target tensors must not participate in residency admission.
    pub(super) fn open_metadata(path: &Path, root: Option<&Path>) -> Result<Self> {
        let cache_limit = CACHE_BYTES;
        let mut s = Self {
            #[cfg(test)]
            fixture: false,
            tensors: HashMap::new(),
            gguf: path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf")),
            metadata: HashMap::new(),
            cache: HashMap::new(),
            order: VecDeque::new(),
            recency: HashMap::new(),
            clock: 0,
            protected: HashSet::new(),
            protected_bytes: 0,
            cache_bytes: 0,
            cache_limit,
            plan: super::memory::Plan {
                budget: cache_limit,
                hot_bytes: 0,
                resident: false,
                policy: "stream".into(),
                available_bytes: None,
                physical_bytes: None,
                resident_bank_bytes: 0,
                resident_expert_layers: 0,
                partial_expert_slots: 0,
            },
            banks: HashMap::new(),
            bank_bytes: 0,
            paired_banks: HashMap::new(),
            slots: HashMap::new(),
            wide_routes: HashMap::new(),
            deferred_reduction_layer: None,
            slot_capacity: None,
            slot_bytes: 0,
            slot_upload_bytes: 0,
            slot_hits: 0,
            slot_misses: 0,
            growth_bytes: 0,
            bytes_read: 0,
            peak_cache_bytes: 0,
            packed: None,
            packed_hits: 0,
            packed_bytes_read: 0,
            lookup_batches: 0,
            lookup_misses: 0,
        };
        if s.gguf {
            s.index_gguf(path)?;
        } else {
            s.index_safe(path)?;
        }
        if s.tensors.is_empty() {
            return Err(err("No model tensors found"));
        }
        if s.gguf && !runtime_flags::is_zero(c"MLX_QWEN4_PACKED_CACHE") {
            let files = s.tensors.values().map(|t| t.path.clone()).collect();
            s.packed = match root {
                Some(root) => super::packed_cache::PackedCache::new_at(root, files),
                None => super::packed_cache::PackedCache::new(path, files),
            }
            .ok();
        }
        Ok(s)
    }
    pub(super) fn cache_budget(&self) -> u64 {
        self.cache_limit
    }
    fn touch(&mut self, key: (String, usize, usize)) {
        self.clock += 1;
        self.recency.insert(key.clone(), self.clock);
        self.order.push_back((key, self.clock));
        // Lazy recency updates avoid scanning thousands of cached experts on
        // every hit. Periodic compaction bounds metadata without GPU work.
        if self.order.len() > (self.cache.len() * 2).max(64) {
            self.order
                .retain(|(key, stamp)| self.recency.get(key) == Some(stamp));
        }
    }
    fn evict_to(&mut self, target: u64) -> Result<()> {
        while self.cache_bytes > target {
            let Some((old, stamp)) = self.order.pop_front() else {
                return Err(err(
                    "Protected Qwen4 weights leave insufficient staging space",
                ));
            };
            if self.recency.get(&old) != Some(&stamp) {
                continue;
            }
            self.recency.remove(&old);
            if let Some(w) = self.cache.remove(&old) {
                self.cache_bytes -= w.bytes()?;
            }
        }
        Ok(())
    }
    fn insert(&mut self, name: String, mut t: Tensor) -> Result<()> {
        let elements = t
            .shape
            .iter()
            .try_fold(1usize, |a, &b| if b == 0 { None } else { a.checked_mul(b) });
        if elements.is_none()
            || t.shape.is_empty()
            || t.shape.len() > 5
            || t.bytes == 0
            || t.offset
                .checked_add(t.bytes)
                .is_none_or(|end| fs::metadata(&t.path).map_or(true, |f| end > f.len()))
        {
            return Err(err(format!("Invalid/truncated SSD tensor {name}")));
        }
        if !t.bytes.is_multiple_of(t.rows()? as u64) {
            return Err(err(format!("Unaligned SSD rows: {name}")));
        }
        // Relative, absolute and symlink aliases must share the same prepared
        // chunk keys. Resolve once while indexing, not on each matrix read.
        t.path = t.path.canonicalize().map_err(err)?;
        if self.tensors.insert(name.clone(), t).is_some() {
            return Err(err(format!("Duplicate tensor {name}")));
        }
        Ok(())
    }
    fn index_safe(&mut self, dir: &Path) -> Result<()> {
        let root = dir.canonicalize().map_err(err)?;
        let index = dir.join("model.safetensors.index.json");
        let files = if index.exists() {
            let v: serde_json::Value =
                serde_json::from_slice(&fs::read(index).map_err(err)?).map_err(err)?;
            let map = v["weight_map"]
                .as_object()
                .ok_or_else(|| err("Missing safetensors weight_map"))?;
            let mut names = std::collections::BTreeSet::new();
            for v in map.values() {
                names.insert(
                    v.as_str()
                        .ok_or_else(|| err("Invalid shard filename"))?
                        .to_string(),
                );
            }
            names.into_iter().map(|f| dir.join(f)).collect::<Vec<_>>()
        } else {
            vec![dir.join("model.safetensors")]
        };
        for path in files {
            if !path.canonicalize().map_err(err)?.starts_with(&root) {
                return Err(err("Shard escapes model directory"));
            }
            let mut f = File::open(&path).map_err(err)?;
            let mut length = [0; 8];
            f.read_exact(&mut length).map_err(err)?;
            let length = u64::from_le_bytes(length);
            if length == 0 || length > MAX_HEADER_BYTES {
                return Err(err("Safetensors header exceeds metadata budget"));
            }
            let mut buf = vec![0; length as usize];
            f.read_exact(&mut buf).map_err(err)?;
            let mut header: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&buf).map_err(err)?;
            header.remove("__metadata__");
            for (name, value) in header {
                // Auxiliary tensors are indexed, never materialized here.
                let d: SafeDescriptor = serde_json::from_value(value).map_err(err)?;
                let size = match d.dtype.as_str() {
                    "BF16" | "F16" => 2,
                    "F32" => 4,
                    "I64" => 8,
                    other => return Err(err(format!("Unsupported SSD safetensors dtype {other}"))),
                };
                let bytes = d
                    .shape
                    .iter()
                    .try_fold(size, |a: u64, &b| a.checked_mul(b as u64))
                    .ok_or_else(|| err("Tensor size overflow"))?;
                if d.data_offsets[1].checked_sub(d.data_offsets[0]) != Some(bytes) {
                    return Err(err(format!("Invalid byte span for {name}")));
                }
                let name = name
                    .strip_prefix("model.language_model.")
                    .or_else(|| name.strip_prefix("language_model.model."))
                    .or_else(|| name.strip_prefix("language_model."))
                    .or_else(|| name.strip_prefix("model."))
                    .unwrap_or(&name)
                    .to_string();
                self.insert(
                    name,
                    Tensor {
                        path: path.clone(),
                        shape: d.shape,
                        offset: (8 + length)
                            .checked_add(d.data_offsets[0])
                            .ok_or_else(|| err("Tensor offset overflow"))?,
                        bytes,
                        encoding: Encoding::Safe(d.dtype),
                    },
                )?;
            }
        }
        Ok(())
    }
    fn index_gguf(&mut self, first: &Path) -> Result<()> {
        let initial = parse_gguf(first)?;
        let count = initial
            .metadata
            .get("split.count")
            .and_then(GgufMetaValue::as_u32)
            .unwrap_or(1);
        if count == 0 || count > 1024 {
            return Err(err("Invalid GGUF split count"));
        }
        if initial
            .metadata
            .get("split.no")
            .and_then(GgufMetaValue::as_u32)
            .unwrap_or(0)
            != 0
        {
            return Err(err("Open the first GGUF split (-00001-of-...)"));
        }
        if initial
            .metadata
            .get("general.architecture")
            .and_then(GgufMetaValue::as_str)
            != Some("qwen4exp")
        {
            return Err(err("Expected qwen4exp GGUF"));
        }
        let paths = crate::utils::gguf::resolve_gguf_shards(first, count)?;
        self.metadata = initial.metadata.clone();
        for (i, path) in paths.into_iter().enumerate() {
            let i = i as u32;
            let g = if i == 0 {
                None
            } else {
                Some(parse_gguf(&path)?)
            };
            let g = g.as_ref().unwrap_or(&initial);
            if g.metadata
                .get("split.no")
                .and_then(GgufMetaValue::as_u32)
                .unwrap_or(0)
                != i
                || g.metadata
                    .get("split.count")
                    .and_then(GgufMetaValue::as_u32)
                    .unwrap_or(1)
                    != count
            {
                return Err(err("Inconsistent GGUF split metadata"));
            }
            for t in &g.tensors {
                if t.tensor_type == GgufTensorType::PQ2_0 {
                    return Err(err(format!(
                        "Qwen4 does not support PQ2_0 tensor '{}'",
                        t.name
                    )));
                }
                if t.dims.is_empty()
                    || t.dims.iter().any(|&d| d == 0 || d > i64::MAX as u64)
                    || t.dims
                        .iter()
                        .try_fold(1u64, |a, &b| a.checked_mul(b))
                        .is_none()
                    || !t.dims[0].is_multiple_of(t.tensor_type.block_size() as u64)
                {
                    return Err(err(format!("Invalid GGUF tensor dimensions: {}", t.name)));
                }
                let shape = t.dims.iter().rev().map(|&d| d as usize).collect();
                let bytes = t.data_size()?;
                self.insert(
                    t.name.clone(),
                    Tensor {
                        path: path.clone(),
                        shape,
                        offset: g
                            .data_offset
                            .checked_add(t.offset)
                            .ok_or_else(|| err("GGUF offset overflow"))?,
                        bytes,
                        encoding: Encoding::Gguf(t.tensor_type),
                    },
                )?;
            }
        }
        if let Some(total) = self
            .metadata
            .get("split.tensors.count")
            .and_then(GgufMetaValue::as_u64)
            && total != self.tensors.len() as u64
        {
            return Err(err("GGUF split tensor count mismatch"));
        }
        Ok(())
    }
    pub fn descriptor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| err(format!("Missing qwen4_exp tensor {name}")))
    }

    /// Attach only the released auxiliary subtrees. Target weights and their
    /// quantization remain those of this store; the caller validates configs.
    pub fn attach_auxiliary(&mut self, source: Store) -> Result<()> {
        if source.gguf {
            return Err(err(
                "Qwen4 auxiliaryModelPath must be a safetensors directory",
            ));
        }
        let additions = source
            .tensors
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with("mtp.")
                    || name.starts_with("visual.")
                    || name.starts_with("vision_tower.")
            })
            .collect::<Vec<_>>();
        if additions
            .iter()
            .any(|(name, _)| self.tensors.contains_key(name))
        {
            return Err(err(
                "Auxiliary checkpoint duplicates embedded Qwen4 tensors",
            ));
        }
        for (name, tensor) in additions {
            self.insert(name, tensor)?;
        }
        Ok(())
    }
    pub fn read(&mut self, name: &str, start: usize, rows: usize) -> Result<Arc<Weight>> {
        self.read_impl(name, start, rows, false, None)
    }

    /// Preload only the already selected expert group. At most 64 MiB of
    /// verified CPU payloads coexist; MLX imports and all Store/slot mutations
    /// stay on the inference thread in original request order.
    fn read_expert_batch(
        &mut self,
        requests: &[(String, usize, usize)],
    ) -> Result<Vec<Arc<Weight>>> {
        if self.packed.is_none() || runtime_flags::is_zero(c"MLX_QWEN4_PARALLEL_READS") {
            return requests
                .iter()
                .map(|(name, start, rows)| self.read(name, *start, *rows))
                .collect();
        }
        let mut jobs = Vec::new();
        let mut indices = Vec::new();
        let mut total = 0u64;
        for (i, (name, start, rows)) in requests.iter().enumerate() {
            if self.banks.contains_key(name)
                || self.cache.contains_key(&(name.clone(), *start, *rows))
            {
                continue;
            }
            let d = self.descriptor(name)?;
            if let Some(key) = d.packed_key(name, *start, *rows)? {
                let bytes = (d.runtime_bytes()? / d.rows()? as u64).saturating_mul(*rows as u64);
                total = total.saturating_add(bytes);
                jobs.push((key, bytes));
                indices.push(i);
            }
        }
        if total > MAX_READ_BYTES || jobs.len() > 24 || jobs.len() < 2 {
            return requests
                .iter()
                .map(|(name, start, rows)| self.read(name, *start, *rows))
                .collect();
        }
        // Packed payload, imported arrays and upload staging coexist. Charge
        // the actual bounded batch, rather than reserving three maximum-size
        // reads even for a few small cache misses.
        let staging_bytes = total * 3;
        self.evict_to(self.cache_limit.saturating_sub(staging_bytes))?;
        super::memory::maintain_freelist(self.plan.physical_bytes);
        super::memory::admit(staging_bytes)?;
        let cache = self
            .packed
            .as_ref()
            .ok_or_else(|| err("Qwen4 packed cache disappeared before the prepared batch read"))?;
        let verified = cache.read_batch(&jobs)?;
        let mut prepared: Vec<_> = (0..requests.len()).map(|_| None).collect();
        for (i, chunk) in indices.into_iter().zip(verified) {
            prepared[i] = Some(chunk);
        }
        requests
            .iter()
            .zip(prepared)
            .map(|((name, start, rows), chunk)| self.read_impl(name, *start, *rows, false, chunk))
            .collect()
    }

    /// Gather a bounded lookup window, sharing duplicate rows and submitting
    /// cold imports/dequantizations together. Decoded rows use the existing
    /// Store-owned LRU and its byte accounting, so no second cache is reserved.
    pub fn lookup_rows(&mut self, name: &str, ids: &[usize]) -> Result<MxArray> {
        let d = self.descriptor(name)?;
        let total_rows = d.rows()?;
        if ids.is_empty()
            || ids.len() > 65536
            || ids.iter().any(|&id| id >= total_rows)
            || (ids.len() as u64)
                .saturating_mul(d.width()? as u64)
                .saturating_mul(4)
                > MAX_READ_BYTES
        {
            return Err(err("Lookup window exceeds tensor or staging bounds"));
        }
        let mut unique = HashMap::new();
        let misses_before = self.lookup_misses;
        let mut rows = Vec::new();
        let mut order = Vec::with_capacity(ids.len());
        for &id in ids {
            let slot = if let Some(&slot) = unique.get(&id) {
                slot
            } else {
                let row = self.read_impl(name, id, 1, true, None)?.dense_pending()?;
                let slot = rows.len() as i32;
                rows.push(row);
                unique.insert(id, slot);
                slot
            };
            order.push(slot);
        }
        let bank = MxArray::concatenate_many(rows.iter().collect(), Some(0))?;
        let out = bank.take(&MxArray::from_int32(&order, &[order.len() as i64])?, 0)?;
        if self.lookup_misses != misses_before {
            MxArray::eval_arrays_with_context(&[&out], "qwen4::lookup_window")?;
        }
        self.lookup_batches += 1;
        Ok(out)
    }

    fn read_impl(
        &mut self,
        name: &str,
        start: usize,
        rows: usize,
        lookup: bool,
        prepared: Option<Option<super::packed_cache::VerifiedChunk>>,
    ) -> Result<Arc<Weight>> {
        if let Some(bank) = self.banks.get(name) {
            let d = self.descriptor(name)?;
            d.range(start, rows)?;
            return Ok(Arc::new(bank.slice_rows(start, rows)?));
        }
        let key = (name.to_string(), start, rows);
        if let Some(w) = self.cache.get(&key).cloned() {
            if !self.protected.contains(&key) {
                self.touch(key);
            }
            return Ok(w);
        }
        let d = self.descriptor(name)?.clone();
        let (offset, bytes) = d.range(start, rows)?;
        let cache_limit = self.cache_limit;
        // Refuse before allocation, also accounting for other models in this process.
        if crate::array::memory::get_active_memory() > (cache_limit + (4 << 30)) as f64 {
            return Err(err(
                "qwen4_exp SSD safety limit: active MLX memory exceeds weight budget plus 4 GiB",
            ));
        }
        super::memory::maintain_freelist(self.plan.physical_bytes);
        if cfg!(target_os = "macos") && self.growth_bytes >= 256 << 20 {
            super::memory::check_growth_headroom()?;
            self.growth_bytes = 0;
        }
        self.evict_to(cache_limit.saturating_sub(MAX_READ_BYTES * 3))?;
        // Do not persist sparse embedding/PLE rows: they are cheap to unpack,
        // and caching the enormous table would consume disk without reuse.
        let packed_key = d.packed_key(name, start, rows)?;
        let prepared = match prepared {
            Some(chunk) => chunk
                .map(super::packed_cache::VerifiedChunk::into_weight)
                .transpose()?,
            None => packed_key
                .as_ref()
                .and_then(|key| self.packed.as_ref()?.read(key).ok().flatten()),
        };
        let prepared_hit = prepared.is_some();
        let shape = [rows as i64, d.width()? as i64];
        let weight = if let Some((weight, bytes)) = prepared {
            self.packed_hits += 1;
            self.packed_bytes_read += bytes;
            weight
        } else {
            match &d.encoding {
                Encoding::Safe(dtype) => {
                    let mut f = File::open(&d.path).map_err(err)?;
                    f.seek(SeekFrom::Start(offset)).map_err(err)?;
                    let mut buf = vec![0; bytes as usize];
                    f.read_exact(&mut buf).map_err(err)?;
                    let values = match dtype.as_str() {
                        "BF16" | "F16" => {
                            let v: Vec<u16> = buf
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                                .collect();
                            if dtype == "BF16" {
                                MxArray::from_bfloat16(&v, &shape)?
                            } else {
                                MxArray::from_float16(&v, &shape)?
                            }
                        }
                        "F32" => MxArray::from_float32(
                            &buf.as_chunks::<4>()
                                .0
                                .iter()
                                .map(|b| f32::from_le_bytes(*b))
                                .collect::<Vec<_>>(),
                            &shape,
                        )?,
                        "I64" => MxArray::from_int64(
                            &buf.as_chunks::<8>()
                                .0
                                .iter()
                                .map(|b| i64::from_le_bytes(*b))
                                .collect::<Vec<_>>(),
                            &shape,
                        )?,
                        _ => unreachable!(),
                    };
                    Weight {
                        dense_bf16: None,
                        values,
                        scales: None,
                        biases: None,
                        group: 0,
                        bits: 0,
                        mode: String::new(),
                    }
                }
                Encoding::Gguf(ty) => {
                    let tensor = GgufTensorInfo {
                        name: "weight".into(),
                        n_dims: 2,
                        dims: vec![d.width()? as u64, rows as u64],
                        tensor_type: *ty,
                        offset: 0,
                    };
                    let g = GgufFile {
                        version: 3,
                        tensor_count: 1,
                        metadata: HashMap::new(),
                        tensors: vec![tensor],
                        alignment: 32,
                        data_offset: offset,
                    };
                    let mut a = load_gguf_tensors(
                        &d.path,
                        &g,
                        GgufLoadOptions {
                            verbose: false,
                            import_k_quants: true,
                        },
                    )?;
                    let values = a
                        .remove("weight")
                        .ok_or_else(|| err("GGUF row reader produced no weight"))?;
                    let scales = a.remove("weight.scales");
                    let mut biases = a.remove("weight.biases");
                    if let Some(zero) = symmetric_zero_point(*ty) {
                        biases = Some(
                            scales
                                .as_ref()
                                .ok_or_else(|| err("Missing affine scales"))?
                                .mul_scalar(-(zero as f64))?,
                        );
                    }
                    let (group, bits, mode) = if let Some(k) = ty.k_quant_format() {
                        (k.group_size() as i32, k.bits() as i32, k.mlx_mode())
                    } else {
                        (
                            32,
                            match ty {
                                GgufTensorType::Q5_1 => 5,
                                GgufTensorType::Q8_0 => 8,
                                _ => 4,
                            },
                            "affine",
                        )
                    };
                    Weight {
                        dense_bf16: None,
                        values,
                        scales,
                        biases,
                        group,
                        bits,
                        mode: mode.into(),
                    }
                }
            }
        };
        let arrays: Vec<_> = [
            Some(&weight.values),
            weight.scales.as_ref(),
            weight.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !lookup {
            MxArray::eval_arrays_with_context(&arrays, "qwen4::weights::loaded")?;
        }
        if !prepared_hit {
            self.bytes_read += bytes;
            if let (Some(cache), Some(key)) = (&self.packed, &packed_key) {
                // Cache I/O is optional. A read-only/full disk keeps the original
                // bounded GGUF path usable, and no partial entry is published.
                let _ = cache.write(key, &weight);
            }
        }
        // Embedding lookup already produces BF16 values. Retain that result
        // for a reused row instead of launching a dequantization on every
        // token/PLE access. Charge its actual bytes to the same bounded LRU;
        // expert matrices and persistent packed entries remain quantized.
        let weight = if rows == 1
            && weight.scales.is_some()
            && (name == "token_embd.weight"
                || name == "embed_tokens.weight"
                || name == "per_layer_token_embd.weight"
                || name.contains("ngram_embedding"))
            && !runtime_flags::is_zero(c"MLX_QWEN4_CACHE_EMBEDDING_ROWS")
        {
            self.lookup_misses += 1;
            Weight {
                dense_bf16: None,
                values: if lookup {
                    weight.dense_pending()?
                } else {
                    weight.dense()?
                },
                scales: None,
                biases: None,
                group: 0,
                bits: 0,
                mode: String::new(),
            }
        } else {
            weight
        };
        let weight = Arc::new(weight);
        // Always-used projections have a protected segment; experts and sparse
        // rows use the remaining LRU. Cap protection at 75%, so new reads retain
        // staging headroom even for a different/BF16 checkpoint topology.
        // Assembled banks and protected chunks share the pinned allowance.
        // Full residency reserves the whole main network during preparation;
        // partial residency keeps at least 25% available to the expert/row LRU.
        let protect_limit = if self.plan.resident {
            cache_limit.saturating_sub(self.plan.hot_bytes.max(self.bank_bytes)) * 3 / 4
        } else {
            (cache_limit.saturating_sub(self.slot_bytes) * 3 / 4).saturating_sub(self.bank_bytes)
        };
        if d.shape.len() <= 2
            && name != "token_embd.weight"
            && name != "embed_tokens.weight"
            && name != "per_layer_token_embd.weight"
            && !name.contains("ngram_embedding")
            && !name.contains(".experts.")
            && !name.contains("_exps.")
            && self.protected_bytes + weight.bytes()? <= protect_limit
        {
            self.protected.insert(key.clone());
            self.protected_bytes += weight.bytes()?;
        }
        self.cache_bytes += weight.bytes()?;
        self.growth_bytes += weight.bytes()?;
        self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
        if !self.protected.contains(&key) {
            self.touch(key.clone());
        }
        self.cache.insert(key, Arc::clone(&weight));
        Ok(weight)
    }
    pub fn dense(&mut self, name: &str) -> Result<MxArray> {
        let d = self.descriptor(name)?;
        let shape = d.shape.iter().map(|&i| i as i64).collect::<Vec<_>>();
        let rows = d.rows()?;
        self.read(name, 0, rows)?.dense()?.reshape(&shape)
    }
    pub fn linear(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        self.linear_impl(name, x, false)
    }

    /// Batch independent vectors while retaining the singleton projection's
    /// accumulation. PLE formerly used GEMV per token; ordinary prompt GEMM
    /// can round differently and change the subsequent greedy token stream.
    /// mlxfast TrackFastModel.pleForward uses the ordinary projection for a
    /// wide window. Keep its M/N/K matrix dispatch, with our existing GGUF
    /// loader, rather than broadcasting one GEMV for every prompt token.
    pub fn linear_ple_window(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        if self.gguf
            && x.dtype()? == DType::BFloat16
            && x.shape()?.len() == 3
            && x.shape()?[0] == 1
            && x.shape()?[1] > 8
            && runtime_flags::is_one(c"MLX_QWEN4_PREFILL_PLE_GEMM")
        {
            self.linear(name, x)
        } else {
            self.linear_vector_window(name, x)
        }
    }

    pub fn linear_vector_window(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        self.linear_impl(name, x, true)
    }

    fn linear_impl(&mut self, name: &str, x: &MxArray, vectors: bool) -> Result<MxArray> {
        let d = self.descriptor(name)?;
        let rows = d.rows()?;
        let shape = x.shape()?.to_vec();
        let (&width, outer_shape) = shape
            .split_last()
            .ok_or_else(|| err("Qwen4 projection input has no final dimension"))?;
        if width != d.width()? as i64 {
            return Err(err(format!("Projection shape mismatch: {name}")));
        }
        let count = x.size()? as i64 / width;
        let project = |w: &Weight| -> Result<MxArray> {
            if !vectors || count == 1 {
                return w.linear(x);
            }
            let input = x.reshape(&[count, 1, width])?;
            let output = if w.scales.is_some() {
                // One shared bank, with every RHS id zero. Gather dispatches
                // M=1 GEMV for each vector, not a wide prompt GEMM.
                w.gather_rows(
                    &input,
                    &MxArray::zeros(&[count], Some(DType::Uint32))?,
                    1,
                    false,
                )?
            } else {
                let parts = (0..count)
                    .map(|i| w.linear(&input.slice_axis(0, i, i + 1)?))
                    .collect::<Result<Vec<_>>>()?;
                MxArray::concatenate_many(parts.iter().collect(), Some(0))?
            };
            let mut out_shape = outer_shape.to_vec();
            out_shape.push(w.values.shape_at(0)?);
            output.reshape(&out_shape)
        };
        // Staging chunks bound SSD reads. A resident matrix can use a single
        // projection, particularly the large vocabulary head, without copying
        // weights or launching one kernel for every former read chunk.
        if let Some(bank) = self.banks.get(name) {
            return project(bank);
        }
        let chunk = (MAX_READ_BYTES as usize / (d.width()? * 4)).clamp(1, 4096);
        let mut outputs = Vec::new();
        for start in (0..rows).step_by(chunk) {
            let weight = self.read(name, start, (rows - start).min(chunk))?;
            outputs.push(project(&weight)?);
        }
        if outputs.len() == 1 {
            Ok(outputs.remove(0))
        } else {
            MxArray::concatenate_many(outputs.iter().collect(), Some(-1))
        }
    }
    pub fn integers(&self, name: &str) -> Result<Vec<u64>> {
        let d = self.descriptor(name)?;
        if !matches!(&d.encoding, Encoding::Safe(t) if t == "I64") || d.bytes > 4096 {
            return Err(err("Invalid PLE integer constants"));
        }
        let mut f = File::open(&d.path).map_err(err)?;
        f.seek(SeekFrom::Start(d.offset)).map_err(err)?;
        let mut b = vec![0; d.bytes as usize];
        f.read_exact(&mut b).map_err(err)?;
        Ok(b.as_chunks::<8>()
            .0
            .iter()
            .map(|x| u64::from_le_bytes(*x))
            .collect())
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn pq2_gguf_is_rejected_during_store_indexing() {
        let dir = std::env::temp_dir().join(format!("qwen4-pq2-reject-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let write = |path: &Path, name: &str, ty: GgufTensorType, index: u32, count: u32| {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.extend(3u64.to_le_bytes());
            string(&mut bytes, "general.architecture");
            bytes.extend(8u32.to_le_bytes());
            string(&mut bytes, "qwen4exp");
            for (key, value) in [("split.no", index), ("split.count", count)] {
                string(&mut bytes, key);
                bytes.extend(4u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            string(&mut bytes, name);
            bytes.extend(2u32.to_le_bytes());
            bytes.extend(128u64.to_le_bytes());
            bytes.extend(1u64.to_le_bytes());
            bytes.extend((ty as u32).to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            let payload_bytes = 128 / ty.block_size() * ty.type_size();
            bytes.resize(bytes.len().div_ceil(32) * 32 + payload_bytes, 0);
            fs::write(path, bytes).unwrap();
        };
        let supported = dir.join("supported.gguf");
        write(&supported, "supported.weight", GgufTensorType::Q4_0, 0, 1);
        let store = Store::open_metadata(&supported, Some(&dir.join("supported-cache"))).unwrap();
        assert_eq!(
            store
                .descriptor("supported.weight")
                .unwrap()
                .width()
                .unwrap(),
            128
        );
        assert_eq!(store.bytes_read, 0);
        let single = dir.join("single.gguf");
        write(&single, "test.pq2.weight", GgufTensorType::PQ2_0, 0, 1);
        let first = dir.join("split-00001-of-00002.gguf");
        let second = dir.join("split-00002-of-00002.gguf");
        write(&first, "supported.weight", GgufTensorType::Q4_0, 0, 2);
        write(&second, "test.pq2.weight", GgufTensorType::PQ2_0, 1, 2);
        for (label, path, pq2_path) in
            [("single", single.clone(), single), ("split", first, second)]
        {
            let parsed = parse_gguf(&pq2_path).unwrap();
            assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::PQ2_0);
            let cache = dir.join(format!("{label}-cache"));
            let error = match Store::open_metadata(&path, Some(&cache)) {
                Ok(_) => panic!("{label}: Qwen4 must reject PQ2 before loading payloads"),
                Err(error) => error,
            };
            assert!(error.reason.contains("PQ2_0"), "{}", error.reason);
            assert!(error.reason.contains("Qwen4"), "{}", error.reason);
            assert!(error.reason.contains("test.pq2.weight"), "{}", error.reason);
            assert!(
                !cache.exists(),
                "{label}: rejection must precede packed-cache setup"
            );
        }
    }

    #[test]
    fn pq2_runtime_residency_is_rejected() {
        let mut tensor = Tensor {
            path: PathBuf::new(),
            shape: vec![1, 128],
            offset: 0,
            bytes: 34,
            encoding: Encoding::Gguf(GgufTensorType::PQ2_0),
        };
        let error = tensor
            .runtime_bytes()
            .expect_err("Qwen4 must not budget unsupported PQ2");
        assert!(error.reason.contains("PQ2_0"), "{}", error.reason);
        tensor.encoding = Encoding::Gguf(GgufTensorType::Q4_0);
        tensor.bytes = 72;
        assert_eq!(tensor.runtime_bytes().unwrap(), 80);
    }

    #[test]
    fn descriptor_dimensions_reject_empty_zero_and_overflowing_shapes() {
        let mut tensor = Tensor {
            path: PathBuf::from("unused.safetensors"),
            shape: vec![2, 3],
            offset: 0,
            bytes: 24,
            encoding: Encoding::Safe("F32".into()),
        };
        assert_eq!(tensor.width().unwrap(), 3);
        assert_eq!(tensor.rows().unwrap(), 2);
        assert_eq!(tensor.range(1, 1).unwrap(), (12, 12));
        for shape in [vec![], vec![0], vec![0, 3], vec![2, 0], vec![usize::MAX, 2]] {
            tensor.shape = shape;
            assert!(tensor.rows().is_err());
            assert!(tensor.range(0, 1).is_err());
        }
    }

    #[test]
    fn fixture_forward_headroom_isolated_from_production_guard() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let fixture = Store::open_fixture(&path, None).unwrap();
        let production = Store::open_metadata(&path, None).unwrap();
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        fixture.check_forward_headroom_with(unavailable).unwrap();
        assert_eq!(checked.get(), 0, "tiny fixture queried production reserves");
        assert!(production.check_forward_headroom_with(unavailable).is_err());
        assert_eq!(checked.get(), 1, "production skipped its headroom check");
    }

    #[test]
    fn reference_shared_prefill_pair_preserves_projections_and_storage() {
        if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            return;
        }
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let (n, k) = (640i64, 2560i64);
        let make = |offset: u32| {
            Arc::new(Weight {
                dense_bf16: None,
                values: MxArray::from_uint32(
                    &(0..n * k / 4)
                        .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(offset))
                        .collect::<Vec<_>>(),
                    &[n, k / 4],
                )
                .unwrap(),
                scales: Some(
                    MxArray::from_float32(
                        &(0..n * k / 32)
                            .map(|i| ((i % 19) as f32 + 1.) / 4096.)
                            .collect::<Vec<_>>(),
                        &[n, k / 32],
                    )
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap(),
                ),
                biases: Some(
                    MxArray::from_float32(
                        &(0..n * k / 32)
                            .map(|i| -((i % 19) as f32 + 1.) / 32.)
                            .collect::<Vec<_>>(),
                        &[n, k / 32],
                    )
                    .unwrap()
                    .astype(DType::Float16)
                    .unwrap(),
                ),
                group: 32,
                bits: 8,
                mode: "affine".into(),
            })
        };
        let banks = [make(13), make(97)];
        let names = ["blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"];
        for (name, bank) in names.iter().zip(&banks) {
            store.tensors.insert(
                (*name).into(),
                Tensor {
                    path: PathBuf::new(),
                    shape: vec![n as usize, k as usize],
                    offset: 0,
                    bytes: (n * k) as u64,
                    encoding: Encoding::Gguf(GgufTensorType::Q8_0),
                },
            );
            store.banks.insert((*name).into(), bank.clone());
        }
        let bytes = banks[0].bytes().unwrap() + banks[1].bytes().unwrap();
        for rows in [511i64, 512, 1024] {
            let x = MxArray::from_float32(
                &(0..rows * k)
                    .map(|i| ((i * 17 % 251) as f32 - 125.) / 128.)
                    .collect::<Vec<_>>(),
                &[1, rows, k],
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
            let expected = [banks[0].linear(&x).unwrap(), banks[1].linear(&x).unwrap()];
            let (a, b) = store.linear_pair(&x, names[0], names[1]).unwrap();
            for (actual, want) in [&a, &b].into_iter().zip(&expected) {
                assert_eq!(
                    &*actual.to_float32().unwrap(),
                    &*want.to_float32().unwrap(),
                    "rows={rows}"
                );
            }
            let key = (names[0].to_owned(), names[1].to_owned());
            let enabled = !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_SHARED_PAIR")
                && !runtime_flags::is_zero(c"MLX_QWEN4_PAIRED_PROJECTIONS")
                && !runtime_flags::is_zero(c"MLX_ENABLE_TF32");
            if rows < 512 {
                assert!(!store.paired_banks.contains_key(&key));
            } else if enabled {
                assert_eq!(store.paired_banks[&key].bytes().unwrap(), bytes);
            }
        }
    }

    #[test]
    fn quantized_ple_windows_preserve_singleton_projection_rounding() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let (n, k) = (64i64, 2560i64);
        let scales = MxArray::from_float32(
            &(0..n * k / 32)
                .map(|i| ((i % 19) as f32 + 1.) / 4096.)
                .collect::<Vec<_>>(),
            &[n, k / 32],
        )
        .unwrap()
        .astype(DType::Float16)
        .unwrap();
        let bank = Arc::new(Weight {
            dense_bf16: None,
            values: MxArray::from_uint32(
                &(0..n * k / 4)
                    .map(|i| (i as u32).wrapping_mul(2654435761))
                    .collect::<Vec<_>>(),
                &[n, k / 4],
            )
            .unwrap(),
            biases: Some(scales.mul_scalar(-128.).unwrap()),
            scales: Some(scales),
            group: 32,
            bits: 8,
            mode: "affine".into(),
        });
        let name = "test.ple_key.weight";
        store.tensors.insert(
            name.into(),
            Tensor {
                path: PathBuf::new(),
                shape: vec![n as usize, k as usize],
                offset: 0,
                bytes: (n * k) as u64,
                encoding: Encoding::Gguf(GgufTensorType::Q8_0),
            },
        );
        store.banks.insert(name.into(), bank.clone());
        for dtype in [DType::BFloat16, DType::Float16, DType::Float32] {
            let x = MxArray::from_float32(
                &(0..17 * k)
                    .map(|i| (i as f32 * 0.13).sin())
                    .collect::<Vec<_>>(),
                &[1, 17, k],
            )
            .unwrap()
            .astype(dtype)
            .unwrap();
            let parts = (0..17)
                .map(|t| bank.linear(&x.slice_axis(1, t, t + 1).unwrap()).unwrap())
                .collect::<Vec<_>>();
            let expected = MxArray::concatenate_many(parts.iter().collect(), Some(1))
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let actual = store
                .linear_vector_window(name, &x)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(&*actual, &*expected, "dtype {dtype:?}");
        }
    }
    #[test]
    fn batched_lookup_reuses_decoded_rows_and_survives_eviction() {
        let dir = std::env::temp_dir().join(format!("qwen4-lookup-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(dir.clone());
        let path = dir.join("rows.gguf");
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        let string = |b: &mut Vec<u8>, s: &str| {
            b.extend((s.len() as u64).to_le_bytes());
            b.extend(s.bytes());
        };
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, "qwen4exp");
        string(&mut b, "token_embd.weight");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(8u64.to_le_bytes());
        b.extend(7u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for row in 0..8u32 {
            b.extend(half::f16::from_f32(0.015625).to_bits().to_le_bytes());
            b.extend(half::f16::from_f32(-0.25).to_bits().to_le_bytes());
            b.extend((row * 7919).to_le_bytes());
            b.extend([row as u8; 16]);
        }
        fs::write(&path, b).unwrap();
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        let mut oracle = Store::open_fixture(&path, Some(&dir)).unwrap();
        let ids = [3usize, 1, 3, 7];
        let expected = ids
            .iter()
            .map(|&id| {
                oracle
                    .read("token_embd.weight", id, 1)
                    .unwrap()
                    .dense()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let expected = MxArray::concatenate_many(expected.iter().collect(), Some(0))
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let cold = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!(store.bytes_read, 3 * 24);
        assert_eq!(store.lookup_misses, 3);
        assert_eq!(store.cache_bytes, 3 * 32 * 2);
        let warm = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!(store.lookup_batches, 2);
        assert_eq!(store.lookup_misses, 3);
        assert_eq!(store.bytes_read, 3 * 24);
        assert!(store.lookup_rows("token_embd.weight", &[0, 8]).is_err());
        assert_eq!(
            store.bytes_read,
            3 * 24,
            "validate the whole window before IO"
        );
        store.evict_to(0).unwrap();
        assert_eq!(store.cache_bytes, 0);
        for output in [cold, warm] {
            assert_eq!(
                &*output.astype(DType::Float32).unwrap().to_float32().unwrap(),
                &expected
            );
        }
    }

    #[test]
    fn partial_banks_share_the_protected_allowance() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        store.bank_bytes = store.cache_limit * 3 / 4 - 768;
        let descriptor = store
            .descriptor("layers.0.attn_hyper_connection.hc_norm.weight")
            .unwrap()
            .clone();
        for i in 0..32 {
            let name = format!("mtp.test_norm_{i}.weight");
            store.tensors.insert(name.clone(), descriptor.clone());
            store.read(&name, 0, descriptor.rows().unwrap()).unwrap();
        }
        assert!(store.protected_bytes > 0);
        assert!(store.protected_bytes <= 768);
        assert!(!store.order.is_empty());
    }

    #[test]
    fn auxiliary_protection_reserves_space_after_pinned_main_weights() {
        let mut store = Store::open_metadata(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        store.plan.resident = true;
        store.plan.hot_bytes = CACHE_BYTES - 1024;
        let descriptor = store
            .descriptor("layers.0.attn_hyper_connection.hc_norm.weight")
            .unwrap()
            .clone();
        for i in 0..32 {
            let name = format!("mtp.test_norm_{i}.weight");
            store.tensors.insert(name.clone(), descriptor.clone());
            store.read(&name, 0, descriptor.rows().unwrap()).unwrap();
        }
        assert!(store.protected_bytes > 0);
        assert!(store.protected_bytes <= 768);
        assert!(
            !store.order.is_empty(),
            "auxiliary weights must remain evictable"
        );
        store.evict_to(store.protected_bytes).unwrap();
        assert_eq!(store.cache_bytes, store.protected_bytes);
    }

    #[test]
    fn expert_lru_preserves_recency_and_bounds_hot_hit_metadata() {
        let mut store = Store::open_fixture(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp"),
            None,
        )
        .unwrap();
        let key = |row| ("embed_tokens.weight".to_string(), row, 1);
        store.read("embed_tokens.weight", 0, 1).unwrap();
        let row_bytes = store.cache_bytes;
        store.read("embed_tokens.weight", 1, 1).unwrap();
        for _ in 0..1000 {
            store.read("embed_tokens.weight", 0, 1).unwrap();
        }
        assert!(store.order.len() <= 64);
        store.evict_to(row_bytes).unwrap();
        assert!(store.cache.contains_key(&key(0)));
        assert!(!store.cache.contains_key(&key(1)));
        store.read("embed_tokens.weight", 2, 1).unwrap();
        store.evict_to(row_bytes).unwrap();
        assert!(store.cache.contains_key(&key(2)));
        assert!(!store.cache.contains_key(&key(0)));
        store.evict_to(0).unwrap();
        assert!(store.cache.is_empty());
    }
}
