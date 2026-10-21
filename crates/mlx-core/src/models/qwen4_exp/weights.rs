//! Bounded, positional SSD reads. Descriptors never create MLX arrays. Only
//! selected expert matrices and embedding rows are read on cache misses.
//! Admitted hot banks can be prepared once; PLE stays row-addressable. Routed
//! experts of a GGUF checkpoint also live in the mapped expert pack.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use napi::{Error, Result};
use serde::Deserialize;

use super::row_fetch::{Fetcher, Job};
use super::tripwire::{Fallback, Live, Tripwire};
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::QuantizedLinear;
use crate::utils::gguf::{
    GgufFile, GgufLoadOptions, GgufMetaValue, GgufTensorInfo, GgufTensorType, load_gguf_tensor,
    parse_gguf, symmetric_zero_point,
};

#[path = "expert_slots.rs"]
mod expert_slots;
#[path = "residency.rs"]
mod residency;

pub const CACHE_BYTES: u64 = 8 << 30;
pub(super) const MAX_READ_BYTES: u64 = 64 << 20;
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// Reader threads for token/PLE rows: 17 random 90 B-2.7 KB reads per token.
/// A layer-major prefill asks for a whole prompt's rows at once while the
/// pack stream yields to them: 32 readers finish a 6k prompt's 58,718 rows
/// in 0.27 s beside the stream, 4 in 1.17 s (P5/C2 M4).
const ROW_READERS: usize = 32;

/// Decoded-row LRU and reader-thread counters, shared with `residencyInfo()`.
#[derive(Default)]
pub struct LookupStats {
    /// Distinct rows asked of the decoded-row LRU per lookup window.
    pub lookups: AtomicU64,
    /// Lookups the LRU had to import.
    pub misses: AtomicU64,
    /// Imports whose bytes a reader thread had ready.
    pub prefetched: AtomicU64,
    /// Imports the model thread read itself or waited for after requesting.
    pub late: AtomicU64,
    /// Decoded rows the LRU bound evicted, and the bytes those evictions
    /// released: a window's shared buffer goes with its last row.
    pub evicted_rows: AtomicU64,
    pub evicted_row_bytes: AtomicU64,
}

impl LookupStats {
    fn count(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    pub fn evicted_rows(&self) -> u64 {
        self.evicted_rows.load(Ordering::Relaxed)
    }
    pub fn evicted_row_bytes(&self) -> u64 {
        self.evicted_row_bytes.load(Ordering::Relaxed)
    }
    pub fn lookups(&self) -> u64 {
        self.lookups.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
    pub fn prefetched(&self) -> u64 {
        self.prefetched.load(Ordering::Relaxed)
    }
    pub fn late(&self) -> u64 {
        self.late.load(Ordering::Relaxed)
    }
}

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
    /// Quantized multi-row GGUF matrices the mapped pack can hold; sparse
    /// embedding/PLE rows are cheap to unpack and never packed.
    fn packable(&self, name: &str, start: usize, rows: usize) -> Result<bool> {
        let (_, bytes) = self.range(start, rows)?;
        Ok(match self.encoding {
            Encoding::Gguf(ty) => {
                rows > 1
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
                    && name != "token_embd.weight"
            }
            Encoding::Safe(_) => false,
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
    pub(super) fn mixer_act(&self, x: &MxArray) -> Result<Option<MxArray>> {
        let decode = x.shape_at(1)? == 1;
        if self.mode != "affine"
            || self.bits != 8
            || self.group != 32
            || !crate::engine::persistence::compiled_forward_backend_available()
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

    pub(super) fn hyper_up(&self, x: &MxArray, normed: &MxArray) -> Result<Option<MxArray>> {
        if self.mode != "affine" || self.bits != 8 || self.group != 32 {
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
                return MxArray::from_handle(compact, "Qwen4 compact dense decode");
            }
        }
        let compact = if self.mode == "affine" && self.bits == 8 && self.group == 32 {
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
            {
                dense
            } else {
                &self.values
            };
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
            bytes += array_bytes(a)?;
        }
        Ok(bytes)
    }
}

fn array_bytes(a: &MxArray) -> Result<u64> {
    Ok(a.size()?
        * match a.dtype()? {
            DType::Uint8 | DType::Int8 => 1,
            DType::Float16 | DType::BFloat16 => 2,
            _ => 4,
        })
}

/// One lookup window's Metal buffer, shared by the rows it imported. It is
/// released with its last row view, so it is charged whole until then.
struct WindowBank {
    bytes: u64,
    views: usize,
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
    mtp_banks_declined: bool,
    /// Draft-bank bytes the mapped plan budgeted beside the dense banks; the
    /// bank admission honours them, the slot path leaves them at zero.
    draft_bank_budget: u64,
    paired_banks: HashMap<(String, String), Arc<Weight>>,
    slots: HashMap<usize, expert_slots::ExpertSlots>,
    mapped_experts: HashMap<usize, expert_slots::MappedExperts>,
    /// Start of the last forward pass in mapped mode (bank re-wire gate).
    mapped_pass_start: Option<std::time::Instant>,
    prefetch: Option<super::prefetch::Pool>,
    /// The turn opened while the pack streams: its passes do not touch, and
    /// the layers already run are kept wired instead.
    streaming_pass: bool,
    last_touch: Option<std::time::Instant>,
    /// A layer-major prefill group runs: the banks of the layers already
    /// swept are kept wired until it commits, through asynchronous touches
    /// whose arrays wait in `pending_touch` for that commit.
    group_in_progress: bool,
    pending_touch: Vec<MxArray>,
    /// Layers covered by every bank touch, in order.
    #[cfg(test)]
    pub(in super::super) touches: Vec<Vec<usize>>,
    /// Whether each touch in `touches` was submitted asynchronously.
    #[cfg(test)]
    pub(in super::super) touches_async: Vec<bool>,
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
    mapped: Option<super::mapped_pack::MappedPack>,
    /// One open descriptor per checkpoint file; row reads `pread` through it.
    files: HashMap<PathBuf, Arc<File>>,
    rows: Option<Fetcher>,
    /// Rows the reader threads or the model thread still have to read;
    /// the mapped pack's stream yields while it is above zero.
    rows_pending: Arc<AtomicUsize>,
    #[cfg(test)]
    pub(in super::super) row_readers: bool,
    /// Rows a `linear_rows` window projected one at a time (no row-batched
    /// kernel admitted the weight); a prefill window must never add to it.
    #[cfg(test)]
    pub(in super::super) single_row_projections: u64,
    /// (vectors, row-kernel calls) of every vector-window projection.
    #[cfg(test)]
    pub(in super::super) vector_windows: Vec<(i64, u64)>,
    /// Measurement hook: `false` leaves every imported row in its own
    /// buffer (the pre-fold shape) so a test can time the fold alone.
    #[cfg(test)]
    pub(in super::super) fold_rows: bool,
    pub lookup: Arc<LookupStats>,
    pub live: Arc<Live>,
    tripwire: Tripwire,
    window_banks: HashMap<u64, WindowBank>,
    row_bank: HashMap<(String, usize, usize), u64>,
    next_window_bank: u64,
}

#[derive(Deserialize)]
struct SafeDescriptor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl Store {
    pub(super) fn check_forward_headroom(&self) -> Result<()> {
        let mapped = self.plan.mapped;
        self.check_forward_headroom_with(|| super::memory::check_growth_headroom(mapped))
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
        let root = root.map_or_else(Self::fixture_pack_root, Path::to_path_buf);
        let mut store = Self::open_metadata(path, Some(&root))?;
        store.fixture = true;
        // Logical ceiling includes the importer staging allowance; the
        // synthetic tensors allocate only their actual payload sizes.
        store.cache_limit = 1 << 30;
        store.plan.budget = store.cache_limit;
        store.plan.hot_bytes = store.hot_bytes()?;
        Ok(store)
    }

    /// `open_metadata`'s `None` resolves the user's real pack cache, whose
    /// sweep deletes idle packs of other builds and processes; a unit test
    /// must never reach it. The root is never created.
    #[cfg(test)]
    fn fixture_pack_root() -> PathBuf {
        std::env::temp_dir().join(format!("mlx-qwen4-fixture-packs-{}", uuid::Uuid::new_v4()))
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
                available_bytes: None,
                physical_bytes: None,
                resident_bank_bytes: Default::default(),
                partial_expert_slots: 0,
                mapped_pack_files: 0,
                mapped: false,
                mapped_bytes: 0,
                mapped_bank_budget: 0,
                live_non_reclaimable: 0,
            },
            banks: HashMap::new(),
            bank_bytes: 0,
            mtp_banks_declined: false,
            draft_bank_budget: 0,
            paired_banks: HashMap::new(),
            slots: HashMap::new(),
            mapped_experts: HashMap::new(),
            mapped_pass_start: None,
            prefetch: None,
            streaming_pass: false,
            last_touch: None,
            group_in_progress: false,
            pending_touch: Vec::new(),
            #[cfg(test)]
            touches: Vec::new(),
            #[cfg(test)]
            touches_async: Vec::new(),
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
            mapped: None,
            files: HashMap::new(),
            rows: None,
            rows_pending: Arc::default(),
            #[cfg(test)]
            row_readers: true,
            #[cfg(test)]
            single_row_projections: 0,
            #[cfg(test)]
            vector_windows: Vec::new(),
            #[cfg(test)]
            fold_rows: true,
            lookup: Arc::default(),
            live: Arc::default(),
            tripwire: Tripwire::default(),
            window_banks: HashMap::new(),
            row_bank: HashMap::new(),
            next_window_bank: 0,
        };
        if s.gguf {
            s.index_gguf(path)?;
        } else {
            s.index_safe(path)?;
        }
        if s.tensors.is_empty() {
            return Err(err("No model tensors found"));
        }
        if s.gguf {
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
            if let Some((_, freed)) = self.uncache(&old)? {
                self.cache_bytes -= freed;
                if old.2 == 1 {
                    LookupStats::count(&self.lookup.evicted_rows);
                    self.lookup
                        .evicted_row_bytes
                        .fetch_add(freed, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }
    /// Drop a cache entry and return it with the bytes its removal released:
    /// a row view of a window's buffer releases that buffer's bytes with the
    /// last view and nothing before, every other entry its own bytes.
    pub(super) fn uncache(
        &mut self,
        key: &(String, usize, usize),
    ) -> Result<Option<(Arc<Weight>, u64)>> {
        let Some(w) = self.cache.remove(key) else {
            return Ok(None);
        };
        let freed = match self.row_bank.remove(key) {
            Some(id) => {
                let bank = self
                    .window_banks
                    .get_mut(&id)
                    .ok_or_else(|| err("Qwen4 decoded row names a released window buffer"))?;
                bank.views -= 1;
                if bank.views == 0 {
                    let bytes = bank.bytes;
                    self.window_banks.remove(&id);
                    bytes
                } else {
                    0
                }
            }
            None => w.bytes()?,
        };
        Ok(Some((w, freed)))
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
        // Relative, absolute and symlink aliases must share one source
        // identity and one descriptor. Resolve once while indexing.
        t.path = t.path.canonicalize().map_err(err)?;
        if self.tensors.insert(name.clone(), t).is_some() {
            return Err(err(format!("Duplicate tensor {name}")));
        }
        Ok(())
    }
    fn index_safe(&mut self, dir: &Path) -> Result<()> {
        let root = dir.canonicalize().map_err(err)?;
        // This store reads packed K-quant rows directly off the file (row
        // windows, no `QuantizedLinear`), so an artifact whose linears were
        // written in the 64-row tiled layout would decode garbage. Refuse by
        // the config marker before indexing anything.
        if let Ok(raw) = fs::read(dir.join("config.json"))
            && let Ok(config) = serde_json::from_slice::<serde_json::Value>(&raw)
            && let Some(key) =
                crate::models::quant_dispatch::config_declares_tiled_kquant_layout(&config)
        {
            return Err(err(format!(
                "Qwen4 safetensors store cannot read '{key}': the checkpoint stores K-quant \
                 linears in the 64-row tiled layout (quantization entry layout=t64), which this \
                 row-window reader does not decode; load the GGUF source instead"
            )));
        }
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
    #[cfg(test)]
    pub(super) fn lru_holds(&self, name: &str) -> bool {
        self.cache.keys().any(|(n, _, _)| n == name)
    }
    #[cfg(test)]
    pub(super) fn cached_bytes(&self) -> u64 {
        self.cache_bytes
    }
    #[cfg(test)]
    pub(super) fn set_cache_limit(&mut self, bytes: u64) {
        self.cache_limit = bytes;
        self.plan.budget = bytes;
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
    /// stay on the inference thread in original request order. Whole-expert
    /// batches of packed tensors come from the mapped pack by range; anything
    /// else is imported from the checkpoint.
    fn read_expert_batch(
        &mut self,
        requests: &[(String, usize, usize)],
    ) -> Result<Vec<Arc<Weight>>> {
        if self.mapped.is_none() {
            return requests
                .iter()
                .map(|(name, start, rows)| self.read(name, *start, *rows))
                .collect();
        }
        let mut indices = Vec::new();
        let mut total = 0u64;
        for (i, (name, start, rows)) in requests.iter().enumerate() {
            if self.banks.contains_key(name)
                || self.cache.contains_key(&(name.clone(), *start, *rows))
            {
                continue;
            }
            let d = self.descriptor(name)?;
            if d.packable(name, *start, *rows)? {
                let bytes = (d.runtime_bytes()? / d.rows()? as u64).saturating_mul(*rows as u64);
                total = total.saturating_add(bytes);
                indices.push(i);
            }
        }
        if total > MAX_READ_BYTES || indices.len() > 24 || indices.len() < 2 {
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
        let subset: Vec<_> = indices.iter().map(|&i| requests[i].clone()).collect();
        let verified = self
            .mapped
            .as_ref()
            .map(|pack| pack.read_batch(&subset))
            .transpose()?
            .flatten();
        let mut prepared: Vec<_> = (0..requests.len()).map(|_| None).collect();
        if let Some(chunks) = verified {
            for (i, chunk) in indices.into_iter().zip(chunks) {
                prepared[i] = Some(chunk);
            }
        }
        requests
            .iter()
            .zip(prepared)
            .map(|((name, start, rows), chunk)| self.read_impl(name, *start, *rows, false, chunk))
            .collect()
    }

    /// Open the mapped expert pack, building whatever is missing from the
    /// checkpoint. A failure leaves the checkpoint reader in charge; it
    /// never fails the load.
    pub(in super::super) fn prepare_mapped_pack(&mut self) -> Result<()> {
        let dir = match &self.packed {
            Some(packed) if self.gguf => packed.mapped_dir().to_path_buf(),
            _ => return Ok(()),
        };
        self.prepare_mapped_pack_at(&dir)
    }

    fn prepare_mapped_pack_at(&mut self, dir: &Path) -> Result<()> {
        let Some(packed) = &self.packed else {
            return Ok(());
        };
        let Some(identity) = packed.identity() else {
            return Ok(());
        };
        let mut routed: Vec<(usize, usize, String, Tensor)> = Vec::new();
        for (name, tensor) in &self.tensors {
            let Some((layer, part)) = routed_expert_tensor(name) else {
                continue;
            };
            if tensor.shape.len() != 3
                || tensor.shape[0] == 0
                || !tensor.packable(name, 0, tensor.shape[1])?
            {
                continue;
            }
            routed.push((layer, part, name.clone(), tensor.clone()));
        }
        if routed.is_empty() {
            return Ok(());
        }
        routed.sort_by_key(|r| (r.0, r.1));
        let sources: Vec<RoutedExpertSource> = routed
            .into_iter()
            .map(|(_, _, name, tensor)| RoutedExpertSource { name, tensor })
            .collect();
        let dyn_sources: Vec<&dyn super::mapped_pack::ExpertSource> = sources
            .iter()
            .map(|s| s as &dyn super::mapped_pack::ExpertSource)
            .collect();
        let started = std::time::Instant::now();
        match super::mapped_pack::MappedPack::open_or_build(
            dir,
            identity,
            Some(packed.manifest()),
            &dyn_sources,
        ) {
            Ok((pack, report)) => {
                tracing::info!(
                    "qwen4 mapped pack: {} opened, {} built, {} unavailable in {:.1} s at {}",
                    report.opened,
                    report.built,
                    report.skipped,
                    started.elapsed().as_secs_f64(),
                    pack.dir().display()
                );
                self.plan.mapped_pack_files = pack.len();
                self.mapped = Some(pack);
            }
            Err(e) => {
                tracing::warn!(
                    "qwen4 mapped pack unavailable, importing experts from the checkpoint: {e}"
                );
            }
        }
        Ok(())
    }

    /// Gather a bounded lookup window, sharing duplicate rows. Decoded rows
    /// use the existing Store-owned LRU and its byte accounting, so no second
    /// cache is reserved. The gather stays lazy: the layer that consumes it
    /// submits the imports and dequantizations with its own graph instead of
    /// draining the GPU at the token start.
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
        let mut rows = Vec::new();
        let mut order = Vec::with_capacity(ids.len());
        let mut imported = Vec::new();
        let banked = self.banks.contains_key(name);
        for &id in ids {
            let slot = if let Some(&slot) = unique.get(&id) {
                slot
            } else {
                LookupStats::count(&self.lookup.lookups);
                let miss = !banked && !self.cache.contains_key(&(name.to_string(), id, 1));
                if miss {
                    LookupStats::count(&self.lookup.misses);
                }
                let row = self.read_impl(name, id, 1, true, None)?.dense_pending()?;
                let slot = rows.len();
                rows.push(row);
                if miss {
                    imported.push((id, slot));
                }
                unique.insert(id, slot);
                slot
            };
            order.push(slot as i32);
        }
        if let [row] = rows.as_slice()
            && order.len() == 1
        {
            return Ok(row.clone());
        }
        // The rows this window imported and still holds share one Metal
        // buffer: each buffer is a Metal resource and MLX aborts at ~499k of
        // them, which one buffer per row reached on the third fresh
        // 10k-token prompt. Rows the LRU already held keep their buffer, so
        // no buffer carries a slot nothing points to, and a window's buffer
        // is charged whole until its last row view is evicted. The window
        // then takes that buffer as ONE concatenation input beside the held
        // rows: one input per imported row again would add a copy dispatch
        // per import to every layer of every decode token (measured 11 us
        // per 2-hit/6-miss lookup, 0.5 ms per token over 48 layers).
        imported.retain(|&(id, _)| self.cache.contains_key(&(name.to_string(), id, 1)));
        #[cfg(test)]
        let fold = self.fold_rows;
        #[cfg(not(test))]
        let fold = true;
        if fold && imported.len() > 1 {
            let fresh = MxArray::concatenate_many(
                imported.iter().map(|&(_, slot)| &rows[slot]).collect(),
                Some(0),
            )?;
            let bytes = array_bytes(&fresh)?;
            let id = self.next_window_bank;
            self.next_window_bank += 1;
            let mut rehomed = 0;
            let mut position = vec![0i32; rows.len()];
            for (i, &(row, slot)) in imported.iter().enumerate() {
                let key = (name.to_string(), row, 1);
                let view = fresh.slice_axis(0, i as i64, i as i64 + 1)?;
                let entry = self
                    .cache
                    .get_mut(&key)
                    .ok_or_else(|| err("Qwen4 decoded row left the LRU inside its window"))?;
                rehomed += entry.bytes()?;
                *entry = Arc::new(Weight {
                    values: view,
                    dense_bf16: None,
                    scales: None,
                    biases: None,
                    group: 0,
                    bits: 0,
                    mode: String::new(),
                });
                self.row_bank.insert(key, id);
                position[slot] = i as i32;
            }
            self.window_banks.insert(
                id,
                WindowBank {
                    bytes,
                    views: imported.len(),
                },
            );
            self.cache_bytes = self.cache_bytes - rehomed + bytes;
            self.peak_cache_bytes = self.peak_cache_bytes.max(self.cache_bytes);
            let mut parts = vec![&fresh];
            for (slot, row) in rows.iter().enumerate() {
                if !imported.iter().any(|&(_, s)| s == slot) {
                    position[slot] = (imported.len() + parts.len() - 1) as i32;
                    parts.push(row);
                }
            }
            let order: Vec<i32> = order.iter().map(|&slot| position[slot as usize]).collect();
            let window = if parts.len() == 1 {
                fresh
            } else {
                MxArray::concatenate_many(parts, Some(0))?
            };
            return window.take(&MxArray::from_int32(&order, &[order.len() as i64])?, 0);
        }
        let window = MxArray::concatenate_many(rows.iter().collect(), Some(0))?;
        window.take(&MxArray::from_int32(&order, &[order.len() as i64])?, 0)
    }

    /// Start reading rows `(tensor, row)` on the reader threads; rows the
    /// LRU or a bank already holds are skipped. `read_impl` takes the bytes
    /// when it imports the row.
    pub fn prefetch_rows(&mut self, keys: &[(String, usize)]) -> Result<()> {
        #[cfg(test)]
        if !self.row_readers {
            return Ok(());
        }
        let mut seen = HashSet::new();
        let mut jobs = Vec::new();
        for (name, row) in keys {
            if !seen.insert((name.as_str(), *row))
                || self.banks.contains_key(name)
                || self.cache.contains_key(&(name.clone(), *row, 1))
            {
                continue;
            }
            let d = self.descriptor(name)?.clone();
            let (offset, len) = d.range(*row, 1)?;
            let file = self.file(&d.path)?;
            jobs.push(Job {
                key: (name.clone(), *row),
                file,
                offset,
                len: len as usize,
            });
        }
        if jobs.is_empty() {
            return Ok(());
        }
        // One ascending sweep per table instead of a random walk of the
        // checkpoint: the rows of a window are read in file order.
        jobs.sort_by_key(|job| (Arc::as_ptr(&job.file) as usize, job.offset));
        if self.rows.is_none() {
            self.rows =
                Some(Fetcher::new(ROW_READERS, Arc::clone(&self.rows_pending)).map_err(err)?);
        }
        if let Some(fetcher) = &self.rows {
            fetcher.request(jobs);
        }
        Ok(())
    }

    /// Finish the reads in flight, then stop the reader threads: the rows
    /// they fetched stay ready, a row requested later is read late.
    #[cfg(test)]
    pub(in super::super) fn pause_rows(&mut self) {
        if let Some(fetcher) = &mut self.rows {
            fetcher.wait_idle();
            fetcher.pause();
        }
    }

    fn file(&mut self, path: &Path) -> Result<Arc<File>> {
        if let Some(file) = self.files.get(path) {
            return Ok(Arc::clone(file));
        }
        let file = Arc::new(File::open(path).map_err(err)?);
        self.files.insert(path.to_path_buf(), Arc::clone(&file));
        Ok(file)
    }

    /// Import `rows` rows of `d`: bytes a reader thread fetched for a single
    /// row, else a `pread` through the shared descriptor. A reader's error is
    /// dropped, never returned: it may be older than this lookup, and only the
    /// read done now says whether the row is readable.
    fn import_rows(&mut self, d: &Tensor, name: &str, start: usize, rows: usize) -> Result<Weight> {
        let fetched = if rows == 1 {
            self.rows
                .as_ref()
                .and_then(|fetcher| fetcher.take(&(name.to_string(), start)))
        } else {
            None
        };
        let bytes = match fetched {
            Some((Ok(bytes), late)) => {
                LookupStats::count(if late {
                    &self.lookup.late
                } else {
                    &self.lookup.prefetched
                });
                bytes
            }
            fetched => {
                if fetched.is_some() {
                    LookupStats::count(&self.lookup.late);
                }
                let file = self.file(&d.path)?;
                d.read_rows(&file, start, rows)?
            }
        };
        d.import_bytes(&bytes, rows)
    }

    fn read_impl(
        &mut self,
        name: &str,
        start: usize,
        rows: usize,
        lookup: bool,
        prepared: Option<super::packed_cache::VerifiedChunk>,
    ) -> Result<Arc<Weight>> {
        let mapped = self.plan.mapped;
        self.read_impl_with(name, start, rows, lookup, prepared, || {
            super::memory::check_growth_headroom(mapped)
        })
    }

    fn read_impl_with(
        &mut self,
        name: &str,
        start: usize,
        rows: usize,
        lookup: bool,
        prepared: Option<super::packed_cache::VerifiedChunk>,
        growth_check: impl FnOnce() -> Result<()>,
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
        let (_, bytes) = d.range(start, rows)?;
        let cache_limit = self.cache_limit;
        // Refuse before allocation, also accounting for other models in this
        // process. A freelist spike is transient, so reclaim it and re-poll
        // briefly under the same safety limit before refusing.
        super::memory::check_active_memory_budget(
            cache_limit,
            || {
                crate::array::memory::get_active_memory()
                    - crate::array::mapped_array_bytes() as f64
            },
            crate::array::memory::clear_cache,
            std::thread::sleep,
        )?;
        super::memory::maintain_freelist(self.plan.physical_bytes);
        // Mapped experts never pass through here, so growth_bytes holds only
        // anonymous bytes: token/PLE rows, MTP/vision weights and fallback
        // projection chunks still consume the system reserve in mapped mode.
        if cfg!(target_os = "macos") && self.growth_bytes >= 256 << 20 {
            growth_check()?;
            self.growth_bytes = 0;
        }
        self.evict_to(cache_limit.saturating_sub(MAX_READ_BYTES * 3))?;
        let prepared = prepared
            .map(super::packed_cache::VerifiedChunk::into_weight)
            .transpose()?;
        let prepared_hit = prepared.is_some();
        let weight = if let Some((weight, bytes)) = prepared {
            self.packed_hits += 1;
            self.packed_bytes_read += bytes;
            weight
        } else {
            self.import_rows(&d, name, start, rows)?
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
        }
        self.retain_read(name, key, weight, lookup, rows)
    }

    /// Cache, protect and account a freshly imported chunk.
    fn retain_read(
        &mut self,
        name: &str,
        key: (String, usize, usize),
        weight: Weight,
        lookup: bool,
        rows: usize,
    ) -> Result<Arc<Weight>> {
        let cache_limit = self.cache_limit;
        let d = self.descriptor(name)?.clone();
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
        {
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
        // Assembled banks and protected chunks share the pinned allowance;
        // at least 25% stays available to the expert/row LRU.
        let protect_limit =
            (cache_limit.saturating_sub(self.slot_bytes) * 3 / 4).saturating_sub(self.bank_bytes);
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
}

impl Tensor {
    /// Import `rows` rows from `start` straight from the checkpoint: raw
    /// safetensors bytes, or GGUF blocks through the K-quant repack. No cache
    /// or accounting is touched; the arrays are lazy.
    pub(super) fn import_rows(&self, start: usize, rows: usize) -> Result<Weight> {
        let file = File::open(&self.path).map_err(err)?;
        self.import_bytes(&self.read_rows(&file, start, rows)?, rows)
    }

    /// The checkpoint bytes of `rows` rows from `start`, through an open
    /// descriptor of the tensor's file.
    pub(super) fn read_rows(&self, file: &File, start: usize, rows: usize) -> Result<Vec<u8>> {
        let (offset, bytes) = self.range(start, rows)?;
        let mut buf = vec![0; bytes as usize];
        file.read_exact_at(&mut buf, offset).map_err(err)?;
        Ok(buf)
    }

    /// Decode the checkpoint bytes of `rows` consecutive rows into lazy arrays.
    pub(super) fn import_bytes(&self, buf: &[u8], rows: usize) -> Result<Weight> {
        if rows == 0 || buf.len() as u64 != self.row_bytes()?.saturating_mul(rows as u64) {
            return Err(err("SSD row payload does not match its descriptor"));
        }
        let shape = [rows as i64, self.width()? as i64];
        Ok(match &self.encoding {
            Encoding::Safe(dtype) => {
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
                    dims: vec![self.width()? as u64, rows as u64],
                    tensor_type: *ty,
                    offset: 0,
                };
                let g = GgufFile {
                    version: 3,
                    tensor_count: 1,
                    metadata: HashMap::new(),
                    tensors: Vec::new(),
                    alignment: 32,
                    data_offset: 0,
                };
                let mut a: HashMap<String, MxArray> = load_gguf_tensor(
                    &mut Cursor::new(buf),
                    &g,
                    &tensor,
                    GgufLoadOptions {
                        verbose: false,
                        import_k_quants: true,
                    },
                )?
                .into_iter()
                .collect();
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
        })
    }
}

impl Store {
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
    /// accumulation: an ordinary prompt GEMM can round differently and change
    /// the subsequent greedy token stream. An admitted Q8 bank runs four
    /// vectors per call of the row-exact decode kernel, every row with the
    /// bits of its single-row projection.
    pub fn linear_ple_window(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        self.linear_vector_window(name, x)
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
        #[cfg(test)]
        let dense_row_calls = std::cell::Cell::new(0u64);
        let project = |w: &Weight| -> Result<MxArray> {
            if !vectors || count == 1 {
                return w.linear(x);
            }
            let mut out_shape = outer_shape.to_vec();
            out_shape.push(w.values.shape_at(0)?);
            let window = x.reshape(&[1, count, width])?;
            let step = super::math::rows::MAX_ROWS;
            let mut parts = Vec::with_capacity(((count + step - 1) / step) as usize);
            for start in (0..count).step_by(step as usize) {
                let rows = window.slice_axis(1, start, (start + step).min(count))?;
                match super::math::rows::dense_decode(w, &rows)? {
                    Some(out) => parts.push(out),
                    None if parts.is_empty() => break,
                    None => return Err(err("Qwen4 row kernel refused part of a vector window")),
                }
                #[cfg(test)]
                dense_row_calls.set(dense_row_calls.get() + 1);
            }
            if !parts.is_empty() {
                return MxArray::concatenate_many(parts.iter().collect(), Some(1))?
                    .reshape(&out_shape);
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
            output.reshape(&out_shape)
        };
        // Staging chunks bound SSD reads. A resident matrix can use a single
        // projection, particularly the large vocabulary head, without copying
        // weights or launching one kernel for every former read chunk.
        let output = if let Some(bank) = self.banks.get(name) {
            project(bank)
        } else {
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
        };
        #[cfg(test)]
        if vectors {
            self.vector_windows.push((count, dense_row_calls.get()));
        }
        output
    }
    /// `x` is `[1, R, K]`, `R <= 4`. Row `j` of the result carries the bits of
    /// `linear` applied to row `j` alone: an admitted Q8 bank streams once
    /// through the row-batched kernel, every other weight runs its single-row
    /// projection per row.
    pub(super) fn linear_rows(&mut self, name: &str, x: &MxArray) -> Result<MxArray> {
        let count = x.shape_at(1)?;
        if count == 1 {
            return self.linear(name, x);
        }
        let d = self.descriptor(name)?;
        let rows = d.rows()?;
        let width = d.width()?;
        if x.shape()?.last().copied() != Some(width as i64) {
            return Err(err(format!("Projection shape mismatch: {name}")));
        }
        let chunk = (MAX_READ_BYTES as usize / (width * 4)).clamp(1, 4096);
        #[cfg(test)]
        let single_rows = std::cell::Cell::new(0u64);
        let project = |w: &Weight| -> Result<MxArray> {
            if let Some(out) = super::math::rows::dense_decode(w, x)? {
                return Ok(out);
            }
            #[cfg(test)]
            single_rows.set(single_rows.get() + count as u64);
            let parts = (0..count)
                .map(|j| w.linear(&x.slice_axis(1, j, j + 1)?))
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(parts.iter().collect(), Some(1))
        };
        let output = if let Some(bank) = self.banks.get(name) {
            project(bank)?
        } else {
            let mut outputs = Vec::new();
            for start in (0..rows).step_by(chunk) {
                let weight = self.read(name, start, (rows - start).min(chunk))?;
                outputs.push(project(&weight)?);
            }
            if outputs.len() == 1 {
                outputs.remove(0)
            } else {
                MxArray::concatenate_many(outputs.iter().collect(), Some(-1))?
            }
        };
        #[cfg(test)]
        {
            self.single_row_projections += single_rows.get();
        }
        Ok(output)
    }

    /// `linear_pair` for a `[1, R, K]` window: the paired bank prepared by
    /// single-row decoding streams once, otherwise each projection runs
    /// through `linear_rows`.
    pub(super) fn linear_pair_rows(
        &mut self,
        x: &MxArray,
        first: &str,
        second: &str,
    ) -> Result<(MxArray, MxArray)> {
        if x.shape_at(1)? == 1 {
            return self.linear_pair(x, first, second);
        }
        let key = (first.to_owned(), second.to_owned());
        if let Some(bank) = self.paired_banks.get(&key)
            && let Some(output) = super::math::rows::dense_decode(bank, x)?
        {
            let n = self.descriptor(first)?.rows()? as i64;
            let axis = output.ndim()? as usize - 1;
            return Ok((
                output.slice_axis(axis, 0, n)?,
                output.slice_axis(axis, n, output.shape()?[axis])?,
            ));
        }
        Ok((self.linear_rows(first, x)?, self.linear_rows(second, x)?))
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

/// `blk.{layer}.ffn_{gate|up|down}_exps.weight` -> `(layer, part)`.
fn routed_expert_tensor(name: &str) -> Option<(usize, usize)> {
    let (layer, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    let part = [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ]
    .iter()
    .position(|p| *p == rest)?;
    Some((layer, part))
}

/// One routed expert tensor as the mapped-pack builder imports it.
struct RoutedExpertSource {
    name: String,
    tensor: Tensor,
}

impl super::mapped_pack::ExpertSource for RoutedExpertSource {
    fn name(&self) -> &str {
        &self.name
    }
    fn experts(&self) -> usize {
        self.tensor.shape[0]
    }
    fn rows_per_expert(&self) -> usize {
        self.tensor.shape[1]
    }
    fn width(&self) -> usize {
        self.tensor.shape[2]
    }
    fn import(&self, expert: usize) -> Result<super::mapped_pack::ExpertRecord> {
        let rows = self.rows_per_expert();
        super::mapped_pack::ExpertRecord::from_weight(
            &self.tensor.import_rows(expert * rows, rows)?,
        )
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
    fn mapped_store_keeps_the_forward_headroom_guard() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut production = Store::open_metadata(&path, None).unwrap();
        production.plan.mapped = true;
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        assert!(production.check_forward_headroom_with(unavailable).is_err());
        assert_eq!(
            checked.get(),
            1,
            "mapped store skipped the forward headroom check"
        );
    }

    /// Only the routed experts are file-backed in mapped mode; every other
    /// read is anonymous growth and keeps the 256 MiB system-headroom recheck.
    #[test]
    fn mapped_store_rechecks_growth_headroom_for_anonymous_rows() {
        if !cfg!(target_os = "macos") {
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut store = Store::open_metadata(&path, None).unwrap();
        store.plan.mapped = true;
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        let row = store
            .read_impl_with("embed_tokens.weight", 0, 1, false, None, unavailable)
            .unwrap();
        assert_eq!(
            checked.get(),
            0,
            "a read under 256 MiB of growth queried headroom"
        );
        assert_eq!(
            store.growth_bytes,
            row.bytes().unwrap(),
            "mapped mode must charge anonymous rows as growth"
        );
        store.growth_bytes = 256 << 20;
        let error =
            match store.read_impl_with("embed_tokens.weight", 1, 1, false, None, unavailable) {
                Ok(_) => panic!("mapped store ignored exhausted growth headroom"),
                Err(error) => error,
            };
        assert!(error.reason.contains("simulated"), "{}", error.reason);
        assert_eq!(checked.get(), 1);
        assert_eq!(
            store.growth_bytes,
            256 << 20,
            "a refused read must leave the growth pending"
        );
        assert!(
            !store
                .cache
                .contains_key(&("embed_tokens.weight".to_string(), 1, 1))
        );
        store
            .read_impl_with("embed_tokens.weight", 1, 1, false, None, || Ok(()))
            .unwrap();
        assert_eq!(
            store.growth_bytes,
            row.bytes().unwrap(),
            "a passing check restarts the growth count"
        );
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
            if rows < 512 {
                assert!(!store.paired_banks.contains_key(&key));
            } else if super::super::tf32_enabled() {
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
    struct Cleanup(PathBuf);
    impl Cleanup {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("qwen4-lookup-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A GGUF of one Q5_1 `token_embd.weight` [8, 32]: 24 bytes per row.
    fn rows_gguf(dir: &Path) -> (PathBuf, Vec<u8>) {
        rows_gguf_n(dir, 8)
    }

    fn rows_gguf_n(dir: &Path, rows: u32) -> (PathBuf, Vec<u8>) {
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
        b.extend(u64::from(rows).to_le_bytes());
        b.extend(7u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for row in 0..rows {
            b.extend(half::f16::from_f32(0.015625).to_bits().to_le_bytes());
            b.extend(half::f16::from_f32(-0.25).to_bits().to_le_bytes());
            b.extend((row * 7919).to_le_bytes());
            b.extend([row as u8; 16]);
        }
        fs::write(&path, &b).unwrap();
        (path, b)
    }

    /// What one decode step pays per layer for the window fold: a PLE lookup
    /// of eight rows where the two unigram heads hit the LRU and six n-gram
    /// heads miss (the steady-decode shape), timed through the eval of its
    /// window, with the fold against the pre-fold shape (one buffer per
    /// row). Prints both medians and the delta scaled to the 48 layers of a
    /// token. The values are bit-identical across both shapes.
    #[test]
    fn decode_row_lookup_fold_cost_per_layer() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, _) = rows_gguf_n(&dir, 4096);
        let name = "token_embd.weight";
        let rounds = 400usize;
        let mut medians = Vec::new();
        let mut sums = Vec::new();
        for fold in [true, false, true, false] {
            let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
            store.fold_rows = fold;
            let warm = store.lookup_rows(name, &[0, 1]).unwrap();
            MxArray::eval_arrays(&[&warm]).unwrap();
            let mut costs = Vec::with_capacity(rounds);
            let mut sum = 0.0f64;
            for round in 0..rounds {
                let m = 2 + round * 6;
                let ids = [0, 1, m, m + 1, m + 2, m + 3, m + 4, m + 5];
                let started = std::time::Instant::now();
                let window = store.lookup_rows(name, &ids).unwrap();
                MxArray::eval_arrays(&[&window]).unwrap();
                costs.push(started.elapsed());
                sum += window
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum::<f64>();
            }
            costs.sort_unstable();
            medians.push((fold, costs[costs.len() / 2], costs[costs.len() * 9 / 10]));
            sums.push(sum);
            assert_eq!(store.lookup.misses(), 2 + 6 * rounds as u64);
        }
        assert!(sums.windows(2).all(|w| w[0] == w[1]), "{sums:?}");
        for (fold, median, p90) in &medians {
            eprintln!(
                "row lookup (2 hits + 6 misses) fold={fold}: median {median:.1?}, p90 {p90:.1?}"
            );
        }
        let fold = medians.iter().filter(|m| m.0).map(|m| m.1).min().unwrap();
        let plain = medians.iter().filter(|m| !m.0).map(|m| m.1).min().unwrap();
        let delta = fold.saturating_sub(plain);
        eprintln!(
            "fold delta per lookup {delta:.1?}; x48 layers = {:.1?} per decode token",
            delta * 48
        );
    }

    /// The same measurement on the real model, one process: a 6k turn's 256
    /// decode steps with the window fold and without it (`fold_rows`), the
    /// prompt being the pinned R0 "6k" case. Opt-in, needs the checkpoint:
    /// `QWEN4_EXP_REAL_MODEL=<gguf> QWEN4_EXP_REAL_INPUTS=<inputs.json>
    /// cargo test -p mlx-core --lib real_6k_turn_fold_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_6k_turn_fold_cost() {
        let (Ok(model), Ok(inputs)) = (
            std::env::var("QWEN4_EXP_REAL_MODEL"),
            std::env::var("QWEN4_EXP_REAL_INPUTS"),
        ) else {
            eprintln!(
                "real_6k_turn_fold_cost: QWEN4_EXP_REAL_MODEL / QWEN4_EXP_REAL_INPUTS not set"
            );
            return;
        };
        let cases: serde_json::Value = serde_json::from_slice(&fs::read(inputs).unwrap()).unwrap();
        let tokens: Vec<u32> = cases
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == "6k")
            .expect("a 6k case")["tokenIds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect();
        let store = Store::open(Path::new(&model)).unwrap();
        let config = super::super::gguf::config(&store).unwrap();
        let mut decoder = super::super::decoder::Decoder::new(config, store).unwrap();
        decoder.register_cache_limit();
        decoder.weights.prepare_mapped_pack().unwrap();
        decoder.weights.select_mapped_residency(0).unwrap();
        decoder.weights.prepare_hot().unwrap();
        decoder.weights.publish_live();
        decoder.residency_changed();
        eprintln!(
            "residency: mapped={} slots={} budget={:.1} GiB window={}",
            decoder.weights.plan.mapped,
            decoder.weights.plan.partial_expert_slots,
            decoder.weights.plan.budget as f64 / (1u64 << 30) as f64,
            decoder.prefill_chunk_size
        );
        let argmax = |logits: &MxArray| -> u32 {
            let last = logits.shape().unwrap().len() as i32 - 1;
            logits
                .argmax(last, Some(false))
                .unwrap()
                .item_at_uint32(0)
                .unwrap()
        };
        let steps = 256usize;
        let mut texts = Vec::new();
        for fold in [true, false, true, false] {
            decoder.weights.fold_rows = fold;
            decoder.reset();
            // Every arm imports the same rows: the LRU would otherwise hold
            // the first arm's rows and the later arms would fold nothing.
            let rows = decoder
                .weights
                .window_banks
                .values()
                .map(|bank| bank.bytes)
                .sum::<u64>()
                + decoder
                    .weights
                    .cache
                    .iter()
                    .filter(|(key, _)| !decoder.weights.row_bank.contains_key(*key))
                    .map(|(_, weight)| weight.bytes().unwrap())
                    .sum::<u64>();
            let target = decoder.weights.cache_bytes - rows;
            decoder.weights.evict_to(target).unwrap();
            assert!(decoder.weights.cache.is_empty(), "the LRU holds rows only");
            decoder.weights.begin_turn().unwrap();
            let started = std::time::Instant::now();
            let mut logits = None;
            let mut start = 0;
            while start < tokens.len() {
                let end = (start + decoder.prefill_slice_size()).min(tokens.len());
                logits = Some(
                    decoder
                        .prefill_chunk(&tokens[start..end], None, end == tokens.len())
                        .unwrap(),
                );
                start = end;
            }
            let mut token = argmax(&logits.unwrap());
            let prefill = started.elapsed();
            let mut generated = vec![token];
            let mut times = Vec::with_capacity(steps);
            for _ in 0..steps {
                let started = std::time::Instant::now();
                let logits = decoder.step(token).unwrap();
                token = argmax(&logits);
                times.push(started.elapsed());
                generated.push(token);
            }
            let steady = &times[32..];
            let mean = steady.iter().sum::<std::time::Duration>() / steady.len() as u32;
            let mut sorted = steady.to_vec();
            sorted.sort_unstable();
            eprintln!(
                "fold={fold}: prefill {prefill:.1?} ({:.0} tok/s), decode steps 32..{steps}: mean {mean:.2?} median {:.2?} -> {:.2} tok/s, lookups {} misses {}",
                tokens.len() as f64 / prefill.as_secs_f64(),
                sorted[sorted.len() / 2],
                1.0 / mean.as_secs_f64(),
                decoder.weights.lookup.lookups(),
                decoder.weights.lookup.misses(),
            );
            texts.push(generated);
        }
        assert!(
            texts.windows(2).all(|w| w[0] == w[1]),
            "the fold must not change the greedy text"
        );
    }

    #[test]
    fn batched_lookup_reuses_decoded_rows_and_survives_eviction() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, _) = rows_gguf(&dir);
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
        assert!(
            !unsafe { mlx_sys::mlx_array_is_available(cold.as_raw_ptr()) },
            "a cold lookup window must stay lazy for its consumer's graph"
        );
        assert_eq!(store.bytes_read, 3 * 24);
        assert_eq!((store.lookup.lookups(), store.lookup.misses()), (3, 3));
        assert_eq!(store.cache_bytes, 3 * 32 * 2);
        let warm = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!((store.lookup.lookups(), store.lookup.misses()), (6, 3));
        assert_eq!(store.bytes_read, 3 * 24);
        assert!(store.lookup_rows("token_embd.weight", &[0, 8]).is_err());
        assert_eq!(
            store.bytes_read,
            3 * 24,
            "validate the whole window before IO"
        );
        store.evict_to(0).unwrap();
        assert_eq!(store.cache_bytes, 0);
        // Reader threads: the same rows, requested ahead and consumed by the
        // import, decode to the same bits through the in-memory GGUF path.
        let keys: Vec<_> = ids
            .iter()
            .map(|&id| ("token_embd.weight".to_string(), id))
            .collect();
        store.prefetch_rows(&keys).unwrap();
        let fetched = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!((store.lookup.lookups(), store.lookup.misses()), (9, 6));
        assert_eq!(store.lookup.prefetched() + store.lookup.late(), 3);
        store.prefetch_rows(&keys).unwrap();
        let hits = store.lookup_rows("token_embd.weight", &ids).unwrap();
        assert_eq!(store.lookup.misses(), 6, "LRU hits request nothing");
        assert_eq!(store.lookup.prefetched() + store.lookup.late(), 3);
        for output in [cold, warm, fetched, hits] {
            assert_eq!(
                &*output.astype(DType::Float32).unwrap().to_float32().unwrap(),
                &expected
            );
        }
    }

    /// A lookup window leaves one Metal buffer behind for the rows it
    /// imported; rows it hit stay where they are; the LRU charges a window's
    /// buffer whole until its last row view is evicted and reports the
    /// bytes it released, so the charge is the memory that is allocated.
    #[test]
    fn decoded_rows_share_their_windows_buffer_and_stay_under_the_lru_bound() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, _) = rows_gguf(&dir);
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        let mut oracle = Store::open_fixture(&path, Some(&dir)).unwrap();
        let name = "token_embd.weight";
        let rb: usize = 32 * 2;
        let key = |id: usize| (name.to_string(), id, 1);
        let cached = |store: &Store, id: usize| store.cache[&key(id)].values.clone();
        let f32s = |a: &MxArray| a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let same_bits = |store: &Store, oracle: &mut Store, id: usize| {
            let expected = oracle.read(name, id, 1).unwrap().dense().unwrap();
            assert_eq!(&*f32s(&cached(store, id)), &*f32s(&expected), "row {id}");
        };
        // What is allocated: every live window buffer whole, plus every row
        // that has its own buffer.
        let allocated = |store: &Store| -> u64 {
            store.window_banks.values().map(|b| b.bytes).sum::<u64>()
                + store
                    .cache
                    .iter()
                    .filter(|(k, _)| !store.row_bank.contains_key(*k))
                    .map(|(_, w)| w.bytes().unwrap())
                    .sum::<u64>()
        };
        let held =
            |store: &Store, ids: &[usize]| ids.iter().all(|&id| store.cache.contains_key(&key(id)));
        let gone = |store: &Store, ids: &[usize]| {
            ids.iter().all(|&id| !store.cache.contains_key(&key(id)))
        };

        let window = store.lookup_rows(name, &[5, 1, 6, 1, 2]).unwrap();
        MxArray::eval_arrays(&[&window]).unwrap();
        let views: Vec<_> = [5usize, 1, 6, 2]
            .iter()
            .map(|&id| cached(&store, id))
            .collect();
        MxArray::eval_arrays(&views.iter().collect::<Vec<_>>()).unwrap();
        let base = views[0].storage_ptr() as usize;
        assert_ne!(base, 0);
        for (slot, (view, id)) in views.iter().zip([5usize, 1, 6, 2]).enumerate() {
            assert_eq!(
                view.storage_ptr() as usize,
                base + slot * rb,
                "row {id} is slot {slot} of the window's buffer"
            );
            same_bits(&store, &mut oracle, id);
        }
        assert_eq!(store.cache_bytes, 4 * rb as u64);
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.window_banks.len(), 1);

        // A hit stays in its buffer; a window's single miss keeps its own.
        let next = store.lookup_rows(name, &[1, 7]).unwrap();
        MxArray::eval_arrays(&[&next]).unwrap();
        let (kept, own) = (cached(&store, 1), cached(&store, 7));
        MxArray::eval_arrays(&[&kept, &own]).unwrap();
        assert_eq!(
            kept.storage_ptr() as usize,
            base + rb,
            "a hit is not copied"
        );
        assert!(!store.row_bank.contains_key(&key(7)));
        assert_eq!(store.cache_bytes, 5 * rb as u64);
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(
            &*f32s(&next),
            &*f32s(&MxArray::concatenate_many(vec![&kept, &own], Some(0)).unwrap())
        );

        // Two misses around a hit: a second buffer holds the two.
        let third = store.lookup_rows(name, &[3, 1, 4]).unwrap();
        MxArray::eval_arrays(&[&third]).unwrap();
        let (a, b) = (cached(&store, 3), cached(&store, 4));
        MxArray::eval_arrays(&[&a, &b]).unwrap();
        assert_eq!(a.storage_ptr() as usize + rb, b.storage_ptr() as usize);
        assert_eq!(cached(&store, 1).storage_ptr() as usize, base + rb);
        assert_eq!(store.cache_bytes, 7 * rb as u64);
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.window_banks.len(), 2);
        for id in [3, 1, 4] {
            same_bits(&store, &mut oracle, id);
        }

        // LRU order is 5, 6, 2, 7, 3, 1, 4. Evicting rows of a buffer that
        // still has a view releases nothing; the last view releases it all,
        // and the counters say what was released.
        store.evict_to(7 * rb as u64 - 1).unwrap();
        assert!(gone(&store, &[5, 6, 2, 7]) && held(&store, &[3, 1, 4]));
        assert_eq!(
            store.cache_bytes,
            6 * rb as u64,
            "the first buffer is charged whole for its one live row"
        );
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(
            (
                store.lookup.evicted_rows(),
                store.lookup.evicted_row_bytes()
            ),
            (4, rb as u64)
        );
        assert_eq!(cached(&store, 1).storage_ptr() as usize, base + rb);
        same_bits(&store, &mut oracle, 1);
        store.evict_to(5 * rb as u64).unwrap();
        assert!(gone(&store, &[3, 1]) && held(&store, &[4]));
        assert_eq!(
            store.cache_bytes,
            2 * rb as u64,
            "the first buffer went with row 1"
        );
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.lookup.evicted_row_bytes(), 5 * rb as u64);
        same_bits(&store, &mut oracle, 4);
        store.evict_to(0).unwrap();
        assert_eq!(store.cache_bytes, 0);
        assert!(
            store.cache.is_empty() && store.window_banks.is_empty() && store.row_bank.is_empty()
        );
        assert_eq!(
            (
                store.lookup.evicted_rows(),
                store.lookup.evicted_row_bytes()
            ),
            (7, 7 * rb as u64)
        );

        // Four rows fit beside the staging reserve; a window of eight evicts
        // in LRU order while it imports and folds the rows that survived.
        store.cache_limit = MAX_READ_BYTES * 3 + 4 * rb as u64;
        let wide = store.lookup_rows(name, &[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        MxArray::eval_arrays(&[&wide]).unwrap();
        assert_eq!(store.cache_bytes, 5 * rb as u64, "LRU bound");
        assert_eq!(store.cache_bytes, allocated(&store));
        assert_eq!(store.window_banks.len(), 1);
        assert!(gone(&store, &[0, 1, 2]) && held(&store, &[3, 4, 5, 6, 7]));
        assert_eq!(
            (
                store.lookup.evicted_rows(),
                store.lookup.evicted_row_bytes()
            ),
            (10, 10 * rb as u64)
        );
        let expected: Vec<_> = (0..8)
            .map(|id| oracle.read(name, id, 1).unwrap().dense().unwrap())
            .collect();
        assert_eq!(
            &*f32s(&wide),
            &*f32s(&MxArray::concatenate_many(expected.iter().collect(), Some(0)).unwrap())
        );
        same_bits(&store, &mut oracle, 7);
    }

    /// A reader's failed read is consumed with its request and never
    /// returned: the import reads the row itself, so the file as it is now
    /// decides, and nothing for the key stays behind.
    #[test]
    fn a_row_readers_error_is_dropped_and_the_import_reads_the_row_itself() {
        let cleanup = Cleanup::new();
        let dir = cleanup.0.clone();
        let (path, bytes) = rows_gguf(&dir);
        let mut store = Store::open_fixture(&path, Some(&dir)).unwrap();
        let mut oracle = Store::open_fixture(&path, Some(&dir)).unwrap();
        let name = "token_embd.weight";
        let keys = [(name.to_string(), 7usize)];
        let (offset, _) = store.descriptor(name).unwrap().range(7, 1).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(offset + 10)
            .unwrap();
        store.prefetch_rows(&keys).unwrap();
        store.rows.as_ref().unwrap().wait_idle();
        fs::write(&path, &bytes).unwrap();
        store.prefetch_rows(&keys).unwrap();
        let row = store.lookup_rows(name, &[7]).unwrap();
        let expected = oracle.read(name, 7, 1).unwrap().dense().unwrap();
        assert_eq!(
            &*row.astype(DType::Float32).unwrap().to_float32().unwrap(),
            &*expected
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
        );
        assert_eq!((store.lookup.prefetched(), store.lookup.late()), (0, 1));
        store.rows.as_ref().unwrap().wait_idle();
        assert!(
            store.rows.as_ref().unwrap().take(&keys[0]).is_none(),
            "nothing for the key may stay behind"
        );
    }

    /// PLE rows of the GGUF fixture through the reader threads against the
    /// direct importer, bit for bit, with every requested row consumed.
    #[test]
    fn prefetched_ple_rows_match_the_direct_importer_and_are_consumed() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/model.gguf");
        let mut store = Store::open_fixture(&path, None).unwrap();
        let name = "per_layer_token_embd.weight";
        let d = store.descriptor(name).unwrap().clone();
        let total = d.rows().unwrap();
        let ids: Vec<usize> = (0..16usize)
            .map(|h| (h * 7919 + 13) % total)
            .chain([13 % total])
            .collect();
        let distinct = ids.iter().collect::<HashSet<_>>().len() as u64;
        let expected = ids
            .iter()
            .map(|&id| d.import_rows(id, 1).unwrap().dense().unwrap())
            .collect::<Vec<_>>();
        let expected = MxArray::concatenate_many(expected.iter().collect(), Some(0))
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let keys: Vec<_> = ids.iter().map(|&id| (name.to_string(), id)).collect();
        store.prefetch_rows(&keys).unwrap();
        let fetched = store.lookup_rows(name, &ids).unwrap();
        assert!(!unsafe { mlx_sys::mlx_array_is_available(fetched.as_raw_ptr()) });
        assert_eq!(
            &*fetched
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap(),
            &expected
        );
        assert_eq!(
            (store.lookup.lookups(), store.lookup.misses()),
            (distinct, distinct)
        );
        assert_eq!(
            store.lookup.prefetched() + store.lookup.late(),
            distinct,
            "every imported row must come from the request"
        );
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
        // Pinned banks leave 768 bytes of the protected allowance.
        store.bank_bytes = CACHE_BYTES * 3 / 4 - 768;
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

#[cfg(test)]
mod mapped_tests {
    use super::*;
    use crate::models::qwen4_exp::memory;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, Instant};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("qwen4-mapped-store-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const EXPERTS: usize = 5;
    const PARTS: [(&str, usize, usize, u32, usize); 3] = [
        // (name, rows per expert, width, GGUF type, block bytes per row)
        ("blk.0.ffn_gate_exps.weight", 32, 256, 12, 144),
        ("blk.0.ffn_up_exps.weight", 32, 256, 12, 144),
        ("blk.0.ffn_down_exps.weight", 128, 64, 7, 48),
    ];

    /// Three routed expert tensors of one layer: Q4_K gate/up and Q5_1 down,
    /// filled with deterministic pseudo-random block bytes.
    fn write_fixture(path: &Path) {
        write_fixture_layers(path, 1)
    }

    /// The same three tensors for layers `0..layers`.
    fn write_fixture_layers(path: &Path, layers: usize) {
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(((PARTS.len() * layers) as u64).to_le_bytes());
        b.extend(1u64.to_le_bytes());
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, "qwen4exp");
        let mut offset = 0u64;
        for layer in 0..layers {
            for (name, rows, width, ty, row_bytes) in PARTS {
                string(&mut b, &name.replace("blk.0.", &format!("blk.{layer}.")));
                b.extend(3u32.to_le_bytes());
                for d in [width as u64, rows as u64, EXPERTS as u64] {
                    b.extend(d.to_le_bytes());
                }
                b.extend(ty.to_le_bytes());
                b.extend(offset.to_le_bytes());
                offset += (EXPERTS * rows * row_bytes) as u64;
            }
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        let mut state = 0x2545_f491u32;
        for _ in 0..offset {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            b.push((state >> 24) as u8);
        }
        fs::write(path, b).unwrap();
    }

    fn encoded(weight: &Weight) -> (Vec<u8>, i32, i32, String) {
        let (arrays, data) = super::super::packed_cache::encode_weight(weight).unwrap();
        let mut tagged = serde_json::to_vec(&arrays).unwrap();
        tagged.extend(data);
        (tagged, weight.group, weight.bits, weight.mode.clone())
    }

    #[test]
    fn mapped_pack_matches_the_checkpoint_and_feeds_the_slot_batch() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        // Every record comes from the GGUF importer.
        let from_gguf = temp.0.join("from-gguf");
        store.prepare_mapped_pack_at(&from_gguf).unwrap();
        assert_eq!(store.plan.mapped_pack_files, 3);
        let mapped = store.mapped.take().unwrap();
        // Oracle: the ordinary reader, expert by expert, encoded as a chunk.
        for (name, rows, _, _, _) in PARTS {
            let file = mapped.file(name).unwrap();
            let h = &file.header;
            assert_eq!(h.experts as usize, EXPERTS);
            assert_eq!(h.rows_per_expert as usize, rows);
            let mut expected: Vec<Vec<u8>> = vec![Vec::new(); h.sections.len()];
            for expert in 0..EXPERTS {
                let w = store.read(name, expert * rows, rows).unwrap();
                let (arrays, data) = super::super::packed_cache::encode_weight(&w).unwrap();
                assert_eq!(arrays.len(), h.sections.len());
                let mut cursor = 0;
                for (part, a) in arrays.iter().enumerate() {
                    expected[part].extend_from_slice(&data[cursor..cursor + a.bytes]);
                    cursor += a.bytes;
                }
                assert_eq!(
                    (w.group, w.bits, w.mode.as_str()),
                    (h.group, h.bits, h.mode.as_str())
                );
            }
            let whole = file.read_range(0, EXPERTS).unwrap();
            for (s, section) in h.sections.iter().enumerate() {
                assert_eq!(section.offset % super::super::mapped_pack::PAGE, 0);
                assert_eq!(section.length % super::super::mapped_pack::PAGE, 0);
                assert_eq!(whole[s], expected[s], "{name} section {s}");
            }
        }
        // A second build is byte-identical.
        let again = temp.0.join("again");
        store.prepare_mapped_pack_at(&again).unwrap();
        for (name, ..) in PARTS {
            let file_name = format!("{name}.mapped");
            assert_eq!(
                fs::read(from_gguf.join(&file_name)).unwrap(),
                fs::read(again.join(&file_name)).unwrap(),
                "{name}: the second build differs from the first"
            );
        }
        // The slot batch through the range reader equals the checkpoint
        // reader.
        let requests: Vec<_> = [4usize, 0, 2]
            .into_iter()
            .flat_map(|expert| PARTS.map(|(name, rows, ..)| (name.to_owned(), expert * rows, rows)))
            .collect();
        store.evict_to(0).unwrap();
        assert_eq!(store.cache.len(), 0);
        let reads = store.mapped.as_ref().unwrap().reads();
        let packed_hits = store.packed_hits;
        let via_pack: Vec<_> = store
            .read_expert_batch(&requests)
            .unwrap()
            .iter()
            .map(|w| encoded(w))
            .collect();
        let pack = store.mapped.take().unwrap();
        assert_eq!(pack.reads(), reads + 9);
        assert_eq!(
            store.packed_hits,
            packed_hits + 9,
            "range reads are prepared hits"
        );
        store.evict_to(0).unwrap();
        let via_checkpoint: Vec<_> = store
            .read_expert_batch(&requests)
            .unwrap()
            .iter()
            .map(|w| encoded(w))
            .collect();
        assert_eq!(store.packed_hits, packed_hits + 9);
        assert_eq!(via_pack, via_checkpoint);
        store.mapped = Some(pack);
        assert_eq!(
            super::routed_expert_tensor("blk.47.ffn_down_exps.weight"),
            Some((47, 2))
        );
        assert_eq!(
            super::routed_expert_tensor("blk.3.ffn_down_shexp.weight"),
            None
        );
        assert_eq!(
            super::routed_expert_tensor("nextn.0.ffn_gate_exps.weight"),
            None
        );
    }

    /// Runs the calling test in a child process. The wired limit is
    /// process-wide: while a sibling test's scheduled turn holds one, the
    /// driver wires this test's mmap-backed banks and the freshly built pack
    /// is never cold. Returns `true` inside the child, `false` in the parent
    /// after the child passed.
    fn in_child(name: &str) -> bool {
        const CHILD: &str = "MLX_QWEN4_STREAM_CHILD";
        if std::env::var_os(CHILD).is_some_and(|v| v == name) {
            return true;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name])
            .env(CHILD, name)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        false
    }

    /// A stream issue that holds every fetch until released, then reads the
    /// file as the real one does: the stream stays "running" for as long as
    /// a test needs.
    #[derive(Default)]
    struct Gate {
        released: Mutex<bool>,
        ready: Condvar,
        started: AtomicUsize,
    }
    impl Gate {
        fn release(&self) {
            *self.released.lock().unwrap() = true;
            self.ready.notify_all();
        }
    }
    impl super::super::prefetch::Issue for Arc<Gate> {
        fn fetch(&self, map: &crate::array::Mmap, stop: &AtomicBool, rows: &AtomicUsize) {
            self.started.fetch_add(1, Ordering::AcqRel);
            let mut released = self.released.lock().unwrap();
            // Polled: a pool dropped by a failing test stops the worker
            // without a release, and nothing signals `ready` then.
            while !*released && !stop.load(Ordering::Acquire) {
                released = self
                    .ready
                    .wait_timeout(released, Duration::from_millis(10))
                    .unwrap()
                    .0;
            }
            drop(released);
            super::super::prefetch::Pread.fetch(map, stop, rows);
        }
    }

    fn mapped_store(temp: &Temp, gguf: &Path, dir: &str) -> Store {
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(gguf, Some(&root)).unwrap();
        store.prepare_mapped_pack_at(&temp.0.join(dir)).unwrap();
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, _, _, _| {
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);
        assert!(
            !store.prefetch.as_ref().unwrap().sampled_resident(),
            "a freshly built pack (F_NOCACHE writes) is cold"
        );
        store
    }

    fn wait_resident(store: &Store) {
        let pool = store.prefetch.as_ref().unwrap();
        pool.wait_idle();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !pool.sampled_resident() {
            assert!(Instant::now() < deadline, "the stream did not land in 10 s");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The load picked the mapped plan: after the dense banks the cold pack
    /// streams; a turn that opens over the stream starts no second one and
    /// its stale first pass does not touch; once the pack is resident and
    /// the stream over, a stale pass touches, a pass inside the gap and a
    /// pass that reads no mapped bank (which also leaves the stamp alone) do
    /// not; slot mode never touches and never streams.
    /// MUTATION (M-P5-touch-cold): drop the `streaming_pass` early return of
    /// `wire_mapped_banks_at` -> "a stale pass over a cold pack streams
    /// instead of touching" fails.
    #[test]
    fn a_cold_mapped_load_streams_after_the_dense_banks_and_a_stale_pass_does_not_touch() {
        if !in_child(
            "models::qwen4_exp::weights::mapped_tests::a_cold_mapped_load_streams_after_the_dense_banks_and_a_stale_pass_does_not_touch",
        ) {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let t0 = Instant::now();
        let mut slots = Store::open_fixture(&gguf, Some(&temp.0.join("cache"))).unwrap();
        slots.start_pack_stream();
        assert!(slots.prefetch.is_none(), "slot mode has no stream");
        assert!(slots.begin_turn().unwrap().is_none());
        assert!(!slots.streaming_pass);
        assert!(
            !slots.wire_mapped_banks_at(t0, true).unwrap(),
            "slot mode never touches"
        );

        let mut store = mapped_store(&temp, &gguf, "mapped");
        let gate = Arc::new(Gate::default());
        store.pool_with(Arc::clone(&gate));
        store.start_pack_stream();
        assert!(
            store.prefetch.as_ref().unwrap().streaming(),
            "the load-time stream runs"
        );
        assert!(!store.streaming_pass, "no turn is open yet");
        let deadline = Instant::now() + Duration::from_secs(5);
        while gate.started.load(Ordering::Acquire) == 0 {
            assert!(Instant::now() < deadline, "no worker started");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(store.begin_turn().unwrap().is_none());
        assert!(store.streaming_pass, "the turn opened over the stream");
        assert!(
            !store.wire_mapped_banks_at(t0, true).unwrap(),
            "a stale pass over a cold pack streams instead of touching"
        );
        assert!(store.touches.is_empty());
        gate.release();
        wait_resident(&store);
        assert!(
            !store
                .wire_mapped_banks_at(t0 + Duration::from_millis(100), true)
                .unwrap(),
            "within the pass"
        );
        store
            .keep_wired_at(t0 + Duration::from_millis(800), 0, true)
            .unwrap();
        assert_eq!(
            store.touches,
            vec![vec![0]],
            "the streaming pass keeps layer 0 wired"
        );
        assert!(!store.streaming_pass, "the stream is over");
        assert!(
            !store
                .wire_mapped_banks_at(t0 + Duration::from_millis(850), true)
                .unwrap()
        );
        assert!(
            store
                .wire_mapped_banks_at(t0 + Duration::from_millis(1800), true)
                .unwrap(),
            "a stale pass over a resident pack touches"
        );
        assert_eq!(store.touches.len(), 2);
        assert!(
            !store
                .wire_mapped_banks_at(t0 + Duration::from_millis(1850), true)
                .unwrap()
        );
        let stamp = store.mapped_pass_start;
        assert!(stamp.is_some());
        for ms in [2600, 3400] {
            assert!(
                !store
                    .wire_mapped_banks_at(t0 + Duration::from_millis(ms), false)
                    .unwrap(),
                "a pass that reads no mapped bank never touches"
            );
            assert_eq!(
                store.mapped_pass_start, stamp,
                "a pass that reads no mapped bank must not refresh the stamp"
            );
        }
        assert!(
            store
                .wire_mapped_banks_at(t0 + Duration::from_millis(4200), true)
                .unwrap(),
            "the main pass after a run of drafts sees the gap since the last main pass"
        );
        assert!(store.begin_turn().unwrap().is_none());
        assert!(
            !store.streaming_pass && !store.prefetch.as_ref().unwrap().streaming(),
            "a turn over a resident pack starts no stream"
        );
    }

    /// While the pack streams, the layer hooks touch the banks of the
    /// layers already run, at most every 700 ms, never the layers ahead
    /// (their pages may be cold), never for a pass that reads no mapped bank,
    /// and not once the stream is over.
    /// MUTATION (M-P5-keep-wired-all): `keep_wired_at` touches every layer
    /// -> "layers 6..7 are not bound yet" fails.
    #[test]
    fn keep_wired_touches_only_bound_layers_every_700ms() {
        if !in_child(
            "models::qwen4_exp::weights::mapped_tests::keep_wired_touches_only_bound_layers_every_700ms",
        ) {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture_layers(&gguf, 8);
        let mut store = mapped_store(&temp, &gguf, "mapped");
        assert_eq!(store.mapped_experts.len(), 8);
        let gate = Arc::new(Gate::default());
        store.pool_with(Arc::clone(&gate));
        let t0 = Instant::now();
        assert!(store.begin_turn().unwrap().is_none());
        assert!(store.streaming_pass);
        assert!(!store.wire_mapped_banks_at(t0, true).unwrap());
        store
            .keep_wired_at(t0 + Duration::from_millis(100), 5, true)
            .unwrap();
        assert!(store.touches.is_empty(), "100 ms after the pass start");
        store
            .keep_wired_at(t0 + Duration::from_millis(800), 5, true)
            .unwrap();
        assert_eq!(
            store.touches,
            vec![vec![0, 1, 2, 3, 4, 5]],
            "layers 6..7 are not bound yet"
        );
        store
            .keep_wired_at(t0 + Duration::from_millis(900), 7, true)
            .unwrap();
        assert_eq!(store.touches.len(), 1, "100 ms after the last touch");
        store
            .keep_wired_at(t0 + Duration::from_millis(1600), 7, false)
            .unwrap();
        assert_eq!(store.touches.len(), 1, "a pass that reads no mapped bank");
        assert!(store.streaming_pass);
        gate.release();
        wait_resident(&store);
        store
            .keep_wired_at(t0 + Duration::from_millis(1600), 7, true)
            .unwrap();
        assert_eq!(store.touches.len(), 2);
        assert_eq!(store.touches[1], (0..8).collect::<Vec<_>>());
        assert!(!store.streaming_pass, "the stream is over");
        store
            .keep_wired_at(t0 + Duration::from_millis(2400), 7, true)
            .unwrap();
        assert_eq!(store.touches.len(), 2, "no streaming pass, no touch");
    }

    /// The tripwire's SSD count starts at the turn: the pack bytes the
    /// load-time stream read before it are below the baseline, and the turn
    /// over a pack the stream made resident starts no stream of its own.
    #[test]
    fn a_streaming_turn_keeps_the_tripwire_baseline() {
        if !in_child(
            "models::qwen4_exp::weights::mapped_tests::a_streaming_turn_keeps_the_tripwire_baseline",
        ) {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let mut store = mapped_store(&temp, &gguf, "mapped");
        let before = memory::process_disk_reads();
        store.start_pack_stream();
        wait_resident(&store);
        let after = memory::process_disk_reads();
        assert!(
            after >= before + store.plan.mapped_bytes,
            "the stream read the pack from disk ({} B over {} B)",
            after - before,
            store.plan.mapped_bytes
        );
        assert!(store.begin_turn().unwrap().is_none());
        assert!(
            store.tripwire_mut().turn_start_reads() >= after,
            "the stream's bytes are not charged to the turn"
        );
        assert!(!store.streaming_pass);
    }

    /// A load up to its last step: the fixture MTP head attached (or not)
    /// before the gate, the mapped plan, a held pack stream and the dense
    /// banks. Returns the gate's anonymous need and the draft bank estimate.
    fn mapped_load(
        temp: &Temp,
        gguf: &Path,
        dir: &str,
        head: bool,
    ) -> (Store, Arc<Gate>, u64, u64) {
        let mut store =
            Store::open_fixture(gguf, Some(&temp.0.join(format!("{dir}-cache")))).unwrap();
        if head {
            let aux = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/qwen4-exp/bf16/paged");
            store
                .attach_auxiliary(Store::open_fixture(&aux, None).unwrap())
                .unwrap();
        }
        let estimate = store.mtp_bank_estimate().unwrap();
        assert_eq!(estimate > 0, head, "the fixture head has routed experts");
        store.prepare_mapped_pack_at(&temp.0.join(dir)).unwrap();
        let mut charged = 0;
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, need, _, _| {
                charged = need;
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);
        let gate = Arc::new(Gate::default());
        store.pool_with(Arc::clone(&gate));
        store.prepare_fixture_hot().unwrap();
        (store, gate, charged, estimate)
    }

    /// T2. With the MTP head attached the load builds the draft banks the
    /// gate charged before any pack fetch starts, then streams the pack,
    /// whether or not a turn will ever ask for MTP; without the head it only
    /// streams.
    #[test]
    fn the_load_builds_the_draft_banks_before_the_pack_stream_starts() {
        if !in_child(
            "models::qwen4_exp::weights::mapped_tests::the_load_builds_the_draft_banks_before_the_pack_stream_starts",
        ) {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        for (head, dir) in [(true, "head"), (false, "plain")] {
            let (mut store, gate, charged, estimate) = mapped_load(&temp, &gguf, dir, head);
            let dense = store.fixed_projection_bytes().unwrap();
            assert_eq!(
                charged,
                dense + estimate,
                "{dir}: the gate charges the bank"
            );
            let mut seen = Vec::new();
            store.finish_load_with(&mut |_| {
                // A pack fetch issued before the build starts within this wait.
                let deadline = Instant::now() + Duration::from_millis(100);
                while gate.started.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                seen.push(gate.started.load(Ordering::Acquire));
                Ok(())
            });
            assert_eq!(seen.is_empty(), !head, "{dir}: the build admits its banks");
            assert!(
                seen.iter().all(|&started| started == 0),
                "{dir}: no pack fetch started before the build ({seen:?})"
            );
            assert_eq!(
                store.mtp_bank_bytes().unwrap(),
                estimate,
                "{dir}: the load built the banks the gate charged"
            );
            assert!(
                store.prefetch.as_ref().unwrap().streaming(),
                "{dir}: the pack streams after the build"
            );
            gate.release();
            wait_resident(&store);
        }
    }

    /// T3a. A load whose host refuses the draft banks still loads: the pack
    /// streams, the drafts stay on the row reader and nothing retries.
    /// MUTATION (M-L2c-load-fails-on-decline): `finish_load_with` returns the
    /// build's error before the stream -> "the declined load still streams"
    /// fails.
    #[test]
    fn a_load_that_cannot_stage_the_draft_banks_still_loads_and_never_retries() {
        if !in_child(
            "models::qwen4_exp::weights::mapped_tests::a_load_that_cannot_stage_the_draft_banks_still_loads_and_never_retries",
        ) {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let (mut store, gate, ..) = mapped_load(&temp, &gguf, "mapped", true);
        let mut admissions = 0;
        store.finish_load_with(&mut |_| {
            admissions += 1;
            Err(err("simulated headroom loss at load"))
        });
        assert!(admissions >= 1);
        assert!(store.mtp_banks_declined);
        assert_eq!(store.mtp_bank_bytes().unwrap(), 0);
        assert!(
            store.prefetch.as_ref().unwrap().streaming(),
            "the declined load still streams the pack"
        );
        assert_eq!(
            store
                .prepare_mtp_expert_banks_with(&mut |_| Ok(()))
                .unwrap(),
            0,
            "a declined head is never built later"
        );
        assert_eq!(store.mtp_bank_bytes().unwrap(), 0);
        gate.release();
        wait_resident(&store);
    }

    /// With the MTP head attached, the mapped plan's anonymous need carries
    /// the Q8 draft banks beside the dense banks, the allowance admits that
    /// bank once the plan is mapped, and leaving the mapped path drops it.
    /// MUTATION (M-INT2-bank): pass `dense` alone to the gate -> the gate sees
    /// the dense bytes without the bank and the first assertion fails.
    #[test]
    fn mapped_plan_charges_the_draft_bank_while_the_mtp_head_is_attached() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        let aux =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/bf16/paged");
        store
            .attach_auxiliary(Store::open_fixture(&aux, None).unwrap())
            .unwrap();
        let bank = store.mtp_bank_estimate().unwrap();
        assert!(bank > 0, "the fixture head has routed experts");
        let dense = store.fixed_projection_bytes().unwrap();
        let dir = temp.0.join("mapped");
        store.prepare_mapped_pack_at(&dir).unwrap();
        let mut charged = None;
        store
            .select_mapped_residency_with(EXPERTS, 1 << 30, &mut |plan, need, _, _| {
                charged = Some(need);
                plan.budget = super::super::memory::mapped_desired(need);
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);
        assert_eq!(
            charged,
            Some(dense + bank),
            "the gate's anonymous need carries the draft bank"
        );
        assert!(
            store.auxiliary_cache_allowance().unwrap() >= bank,
            "the mapped budget admits the bank it charged"
        );
        let slot_rule = store
            .tensors
            .iter()
            .filter(|(name, _)| super::residency::is_auxiliary(name))
            .map(|(_, t)| t.runtime_bytes().unwrap())
            .sum::<u64>()
            .min(store.cache_limit / 8)
            .max(MAX_READ_BYTES * 6);
        store
            .leave_mapped_with(
                &Fallback::LoadGate { pack: 0, budget: 0 },
                &mut super::super::memory::host_memory,
                &mut |_| {},
            )
            .unwrap();
        assert!(!store.plan.mapped);
        assert_eq!(
            store.auxiliary_cache_allowance().unwrap(),
            slot_rule,
            "the slot path keeps its own allowance rule"
        );
    }

    /// Mapped mode on the fixture pack: identity routing, capacity == experts,
    /// never a miss or an upload, bank bytes and kernel reads identical to
    /// the chunk-built banks, leases inert, out-of-range ids refused.
    #[test]
    fn mapped_residency_addresses_every_expert_in_place_without_misses() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        let dir = temp.0.join("mapped");
        store.prepare_mapped_pack_at(&dir).unwrap();
        assert_eq!(store.plan.mapped_pack_files, 3);
        // Oracle banks: every expert through the ordinary reader, in id order.
        let oracle: Vec<Arc<Weight>> = PARTS
            .iter()
            .map(|(name, rows, ..)| {
                let chunks: Vec<_> = (0..EXPERTS)
                    .map(|e| store.read(name, e * *rows, *rows).unwrap())
                    .collect();
                Arc::new(Weight::concatenate_rows(&chunks).unwrap())
            })
            .collect();
        store.evict_to(0).unwrap();
        let expected_bytes: u64 = PARTS
            .iter()
            .map(|(name, ..)| {
                let h = &store.mapped.as_ref().unwrap().file(name).unwrap().header;
                h.sections.iter().map(|s| s.length).sum::<u64>()
            })
            .sum();
        let before = store.cache_bytes;
        store
            .select_mapped_residency_with(EXPERTS, 1 << 30, &mut |plan, dense, pool, pack| {
                // The fixture holds routed experts only; the gate sees
                // the pool it was given and the pack's section bytes.
                assert_eq!((dense, pool), (0, 1 << 30));
                assert_eq!(pack, expected_bytes);
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);
        assert_eq!(store.plan.mapped_bytes, expected_bytes);
        assert_eq!(store.plan.partial_expert_slots, EXPERTS);
        assert_eq!(store.cache_limit, 1 << 30);
        assert_eq!(store.expert_slot_capacity(EXPERTS, 2).unwrap(), EXPERTS);
        assert!(store.device_slots_ready(1) && store.prefill_device_slots_ready(1));
        assert!(!store.device_slots_ready(2));
        assert_eq!(store.cache_bytes, before, "mapped banks are not anonymous");
        assert_eq!(store.slot_bytes, 0);

        let ids = [4u32, 0, 2, 2, 1, 3];
        let (banks, local, capacity) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_eq!(local, ids, "slot id must be the expert id");
        assert_eq!(capacity, EXPERTS);
        let device = MxArray::from_uint32(&ids, &[2, 3]).unwrap();
        let (_, device_local) = store.device_expert_slots(0, &device).unwrap().unwrap();
        assert_eq!(&*device_local.to_uint32().unwrap(), &ids);
        let (_, safe, count) = store.prefill_device_expert_slots(0, &device).unwrap();
        assert_eq!(&*safe.to_uint32().unwrap(), &ids);
        assert_eq!(count, EXPERTS);
        for round in 0..3 {
            assert!(
                store.commit_device_routes(&[0, 0, 0], &ids, 2).unwrap(),
                "round {round}"
            );
        }
        assert_eq!(store.slot_misses, 0);
        assert_eq!(store.slot_upload_bytes, 0);
        // 5 unique ids in the window, then 2 + 1 + 2 unique per committed tape.
        assert_eq!(store.slot_hits, 5 + 3 * 5);
        assert!(store.slots.is_empty(), "the slot machinery stays unused");

        // The mapped banks are the pack's bytes: the same arrays the chunk
        // reader builds, byte for byte, with the same quantization geometry.
        for (part, (name, rows, ..)) in PARTS.iter().enumerate() {
            let bank = &banks[part];
            assert!(!bank.values.storage_ptr().is_null(), "{name} is not mapped");
            assert_eq!(
                encoded(bank),
                encoded(&oracle[part]),
                "{name}: mapped bank differs from the chunk-built bank"
            );
            for e in [0usize, 2, 4] {
                assert_eq!(
                    encoded(&bank.slice_rows(e * rows, *rows).unwrap()),
                    encoded(&store.read(name, e * rows, *rows).unwrap()),
                    "{name} expert {e}"
                );
            }
        }
        // A kernel reading the mapped buffers equals the same kernel on the
        // anonymous copy of the same bytes.
        let x = MxArray::from_float32(
            &(0..6 * 256)
                .map(|n| ((n % 97) as f32 - 48.) / 64.)
                .collect::<Vec<_>>(),
            &[6, 1, 256],
        )
        .unwrap()
        .astype(DType::BFloat16)
        .unwrap();
        let selected = MxArray::from_uint32(&ids, &[6]).unwrap();
        let mapped_out = banks[0].gather_rows(&x, &selected, EXPERTS, false).unwrap();
        let anonymous_out = oracle[0]
            .gather_rows(&x, &selected, EXPERTS, false)
            .unwrap();
        // Bit patterns: the pseudo-random fixture codes decode to NaNs too.
        let bits = |a: &MxArray| {
            a.to_float32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<u32>>()
        };
        assert_eq!(bits(&mapped_out), bits(&anonymous_out));
        // Leases are inert: nothing is retained, completion still evaluates.
        store.finish_expert_slots(0, &mapped_out).unwrap();
        store.defer_expert_reduction(0, &mapped_out).unwrap();
        assert_eq!(store.deferred_reduction_layer, None);
        store.complete_device_readers().unwrap();
        store.complete_expert_window(0, &anonymous_out).unwrap();
        assert!(
            store
                .expert_slots(0, EXPERTS, 2, &[EXPERTS as u32])
                .is_err()
        );
        assert!(
            store
                .commit_device_routes(&[0], &[EXPERTS as u32, 0], 2)
                .is_err()
        );
        assert!(store.device_expert_slots(1, &device).unwrap().is_none());
    }

    /// A pack whose header page was damaged is refused at open and rebuilt
    /// before anything is mapped; a pack whose expert count disagrees with
    /// the checkpoint keeps the slot path.
    #[test]
    fn mapped_residency_refuses_a_damaged_or_mismatched_pack() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        let dir = temp.0.join("mapped");
        store.prepare_mapped_pack_at(&dir).unwrap();
        let name = PARTS[1].0;
        let header = store
            .mapped
            .as_ref()
            .unwrap()
            .file(name)
            .unwrap()
            .header
            .clone();
        store.mapped = None;
        // Flip one hex digit of a section checksum inside the header JSON:
        // still valid JSON, still a 64-char digest, only the page checksum
        // can tell.
        let path = dir.join(format!("{name}.mapped"));
        let mut bytes = fs::read(&path).unwrap();
        let sha = header.sections[0].sha256.as_bytes();
        let at = bytes[..super::super::mapped_pack::PAGE as usize]
            .windows(sha.len())
            .position(|w| w == sha)
            .unwrap();
        bytes[at] = if bytes[at] == b'0' { b'1' } else { b'0' };
        fs::write(&path, &bytes).unwrap();
        let opened = fs::metadata(&path).unwrap().len();
        store.prepare_mapped_pack_at(&dir).unwrap();
        let rebuilt = store.mapped.as_ref().unwrap().file(name).unwrap();
        assert_eq!(
            rebuilt.header, header,
            "{name} must be rebuilt from the sources"
        );
        assert_eq!(fs::metadata(&path).unwrap().len(), opened);
        assert_ne!(
            fs::read(&path).unwrap()[at],
            bytes[at],
            "the damaged page must not survive"
        );
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, _, _, _| {
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);

        let mut other = Store::open_fixture(&gguf, Some(&root)).unwrap();
        other.prepare_mapped_pack_at(&dir).unwrap();
        other
            .select_mapped_residency_with(EXPERTS + 1, 0, &mut |_, _, _, _| {
                panic!("an incomplete pack must not be admitted")
            })
            .unwrap();
        assert!(!other.plan.mapped);
        assert_eq!(other.plan.mapped_bytes, 0);
        assert!(other.mapped_experts.is_empty());
    }

    /// The live-memory gate says no: the store stays on slots, maps nothing,
    /// keeps its slot budget and reports the refusal with the gate's budget.
    #[test]
    fn a_refused_gate_keeps_the_slot_path_and_reports_the_bank_budget() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        store
            .prepare_mapped_pack_at(&temp.0.join("mapped"))
            .unwrap();
        let budget_before = store.plan.budget;
        let limit_before = store.cache_limit;
        let mut seen_pack = 0;
        store
            .select_mapped_residency_with(EXPERTS, 1 << 30, &mut |plan, _, _, pack| {
                seen_pack = pack;
                plan.mapped_bank_budget = pack - 1;
                plan.live_non_reclaimable = 40 << 30;
                plan.budget = 1;
                plan.mapped = true;
                Err(err("mapped banks need more than the host leaves"))
            })
            .unwrap();
        assert!(seen_pack > 0);
        assert!(!store.plan.mapped);
        assert!(store.mapped_experts.is_empty() && store.prefetch.is_none());
        assert_eq!(
            (store.plan.budget, store.cache_limit),
            (budget_before, limit_before)
        );
        assert_eq!(store.plan.mapped_bank_budget, seen_pack - 1);
        assert_eq!(store.plan.live_non_reclaimable, 40 << 30);
        assert_eq!(
            store.live.fallback(),
            Some(Fallback::LoadGate {
                pack: seen_pack,
                budget: seen_pack - 1
            })
        );
        assert_eq!(store.expert_slot_capacity(EXPERTS, 2).unwrap(), EXPERTS);
        assert!(store.slots.is_empty(), "slots are built by the first turn");
    }

    /// A pass that trips the wire is the request's last mapped one: the
    /// store leaves the mapped path between that pass and the next (banks
    /// unmapped, read-ahead released, the slot budget taken from the live
    /// host, slots built on demand, the reason published) and the same
    /// request's next pass maps its expert ids through slots.
    ///
    /// MUTATION (M-M6-switch): `observe_pass_at` logs the trip and returns
    /// `Ok(None)` without `apply_fallback`; the request finishes mapped.
    #[test]
    fn a_trip_observed_mid_turn_leaves_the_mapped_path_at_that_pass() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        store
            .prepare_mapped_pack_at(&temp.0.join("mapped"))
            .unwrap();
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, _, _, _| {
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);
        let mapped = store.plan.mapped_bytes;
        assert!(store.begin_turn().unwrap().is_none());
        let base = store.tripwire_mut().turn_start_reads();
        let ids = [4u32, 0, 2];
        let (banks, _, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let mapped_bank = Arc::as_ptr(&banks[0]);
        // Passes 1..k read the pack once between them: mapped throughout.
        for pass in 1..=4u64 {
            assert!(
                store
                    .observe_pass_at(1, Duration::from_millis(25), base + pass * mapped / 4)
                    .unwrap()
                    .is_none(),
                "pass {pass}"
            );
            assert!(store.plan.mapped);
            let (same, _, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
            assert_eq!(Arc::as_ptr(&same[0]), mapped_bank);
        }
        // Pass k+1 crosses twice the pack: the store leaves mapped mode now.
        let why = store
            .observe_pass_at(1, Duration::from_millis(25), base + 2 * mapped + 1)
            .unwrap()
            .expect("the tripping pass applies the switch");
        assert_eq!(
            why,
            Fallback::SsdReads {
                read: 2 * mapped + 1,
                mapped
            }
        );
        assert!(!store.plan.mapped);
        assert_eq!(store.plan.mapped_bytes, 0);
        assert!(store.mapped_experts.is_empty() && store.prefetch.is_none());
        assert_eq!(store.cache_limit, store.plan.budget);
        assert!(store.cache_limit >= store.cache_bytes + MAX_READ_BYTES * 3);
        assert!(!store.live.mapped.load(Ordering::Acquire));
        assert_eq!(store.live.fallback(), Some(why));
        assert_eq!(
            store.tripwire_mut().turn_start_reads(),
            base,
            "the window in progress keeps its baseline"
        );
        let (slots, local, capacity) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_ne!(Arc::as_ptr(&slots[0]), mapped_bank, "anonymous slot banks");
        assert_eq!(capacity, EXPERTS);
        assert_eq!(local.len(), ids.len());
        assert!(store.slot_bytes > 0 && store.slots.contains_key(&0));
        // Later passes of the same request and the next turn change nothing.
        assert!(
            store
                .observe_pass_at(1, Duration::from_millis(25), base + 3 * mapped)
                .unwrap()
                .is_none()
        );
        assert!(store.begin_turn().unwrap().is_none(), "applied once");
        assert!(!store.plan.mapped);
    }

    /// The per-pass bookkeeping of a mapped turn (two `Instant::now`, the
    /// process disk-read counter, the tripwire's median and reference scan)
    /// costs microseconds, not the 0.3 ms per token that would show in a
    /// 20 ms decode step. Prints the median and p99 per pass.
    #[test]
    fn per_pass_bookkeeping_costs_microseconds() {
        let mapped = 73u64 << 30;
        let mut wire = Tripwire::default();
        wire.begin_turn(memory::process_disk_reads());
        for window in [12_000u64, 900, 850, 800, 820, 810] {
            wire.observe(1024, Duration::from_millis(window), 0, mapped);
        }
        for _ in 0..64 {
            wire.observe(1, Duration::from_millis(25), 0, mapped);
        }
        let mut costs: Vec<Duration> = (0..4096)
            .map(|_| {
                let started = std::time::Instant::now();
                let reads = memory::process_disk_reads();
                let elapsed = started.elapsed();
                assert!(wire.observe(1, elapsed, reads, mapped).is_none());
                wire.observe(4, Duration::from_millis(30), reads, mapped);
                started.elapsed()
            })
            .collect();
        costs.sort_unstable();
        let median = costs[costs.len() / 2];
        let p99 = costs[costs.len() * 99 / 100];
        eprintln!(
            "per-pass bookkeeping: median {median:.1?}, p99 {p99:.1?}, max {:.1?}",
            costs[costs.len() - 1]
        );
        assert!(median < Duration::from_micros(300), "median {median:?}");
    }

    /// A trip taken through the wire directly (its switch did not run at
    /// the pass) stays armed: the next turn to open applies it, banks
    /// unmapped, the slot budget taken from the live host, slots built on
    /// demand, the reason published.
    #[test]
    fn a_tripped_mapped_store_runs_its_next_turn_on_slots() {
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        store
            .prepare_mapped_pack_at(&temp.0.join("mapped"))
            .unwrap();
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, _, _, _| {
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped && store.live.mapped.load(Ordering::Acquire));
        let mapped = store.plan.mapped_bytes;
        assert!(store.begin_turn().unwrap().is_none(), "nothing tripped yet");
        let ids = [4u32, 0, 2];
        let (banks, _, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let mapped_bank = Arc::as_ptr(&banks[0]);
        store.trip(Fallback::SsdReads {
            read: 2 * mapped + 1,
            mapped,
        });
        // The turn that tripped keeps its mapped banks.
        assert!(store.plan.mapped);
        let (same, _, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_eq!(Arc::as_ptr(&same[0]), mapped_bank);
        assert!(store.live.fallback().is_none());

        let why = store
            .begin_turn()
            .unwrap()
            .expect("the next turn applies the fallback");
        assert_eq!(
            why,
            Fallback::SsdReads {
                read: 2 * mapped + 1,
                mapped
            }
        );
        assert!(!store.plan.mapped);
        assert_eq!(store.plan.mapped_bytes, 0);
        assert!(store.mapped_experts.is_empty() && store.prefetch.is_none());
        assert_eq!(store.cache_limit, store.plan.budget);
        assert!(store.cache_limit >= store.cache_bytes + MAX_READ_BYTES * 3);
        assert_eq!(store.expert_slot_capacity(EXPERTS, 2).unwrap(), EXPERTS);
        assert!(!store.live.mapped.load(Ordering::Acquire));
        assert_eq!(store.live.fallback(), Some(why));
        assert!(
            !store
                .wire_mapped_banks_at(std::time::Instant::now(), true)
                .unwrap(),
            "a slot store never touches banks"
        );
        let (slots, local, capacity) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_ne!(Arc::as_ptr(&slots[0]), mapped_bank, "anonymous slot banks");
        assert_eq!(capacity, EXPERTS);
        assert_eq!(local.len(), ids.len());
        assert!(store.slot_bytes > 0 && store.slots.contains_key(&0));
        assert!(store.begin_turn().unwrap().is_none(), "applied once");
    }

    /// A turn admitted while another is open applies a trip without starting
    /// a window: the store leaves mapped mode (every open turn runs on slots
    /// from its next pass) and the SSD baseline of the window in progress
    /// stays.
    ///
    /// MUTATION (M-M5R2-1b): `apply_fallback` returns `Ok(None)` without
    /// leaving; the store stays mapped.
    #[test]
    fn an_admission_beside_an_open_turn_applies_the_trip_and_keeps_the_window() {
        const SENTINEL: u64 = u64::MAX - 1;
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        store
            .prepare_mapped_pack_at(&temp.0.join("mapped"))
            .unwrap();
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, _, _, _| {
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.begin_turn().unwrap().is_none());
        let mapped = store.plan.mapped_bytes;
        let ids = [4u32, 0, 2];
        let (banks, _, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let mapped_bank = Arc::as_ptr(&banks[0]);
        store.trip(Fallback::SsdReads {
            read: 2 * mapped + 1,
            mapped,
        });
        store.tripwire_mut().begin_turn(SENTINEL);
        let why = store
            .apply_fallback()
            .unwrap()
            .expect("the admission applies the trip");
        assert_eq!(
            why,
            Fallback::SsdReads {
                read: 2 * mapped + 1,
                mapped
            }
        );
        assert!(!store.plan.mapped && store.mapped_experts.is_empty() && store.prefetch.is_none());
        assert_eq!(store.cache_limit, store.plan.budget);
        assert_eq!(store.plan.mapped_bytes, 0);
        assert_eq!(
            store.tripwire_mut().turn_start_reads(),
            SENTINEL,
            "the window in progress keeps its baseline"
        );
        assert_eq!(store.live.fallback(), Some(why));
        let (slots, _, capacity) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_ne!(Arc::as_ptr(&slots[0]), mapped_bank, "anonymous slot banks");
        assert_eq!(capacity, EXPERTS);
        assert!(store.apply_fallback().unwrap().is_none(), "applied once");
    }

    /// The slot budget is sized after the banks are released: the host is
    /// read with the store's maps already dropped, from the sample after
    /// wired memory settles, and that budget is the one the store keeps.
    #[test]
    fn leaving_mapped_mode_releases_the_banks_before_sizing_the_slots() {
        use super::super::memory::HostMemory;
        let temp = Temp::new();
        let gguf = temp.0.join("experts.gguf");
        write_fixture(&gguf);
        let root = temp.0.join("cache");
        let mut store = Store::open_fixture(&gguf, Some(&root)).unwrap();
        store
            .prepare_mapped_pack_at(&temp.0.join("mapped"))
            .unwrap();
        store
            .select_mapped_residency_with(EXPERTS, 0, &mut |plan, _, _, _| {
                plan.budget = 1 << 30;
                plan.mapped = true;
                Ok(())
            })
            .unwrap();
        assert!(store.plan.mapped);
        store.plan.physical_bytes = Some(128 << 30);
        store.plan.hot_bytes = 83 << 30;
        let map = Arc::clone(&store.mapped_experts[&0].maps[0]);
        let holders = Arc::strong_count(&map);
        assert!(holders >= 2, "the store holds the mapping: {holders}");
        let wired = HostMemory {
            physical: 128 << 30,
            free: 1 << 30,
            file: 20 << 30,
            anonymous: 30 << 30,
            purgeable: 0,
            wired: 50 << 30,
            compressor: 10 << 30,
            uncompressed: 10 << 30,
            swapped: 0,
            faults: 1,
        };
        let released = |faults: u64| HostMemory {
            file: 54 << 30,
            wired: 16 << 30,
            faults,
            ..wired
        };
        let mut samples = std::collections::VecDeque::from([
            wired,
            released(2),
            released(3),
            released(4),
            released(5),
            released(6),
        ]);
        let mut seen = Vec::new();
        let mut slept = 0;
        let why = Fallback::SsdReads {
            read: 3 << 30,
            mapped: 1 << 30,
        };
        store
            .leave_mapped_with(
                &why,
                &mut || {
                    seen.push(Arc::strong_count(&map));
                    Ok(samples.pop_front().expect("more samples than polls"))
                },
                &mut |_| slept += 1,
            )
            .unwrap();
        assert_eq!(seen.len(), 6);
        assert!(
            seen.iter().all(|&n| n < holders),
            "sampled while the store still held its banks: {seen:?} of {holders}"
        );
        assert_eq!(slept, 5);
        assert!(!store.plan.mapped && store.mapped_experts.is_empty() && store.prefetch.is_none());
        assert_eq!(store.plan.available_bytes, Some(released(6).available()));
        assert_eq!(store.plan.budget, 35 << 30, "sized from the released host");
        assert_eq!(store.cache_limit, store.plan.budget);
        assert_eq!(store.live.fallback(), Some(why));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while Arc::strong_count(&map) > 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the pack mapping outlived the fallback"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
