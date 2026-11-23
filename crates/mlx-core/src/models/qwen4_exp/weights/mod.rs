//! Bounded, positional SSD reads. Descriptors never create MLX arrays. Only
//! selected expert matrices and embedding rows are read on cache misses.
//! Admitted hot banks can be prepared once; PLE stays row-addressable. Routed
//! GGUF experts fill anonymous banks directly from the source checkpoint.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Cursor, Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use napi::{Error, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::live::{Fallback, Live};
use super::row_fetch::Fetcher;
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::QuantizedLinear;
use crate::utils::gguf::{
    GgufFile, GgufLoadOptions, GgufMetaValue, GgufTensorInfo, GgufTensorType, load_gguf_tensor,
    parse_gguf, source_file_identity_digest, symmetric_zero_point,
};
use crate::utils::gguf_read::ReadAt;

pub(super) mod expert_slots;
mod linear;
mod lookup;
pub(super) mod residency;
#[cfg(test)]
mod tests;

use lookup::WindowBank;

pub const CACHE_BYTES: u64 = 8 << 30;
pub(super) const MAX_READ_BYTES: u64 = 64 << 20;
const MAX_HEADER_BYTES: u64 = 16 << 20;

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

pub(super) fn err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(e.to_string())
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
#[cfg(test)]
impl Tensor {
    /// A safetensors descriptor of `shape` in `dtype` with no payload.
    pub(super) fn safe_descriptor(shape: Vec<usize>, dtype: &str, bytes: u64) -> Self {
        Self {
            path: PathBuf::new(),
            shape,
            offset: 0,
            bytes,
            encoding: Encoding::Safe(dtype.into()),
        }
    }
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
}

/// One main-checkpoint shard in declared split order. `base` is aligned so
/// the direct window planner can address the whole set without losing the
/// per-file alignment required by uncached reads. Auxiliary checkpoints are
/// deliberately not part of this set.
#[derive(Clone)]
struct DirectGgufShard {
    path: PathBuf,
    identity: String,
    len: u64,
    base: u64,
}

/// The main GGUF shard set whose identities guard direct expert fills.
#[derive(Clone)]
pub(super) struct DirectGgufSource {
    shards: Vec<DirectGgufShard>,
    identity: String,
    len: u64,
}

impl DirectGgufSource {
    fn set_identity(shards: &[DirectGgufShard]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"qwen4-direct-gguf-v1");
        for (ordinal, shard) in shards.iter().enumerate() {
            hasher.update((ordinal as u64).to_le_bytes());
            hasher.update(shard.base.to_le_bytes());
            let path = shard.path.as_os_str().as_encoded_bytes();
            hasher.update((path.len() as u64).to_le_bytes());
            hasher.update(path);
            hasher.update((shard.identity.len() as u64).to_le_bytes());
            hasher.update(shard.identity.as_bytes());
        }
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn new(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            return Err(err("Qwen4 direct GGUF source has no shards"));
        }
        let mut shards = Vec::with_capacity(paths.len());
        let mut seen = HashSet::new();
        let mut next_base = 0u64;
        for path in paths {
            let path = path.canonicalize().map_err(err)?;
            if !seen.insert(path.clone()) {
                return Err(err(format!(
                    "GGUF split set contains duplicate shard '{}'",
                    path.display()
                )));
            }
            let meta = fs::metadata(&path).map_err(err)?;
            let base = next_base;
            let len = meta.len();
            next_base = base
                .checked_add(len)
                .and_then(|end| end.checked_next_multiple_of(crate::utils::gguf_read::ALIGN))
                .ok_or_else(|| err("Qwen4 direct GGUF logical size overflow"))?;
            shards.push(DirectGgufShard {
                identity: source_file_identity_digest(&path, &meta),
                path,
                len,
                base,
            });
        }
        let last = shards
            .last()
            .ok_or_else(|| err("Qwen4 direct GGUF source has no shards"))?;
        let len = last
            .base
            .checked_add(last.len)
            .ok_or_else(|| err("Qwen4 direct GGUF logical size overflow"))?;
        let identity = Self::set_identity(&shards);
        Ok(Self {
            shards,
            identity,
            len,
        })
    }

    fn has_path(&self, path: &Path) -> bool {
        self.shards.iter().any(|shard| shard.path == path)
    }

    fn open_files(&self) -> Result<Vec<File>> {
        self.shards
            .iter()
            .map(|shard| {
                File::open(&shard.path).map_err(|error| {
                    err(format!(
                        "Failed to open GGUF '{}': {error}",
                        shard.path.display()
                    ))
                })
            })
            .collect()
    }

    fn verify_path(&self) -> Result<()> {
        let mut live_shards = Vec::with_capacity(self.shards.len());
        for expected in &self.shards {
            let live = fs::metadata(&expected.path).map_err(|e| {
                err(format!(
                    "Failed to stat GGUF '{}': {e}",
                    expected.path.display()
                ))
            })?;
            if live.len() < expected.len {
                return Err(err(format!(
                    "GGUF '{}' is truncated: expected {} bytes, found {}",
                    expected.path.display(),
                    expected.len,
                    live.len()
                )));
            }
            let identity = source_file_identity_digest(&expected.path, &live);
            if live.len() != expected.len || identity != expected.identity {
                return Err(err(format!(
                    "GGUF '{}' changed since open; retry the request",
                    expected.path.display()
                )));
            }
            live_shards.push(DirectGgufShard {
                identity,
                path: expected.path.clone(),
                len: live.len(),
                base: expected.base,
            });
        }
        if Self::set_identity(&live_shards) != self.identity {
            return Err(err("GGUF split set changed since open; retry the request"));
        }
        Ok(())
    }

    fn verify(&self, files: &[&File]) -> Result<()> {
        if files.len() != self.shards.len() {
            return Err(err("Qwen4 direct GGUF shard set is incomplete"));
        }
        for (expected, file) in self.shards.iter().zip(files) {
            let live = fs::metadata(&expected.path).map_err(|e| {
                err(format!(
                    "Failed to stat GGUF '{}': {e}",
                    expected.path.display()
                ))
            })?;
            let held = file.metadata().map_err(|e| {
                err(format!(
                    "Failed to stat open GGUF '{}': {e}",
                    expected.path.display()
                ))
            })?;
            #[cfg(unix)]
            if live.len() < expected.len && live.dev() == held.dev() && live.ino() == held.ino() {
                return Err(err(format!(
                    "GGUF '{}' is truncated: expected {} bytes, found {}",
                    expected.path.display(),
                    expected.len,
                    live.len()
                )));
            }
            if live.len() != expected.len
                || held.len() != expected.len
                || source_file_identity_digest(&expected.path, &live) != expected.identity
                || source_file_identity_digest(&expected.path, &held) != expected.identity
            {
                return Err(err(format!(
                    "GGUF '{}' changed since open; retry the request",
                    expected.path.display()
                )));
            }
        }
        Ok(())
    }
}

/// Positional reads over the aligned logical shard address space. A window
/// may cross a shard boundary; only the virtual alignment gap is synthesized.
pub(super) struct DirectGgufReader<'a> {
    source: &'a DirectGgufSource,
    files: &'a [&'a File],
}

impl<'a> DirectGgufReader<'a> {
    pub(super) fn new(source: &'a DirectGgufSource, files: &'a [&'a File]) -> Result<Self> {
        if files.len() != source.shards.len() {
            return Err(err("Qwen4 direct GGUF shard set is incomplete"));
        }
        Ok(Self { source, files })
    }
}

/// A [`DirectGgufReader`] that owns its source and descriptors, for readers
/// that outlive one call.
pub(super) struct OwnedDirectGgufReader {
    source: DirectGgufSource,
    files: Vec<Arc<File>>,
}

impl OwnedDirectGgufReader {
    pub(super) fn new(source: DirectGgufSource, files: Vec<Arc<File>>) -> Result<Self> {
        if files.len() != source.shards.len() {
            return Err(err("Qwen4 direct GGUF shard set is incomplete"));
        }
        Ok(Self { source, files })
    }
}

impl crate::utils::gguf_read::ReadAt for OwnedDirectGgufReader {
    fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        let files: Vec<&File> = self.files.iter().map(AsRef::as_ref).collect();
        DirectGgufReader {
            source: &self.source,
            files: &files,
        }
        .read_exact_at(buf, off)
    }
}

impl crate::utils::gguf_read::ReadAt for DirectGgufReader<'_> {
    fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let last_path = &self.source.shards[self.source.shards.len() - 1].path;
        let end = off.checked_add(buf.len() as u64).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "GGUF shard '{}': logical read offset overflows",
                    last_path.display()
                ),
            )
        })?;
        if end > self.source.len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "GGUF shard '{}': logical read ends at {end}, beyond {} bytes",
                    last_path.display(),
                    self.source.len
                ),
            ));
        }
        let mut index = match self
            .source
            .shards
            .binary_search_by(|shard| shard.base.cmp(&off))
        {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let mut offset = off;
        let mut remaining = buf;
        while !remaining.is_empty() {
            let shard = self.source.shards.get(index).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "GGUF shard '{}': split set ended early",
                        last_path.display()
                    ),
                )
            })?;
            if offset < shard.base {
                let n = (shard.base - offset).min(remaining.len() as u64) as usize;
                remaining[..n].fill(0);
                offset += n as u64;
                remaining = &mut remaining[n..];
                continue;
            }
            let shard_end = shard.base + shard.len;
            if offset >= shard_end {
                index += 1;
                continue;
            }
            let in_shard = offset - shard.base;
            let n = (shard.len - in_shard).min(remaining.len() as u64) as usize;
            self.files[index]
                .read_exact_at(&mut remaining[..n], in_shard)
                .map_err(|error| {
                    if self.source.shards.len() == 1 {
                        error
                    } else {
                        std::io::Error::new(
                            error.kind(),
                            format!("GGUF shard '{}': {error}", shard.path.display()),
                        )
                    }
                })?;
            offset += n as u64;
            remaining = &mut remaining[n..];
        }
        Ok(())
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
    /// True when this affine Q8_0 bank was imported from GGUF `Q8_0` where
    /// every block's bias is strictly `-128 * scale` (enables symmetric INT8
    /// `q8_prefill_cuda` and fused BF16 `hc_up_mix_prefill_cuda`).
    pub(super) q8_symmetric_128: bool,
    // Persistent CUDA `QuantizedLinear` wrapper so repeated `Weight::linear`
    // calls share `cuda_dense_prefill` across prefill chunks instead of
    // re-allocating a throwaway wrapper on every projection. `None` on Metal/CPU
    // and on non-quantized or 3-D expert banks.
    pub(super) qlinear: Option<Arc<QuantizedLinear>>,
}
impl Weight {
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
            q8_symmetric_128: weights.iter().all(|w| w.q8_symmetric_128),
            qlinear: None,
        }))
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
        if super::cuda_qwen4_fast_enabled()
            && self.mode == "affine"
            && self.bits == 8
            && self.group == 32
            && let Some(biases) = &self.biases
            && let Some(fast) =
                crate::kernels::linear::dequant_q8_bf16_cuda(&self.values, scales, biases)?
        {
            return Ok(fast);
        }
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
    pub(super) fn qlinear(&self) -> Option<&QuantizedLinear> {
        self.qlinear.as_deref()
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
    /// Draft-bank bytes the direct plan budgeted beside the dense banks; the
    /// bank admission honours them, the partial path leaves them at zero.
    draft_bank_budget: u64,
    paired_banks: HashMap<(String, String), Arc<Weight>>,
    /// Cached reshaped dense arrays for resident banks so `Store::dense()` skips
    /// per-call `slice_rows` + `Arc<Weight>` + `with_cached_qlinear()` + `reshape`.
    dense_cache: HashMap<String, MxArray>,
    /// Bounded host-RAM raw-row cache for CUDA batched `lookup_rows` (`token_embd`,
    /// `per_layer_token_embd`): replaces 16,384 per-row GPU dequant/concat kernels
    /// with ONE batched `import_bytes` + ONE `mlx_dequantize` launch.
    cuda_raw_rows: HashMap<(String, usize), Vec<u8>>,
    cuda_raw_rows_order: VecDeque<(String, usize)>,
    cuda_raw_rows_bytes: u64,
    file_paths: HashMap<PathBuf, Arc<PathBuf>>,
    slots: HashMap<usize, expert_slots::ExpertSlots>,
    direct_fill: Option<expert_slots::DirectFill>,
    direct_fill_remaining: usize,
    direct_error: Option<String>,
    /// This turn admitted a layer-major prefill beside a direct fill.
    direct_fill_pass: bool,
    /// Experts a preload of the decode token in flight read, per layer: they
    /// counted as misses, so the routes that use them do not count hits.
    preloaded: HashMap<usize, Vec<u32>>,
    deferred_reduction_layer: Option<usize>,
    slot_capacity: Option<usize>,
    /// The bank ceiling of partial direct slots: the streamed store's cap
    /// less the slots' staging, or what a refusing host admits.
    slot_bank_budget: Option<u64>,
    /// The memory plan's store: whole experts or a cap, and the row cache.
    store_plan: Option<crate::engine::memory_plan::StorePlan>,
    /// The memory plan's ceiling; MLX's active bytes stay within it less the
    /// registered KV pools. Unset for a store no plan sized (fixtures).
    active_ceiling: Option<u64>,
    slot_bytes: u64,
    pub slot_upload_bytes: u64,
    pub slot_hits: u64,
    pub slot_misses: u64,
    growth_bytes: u64,
    pub bytes_read: u64,
    pub peak_cache_bytes: u64,
    gguf_source: Option<DirectGgufSource>,
    slot_fill: Option<expert_slots::SlotFill>,
    /// A slot fill whose reads run while the GPU computes the experts that
    /// hold slots; the next slot fill finishes or abandons it.
    pending_fill: Option<expert_slots::PendingFill>,
    /// No slot fill reads ahead of its lease.
    #[cfg(test)]
    pub(in super::super) no_fill_ahead: bool,
    /// Slot fills take the loop they had before one fill per lease: eight
    /// misses per fill, each through the whole-store fill's reads.
    #[cfg(test)]
    pub(in super::super) old_slot_fill: bool,
    /// The slot fills' batch (and staging buffer) bytes, set before the
    /// first slot fill.
    #[cfg(test)]
    pub(in super::super) fill_batch_bytes: Option<usize>,
    /// A slot fill fails at this batch, as on a read error.
    #[cfg(test)]
    pub(in super::super) fail_fill_batch: Option<usize>,
    /// Slot fill reader threads started and staging buffers allocated.
    #[cfg(test)]
    pub(in super::super) slot_fill_builds: [usize; 2],
    #[cfg(test)]
    direct_gguf: bool,
    #[cfg(test)]
    pub(in super::super) direct_read_windows: Vec<(u64, usize)>,
    /// Per layer of a partial store: the experts its slot leases wanted and
    /// the misses they counted.
    #[cfg(test)]
    pub(in super::super) layer_leases: HashMap<usize, (HashSet<u32>, u64)>,
    /// Per (layer, expert) of a partial store: the slot fills that read it.
    #[cfg(test)]
    pub(in super::super) expert_fills: HashMap<(usize, u32), u32>,
    /// One open descriptor per checkpoint file; row reads `pread` through it.
    files: HashMap<PathBuf, Arc<File>>,
    rows: Option<Fetcher>,
    /// Rows the reader threads or the model thread still have to read;
    /// direct expert fills yield while it is above zero.
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
    pub lookup: Arc<LookupStats>,
    pub live: Arc<Live>,
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
    /// The digest of every GGUF shard's path, size and mtime; `None` for a
    /// load that serves no GGUF source.
    pub(super) fn checkpoint_identity(&self) -> Option<&str> {
        self.gguf_source
            .as_ref()
            .map(|source| source.identity.as_str())
    }

    /// Count `bytes` read from the checkpoint files, here and in the live
    /// counters `residencyInfo` reads.
    pub(super) fn count_bytes_read(&mut self, bytes: u64) {
        self.bytes_read += bytes;
        self.live.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(super) fn check_forward_headroom(&self) -> Result<()> {
        self.check_forward_headroom_with(|| {
            super::memory::check_growth_headroom(self.plan.use_direct_turn_floor())
        })
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

    /// The store from its descriptors; `set_store_plan` sizes it before any
    /// payload read.
    pub fn open(path: &Path) -> Result<Self> {
        let mut s = Self::open_metadata(path, None)?;
        s.plan.hot_bytes = s.hot_bytes()?;
        super::memory::host_snapshot(&mut s.plan)?;
        Ok(s)
    }

    /// Hold MLX's active bytes within the memory plan's `ceiling` less the
    /// registered KV pools.
    pub(super) fn set_active_ceiling(&mut self, ceiling: u64) {
        self.active_ceiling = Some(ceiling);
    }

    /// The active-memory limit of an SSD read: the plan's ceiling less the
    /// `pool_bytes` registered now.
    pub(super) fn active_limit(&self, pool_bytes: u64) -> Option<u64> {
        self.active_ceiling
            .map(|ceiling| ceiling.saturating_sub(pool_bytes))
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

    /// The fixture under the production forward reserve from here on.
    #[cfg(test)]
    pub(in super::super) fn end_fixture_exemption(&mut self) {
        self.fixture = false;
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
    pub(super) fn open_metadata(path: &Path, _root: Option<&Path>) -> Result<Self> {
        let s = Self::open_descriptors(path)?;
        #[cfg(unix)]
        if s.gguf {
            let root =
                _root.map_or_else(|| super::packed_cache::cache_root(path), Path::to_path_buf);
            super::packed_cache::sweep_old_packs(&root);
        }
        Ok(s)
    }

    /// The tensors' descriptors and nothing else: no pack sweep, no payload.
    pub(super) fn open_descriptors(path: &Path) -> Result<Self> {
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
                direct: false,
                partial_direct_gate: false,
                direct_bytes: 0,
                direct_layers: 0,
                bank_budget: 0,
                live_non_reclaimable: 0,
                host: None,
                anonymous_need: 0,
                self_mlx_bytes: 0,
            },
            banks: HashMap::new(),
            bank_bytes: 0,
            mtp_banks_declined: false,
            draft_bank_budget: 0,
            paired_banks: HashMap::new(),
            dense_cache: HashMap::new(),
            cuda_raw_rows: HashMap::new(),
            cuda_raw_rows_order: VecDeque::new(),
            cuda_raw_rows_bytes: 0,
            file_paths: HashMap::new(),
            slots: HashMap::new(),
            direct_fill: None,
            direct_fill_remaining: 0,
            direct_error: None,
            direct_fill_pass: false,
            preloaded: HashMap::new(),
            deferred_reduction_layer: None,
            slot_capacity: None,
            slot_bank_budget: None,
            store_plan: None,
            active_ceiling: None,
            slot_bytes: 0,
            slot_upload_bytes: 0,
            slot_hits: 0,
            slot_misses: 0,
            growth_bytes: 0,
            bytes_read: 0,
            peak_cache_bytes: 0,
            gguf_source: None,
            slot_fill: None,
            pending_fill: None,
            #[cfg(test)]
            no_fill_ahead: false,
            #[cfg(test)]
            old_slot_fill: false,
            #[cfg(test)]
            fill_batch_bytes: None,
            #[cfg(test)]
            fail_fill_batch: None,
            #[cfg(test)]
            slot_fill_builds: [0; 2],
            #[cfg(test)]
            direct_gguf: true,
            #[cfg(test)]
            direct_read_windows: Vec::new(),
            #[cfg(test)]
            layer_leases: HashMap::new(),
            #[cfg(test)]
            expert_fills: HashMap::new(),
            files: HashMap::new(),
            rows: None,
            rows_pending: Arc::default(),
            #[cfg(test)]
            row_readers: true,
            #[cfg(test)]
            single_row_projections: 0,
            #[cfg(test)]
            vector_windows: Vec::new(),
            lookup: Arc::default(),
            live: Arc::default(),
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
        for (i, path) in paths.iter().enumerate() {
            let i = i as u32;
            let g = if i == 0 {
                None
            } else {
                Some(parse_gguf(path)?)
            };
            let g = g.as_ref().unwrap_or(&initial);
            crate::utils::gguf_source::check_tensor_bounds(
                g,
                fs::metadata(path).map_err(err)?.len(),
                path,
            )?;
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
        let source = DirectGgufSource::new(&paths)?;
        source.verify_path()?;
        let files = source.open_files()?;
        let file_refs: Vec<&File> = files.iter().collect();
        source.verify(&file_refs)?;
        self.gguf_source = Some(source);
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
        self.slot_bank_budget = None;
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
        self.read_impl(name, start, rows, false)
    }

    /// Import a bounded group through the ordinary checkpoint reader.
    fn read_expert_batch(
        &mut self,
        requests: &[(String, usize, usize)],
    ) -> Result<Vec<Arc<Weight>>> {
        requests
            .iter()
            .map(|(name, start, rows)| self.read(name, *start, *rows))
            .collect()
    }

    fn file(&mut self, path: &Path) -> Result<Arc<File>> {
        if let Some(file) = self.files.get(path) {
            return Ok(Arc::clone(file));
        }
        let file = Arc::new(File::open(path).map_err(err)?);
        self.files.insert(path.to_path_buf(), Arc::clone(&file));
        Ok(file)
    }

    fn path_arc(&mut self, path: &Path) -> Arc<PathBuf> {
        if let Some(arc) = self.file_paths.get(path) {
            return Arc::clone(arc);
        }
        let arc = Arc::new(path.to_path_buf());
        self.file_paths.insert(path.to_path_buf(), Arc::clone(&arc));
        arc
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
    ) -> Result<Arc<Weight>> {
        let direct_gate = self.plan.use_direct_turn_floor();
        self.read_impl_with(name, start, rows, lookup, || {
            super::memory::check_growth_headroom(direct_gate)
        })
    }

    fn read_impl_with(
        &mut self,
        name: &str,
        start: usize,
        rows: usize,
        lookup: bool,
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
        if let Some(limit) =
            self.active_limit(crate::cache_limit::coordinator().registered_pool_bytes())
        {
            super::memory::check_active_memory_budget(
                limit,
                crate::array::memory::anonymous_active_memory,
                crate::array::memory::clear_cache,
                std::thread::sleep,
            )?;
        }
        super::memory::maintain_freelist(self.plan.physical_bytes);
        // Full direct experts do not pass through this per-row importer;
        // token/PLE rows, auxiliary weights and fallback projection chunks
        // still consume the applicable headroom reserve.
        if cfg!(target_os = "macos") && self.growth_bytes >= 256 << 20 {
            growth_check()?;
            self.growth_bytes = 0;
        }
        self.evict_to(cache_limit.saturating_sub(MAX_READ_BYTES * 3))?;
        let weight = self.import_rows(&d, name, start, rows)?;
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
        self.count_bytes_read(bytes);
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
                q8_symmetric_128: false,
                qlinear: None,
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
        self.import_bytes_with_cache(buf, rows, true)
    }

    pub(super) fn import_bytes_with_cache(
        &self,
        buf: &[u8],
        rows: usize,
        cache_qlinear: bool,
    ) -> Result<Weight> {
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
                    q8_symmetric_128: false,
                    qlinear: None,
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
                let w = Weight {
                    dense_bf16: None,
                    values,
                    scales,
                    biases,
                    group,
                    bits,
                    mode: mode.into(),
                    q8_symmetric_128: matches!(ty, GgufTensorType::Q8_0),
                    qlinear: None,
                };
                if cache_qlinear {
                    w.with_cached_qlinear()
                } else {
                    w
                }
            }
        })
    }
}

impl Store {
    /// True when the checkpoint's token embedding produces BF16 activations
    /// (GGUF quantized / BF16 or safetensors BF16), as opposed to FP32 fixtures.
    pub(super) fn embedding_is_bf16(&self) -> bool {
        let Some(d) = self
            .tensors
            .get("token_embd.weight")
            .or_else(|| self.tensors.get("embed_tokens.weight"))
        else {
            return false;
        };
        match &d.encoding {
            Encoding::Gguf(ty) => !matches!(ty, GgufTensorType::F32),
            Encoding::Safe(dt) => dt == "BF16",
        }
    }

    pub fn dense(&mut self, name: &str) -> Result<MxArray> {
        if let Some(cached) = self.dense_cache.get(name) {
            return Ok(cached.clone());
        }
        let d = self.descriptor(name)?;
        let shape = d.shape.iter().map(|&i| i as i64).collect::<Vec<_>>();
        let rows = d.rows()?;
        let out = self.read(name, 0, rows)?.dense()?.reshape(&shape)?;
        if self.banks.contains_key(name) {
            self.dense_cache.insert(name.to_owned(), out.clone());
        }
        Ok(out)
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
pub(super) fn routed_expert_tensor(name: &str) -> Option<(usize, usize)> {
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
