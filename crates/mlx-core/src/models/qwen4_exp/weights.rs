//! Bounded, positional SSD reads. Descriptors never create MLX arrays. Only
//! selected expert matrices and embedding rows are read on cache misses.
//! Admitted hot banks can be prepared once; PLE stays row-addressable. Routed
//! GGUF experts fill anonymous banks directly from the source checkpoint.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use napi::{Error, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::live::{Fallback, Live};
use super::row_fetch::{Fetcher, Job};
use crate::array::{DType, MxArray};
use crate::models::quantized_linear::QuantizedLinear;
use crate::utils::gguf::{
    GgufFile, GgufLoadOptions, GgufMetaValue, GgufTensorInfo, GgufTensorType, load_gguf_tensor,
    parse_gguf, source_file_identity_digest, symmetric_zero_point,
};

#[path = "expert_slots.rs"]
mod expert_slots;
#[path = "residency.rs"]
mod residency;

pub const CACHE_BYTES: u64 = 8 << 30;
pub(super) const MAX_READ_BYTES: u64 = 64 << 20;
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// Reader threads for token/PLE rows: 17 random 90 B-2.7 KB reads per token.
/// Direct expert fills yield while these rows are pending.
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
            std::os::unix::fs::FileExt::read_exact_at(
                self.files[index],
                &mut remaining[..n],
                in_shard,
            )
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
    /// Draft-bank bytes the direct plan budgeted beside the dense banks; the
    /// bank admission honours them, the partial path leaves them at zero.
    draft_bank_budget: u64,
    paired_banks: HashMap<(String, String), Arc<Weight>>,
    slots: HashMap<usize, expert_slots::ExpertSlots>,
    direct_fill: Option<expert_slots::DirectFill>,
    direct_fill_remaining: usize,
    direct_error: Option<String>,
    /// This turn admitted a layer-major prefill beside a direct fill.
    direct_fill_pass: bool,
    /// A slow source keeps its filled banks instead of purging them at idle.
    direct_slow_source: bool,
    slot_reclaim_since: Option<std::time::Instant>,
    wide_routes: HashMap<usize, Vec<u32>>,
    deferred_reduction_layer: Option<usize>,
    slot_capacity: Option<usize>,
    /// Gate-derived ceiling for direct partial banks. Its presence also keeps
    /// load reconciliation from replacing the live-pressure plan with the
    /// older bootstrap rule.
    slot_bank_budget: Option<u64>,
    slot_bytes: u64,
    pub slot_upload_bytes: u64,
    pub slot_hits: u64,
    pub slot_misses: u64,
    growth_bytes: u64,
    pub bytes_read: u64,
    pub peak_cache_bytes: u64,
    gguf_source: Option<DirectGgufSource>,
    #[cfg(test)]
    direct_gguf: bool,
    #[cfg(test)]
    direct_read_windows: Vec<(u64, usize)>,
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
    pub(super) fn check_forward_headroom(&self) -> Result<()> {
        self.check_forward_headroom_with(|| super::memory::check_growth_headroom(self.plan.direct))
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
                direct: false,
                direct_bytes: 0,
                direct_layers: 0,
                bank_budget: 0,
                live_non_reclaimable: 0,
            },
            banks: HashMap::new(),
            bank_bytes: 0,
            mtp_banks_declined: false,
            draft_bank_budget: 0,
            paired_banks: HashMap::new(),
            slots: HashMap::new(),
            direct_fill: None,
            direct_fill_remaining: 0,
            direct_error: None,
            direct_fill_pass: false,
            direct_slow_source: false,
            slot_reclaim_since: None,
            wide_routes: HashMap::new(),
            deferred_reduction_layer: None,
            slot_capacity: None,
            slot_bank_budget: None,
            slot_bytes: 0,
            slot_upload_bytes: 0,
            slot_hits: 0,
            slot_misses: 0,
            growth_bytes: 0,
            bytes_read: 0,
            peak_cache_bytes: 0,
            gguf_source: None,
            #[cfg(test)]
            direct_gguf: true,
            #[cfg(test)]
            direct_read_windows: Vec::new(),
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
        if s.gguf {
            let root =
                root.map_or_else(|| super::packed_cache::cache_root(path), Path::to_path_buf);
            super::packed_cache::sweep_old_packs(&root);
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
                let row = self.read_impl(name, id, 1, true)?.dense_pending()?;
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
        if imported.len() > 1 {
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
    ) -> Result<Arc<Weight>> {
        let direct = self.plan.direct;
        self.read_impl_with(name, start, rows, lookup, || {
            super::memory::check_growth_headroom(direct)
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
        super::memory::check_active_memory_budget(
            cache_limit,
            crate::array::memory::anonymous_active_memory,
            crate::array::memory::clear_cache,
            std::thread::sleep,
        )?;
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
        self.bytes_read += bytes;
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
    fn direct_store_keeps_the_forward_headroom_guard() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut production = Store::open_metadata(&path, None).unwrap();
        production.plan.direct = true;
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        assert!(production.check_forward_headroom_with(unavailable).is_err());
        assert_eq!(
            checked.get(),
            1,
            "direct store skipped the forward headroom check"
        );
    }

    /// Full direct experts bypass the row importer; every other
    /// read is anonymous growth and keeps the 256 MiB system-headroom recheck.
    #[test]
    fn direct_store_rechecks_growth_headroom_for_anonymous_rows() {
        if !cfg!(target_os = "macos") {
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp");
        let mut store = Store::open_metadata(&path, None).unwrap();
        store.plan.direct = true;
        let checked = std::cell::Cell::new(0);
        let unavailable = || {
            checked.set(checked.get() + 1);
            Err(err("simulated exhausted system headroom"))
        };
        let row = store
            .read_impl_with("embed_tokens.weight", 0, 1, false, unavailable)
            .unwrap();
        assert_eq!(
            checked.get(),
            0,
            "a read under 256 MiB of growth queried headroom"
        );
        assert_eq!(
            store.growth_bytes,
            row.bytes().unwrap(),
            "direct mode must charge anonymous rows as growth"
        );
        store.growth_bytes = 256 << 20;
        let error = match store.read_impl_with("embed_tokens.weight", 1, 1, false, unavailable) {
            Ok(_) => panic!("direct store ignored exhausted growth headroom"),
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
            .read_impl_with("embed_tokens.weight", 1, 1, false, || Ok(()))
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
        let rows = 8u32;
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
mod direct_tests {
    use super::*;
    use crate::models::qwen4_exp::memory;
    use std::time::{Duration, Instant};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("qwen4-direct-store-{}", uuid::Uuid::new_v4()));
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
    type Part = (&'static str, usize, usize, u32, usize);
    const PARTS: [Part; 3] = [
        // (name, rows per expert, width, GGUF type, block bytes per row)
        ("blk.0.ffn_gate_exps.weight", 32, 256, 12, 144),
        ("blk.0.ffn_up_exps.weight", 32, 256, 12, 144),
        ("blk.0.ffn_down_exps.weight", 128, 64, 7, 48),
    ];
    const Q8_PARTS: [Part; 3] = [
        PARTS[0],
        PARTS[1],
        ("blk.0.ffn_down_exps.weight", 128, 64, 8, 68),
    ];
    const Q5K_PARTS: [Part; 3] = [
        ("blk.0.ffn_gate_exps.weight", 32, 256, 13, 176),
        ("blk.0.ffn_up_exps.weight", 32, 256, 13, 176),
        PARTS[2],
    ];
    const FLASH_PARTS: [Part; 3] = [
        ("blk.0.ffn_gate_exps.weight", 640, 2560, 12, 1440),
        ("blk.0.ffn_up_exps.weight", 640, 2560, 12, 1440),
        ("blk.0.ffn_down_exps.weight", 2560, 640, 7, 480),
    ];

    /// Three routed expert tensors of one layer: Q4_K gate/up and Q5_1 down,
    /// filled with deterministic pseudo-random block bytes.
    fn write_fixture(path: &Path) {
        write_fixture_layers(path, 1)
    }

    /// The same three tensors for layers `0..layers`.
    fn write_fixture_layers(path: &Path, layers: usize) {
        write_fixture_layout(path, layers, EXPERTS, &PARTS, 0, 0);
    }

    fn write_fixture_layout(
        path: &Path,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
        header_pad: usize,
        trailing: usize,
    ) {
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(((parts.len() * layers) as u64).to_le_bytes());
        b.extend(1u64.to_le_bytes());
        string(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        string(&mut b, "qwen4exp");
        let mut offset = 0u64;
        for layer in 0..layers {
            for (name, rows, width, ty, row_bytes) in parts {
                string(&mut b, &name.replace("blk.0.", &format!("blk.{layer}.")));
                b.extend(3u32.to_le_bytes());
                for d in [*width as u64, *rows as u64, experts as u64] {
                    b.extend(d.to_le_bytes());
                }
                b.extend(ty.to_le_bytes());
                b.extend(offset.to_le_bytes());
                offset += (experts * *rows * *row_bytes) as u64;
            }
        }
        b.resize(b.len() + header_pad, 0);
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        let mut state = 0x2545_f491u32;
        for _ in 0..offset {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            b.push((state >> 24) as u8);
        }
        b.resize(b.len() + trailing, 0x5a);
        fs::write(path, b).unwrap();
    }

    fn write_split_fixture_layout(
        dir: &Path,
        label: &str,
        shards: usize,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
    ) -> Vec<PathBuf> {
        write_split_fixture_layout_with_empty_first(
            dir, label, shards, layers, experts, parts, false,
        )
    }

    fn write_split_fixture_layout_with_empty_first(
        dir: &Path,
        label: &str,
        shards: usize,
        layers: usize,
        experts: usize,
        parts: &[Part; 3],
        empty_first: bool,
    ) -> Vec<PathBuf> {
        assert!(shards > 1);
        let paths: Vec<PathBuf> = (1..=shards)
            .map(|ordinal| dir.join(format!("{label}-{ordinal:05}-of-{shards:05}.gguf")))
            .collect();
        let mut assignments = vec![Vec::new(); shards];
        for (index, (layer, part)) in (0..layers)
            .flat_map(|layer| (0..parts.len()).map(move |part| (layer, part)))
            .enumerate()
        {
            let target = if empty_first {
                (index % (shards - 1)) + 1
            } else {
                index % shards
            };
            assignments[target].push((layer, part));
        }
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.bytes());
        };
        let mut state = 0x2545_f491u32;
        for (shard, assigned) in assignments.into_iter().enumerate() {
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend((assigned.len() as u64).to_le_bytes());
            b.extend(7u64.to_le_bytes());
            string(&mut b, "general.architecture");
            b.extend(8u32.to_le_bytes());
            string(&mut b, "qwen4exp");
            for (key, value) in [
                ("split.no", shard as u64),
                ("split.count", shards as u64),
                ("split.tensors.count", (layers * parts.len()) as u64),
                ("qwen4exp.expert_count", experts as u64),
                ("qwen4exp.expert_used_count", 2),
            ] {
                string(&mut b, key);
                b.extend(10u32.to_le_bytes());
                b.extend(value.to_le_bytes());
            }
            string(&mut b, "general.alignment");
            b.extend(4u32.to_le_bytes());
            b.extend(32u32.to_le_bytes());
            let mut offset = 0u64;
            for &(layer, part) in &assigned {
                let (name, rows, width, ty, row_bytes) = parts[part];
                string(&mut b, &name.replace("blk.0.", &format!("blk.{layer}.")));
                b.extend(3u32.to_le_bytes());
                for d in [width as u64, rows as u64, experts as u64] {
                    b.extend(d.to_le_bytes());
                }
                b.extend(ty.to_le_bytes());
                b.extend(offset.to_le_bytes());
                offset += (experts * rows * row_bytes) as u64;
            }
            while !b.len().is_multiple_of(32) {
                b.push(0);
            }
            for _ in 0..offset {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                b.push((state >> 24) as u8);
            }
            fs::write(&paths[shard], b).unwrap();
        }
        paths
    }

    fn open_split_slots(
        temp: &Temp,
        label: &str,
        parts: &[Part; 3],
        layers: usize,
        shards: usize,
    ) -> (Store, Vec<PathBuf>) {
        let paths = write_split_fixture_layout(&temp.0, label, shards, layers, EXPERTS, parts);
        let mut store =
            Store::open_fixture(&paths[0], Some(&temp.0.join(format!("{label}-cache")))).unwrap();
        store.slot_capacity = Some(EXPERTS);
        (store, paths)
    }

    fn assert_split_read_windows(store: &Store) {
        let source = store.gguf_source.as_ref().unwrap();
        assert!(source.shards.len() > 1);
        assert!(!store.direct_read_windows.is_empty());
        let mut crossed = false;
        for &(off, len) in &store.direct_read_windows {
            let len = len as u64;
            assert!(len <= 8 << 20 && off % (16 << 10) == 0);
            for shard in &source.shards {
                let actual_start = off.max(shard.base);
                let actual_end = (off + len).min(shard.base + shard.len);
                if actual_start < actual_end {
                    assert_eq!((actual_start - shard.base) % (16 << 10), 0);
                }
                if off < shard.base && off + len > shard.base {
                    crossed = true;
                }
            }
        }
        assert!(crossed, "no direct read window crossed a shard boundary");
    }

    fn encoded(weight: &Weight) -> (Vec<u8>, i32, i32, String) {
        let mut data = Vec::new();
        for array in [
            Some(&weight.values),
            weight.scales.as_ref(),
            weight.biases.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let dtype = array.dtype().unwrap();
            data.extend(format!("{dtype:?}:{:?};", array.shape().unwrap().to_vec()).as_bytes());
            match dtype {
                DType::Uint32 => data.extend(
                    array
                        .to_uint32()
                        .unwrap()
                        .iter()
                        .flat_map(|value| value.to_le_bytes()),
                ),
                DType::Float32 => data.extend(
                    array
                        .to_float32()
                        .unwrap()
                        .iter()
                        .flat_map(|value| value.to_le_bytes()),
                ),
                DType::Float16 | DType::BFloat16 => data.extend(
                    array
                        .to_uint16_native()
                        .unwrap()
                        .iter()
                        .flat_map(|value| value.to_le_bytes()),
                ),
                DType::Uint8 => data.extend(array.to_uint8().unwrap()),
                DType::Int8 => data.extend(array.to_int8().unwrap().into_iter().map(|v| v as u8)),
                other => panic!("unsupported test oracle dtype {other:?}"),
            }
        }
        (data, weight.group, weight.bits, weight.mode.clone())
    }

    #[test]
    fn routed_expert_names_are_the_only_direct_bank_sources() {
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

    fn open_slots(temp: &Temp, gguf: &Path, label: &str, capacity: usize, direct: bool) -> Store {
        let mut store =
            Store::open_fixture(gguf, Some(&temp.0.join(format!("{label}-cache")))).unwrap();
        store.slot_capacity = Some(capacity);
        store.direct_gguf = direct;
        store
    }

    fn open_full(temp: &Temp, gguf: &Path, label: &str, layers: usize, experts: usize) -> Store {
        let mut store =
            Store::open_fixture(gguf, Some(&temp.0.join(format!("{label}-cache")))).unwrap();
        let handled = store
            .select_direct_residency_with(experts, 0, &mut memory::direct_plan, &mut |_| Ok(()))
            .unwrap();
        assert!(
            handled,
            "the single-file fixture must take direct selection"
        );
        assert!(store.plan.direct);
        assert_eq!(store.plan.direct_layers, layers);
        assert_eq!(store.plan.partial_expert_slots, experts);
        assert_eq!(store.plan.direct_bytes, store.slot_bytes);
        assert_eq!(store.cache_bytes, store.slot_bytes);
        assert_eq!(store.slots.len(), layers);
        store
    }

    fn raw_fill_bytes(store: &Store, parts: &[Part; 3], ids: &[u32]) -> u64 {
        ids.iter()
            .flat_map(|&expert| {
                parts.iter().map(move |&(name, rows, ..)| {
                    store
                        .descriptor(name)
                        .unwrap()
                        .range(expert as usize * rows, rows)
                        .unwrap()
                        .1
                })
            })
            .sum()
    }

    /// Compare every value/scale/bias byte of every requested filled slot
    /// with the ordinary CPU checkpoint importer for that expert.
    fn assert_banks_match_checkpoint(
        store: &Store,
        parts: &[Part; 3],
        banks: &[Arc<Weight>; 3],
        ids: &[u32],
        local: &[u32],
    ) -> Vec<u8> {
        assert_banks_match_checkpoint_layer(store, 0, parts, banks, ids, local)
    }

    fn assert_banks_match_checkpoint_layer(
        store: &Store,
        layer: usize,
        parts: &[Part; 3],
        banks: &[Arc<Weight>; 3],
        ids: &[u32],
        local: &[u32],
    ) -> Vec<u8> {
        let mut all = Vec::new();
        for (position, &expert) in ids.iter().enumerate() {
            for (part, &(name, rows, ..)) in parts.iter().enumerate() {
                let name = name.replace("blk.0.", &format!("blk.{layer}."));
                let expected = store
                    .descriptor(&name)
                    .unwrap()
                    .import_rows(expert as usize * rows, rows)
                    .unwrap();
                let actual = banks[part]
                    .slice_rows(local[position] as usize * rows, rows)
                    .unwrap();
                let actual = encoded(&actual);
                assert_eq!(actual, encoded(&expected), "{name} expert {expert}");
                all.extend(actual.0);
            }
        }
        all
    }

    #[test]
    fn direct_gguf_split_reader_preserves_bytes_gaps_and_bounds() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let paths = write_split_fixture_layout_with_empty_first(
            &temp.0,
            "split-reader",
            4,
            2,
            EXPERTS,
            &PARTS,
            true,
        );
        assert!(parse_gguf(&paths[0]).unwrap().tensors.is_empty());
        let source = DirectGgufSource::new(&paths).unwrap();
        let files = source.open_files().unwrap();
        let refs: Vec<&File> = files.iter().collect();
        let reader = DirectGgufReader::new(&source, &refs).unwrap();
        let mut actual = vec![0u8; source.len as usize];
        crate::utils::gguf_read::ReadAt::read_exact_at(&reader, &mut actual, 0).unwrap();
        for (index, shard) in source.shards.iter().enumerate() {
            let begin = shard.base as usize;
            let end = begin + shard.len as usize;
            assert_eq!(&actual[begin..end], fs::read(&paths[index]).unwrap());
            if let Some(next) = source.shards.get(index + 1) {
                assert!(
                    actual[end..next.base as usize]
                        .iter()
                        .all(|&byte| byte == 0)
                );
            }
        }
        let error =
            crate::utils::gguf_read::ReadAt::read_exact_at(&reader, &mut [0u8; 2], source.len - 1)
                .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn direct_gguf_split_formats_match_checkpoint_and_cross_shards() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        for (label, parts) in [
            ("split-q4k-q5-1", &PARTS),
            ("split-q5k", &Q5K_PARTS),
            ("split-q8-bias", &Q8_PARTS),
        ] {
            let (mut store, _) = open_split_slots(&temp, label, parts, 2, 3);
            let ids: Vec<u32> = (0..EXPERTS as u32).collect();
            for layer in 0..2 {
                let (banks, local, _) = store
                    .expert_slots(layer, EXPERTS, 2, &ids)
                    .unwrap()
                    .unwrap();
                assert_banks_match_checkpoint_layer(&store, layer, parts, &banks, &ids, &local);
            }
            assert_split_read_windows(&store);
        }
    }

    #[test]
    fn direct_gguf_split_full_capacity_and_refill_match_checkpoint() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let paths = write_split_fixture_layout_with_empty_first(
            &temp.0,
            "split-full",
            4,
            2,
            EXPERTS,
            &PARTS,
            true,
        );
        let mut store = open_full(&temp, &paths[0], "split-full-direct", 2, EXPERTS);
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        store.activate_expert_slots().unwrap();
        for refill in 0..2 {
            for layer in 0..2 {
                let (banks, local, capacity) = store
                    .expert_slots(layer, EXPERTS, 2, &ids)
                    .unwrap()
                    .unwrap();
                assert_eq!(capacity, EXPERTS);
                assert_eq!(local, ids);
                assert_banks_match_checkpoint_layer(&store, layer, &PARTS, &banks, &ids, &local);
            }
            assert!(store.device_slots_ready(2));
            assert_eq!(store.slot_misses, 0);
            assert!(store.plan.direct);
            assert_eq!(store.plan.direct_layers, 2);
            assert_eq!(store.plan.direct_bytes, store.slot_bytes);
            if refill == 0 {
                store.purge_expert_slots().unwrap();
                store.activate_expert_slots().unwrap();
            }
        }
        assert_eq!(store.slot_upload_bytes, 2 * store.slot_bytes);
        assert_split_read_windows(&store);
    }

    #[test]
    fn direct_gguf_split_source_failures_name_the_failed_shard() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let (mut flipped, paths) = open_split_slots(&temp, "split-flip", &PARTS, 1, 3);
        let parsed = parse_gguf(&paths[1]).unwrap();
        let tensor = parsed.tensors.first().unwrap();
        let at = (parsed.data_offset + tensor.offset) as usize + 8;
        let mut bytes = fs::read(&paths[1]).unwrap();
        bytes[at] ^= 0x20;
        fs::write(&paths[1], bytes).unwrap();
        let error = flipped.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("changed since open"), "{error}");
        assert!(
            error.reason.contains("split-flip-00002-of-00003.gguf"),
            "{error}"
        );
        assert!(!flipped.device_slots_ready(1));

        let (mut missing, paths) = open_split_slots(&temp, "split-missing", &PARTS, 1, 3);
        fs::remove_file(&paths[2]).unwrap();
        let error = missing.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("Failed to stat GGUF"), "{error}");
        assert!(
            error.reason.contains("split-missing-00003-of-00003.gguf"),
            "{error}"
        );
        assert!(!missing.device_slots_ready(1));

        let (mut truncated, paths) = open_split_slots(&temp, "split-truncated", &PARTS, 1, 3);
        let len = fs::metadata(&paths[1]).unwrap().len();
        fs::OpenOptions::new()
            .write(true)
            .open(&paths[1])
            .unwrap()
            .set_len(len - 1)
            .unwrap();
        let error = truncated.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("truncated"), "{error}");
        assert!(
            error.reason.contains("split-truncated-00002-of-00003.gguf"),
            "{error}"
        );
        assert!(!truncated.device_slots_ready(1));
    }

    #[test]
    fn direct_gguf_incomplete_or_reordered_splits_fail_before_residency() {
        let temp = Temp::new();
        let paths = write_split_fixture_layout(&temp.0, "split-incomplete", 3, 1, EXPERTS, &PARTS);
        fs::remove_file(&paths[1]).unwrap();
        let error = match Store::open_fixture(&paths[0], Some(&temp.0.join("incomplete-cache"))) {
            Ok(_) => panic!("an incomplete split set was accepted"),
            Err(error) => error,
        };
        assert!(error.reason.contains("Expected one GGUF split"), "{error}");
        assert!(
            error.reason.contains("split-incomplete-00002-of-00003"),
            "{error}"
        );

        let paths = write_split_fixture_layout(&temp.0, "split-reordered", 3, 1, EXPERTS, &PARTS);
        let held = temp.0.join("split-reordered.held");
        fs::rename(&paths[0], &held).unwrap();
        fs::rename(&paths[1], &paths[0]).unwrap();
        fs::rename(&held, &paths[1]).unwrap();
        let error = match Store::open_fixture(&paths[0], Some(&temp.0.join("reordered-cache"))) {
            Ok(_) => panic!("a reordered split set was accepted"),
            Err(error) => error,
        };
        assert!(
            error.reason.contains("Inconsistent GGUF split metadata")
                || error.reason.contains("Open the first GGUF split"),
            "{error}"
        );
    }

    #[test]
    fn direct_gguf_full_gate_refusal_selects_bounded_partial_slots() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("full-gate-refusal.gguf");
        write_fixture(&gguf);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("cache"))).unwrap();
        let anonymous_budget = 512 << 20;
        let handled = store
            .select_direct_residency_with(
                EXPERTS,
                0,
                &mut |plan, _, _, bank_bytes| {
                    plan.budget = anonymous_budget;
                    plan.bank_budget = bank_bytes * 2 / EXPERTS as u64;
                    plan.live_non_reclaimable = 40 << 30;
                    plan.available_bytes = Some(88 << 30);
                    plan.physical_bytes = Some(128 << 30);
                    Err(err(
                        "simulated full-bank refusal beside a 40 GiB live-pressure row",
                    ))
                },
                &mut |_| panic!("a refused gate must not allocate"),
            )
            .unwrap();
        assert!(handled);
        assert!(!store.plan.direct);
        assert!(store.slots.is_empty());
        assert_eq!(store.slot_bytes, 0);
        assert_eq!(store.cache_bytes, 0);
        assert_eq!(store.expert_slot_capacity(EXPERTS, 2).unwrap(), 2);
        let bank_budget = store.plan.bank_budget;
        assert_eq!(store.slot_bank_budget, Some(bank_budget));
        assert_eq!(store.plan.budget, anonymous_budget + bank_budget);
        assert!(store.planned_slot_bytes().unwrap() <= bank_budget);
        assert_eq!(store.plan.live_non_reclaimable, 40 << 30);
        assert_eq!(store.plan.direct_bytes, 0);
        assert_eq!(store.plan.direct_layers, 0);
        assert!(matches!(
            store.live.fallback(),
            Some(Fallback::LoadGate { budget, .. }) if budget == bank_budget
        ));
    }

    /// Real fixture inference through the partial direct fill path. The gate
    /// rows are injected so the two capacities and the 40/20 GiB pressure
    /// cases are deterministic; they are not real-model throughput rows.
    #[test]
    fn direct_partial_slot_text_matches_cpu_import_at_two_capacities() {
        use super::super::decoder;
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let path = temp.0.join("quantized-fixture.gguf");
        let base_config = super::super::tests::write_direct_gguf_fixture(&path);
        let experts = base_config.num_experts;
        let make_store = || Store::open_fixture(&path, Some(&temp.0.join("unused-cache"))).unwrap();
        let prompt = [3u32, 9, 4, 7, 1, 5, 8, 2, 3, 6, 9, 4];
        // Oracle: the same checkpoint through the bounded per-row CPU
        // importer, with no direct slots or derived pack involved.
        let store = make_store();
        let mut oracle = decoder::Decoder::new(base_config.clone(), store).unwrap();
        oracle.register_cache_limit();
        oracle.weights.prepare_fixture_hot().unwrap();
        oracle.weights.slot_capacity = Some(0);
        oracle.weights.publish_live();
        oracle.residency_changed();
        oracle.weights.begin_turn().unwrap();
        let mut logits = oracle.prefill_prompt(&prompt).unwrap();
        let mut expected = Vec::new();
        for step in 0..6 {
            let id = super::super::tests::argmax(&logits);
            expected.push(id);
            if step < 5 {
                logits = oracle.step(id).unwrap();
            }
        }

        for (capacity, live_gib) in [(2usize, 40u64), (3, 20)] {
            let store = make_store();
            let mut decoder = decoder::Decoder::new(base_config.clone(), store).unwrap();
            decoder.register_cache_limit();
            let anonymous_budget = 512 << 20;
            let handled = decoder
                .weights
                .select_direct_residency_with(
                    experts,
                    0,
                    &mut |plan, _, _, bank_bytes| {
                        plan.budget = anonymous_budget;
                        plan.bank_budget = bank_bytes * capacity as u64 / experts as u64;
                        plan.live_non_reclaimable = live_gib << 30;
                        plan.available_bytes = Some((128 - live_gib) << 30);
                        plan.physical_bytes = Some(128 << 30);
                        Err(err("injected partial-capacity gate row"))
                    },
                    &mut |_| panic!("partial selection must not allocate full banks"),
                )
                .unwrap();
            assert!(handled);
            let bank_budget = decoder.weights.plan.bank_budget;
            assert!(bank_budget > 0);
            assert!(!decoder.weights.plan.direct);
            assert_eq!(decoder.weights.plan.partial_expert_slots, capacity);
            assert_eq!(decoder.weights.slot_capacity, Some(capacity));
            assert_eq!(decoder.weights.slot_bank_budget, Some(bank_budget));
            assert_eq!(decoder.weights.plan.live_non_reclaimable, live_gib << 30);
            assert!(decoder.weights.planned_slot_bytes().unwrap() <= bank_budget);
            decoder.weights.prepare_fixture_hot().unwrap();
            decoder.weights.publish_live();
            decoder.residency_changed();
            assert_eq!(
                decoder.weights.cache_budget(),
                anonymous_budget + bank_budget
            );

            decoder.weights.begin_turn().unwrap();
            let mut logits = decoder.prefill_prompt(&prompt).unwrap();
            let unique = decoder
                .weights
                .wide_routes
                .values()
                .map(Vec::len)
                .sum::<usize>() as u64;
            assert!(unique > 0);
            assert_eq!(
                decoder.weights.slot_misses, unique,
                "last-assignment expert chunks re-read an expert during prefill"
            );
            assert!(decoder.weights.slot_bytes <= bank_budget);
            assert!(!decoder.weights.direct_read_windows.is_empty());
            assert!(
                decoder
                    .weights
                    .direct_read_windows
                    .iter()
                    .all(|&(offset, len)| len <= 8 << 20 && offset % (16 << 10) == 0)
            );
            let mut actual = Vec::new();
            for step in 0..6 {
                let id = super::super::tests::argmax(&logits);
                actual.push(id);
                if step < 5 {
                    logits = decoder.step(id).unwrap();
                }
            }
            assert_eq!(actual, expected, "capacity {capacity}, live {live_gib} GiB");
            assert_eq!(
                decoder.weights.slots.len(),
                decoder.config.num_hidden_layers
            );
            assert!(
                decoder
                    .weights
                    .device_slots_ready(decoder.config.num_hidden_layers)
            );
            assert!(decoder.weights.slot_bytes <= bank_budget);
            assert!(decoder.weights.cache_bytes <= decoder.weights.cache_limit);
            assert!(decoder.weights.peak_cache_bytes <= decoder.weights.cache_limit);
            eprintln!(
                "synthetic pressure row (not a real-model benchmark): live={live_gib} GiB, capacity={capacity}, prefill unique misses={unique}, slot bytes={}, text={actual:?}",
                decoder.weights.slot_bytes,
            );
        }
    }

    #[test]
    fn direct_gguf_full_capacity_fills_layers_with_identity_and_refills_empty() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        const LAYERS: usize = 3;
        let temp = Temp::new();
        let gguf = temp.0.join("full-capacity.gguf");
        write_fixture_layers(&gguf, LAYERS);
        let mut store = open_full(&temp, &gguf, "full-capacity-direct", LAYERS, EXPERTS);
        let source_bytes: u64 = store
            .tensors
            .iter()
            .filter(|(name, _)| name.contains("_exps.weight"))
            .map(|(_, tensor)| tensor.bytes)
            .sum();
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        assert!(!store.device_slots_ready(LAYERS));
        assert!(store.prefill_device_slots_ready(LAYERS));
        store.activate_expert_slots().unwrap();
        assert!(store.direct_fill_pass());
        for layer in [1, 2, 0] {
            let (banks, local, capacity) = store
                .expert_slots(layer, EXPERTS, 2, &ids)
                .unwrap()
                .unwrap();
            assert_eq!(capacity, EXPERTS);
            assert_eq!(local, ids);
            assert_banks_match_checkpoint_layer(&store, layer, &PARTS, &banks, &ids, &local);
        }
        assert!(store.device_slots_ready(LAYERS));
        assert_eq!(store.slot_misses, 0);
        assert_eq!(store.bytes_read, source_bytes);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);
        assert_eq!(store.cache_bytes, store.slot_bytes);
        assert_eq!(
            store.live.direct_filled_layers.load(Ordering::Acquire),
            LAYERS as u64
        );
        assert!(!store.direct_read_windows.is_empty());
        assert!(
            store
                .direct_read_windows
                .iter()
                .all(|&(off, len)| len <= 8 << 20 && off % (16 << 10) == 0)
        );
        assert!(
            store
                .commit_device_routes(&[0, 1, 2], &[0, 1, 2], 1)
                .unwrap()
        );

        let allocated = store.slot_bytes;
        let first_upload = store.slot_upload_bytes;
        store.purge_expert_slots().unwrap();
        assert!(!store.device_slots_ready(LAYERS));
        store.activate_expert_slots().unwrap();
        assert!(store.direct_fill_pass());
        let before = store.bytes_read;
        for layer in 0..LAYERS {
            let (banks, local, _) = store
                .expert_slots(layer, EXPERTS, 2, &ids)
                .unwrap()
                .unwrap();
            assert_eq!(local, ids);
            assert_banks_match_checkpoint_layer(&store, layer, &PARTS, &banks, &ids, &local);
        }
        assert_eq!(store.bytes_read, before + source_bytes);
        assert_eq!(store.slot_upload_bytes, first_upload + allocated);
        assert_eq!(store.slot_bytes, allocated);
        assert_eq!(store.cache_bytes, allocated);
        assert_eq!(store.slot_misses, 0);
        assert!(store.device_slots_ready(LAYERS));
    }

    #[test]
    fn direct_gguf_full_fill_error_stays_unpublished_and_rejects_changed_source() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("full-source-error.gguf");
        write_fixture(&gguf);
        let mut store = open_full(&temp, &gguf, "full-source-error-direct", 1, EXPERTS);
        let held = temp.0.join("full-source-error.held");
        fs::rename(&gguf, &held).unwrap();
        store.start_direct_fill().unwrap();
        let error = store.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("Failed to open GGUF"), "{error}");
        assert!(!store.device_slots_ready(1));
        fs::rename(&held, &gguf).unwrap();
        store.activate_expert_slots().unwrap();
        let error = store.expert_slots(0, EXPERTS, 2, &[0]).err().unwrap();
        assert!(error.reason.contains("changed since open"), "{error}");
        assert_eq!(store.slot_misses, 0);
        assert!(!store.device_slots_ready(1));
    }

    #[test]
    fn direct_gguf_q5_1_slots_match_cpu_import_eviction_and_purge_refill() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("q5_1.gguf");
        write_fixture_layout(&gguf, 1, EXPERTS, &PARTS, 32, 17);
        let mut store = open_slots(&temp, &gguf, "q5_1-direct", 2, true);
        let ids = [0u32, 1];
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let mut compared =
            assert_banks_match_checkpoint(&store, &PARTS, &banks, &ids, &local).len();
        let first_fill = raw_fill_bytes(&store, &PARTS, &ids);
        assert_eq!(store.bytes_read, first_fill);

        // One miss evicts expert 0 and reuses its bank; expert 1 stays warm.
        let next = [2u32, 1];
        let one_fill = raw_fill_bytes(&store, &PARTS, &[2]);
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &next).unwrap().unwrap();
        assert_eq!(local, [0, 1], "the LRU bank was reused");
        compared += assert_banks_match_checkpoint(&store, &PARTS, &banks, &next, &local).len();
        assert_eq!(store.bytes_read, first_fill + one_fill);

        // Volatile banks either survive or take the same Empty refill path;
        // explicit Empty guarantees the fallback and requires a full refill.
        store.reclaim_expert_slots().unwrap();
        let before = store.bytes_read;
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &next).unwrap().unwrap();
        compared += assert_banks_match_checkpoint(&store, &PARTS, &banks, &next, &local).len();
        assert!((store.bytes_read - before) <= first_fill);
        store.purge_expert_slots().unwrap();
        let before = store.bytes_read;
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &next).unwrap().unwrap();
        compared += assert_banks_match_checkpoint(&store, &PARTS, &banks, &next, &local).len();
        assert_eq!(store.bytes_read, before + first_fill);
        assert!(store.device_slots_ready(1));
        eprintln!(
            "P3.0-d Q5_1 parity: {compared} bank bytes, eviction + volatile + explicit Empty refill"
        );
    }

    #[test]
    fn direct_gguf_partial_source_error_invalidates_changed_source() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("partial-source-error.gguf");
        write_fixture(&gguf);
        let mut store = open_slots(&temp, &gguf, "partial-source-error-direct", 2, true);
        store.expert_slots(0, EXPERTS, 2, &[0, 1]).unwrap();
        let held = temp.0.join("partial-source-error.held");
        fs::rename(&gguf, &held).unwrap();
        let error = store.expert_slots(0, EXPERTS, 2, &[2, 3]).err().unwrap();
        assert!(error.reason.contains("Failed to stat GGUF"), "{error}");
        assert!(!store.device_slots_ready(1));

        fs::rename(&held, &gguf).unwrap();
        let before = store.bytes_read;
        let error = store.expert_slots(0, EXPERTS, 2, &[2, 3]).err().unwrap();
        assert!(error.reason.contains("changed since open"), "{error}");
        assert_eq!(store.bytes_read, before);
        assert!(!store.device_slots_ready(1));
    }

    #[test]
    fn direct_gguf_idle_fallback_forces_a_fresh_fill() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("idle-fallback.gguf");
        write_fixture(&gguf);
        let mut store = open_slots(&temp, &gguf, "idle-fallback-direct", EXPERTS, true);
        let ids = [0u32, 1];
        store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        let first_fill = raw_fill_bytes(&store, &PARTS, &ids);
        store.reclaim_expert_slots().unwrap();
        store.slot_reclaim_since = Some(Instant::now() - Duration::from_secs(31));
        store.activate_expert_slots().unwrap();
        let before = store.bytes_read;
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_banks_match_checkpoint(&store, &PARTS, &banks, &ids, &local);
        assert_eq!(store.bytes_read, before + first_fill);
        assert!(store.slot_reclaim_since.is_none());
    }

    #[test]
    fn direct_gguf_source_flips_match_cpu_import_for_q5_1_and_q8_0_bias() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        for (label, parts, flip) in [
            ("q5_1", &PARTS, 8usize),
            ("q5_k", &Q5K_PARTS, 8),
            ("q8_0", &Q8_PARTS, 1),
        ] {
            let gguf = temp.0.join(format!("{label}-flip.gguf"));
            write_fixture_layout(&gguf, 1, EXPERTS, parts, 64, 13);
            let mut before = open_slots(&temp, &gguf, &format!("{label}-before"), EXPERTS, true);
            let ids = [0u32];
            let (banks, local, _) = before.expert_slots(0, EXPERTS, 1, &ids).unwrap().unwrap();
            let before = assert_banks_match_checkpoint(&before, parts, &banks, &ids, &local);

            let parsed = parse_gguf(&gguf).unwrap();
            let tensor = parsed
                .tensors
                .iter()
                .find(|t| t.name == "blk.0.ffn_down_exps.weight")
                .unwrap();
            let at = (parsed.data_offset + tensor.offset) as usize + flip;
            let mut bytes = fs::read(&gguf).unwrap();
            bytes[at] ^= 0x20;
            fs::write(&gguf, &bytes).unwrap();
            let mut after = open_slots(&temp, &gguf, &format!("{label}-after"), EXPERTS, true);
            let (banks, local, _) = after.expert_slots(0, EXPERTS, 1, &ids).unwrap().unwrap();
            let after_bytes = assert_banks_match_checkpoint(&after, parts, &banks, &ids, &local);
            assert_ne!(before, after_bytes, "{label}: source flip had no effect");

            // A source that changes after open is never mixed with the old
            // identity across repeated direct fills.
            let mut bytes = fs::read(&gguf).unwrap();
            bytes[at + 1] ^= 0x04;
            fs::write(&gguf, &bytes).unwrap();
            let error = after.expert_slots(0, EXPERTS, 1, &[1]).err().unwrap();
            assert!(error.reason.contains("changed since open"), "{error}");
            assert!(!after.device_slots_ready(1));
        }
    }

    #[test]
    fn direct_gguf_missing_and_truncated_sources_fail_without_stale_slots() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        for truncated in [false, true] {
            let temp = Temp::new();
            let gguf = temp.0.join("source-error.gguf");
            write_fixture(&gguf);
            let mut store = open_slots(&temp, &gguf, "source-error-direct", EXPERTS, true);
            if truncated {
                let len = fs::metadata(&gguf).unwrap().len();
                fs::OpenOptions::new()
                    .write(true)
                    .open(&gguf)
                    .unwrap()
                    .set_len(len - 1)
                    .unwrap();
            } else {
                fs::remove_file(&gguf).unwrap();
            }
            let error = store.expert_slots(0, EXPERTS, 2, &[0, 1]).err().unwrap();
            if truncated {
                assert!(error.reason.contains("truncated"), "{error}");
            } else {
                assert!(error.reason.contains("Failed to stat GGUF"), "{error}");
            }
            assert!(error.reason.contains("source-error.gguf"), "{error}");
            assert!(!store.device_slots_ready(1));
            assert!(store.direct_read_windows.is_empty());
        }
    }

    #[test]
    fn direct_gguf_kernel_unaligned_offsets_use_cpu_import() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("unaligned.gguf");
        write_fixture(&gguf);
        let mut store = open_slots(&temp, &gguf, "unaligned-direct", EXPERTS, true);
        let names = ["gate", "up", "down"].map(|part| format!("blk.0.ffn_{part}_exps.weight"));
        store.tensors.get_mut(&names[0]).unwrap().offset += 1;
        assert!(!store.direct_layer(&names));
        assert!(store.expert_slots(0, EXPERTS, 2, &[0]).unwrap().is_none());
        store.select_residency(0).unwrap();
        assert_eq!(store.plan.partial_expert_slots, 0);
        assert_eq!(store.slot_capacity, Some(0));
    }

    #[test]
    fn direct_gguf_flash_geometry_matches_checkpoint_and_layers_reads_at_8mib() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        const EXPERTS: usize = 8;
        let temp = Temp::new();
        let gguf = temp.0.join("flash-geometry.gguf");
        write_fixture_layout(&gguf, 1, EXPERTS, &FLASH_PARTS, 0, 0);
        let mut store = open_slots(&temp, &gguf, "flash-geometry-direct", EXPERTS, true);
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 8, &ids).unwrap().unwrap();
        let compared =
            assert_banks_match_checkpoint(&store, &FLASH_PARTS, &banks, &ids, &local).len();
        assert_eq!(store.bytes_read, 24_576_000);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);
        assert!(store.direct_read_windows.len() > 1);
        assert!(
            store
                .direct_read_windows
                .iter()
                .all(|&(off, len)| len <= 8 << 20 && off % (16 << 10) == 0),
            "{:?}",
            store.direct_read_windows
        );
        eprintln!(
            "P3.0-d Flash geometry parity: {compared} bank bytes in {} aligned windows",
            store.direct_read_windows.len()
        );
    }

    #[test]
    fn direct_full_banks_preserve_identity_routing_and_kernel_results() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("identity.gguf");
        write_fixture(&gguf);
        let mut store = open_full(&temp, &gguf, "identity-direct", 1, EXPERTS);
        store.activate_expert_slots().unwrap();
        let ids = [4u32, 0, 2, 2, 1, 3];
        let (banks, local, capacity) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_eq!(local, ids);
        assert_eq!(capacity, EXPERTS);
        let device = MxArray::from_uint32(&ids, &[2, 3]).unwrap();
        let (_, device_local) = store.device_expert_slots(0, &device).unwrap().unwrap();
        assert_eq!(&*device_local.to_uint32().unwrap(), &ids);
        let (_, safe, count) = store.prefill_device_expert_slots(0, &device).unwrap();
        assert_eq!(&*safe.to_uint32().unwrap(), &ids);
        assert_eq!(count, EXPERTS);
        for _ in 0..3 {
            assert!(store.commit_device_routes(&[0, 0, 0], &ids, 2).unwrap());
        }
        assert_eq!(store.slot_misses, 0);
        assert_eq!(store.slot_hits, 5 + 3 * 5);
        assert_eq!(store.slot_upload_bytes, store.slot_bytes);

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
        let bits = |array: &MxArray| {
            array
                .to_float32()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        for (part, &(name, rows, ..)) in PARTS.iter().enumerate() {
            let chunks: Vec<_> = (0..EXPERTS)
                .map(|expert| {
                    store
                        .descriptor(name)
                        .unwrap()
                        .import_rows(expert * rows, rows)
                        .map(Arc::new)
                })
                .collect::<Result<_>>()
                .unwrap();
            let oracle = Weight::concatenate_rows(&chunks).unwrap();
            assert_eq!(encoded(&banks[part]), encoded(&oracle), "{name}");
            if part == 0 {
                let actual = banks[0].gather_rows(&x, &selected, EXPERTS, false).unwrap();
                let expected = oracle.gather_rows(&x, &selected, EXPERTS, false).unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                store.finish_expert_slots(0, &actual).unwrap();
                store.complete_expert_window(0, &expected).unwrap();
            }
        }
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
    }

    #[test]
    fn direct_full_load_writes_no_derived_weight_pack() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("no-pack.gguf");
        write_fixture(&gguf);
        let cache = temp.0.join("fresh-research-cache");
        let mut store = Store::open_fixture(&gguf, Some(&cache)).unwrap();
        assert!(
            store
                .select_direct_residency_with(EXPERTS, 0, &mut memory::direct_plan, &mut |_| Ok(()))
                .unwrap()
        );
        store.activate_expert_slots().unwrap();
        let ids: Vec<u32> = (0..EXPERTS as u32).collect();
        let (banks, local, _) = store.expert_slots(0, EXPERTS, 2, &ids).unwrap().unwrap();
        assert_banks_match_checkpoint(&store, &PARTS, &banks, &ids, &local);
        assert!(
            !cache.exists(),
            "the direct load created a weight-pack root"
        );
        let entries: Vec<_> = fs::read_dir(&temp.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries.len(), 1, "unexpected derived files: {entries:?}");
        assert!(entries[0].to_string_lossy().ends_with(".gguf"));
    }

    #[test]
    fn direct_load_admits_draft_banks_before_starting_expert_fill() {
        if !crate::utils::gguf_gpu::gpu_repack_available() {
            return;
        }
        let temp = Temp::new();
        let gguf = temp.0.join("draft-order.gguf");
        write_fixture(&gguf);
        let mut store = Store::open_fixture(&gguf, Some(&temp.0.join("draft-cache"))).unwrap();
        let aux =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/bf16/paged");
        store
            .attach_auxiliary(Store::open_fixture(&aux, None).unwrap())
            .unwrap();
        let estimate = store.mtp_bank_estimate().unwrap();
        let dense = store.fixed_projection_bytes().unwrap();
        assert!(estimate > 0);
        let mut charged = None;
        assert!(
            store
                .select_direct_residency_with(
                    EXPERTS,
                    1 << 30,
                    &mut |plan, need, pool, banks| {
                        charged = Some((need, pool, banks));
                        memory::direct_plan(plan, need, pool, banks)
                    },
                    &mut |_| Ok(()),
                )
                .unwrap()
        );
        assert_eq!(
            charged,
            Some((dense + estimate, 1 << 30, store.plan.direct_bytes))
        );
        assert_eq!(store.draft_bank_budget, estimate);
        store.prepare_fixture_hot().unwrap();
        let live = Arc::clone(&store.live);
        store
            .finish_load_with(&mut |_| {
                assert!(!live.direct_fill_in_progress.load(Ordering::Acquire));
                Ok(())
            })
            .unwrap();
        assert_eq!(store.mtp_bank_bytes().unwrap(), estimate);
        assert!(
            store.direct_fill.is_some(),
            "the direct fill starts after the draft banks"
        );
    }
}
