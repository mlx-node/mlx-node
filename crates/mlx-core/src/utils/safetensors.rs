/**
 * SafeTensors Format Loader
 *
 * Loads weights from the SafeTensors format (https://github.com/huggingface/safetensors).
 *
 * SafeTensors Format:
 * - 8 bytes: Header length N (u64, little-endian)
 * - N bytes: JSON header with tensor metadata
 * - Remaining bytes: Raw tensor data in the order specified in header
 *
 * Header JSON format:
 * {
 *   "tensor_name": {
 *     "dtype": "F32"|"F16"|"BF16"|"I32"|...,
 *     "shape": [dim1, dim2, ...],
 *     "data_offsets": [start, end]
 *   },
 *   ...
 *   "__metadata__": { ... }  // Optional metadata
 * }
 */
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use napi::bindgen_prelude::*;
use serde::{Deserialize, Serialize};

use crate::array::{DType, MxArray};

/// Supported tensor data types in SafeTensors format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[allow(non_camel_case_types)]
pub enum SafeTensorDType {
    F32,     // float32
    F16,     // float16
    BF16,    // bfloat16
    I32,     // int32
    I64,     // int64
    U8,      // uint8
    U32,     // uint32 (packed quantized weights)
    I8,      // int8
    F64,     // float64
    BOOL,    // boolean
    F8_E4M3, // FP8 E4M3 (8-bit float, used by DeepSeek/Qwen FP8 models)
}

impl SafeTensorDType {
    /// Get the byte size of this dtype
    pub fn byte_size(&self) -> usize {
        match self {
            SafeTensorDType::F32 => 4,
            SafeTensorDType::F16 => 2,
            SafeTensorDType::BF16 => 2,
            SafeTensorDType::I32 => 4,
            SafeTensorDType::I64 => 8,
            SafeTensorDType::U8 => 1,
            SafeTensorDType::U32 => 4,
            SafeTensorDType::I8 => 1,
            SafeTensorDType::F64 => 8,
            SafeTensorDType::BOOL => 1,
            SafeTensorDType::F8_E4M3 => 1,
        }
    }

    /// Convert to MLX DType if supported
    pub fn to_mlx_dtype(&self) -> Option<DType> {
        match self {
            SafeTensorDType::F32 => Some(DType::Float32),
            SafeTensorDType::F16 => Some(DType::Float16),
            SafeTensorDType::BF16 => Some(DType::BFloat16),
            SafeTensorDType::I32 => Some(DType::Int32),
            SafeTensorDType::U8 => Some(DType::Uint8),
            SafeTensorDType::U32 => Some(DType::Uint32),
            SafeTensorDType::I8 => Some(DType::Int8),
            _ => None, // Unsupported dtypes
        }
    }
}

/// Tensor metadata from SafeTensors header
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: SafeTensorDType,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2], // [start, end]
}

impl TensorInfo {
    /// Calculate the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate expected byte size
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }

    /// Validate that offsets match expected size
    pub fn validate(&self) -> Result<()> {
        let expected_bytes = self.byte_size();
        let actual_bytes = self.data_offsets[1] - self.data_offsets[0];

        if expected_bytes != actual_bytes {
            return Err(Error::from_reason(format!(
                "Tensor size mismatch: expected {} bytes, got {}",
                expected_bytes, actual_bytes
            )));
        }

        Ok(())
    }
}

/// SafeTensors file structure
pub struct SafeTensorsFile {
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: Option<serde_json::Value>,
    data_offset: usize, // Offset where tensor data begins
}

impl SafeTensorsFile {
    /// Load SafeTensors file from path
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::from_reason(format!("Failed to open file: {}", e)))?;

        // Read header length (first 8 bytes, little-endian u64)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)
            .map_err(|e| Error::from_reason(format!("Failed to read header length: {}", e)))?;

        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // Validate header length (prevent absurdly large allocations)
        if header_len > 100_000_000 {
            // 100MB max header
            return Err(Error::from_reason(format!(
                "Invalid header length: {} bytes (too large)",
                header_len
            )));
        }

        // Read header JSON
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|e| Error::from_reason(format!("Failed to read header: {}", e)))?;

        let header_str = String::from_utf8(header_bytes)
            .map_err(|e| Error::from_reason(format!("Invalid UTF-8 in header: {}", e)))?;

        // Parse JSON header
        let header: serde_json::Value = serde_json::from_str(&header_str)
            .map_err(|e| Error::from_reason(format!("Failed to parse header JSON: {}", e)))?;

        let header_obj = header
            .as_object()
            .ok_or_else(|| Error::from_reason("Header must be a JSON object".to_string()))?;

        // Extract tensors and metadata
        let mut tensors = HashMap::new();
        let mut metadata = None;

        for (key, value) in header_obj.iter() {
            if key == "__metadata__" {
                metadata = Some(value.clone());
            } else {
                // Parse tensor info
                let tensor_info: TensorInfo =
                    serde_json::from_value(value.clone()).map_err(|e| {
                        Error::from_reason(format!(
                            "Failed to parse tensor info for {}: {}",
                            key, e
                        ))
                    })?;

                // Validate tensor info
                tensor_info.validate()?;

                tensors.insert(key.clone(), tensor_info);
            }
        }

        // Data starts after header
        let data_offset = 8 + header_len;

        Ok(SafeTensorsFile {
            tensors,
            metadata,
            data_offset,
        })
    }

    /// Load all tensors into MxArrays
    pub fn load_tensors<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, MxArray>> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| Error::from_reason(format!("Failed to open file: {}", e)))?;

        let mut result = HashMap::new();

        for (name, info) in self.tensors.iter() {
            let array = self.load_tensor(&mut file, name, info)?;
            result.insert(name.clone(), array);
        }

        Ok(result)
    }

    /// Load a single tensor
    fn load_tensor(&self, file: &mut File, name: &str, info: &TensorInfo) -> Result<MxArray> {
        // Seek to tensor data
        let absolute_offset = self.data_offset + info.data_offsets[0];
        file.seek(SeekFrom::Start(absolute_offset as u64))
            .map_err(|e| Error::from_reason(format!("Failed to seek to tensor {}: {}", name, e)))?;

        // Read tensor data
        let byte_size = info.byte_size();
        let mut buffer = vec![0u8; byte_size];
        file.read_exact(&mut buffer).map_err(|e| {
            Error::from_reason(format!("Failed to read tensor data for {}: {}", name, e))
        })?;

        // Convert to MxArray based on dtype
        let shape: Vec<i64> = info.shape.iter().map(|&x| x as i64).collect();

        match &info.dtype {
            SafeTensorDType::F32 => {
                // Convert bytes to f32 array
                let float_data = bytes_to_f32(&buffer);
                MxArray::from_float32(&float_data, &shape)
            }
            SafeTensorDType::F16 => {
                // Load f16 directly without f32 conversion - saves ~2x memory
                let u16_data = bytes_to_u16(&buffer);
                MxArray::from_float16(&u16_data, &shape)
            }
            SafeTensorDType::BF16 => {
                // Load bf16 directly without f32 conversion - saves ~2x memory
                // Previously this created 3 copies: f32 Vec + f32 MxArray + bf16 MxArray
                // Now we create just 1 copy: bf16 MxArray directly from raw bytes
                let u16_data = bytes_to_u16(&buffer);
                MxArray::from_bfloat16(&u16_data, &shape)
            }
            SafeTensorDType::I32 => {
                // Convert bytes to i32 array
                let int_data = bytes_to_i32(&buffer);
                MxArray::from_int32(&int_data, &shape)
            }
            SafeTensorDType::F8_E4M3 | SafeTensorDType::U8 => {
                // Load as raw uint8 - FP8 dequantization or MXFP8 scales
                MxArray::from_uint8(&buffer, &shape)
            }
            SafeTensorDType::I8 => {
                // Load as raw int8 - sym8 per-channel symmetric quantized
                // weights. Bit-reinterpret the bytes (from_int8), NOT
                // from_uint8 + astype (that converts numerically: 0xFF would
                // become 255, not -1).
                let i8_data: &[i8] = unsafe {
                    std::slice::from_raw_parts(buffer.as_ptr() as *const i8, buffer.len())
                };
                MxArray::from_int8(i8_data, &shape)
            }
            SafeTensorDType::U32 => {
                // Load as uint32 - packed quantized weights
                let u32_data = bytes_to_u32(&buffer);
                MxArray::from_uint32(&u32_data, &shape)
            }
            _ => Err(Error::from_reason(format!(
                "Unsupported dtype for tensor {}: {:?}. Supported: F32, F16, BF16, I8, I32, U8, U32, F8_E4M3",
                name, info.dtype
            ))),
        }
    }

    /// Get information about all tensors
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.tensors.values().map(|t| t.numel()).sum()
    }
}

// Helper functions for byte conversion

/// Convert bytes to f32 array (little-endian)
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert bytes to i32 array (little-endian)
fn bytes_to_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert bytes to u32 array (little-endian)
/// Used for packed quantized weight loading
fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert bytes to u16 array (little-endian)
/// Used for direct bf16/f16 loading without f32 conversion
fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

// ============================================================================
// SafeTensors Writer
// ============================================================================

/// Maximum shard size in bytes (5 GB), matching mlx-lm and mlx-vlm.
const MAX_SHARD_SIZE: usize = 5 << 30;

/// Default minimum byte size at which a passthrough tensor is written by
/// reading the source file directly instead of materializing it through MLX.
///
/// Chosen safely below the smallest real Metal per-buffer cap (3.5 GiB on the
/// memory-constrained CI runner that triggered the original OOM). A whole-tensor
/// MLX `eval()` over a tensor at or above this size risks allocating one Metal
/// buffer that trips `metal::malloc`. Below it, the normal `array_to_bytes`
/// path is unchanged.
const DEFAULT_RAW_PASSTHROUGH_THRESHOLD_BYTES: u64 = 2 << 30; // 2 GiB

/// Resolve the raw-passthrough byte threshold, honoring the
/// `MLX_CONVERT_RAW_THRESHOLD_BYTES` override (used by tests / large-tensor
/// debugging). Falls back to [`DEFAULT_RAW_PASSTHROUGH_THRESHOLD_BYTES`] when
/// the var is unset or unparseable.
fn raw_passthrough_threshold_bytes() -> u64 {
    std::env::var("MLX_CONVERT_RAW_THRESHOLD_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_RAW_PASSTHROUGH_THRESHOLD_BYTES)
}

/// Where to find a dest tensor's raw bytes in a SOURCE safetensors file.
///
/// A dest tensor is eligible for this path only when convert proved it is an
/// unmodified passthrough of a source tensor (same MLX array handle after
/// sanitize ⇒ no astype / quant / slice / stack happened) AND it is a 2-byte
/// float (bf16/f16), whose safetensors on-disk encoding is a pure
/// little-endian bit layout identical to what the MLX writer would emit.
/// The bytes are therefore byte-identical to `array_to_bytes` by construction.
#[derive(Debug, Clone)]
pub struct PassthroughSource {
    /// Absolute path to the source safetensors shard file.
    pub file_path: std::path::PathBuf,
    /// Absolute byte offset of this tensor's data within the file
    /// (`8 + header_len + data_offsets[begin]`).
    pub file_offset: u64,
    /// Tensor byte length (`data_offsets[end] - data_offsets[begin]`).
    pub byte_len: usize,
    /// Source dtype, used as a cheap pre-read guard against the dest header.
    pub dtype: DType,
}

/// A recorded source tensor's provenance, keyed during convert by the loaded
/// MLX array's raw handle pointer.
///
/// `_keep_alive` pins the source `MxArray` (a clone shares the `Arc<MxHandle>`)
/// for the whole convert so its handle pointer can never be freed and reused by
/// a later allocation — without this, a dropped source's address could be
/// recycled by an `astype`/quantize result and falsely match as a passthrough.
/// The clone is a lazy handle (no materialized data), so the memory cost is just
/// the handle structs.
pub struct SourceProvenance {
    _keep_alive: MxArray,
    pub source: PassthroughSource,
}

/// For one SOURCE safetensors file, record the on-disk location of every tensor
/// that is still present (by name) in `loaded`, keyed by the loaded MLX array's
/// raw handle pointer (as `usize`).
///
/// This is the basis of convert's passthrough provenance: the handle pointer is
/// stable for the lifetime of the lazily-loaded array, and a pure HashMap rename
/// preserves it (move, no clone of the underlying array), whereas any transform
/// (`astype` / quantize / slice / stack) constructs a NEW array with a new
/// handle. So a dest tensor whose handle is still in this map is provably an
/// unmodified passthrough of this file's tensor.
///
/// Only 2-byte float source tensors (bf16/f16) are recorded — their safetensors
/// on-disk bytes are a pure little-endian bit layout byte-identical to what the
/// MLX writer emits. Other dtypes are excluded conservatively.
pub fn record_passthrough_sources<P: AsRef<Path>>(
    source_file: P,
    loaded: &HashMap<String, MxArray>,
    out: &mut HashMap<usize, SourceProvenance>,
) -> Result<()> {
    let path = source_file.as_ref();
    let st = SafeTensorsFile::load(path)?;
    for (name, info) in st.tensors.iter() {
        let Some(array) = loaded.get(name) else {
            continue;
        };
        // Only bf16/f16 — see fn doc.
        let dtype = match info.dtype {
            SafeTensorDType::BF16 => DType::BFloat16,
            SafeTensorDType::F16 => DType::Float16,
            _ => continue,
        };
        let byte_len = info.data_offsets[1] - info.data_offsets[0];
        let file_offset = (st.data_offset + info.data_offsets[0]) as u64;
        // Keep the source array alive (clone shares the `Arc<MxHandle>`) so its
        // raw handle pointer is PINNED for the whole convert. This closes an ABA
        // hazard: if a source array were dropped and its MLX buffer freed, a
        // later `astype`/quantize allocation could reuse the same address and
        // collide with a stale entry — a false passthrough = corrupt output.
        // Pinning the handle guarantees no other array can ever hold this
        // pointer, so a dest handle match is genuinely the same array.
        out.insert(
            array.as_raw_ptr() as usize,
            SourceProvenance {
                _keep_alive: array.clone(),
                source: PassthroughSource {
                    file_path: path.to_path_buf(),
                    file_offset,
                    byte_len,
                    dtype,
                },
            },
        );
    }
    Ok(())
}

/// Read a passthrough tensor's raw bytes straight from the source file via FFI,
/// constructing no MLX array (so no Metal per-buffer cap applies). The host
/// `Vec` is sized to the tensor; host RAM has no per-buffer cap.
fn read_passthrough_bytes(src: &PassthroughSource) -> Result<Vec<u8>> {
    use mlx_sys as sys;

    let path_str = src
        .file_path
        .to_str()
        .ok_or_else(|| Error::from_reason("Passthrough source path is not valid UTF-8"))?;
    let c_path = std::ffi::CString::new(path_str)
        .map_err(|_| Error::from_reason("Passthrough source path contains null byte"))?;

    let mut buf = vec![0u8; src.byte_len];
    let ok = unsafe {
        sys::mlx_safetensor_read_raw(
            c_path.as_ptr(),
            src.file_offset,
            buf.as_mut_ptr(),
            buf.len(),
        )
    };
    if !ok {
        return Err(Error::from_reason(format!(
            "Failed to read passthrough tensor bytes from {} (offset={}, len={})",
            path_str, src.file_offset, src.byte_len
        )));
    }
    Ok(buf)
}

/// Load a dense bf16 2-D tensor as axis-0 row shards, each `<= shard_byte_budget`
/// bytes, streaming the rows straight from the source file (no whole-tensor MLX
/// array is ever built, so the Metal per-buffer cap never applies).
///
/// Returns `(shards, rows_per_shard)`: `shards` concatenated along axis 0 equal
/// the full `[vocab, cols]` table; `rows_per_shard` is the uniform axis-0 stride
/// (every shard but the last has exactly this many rows). Used for tables like
/// gemma-4-E2B's ~4GB `embed_tokens_per_layer.weight` that exceed the cap.
/// The canonical checkpoint `.safetensors` under `dir`, matching exactly what
/// the main loader (`crate::engine::persistence::load_all_safetensors`) reads:
/// a single `weights.safetensors`/`model.safetensors` when present, otherwise
/// the sorted `model-*-of-*.safetensors` shards. Stray or auxiliary
/// `.safetensors` files in the directory are deliberately ignored so a sharded
/// lookup resolves against the same files that produced the sanitized params.
fn checkpoint_safetensors_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    for single in ["weights.safetensors", "model.safetensors"] {
        let p = dir.join(single);
        if p.exists() {
            return vec![p];
        }
    }
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_shard = (name.starts_with("model-") || name.starts_with("model.safetensors-"))
                && name.ends_with(".safetensors")
                && name.contains("-of-");
            if is_shard {
                shards.push(entry.path());
            }
        }
    }
    shards.sort();
    shards
}

pub(crate) fn load_bf16_tensor_sharded(
    model_dir: &std::path::Path,
    key: &str,
    shard_byte_budget: usize,
) -> Result<(Vec<MxArray>, i64)> {
    use mlx_sys as sys;

    // Locate the file holding `key`, reading only each file's header for the
    // data offset + shape + dtype, and only over the canonical checkpoint files
    // (`checkpoint_safetensors_files`) so a stray `.safetensors` can neither win
    // the lookup nor abort it. `key` is the sanitized (prefix-stripped) name,
    // but checkpoints store tensors under their raw names (gemma-4 wraps
    // everything in `model.language_model.`). Resolution is decided GLOBALLY
    // across the shard files rather than file-by-file — file order is
    // unspecified and a tensor lives in exactly one shard:
    //   * an exact `key` match anywhere wins over any `*.{key}` suffix match;
    //   * a `*.{key}` suffix match is taken only when no exact match exists;
    //   * 2+ exact matches, or 2+ suffix matches with no exact, is a loud error
    //     rather than silently gathering the wrong tensor.
    let suffix = format!(".{key}");
    let mut exact: Vec<(std::path::PathBuf, TensorInfo, usize)> = Vec::new();
    let mut suffixed: Vec<(std::path::PathBuf, String, TensorInfo, usize)> = Vec::new();
    for p in checkpoint_safetensors_files(model_dir) {
        let st = SafeTensorsFile::load(&p)?;
        if let Some(info) = st.tensors.get(key) {
            exact.push((p.clone(), info.clone(), st.data_offset));
        }
        for (name, info) in &st.tensors {
            if name != key && name.ends_with(&suffix) {
                suffixed.push((p.clone(), name.clone(), info.clone(), st.data_offset));
            }
        }
    }
    let (file_path, info, data_offset) = if exact.len() > 1 {
        return Err(Error::from_reason(format!(
            "tensor {key} appears as an exact key in {} safetensors under {model_dir:?}; refusing to guess",
            exact.len()
        )));
    } else if let Some(hit) = exact.pop() {
        hit
    } else if suffixed.len() > 1 {
        let names: Vec<&str> = suffixed
            .iter()
            .map(|(_, name, _, _)| name.as_str())
            .collect();
        return Err(Error::from_reason(format!(
            "ambiguous `*.{key}` suffix matched {} tensors under {model_dir:?}: {names:?}; refusing to guess",
            names.len()
        )));
    } else if let Some((p, _name, info, off)) = suffixed.pop() {
        (p, info, off)
    } else {
        return Err(Error::from_reason(format!(
            "tensor {key} (or a `*.{key}` suffix) not found in any safetensors under {model_dir:?}"
        )));
    };

    if !matches!(info.dtype, SafeTensorDType::BF16) {
        return Err(Error::from_reason(format!(
            "load_bf16_tensor_sharded: {key} is {:?}, expected BF16",
            info.dtype
        )));
    }
    if info.shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "load_bf16_tensor_sharded: {key} is {}-D, expected 2-D",
            info.shape.len()
        )));
    }
    let vocab = info.shape[0] as i64;
    let cols = info.shape[1] as i64;
    let row_bytes = cols as usize * 2; // bf16 = 2 bytes/element
    if row_bytes == 0 {
        return Err(Error::from_reason(format!(
            "load_bf16_tensor_sharded: {key} has zero columns"
        )));
    }
    let rows_per_shard = ((shard_byte_budget / row_bytes).max(1)) as i64;
    // Absolute file offset where this tensor's data begins.
    let tensor_base = (data_offset + info.data_offsets[0]) as u64;

    let path_str = file_path
        .to_str()
        .ok_or_else(|| Error::from_reason("safetensors path is not valid UTF-8"))?;
    let c_path = std::ffi::CString::new(path_str)
        .map_err(|_| Error::from_reason("safetensors path contains a null byte"))?;

    let mut shards: Vec<MxArray> = Vec::new();
    let mut base: i64 = 0;
    while base < vocab {
        let rows_s = (vocab - base).min(rows_per_shard);
        let shard_offset = tensor_base + (base as u64) * (row_bytes as u64);
        let shard_len = rows_s as usize * row_bytes;
        let mut buf = vec![0u8; shard_len];
        let ok = unsafe {
            sys::mlx_safetensor_read_raw(c_path.as_ptr(), shard_offset, buf.as_mut_ptr(), buf.len())
        };
        if !ok {
            return Err(Error::from_reason(format!(
                "Failed to read shard of {key} from {path_str} (offset={shard_offset}, len={shard_len})"
            )));
        }
        let u16_data = bytes_to_u16(&buf);
        shards.push(MxArray::from_bfloat16(&u16_data, &[rows_s, cols])?);
        base += rows_s;
    }
    Ok((shards, rows_per_shard))
}

/// Snapshot MLX's allocator counters in MB. Returns `(active, peak, cache)`.
/// Each accessor is fallible (returns -1 if the Metal allocator is
/// uninitialised, e.g. CPU-only host or convert path that never touched the
/// GPU), in which case we report `0` so the log line still emits cleanly.
fn current_mlx_memory_stats_mb() -> (u64, u64, u64) {
    use mlx_sys as sys;
    let to_mb = |bytes: u64| bytes / (1u64 << 20);
    let mut active: u64 = 0;
    let mut peak: u64 = 0;
    let mut cache: u64 = 0;
    unsafe {
        let _ = sys::mlx_get_active_memory(&mut active);
        let _ = sys::mlx_get_peak_memory(&mut peak);
        let _ = sys::mlx_get_cache_memory(&mut cache);
    }
    (to_mb(active), to_mb(peak), to_mb(cache))
}

/// Save tensors to SafeTensors format.
///
/// Uses a two-pass streaming approach to avoid materializing all tensor bytes
/// in memory at once — critical for large models (e.g. 27B params / 52 GB).
///
/// Pass 1: Compute byte sizes from array metadata (no evaluation), build header.
/// Pass 2: Write header, then stream each tensor's bytes to disk one at a time.
///
/// `tensors` is borrowed `&mut` so each entry can be `.remove(name)`d after
/// its bytes hit disk. Dropping the `MxArray` releases the MLX-allocated
/// backing buffer immediately, which is critical for very large MoE
/// checkpoints — otherwise materialized contiguous buffers for 144+ expert
/// tensors stay live across the whole sharded save and exhaust RAM
/// (observed: 162 GB MLX active memory at shard 34 of 49 on a 128 GB host,
/// triggering silent OOM-kill).
fn save_safetensors_single<P: AsRef<Path>>(
    path: P,
    tensors: &mut HashMap<String, MxArray>,
    names: &[String],
    metadata: Option<serde_json::Value>,
    passthrough: Option<&HashMap<String, PassthroughSource>>,
) -> Result<()> {
    use std::io::{BufWriter, Write};

    let raw_threshold = raw_passthrough_threshold_bytes();

    // --- Pass 1: Build header from metadata only (no tensor evaluation) ---
    let mut header = serde_json::Map::new();
    let mut current_offset = 0usize;

    for name in names {
        let array = tensors.get(name).unwrap();
        let shape = array.shape()?;
        let shape_vec: Vec<usize> = shape.as_ref().iter().map(|&x| x as usize).collect();
        let dtype = array.dtype()?;

        // Compute byte size without evaluating the array
        let size = array.size()? as usize;
        let byte_size = size * dtype.byte_size();

        let tensor_info = serde_json::json!({
            "dtype": dtype_to_safetensor_str(dtype),
            "shape": shape_vec,
            "data_offsets": [current_offset, current_offset + byte_size]
        });

        header.insert(name.clone(), tensor_info);
        current_offset += byte_size;
    }

    if let Some(meta) = metadata {
        header.insert("__metadata__".to_string(), meta);
    }

    let header_json = serde_json::to_string(&header)
        .map_err(|e| Error::from_reason(format!("Failed to serialize header: {}", e)))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // --- Pass 2: Stream header + tensor data to file ---
    let file = File::create(path.as_ref())
        .map_err(|e| Error::from_reason(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);

    writer
        .write_all(&header_len.to_le_bytes())
        .map_err(|e| Error::from_reason(format!("Failed to write header length: {}", e)))?;

    writer
        .write_all(header_bytes)
        .map_err(|e| Error::from_reason(format!("Failed to write header: {}", e)))?;

    // Stream each tensor: materialize → write → drop.
    // - DEBUG: per-tensor timing (off in production)
    // - INFO:  any single tensor that took >2 s to materialize, since on a
    //          GPU stream that means we're approaching the macOS Metal
    //          command-buffer watchdog (~5 s) and on a CPU stream it means
    //          we're likely paging in a cold mmap region — both are signal
    //          you want in production logs.
    let trace_enabled = tracing::enabled!(tracing::Level::DEBUG);
    for name in names {
        let array = tensors.get(name).unwrap();
        let t0 = std::time::Instant::now();
        if trace_enabled {
            let dtype = array.dtype().ok();
            let shape = array.shape().ok();
            tracing::debug!(
                tensor = %name,
                ?dtype,
                shape = ?shape.as_ref().map(|s| s.as_ref().to_vec()),
                "materializing tensor"
            );
        }
        // Compute this tensor's dest byte size + dtype from metadata (no eval).
        // Used both to gate the raw-passthrough path by size and to validate a
        // passthrough source against the dest header before trusting the file.
        let dest_dtype = array.dtype().ok();
        let dest_byte_size = match (array.size().ok(), dest_dtype) {
            (Some(size), Some(dt)) => Some(size as usize * dt.byte_size()),
            _ => None,
        };

        // Oversized passthrough fast path: for a verified unmodified passthrough
        // of a source tensor whose byte size exceeds the threshold, read its
        // bytes straight from the source file (no MLX array → no Metal
        // per-buffer cap). Only taken when the source's dtype + byte length
        // match the dest header exactly; any mismatch falls back to the
        // standard `array_to_bytes` path.
        let raw_bytes = match (
            passthrough.and_then(|m| m.get(name)),
            dest_byte_size,
            dest_dtype,
        ) {
            (Some(src), Some(dest_len), Some(dest_dt))
                if dest_len as u64 >= raw_threshold
                    && src.byte_len == dest_len
                    && src.dtype == dest_dt =>
            {
                let bytes = read_passthrough_bytes(src).map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to read passthrough tensor '{}' from source: {}",
                        name, e
                    ))
                })?;
                tracing::info!(
                    tensor = %name,
                    bytes = bytes.len(),
                    source = %src.file_path.display(),
                    "wrote oversized tensor via direct source file read (bypassed MLX)"
                );
                if std::env::var("MLX_CONVERT_RAW_DEBUG").is_ok() {
                    eprintln!(
                        "[raw-passthrough] tensor={} bytes={} source={}",
                        name,
                        bytes.len(),
                        src.file_path.display()
                    );
                }
                Some(bytes)
            }
            _ => None,
        };

        let bytes = match raw_bytes {
            Some(b) => b,
            None => array_to_bytes(array).map_err(|e| {
                let dtype = array.dtype().ok();
                let shape = array.shape().ok();
                Error::from_reason(format!(
                    "Failed to serialize tensor '{}' (dtype={:?}, shape={:?}): {}",
                    name,
                    dtype,
                    shape.as_ref().map(|s| s.as_ref()),
                    e
                ))
            })?,
        };
        let elapsed = t0.elapsed();
        if trace_enabled {
            tracing::debug!(
                tensor = %name,
                bytes = bytes.len(),
                elapsed_ms = elapsed.as_millis() as u64,
                "tensor written"
            );
        } else if elapsed.as_millis() >= 2000 {
            tracing::info!(
                tensor = %name,
                bytes = bytes.len(),
                elapsed_ms = elapsed.as_millis() as u64,
                "slow tensor materialization"
            );
        }
        writer
            .write_all(&bytes)
            .map_err(|e| Error::from_reason(format!("Failed to write tensor {}: {}", name, e)))?;
        drop(bytes);

        // Release the MLX backing buffer for this tensor now that its bytes
        // are on disk. See the function-level doc on `tensors: &mut` for the
        // reason this is load-bearing on huge MoE checkpoints.
        tensors.remove(name);
    }

    writer
        .flush()
        .map_err(|e| Error::from_reason(format!("Failed to flush file: {}", e)))?;

    Ok(())
}

/// Save tensors to a single SafeTensors file (legacy API for non-sharded use cases).
///
/// Note: drains `tensors` as it writes — see `save_safetensors_single`'s docs.
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &mut HashMap<String, MxArray>,
    metadata: Option<serde_json::Value>,
) -> Result<()> {
    let mut names: Vec<String> = tensors.keys().cloned().collect();
    names.sort();
    save_safetensors_single(path, tensors, &names, metadata, None)
}

/// Save tensors as sharded SafeTensors files with an index, matching mlx-lm/mlx-vlm output.
///
/// - Splits into 5GB shards: `model-00001-of-XXXXX.safetensors`
/// - If total size ≤ 5GB, writes a single `model.safetensors`
/// - Always writes `model.safetensors.index.json` with `{metadata: {total_size}, weight_map}`
/// - SafeTensors metadata is `{"format": "mlx"}` for compatibility
///
/// Note: drains `tensors` as it writes — see `save_safetensors_single`'s docs.
/// On return (success or failure) `tensors` may be partially drained; callers
/// shouldn't rely on its contents post-call.
pub fn save_safetensors_sharded(
    output_dir: &Path,
    tensors: &mut HashMap<String, MxArray>,
    passthrough: Option<&HashMap<String, PassthroughSource>>,
) -> Result<()> {
    use tracing::info;

    let metadata = serde_json::json!({"format": "mlx"});

    // Sort tensor names for deterministic output
    let mut all_names: Vec<String> = tensors.keys().cloned().collect();
    all_names.sort();

    // Compute per-tensor byte sizes
    let mut tensor_sizes: Vec<(String, usize)> = Vec::with_capacity(all_names.len());
    let mut total_size: usize = 0;
    for name in &all_names {
        let array = tensors.get(name).unwrap();
        let size = array.size()? as usize;
        let byte_size = size * array.dtype()?.byte_size();
        tensor_sizes.push((name.clone(), byte_size));
        total_size += byte_size;
    }

    // Split into shards (greedy bin-packing like mlx-lm/mlx-vlm)
    let mut shards: Vec<Vec<String>> = Vec::new();
    let mut current_shard: Vec<String> = Vec::new();
    let mut current_size: usize = 0;

    for (name, byte_size) in &tensor_sizes {
        if !current_shard.is_empty() && current_size + byte_size > MAX_SHARD_SIZE {
            shards.push(std::mem::take(&mut current_shard));
            current_size = 0;
        }
        current_shard.push(name.clone());
        current_size += byte_size;
    }
    if !current_shard.is_empty() {
        shards.push(current_shard);
    }

    let num_shards = shards.len();
    info!(
        "Saving {} tensors ({:.2} GB) in {} shard(s)",
        all_names.len(),
        total_size as f64 / (1 << 30) as f64,
        num_shards
    );

    // Build weight_map and write each shard
    let mut weight_map: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    let convert_start = std::time::Instant::now();
    let mut bytes_written_total: u64 = 0;

    for (i, shard_names) in shards.iter().enumerate() {
        let shard_filename = if num_shards == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}-of-{:05}.safetensors", i + 1, num_shards)
        };

        let shard_path = output_dir.join(&shard_filename);
        let shard_bytes: u64 = shard_names
            .iter()
            .map(|n| {
                let a = tensors.get(n).unwrap();
                a.size().unwrap_or(0) * (a.dtype().map(|d| d.byte_size()).unwrap_or(0) as u64)
            })
            .sum();
        info!(
            shard_index = i + 1,
            shard_count = num_shards,
            shard_file = %shard_filename,
            tensors = shard_names.len(),
            shard_mb = (shard_bytes as f64 / (1u64 << 20) as f64),
            "writing shard"
        );

        let shard_start = std::time::Instant::now();
        save_safetensors_single(
            &shard_path,
            tensors,
            shard_names,
            Some(metadata.clone()),
            passthrough,
        )?;
        let shard_elapsed = shard_start.elapsed();
        bytes_written_total += shard_bytes;
        let convert_elapsed_s = convert_start.elapsed().as_secs_f64().max(1e-6);

        // Snapshot MLX memory after each shard. Helps catch lazy-graph
        // accumulation, allocator leaks, or runaway page-cache growth that
        // could silently OOM the process mid-convert on huge MoE checkpoints.
        let (active_mb, peak_mb, cache_mb) = current_mlx_memory_stats_mb();
        info!(
            shard_index = i + 1,
            shard_count = num_shards,
            shard_file = %shard_filename,
            shard_ms = shard_elapsed.as_millis() as u64,
            shard_mb = (shard_bytes as f64 / (1u64 << 20) as f64),
            avg_mbps = (bytes_written_total as f64 / (1u64 << 20) as f64) / convert_elapsed_s,
            mlx_active_mb = active_mb,
            mlx_peak_mb = peak_mb,
            mlx_cache_mb = cache_mb,
            "shard written"
        );

        for name in shard_names {
            weight_map.insert(name.clone(), shard_filename.clone());
        }
    }

    // Write index file (always, even for single shard — matches mlx-lm behavior)
    let index = serde_json::json!({
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    });

    let index_path = output_dir.join("model.safetensors.index.json");
    let index_str = serde_json::to_string_pretty(&index)
        .map_err(|e| Error::from_reason(format!("Failed to serialize index: {}", e)))?;
    std::fs::write(&index_path, index_str)
        .map_err(|e| Error::from_reason(format!("Failed to write index: {}", e)))?;

    info!("  Wrote model.safetensors.index.json");

    Ok(())
}

/// Convert MxArray to bytes for SafeTensors format.
///
/// For bf16/f16 arrays, extracts raw 16-bit values directly via FFI instead of
/// round-tripping through f32, which would triple per-tensor memory usage.
fn array_to_bytes(array: &MxArray) -> Result<Vec<u8>> {
    let dtype = array.dtype()?;

    match dtype {
        DType::Float32 => {
            let data = array.to_float32()?;
            Ok(f32_to_bytes(&data))
        }
        DType::Float16 | DType::BFloat16 => {
            let u16_data = array.to_uint16_native()?;
            Ok(u16_to_bytes(&u16_data))
        }
        DType::Int32 => {
            let data = array.to_int32()?;
            Ok(i32_to_bytes(&data))
        }
        DType::Uint32 => {
            let data = array.to_uint32()?;
            Ok(data.iter().flat_map(|&x| x.to_le_bytes()).collect())
        }
        DType::Uint8 => {
            let data = array.to_uint8()?;
            Ok(data)
        }
        DType::Int8 => {
            // sym8 quantized weights: int8 bytes are bit-identical to their
            // unsigned reinterpretation, so the cast is lossless.
            let data = array.to_int8()?;
            Ok(data.into_iter().map(|x| x as u8).collect())
        }
    }
}

/// Convert dtype to SafeTensors string representation
fn dtype_to_safetensor_str(dtype: DType) -> String {
    match dtype {
        DType::Float32 => "F32".to_string(),
        DType::Float16 => "F16".to_string(),
        DType::BFloat16 => "BF16".to_string(),
        DType::Int32 => "I32".to_string(),
        DType::Uint32 => "U32".to_string(),
        DType::Uint8 => "U8".to_string(),
        DType::Int8 => "I8".to_string(),
    }
}

/// Convert f32 array to bytes (little-endian)
fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|&x| x.to_le_bytes()).collect()
}

/// Convert i32 array to bytes (little-endian)
fn i32_to_bytes(data: &[i32]) -> Vec<u8> {
    data.iter().flat_map(|&x| x.to_le_bytes()).collect()
}

/// Convert u16 array to bytes (little-endian)
/// Used for writing bf16/f16 tensors extracted via direct FFI
fn u16_to_bytes(data: &[u16]) -> Vec<u8> {
    data.iter().flat_map(|&x| x.to_le_bytes()).collect()
}

/// Load tensors from a safetensors file using MLX's native lazy loader.
/// Arrays are backed by deferred disk reads — data is only materialized on eval.
/// This uses near-zero memory at load time vs the eager `SafeTensorsFile::load_tensors`.
pub fn load_safetensors_lazy<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<HashMap<String, crate::array::MxArray>> {
    use mlx_sys as sys;
    use std::collections::HashMap;

    let path_str = path
        .as_ref()
        .to_str()
        .ok_or_else(|| Error::from_reason("Path is not valid UTF-8"))?;
    let c_path = std::ffi::CString::new(path_str)
        .map_err(|_| Error::from_reason("Path contains null byte"))?;

    // Context struct passed through FFI callback
    struct LoadCtx {
        tensors: HashMap<String, crate::array::MxArray>,
    }

    unsafe extern "C-unwind" fn on_tensor(
        name: *const std::os::raw::c_char,
        name_len: usize,
        handle: *mut sys::mlx_array,
        ctx: *mut std::os::raw::c_void,
    ) {
        unsafe {
            let ctx = &mut *(ctx as *mut LoadCtx);
            let name_bytes = std::slice::from_raw_parts(name as *const u8, name_len);
            let name = String::from_utf8_lossy(name_bytes).to_string();
            if let Ok(arr) = crate::array::MxArray::from_handle(handle, "lazy_load") {
                ctx.tensors.insert(name, arr);
            }
        }
    }

    let mut ctx = LoadCtx {
        tensors: HashMap::new(),
    };

    let count = unsafe {
        sys::mlx_load_safetensors(
            c_path.as_ptr(),
            on_tensor,
            &mut ctx as *mut LoadCtx as *mut std::os::raw::c_void,
        )
    };

    if count < 0 {
        return Err(Error::from_reason(format!(
            "Failed to load safetensors: {}",
            path_str
        )));
    }

    Ok(ctx.tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_byte_sizes() {
        assert_eq!(SafeTensorDType::F32.byte_size(), 4);
        assert_eq!(SafeTensorDType::F16.byte_size(), 2);
        assert_eq!(SafeTensorDType::BF16.byte_size(), 2);
        assert_eq!(SafeTensorDType::I32.byte_size(), 4);
    }

    #[test]
    fn test_bytes_to_f32() {
        let bytes = vec![0x00, 0x00, 0x80, 0x3f]; // 1.0 in little-endian f32
        let floats = bytes_to_f32(&bytes);
        assert_eq!(floats.len(), 1);
        assert_eq!(floats[0], 1.0);
    }

    #[test]
    fn test_bytes_to_i32() {
        let bytes = vec![0x0a, 0x00, 0x00, 0x00]; // 10 in little-endian i32
        let ints = bytes_to_i32(&bytes);
        assert_eq!(ints.len(), 1);
        assert_eq!(ints[0], 10);
    }

    // sym8 contract: an Int8 tensor written by save_safetensors must round-trip
    // bit-exactly through the eager SafeTensorsFile reader (I8 arm). Negative
    // values prove the bit-reinterpret path (from_uint8+astype would map
    // 0xFF -> 255, not -1).
    #[test]
    fn test_int8_save_load_round_trip() {
        let dir = std::env::temp_dir().join(format!("st_i8_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("i8.safetensors");

        let vals: Vec<i8> = vec![-127, -1, 0, 1, 64, 127];
        let arr = MxArray::from_int8(&vals, &[2, 3]).unwrap();
        assert_eq!(arr.dtype().unwrap(), DType::Int8);
        let mut tensors = HashMap::new();
        tensors.insert("w".to_string(), arr);
        save_safetensors(&path, &mut tensors, None).unwrap();

        let f = SafeTensorsFile::load(&path).unwrap();
        let loaded = f.load_tensors(&path).unwrap();
        let w = loaded.get("w").unwrap();
        assert_eq!(w.dtype().unwrap(), DType::Int8);
        assert_eq!(w.shape().unwrap().to_vec(), vec![2i64, 3]);
        assert_eq!(w.to_int8().unwrap(), vals);

        std::fs::remove_dir_all(&dir).ok();
    }

    // Byte-identity lock: a bf16 tensor's on-disk bytes read by the raw
    // source-file path (`mlx_safetensor_read_raw` via `read_passthrough_bytes`)
    // must be byte-identical to the bytes the MLX writer produces
    // (`array_to_bytes`). This is the invariant the convert OOM fix relies on —
    // routing an oversized passthrough tensor through the file read instead of
    // an MLX `eval()` must not change a single byte. Uses raw u16 bit patterns
    // (incl. NaN/Inf/denormal/sign-bit) to prove a pure bit-reinterpret, not a
    // numeric round-trip.
    #[test]
    fn test_passthrough_raw_read_matches_array_to_bytes_bf16() {
        let dir = std::env::temp_dir().join(format!("st_raw_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bf16.safetensors");

        // bf16 bit patterns: 0, +1.0, -1.0, +Inf, NaN, denormal, sign-only, max.
        let bits: Vec<u16> = vec![
            0x0000, 0x3F80, 0xBF80, 0x7F80, 0x7FC0, 0x0001, 0x8000, 0x7F7F, 0x4049, 0xC2F7, 0x1234,
            0xABCD,
        ];
        let arr = MxArray::from_bfloat16(&bits, &[3, 4]).unwrap();
        assert_eq!(arr.dtype().unwrap(), DType::BFloat16);
        let mut tensors = HashMap::new();
        tensors.insert("emb".to_string(), arr);
        save_safetensors(&path, &mut tensors, None).unwrap();

        // Parse the saved file header to locate the tensor's on-disk bytes.
        let st = SafeTensorsFile::load(&path).unwrap();
        let info = st.tensors.get("emb").unwrap();
        let byte_len = info.data_offsets[1] - info.data_offsets[0];
        let file_offset = (st.data_offset + info.data_offsets[0]) as u64;
        assert_eq!(byte_len, bits.len() * 2);

        // Path 1: MLX writer bytes via array_to_bytes on the reloaded array.
        let loaded = st.load_tensors(&path).unwrap();
        let mlx_bytes = array_to_bytes(loaded.get("emb").unwrap()).unwrap();

        // Path 2: raw source-file read via the new FFI.
        let raw_bytes = read_passthrough_bytes(&PassthroughSource {
            file_path: path.clone(),
            file_offset,
            byte_len,
            dtype: DType::BFloat16,
        })
        .unwrap();

        assert_eq!(
            raw_bytes, mlx_bytes,
            "raw source-file bytes must be byte-identical to the MLX writer bytes"
        );
        // And both must equal the original little-endian u16 input.
        let expected: Vec<u8> = bits.iter().flat_map(|&x| x.to_le_bytes()).collect();
        assert_eq!(raw_bytes, expected);

        std::fs::remove_dir_all(&dir).ok();
    }

    // Streaming a bf16 table into axis-0 row shards must reconstruct the source
    // rows exactly — exercises the file discovery + per-shard offset math
    // (tensor_base + base*row_bytes) and the short final shard.
    #[test]
    fn load_bf16_tensor_sharded_matches_source_rows() {
        let dir = std::env::temp_dir().join(format!("st_shard_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        // [7, 4] bf16 with every element distinct, so a wrong offset is obvious.
        const V: usize = 7;
        const C: usize = 4;
        let bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x4000 + i).collect();
        let arr = MxArray::from_bfloat16(&bits, &[V as i64, C as i64]).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("embed_tokens_per_layer.weight".to_string(), arr);
        save_safetensors(&path, &mut tensors, None).unwrap();

        // Budget = 2 rows (2*C*2 bytes) → rows_per_shard 2 → shards [2,2,2,1].
        let budget = 2 * C * 2;
        let (shards, rows_per_shard) =
            load_bf16_tensor_sharded(&dir, "embed_tokens_per_layer.weight", budget).unwrap();
        assert_eq!(rows_per_shard, 2);
        assert_eq!(shards.len(), 4);

        // Concatenated shard rows (in order) must equal the source, byte-exact.
        let mut got: Vec<u16> = Vec::new();
        for (s, shard) in shards.iter().enumerate() {
            let rows_s = if s < 3 { 2 } else { 1 };
            assert_eq!(shard.shape_at(0).unwrap(), rows_s as i64);
            assert_eq!(shard.shape_at(1).unwrap(), C as i64);
            got.extend(shard.to_uint16_native().unwrap());
        }
        assert_eq!(
            got, bits,
            "sharded load must reconstruct the source tensor rows exactly"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Checkpoints store tensors under their raw, prefixed names (gemma-4 wraps
    /// everything in `model.language_model.`), but callers pass the sanitized
    /// bare key. The loader must resolve the bare key against the unique
    /// `*.{key}` suffix in the file — without this, the gemma-4-E2B PLE shard
    /// load fails with "tensor not found".
    #[test]
    fn load_bf16_tensor_sharded_resolves_prefixed_key() {
        let dir = std::env::temp_dir().join(format!("st_shard_prefixed_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        const V: usize = 5;
        const C: usize = 3;
        let bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x4000 + i).collect();
        let arr = MxArray::from_bfloat16(&bits, &[V as i64, C as i64]).unwrap();
        let mut tensors = HashMap::new();
        // Stored under the raw prefixed name, exactly as gemma-4 ships it.
        tensors.insert(
            "model.language_model.embed_tokens_per_layer.weight".to_string(),
            arr,
        );
        save_safetensors(&path, &mut tensors, None).unwrap();

        // Requested by the sanitized bare key — must resolve via the suffix.
        let budget = 2 * C * 2;
        let (shards, rows_per_shard) =
            load_bf16_tensor_sharded(&dir, "embed_tokens_per_layer.weight", budget).unwrap();
        assert_eq!(rows_per_shard, 2);
        assert_eq!(shards.len(), 3); // [2, 2, 1]

        let mut got: Vec<u16> = Vec::new();
        for shard in &shards {
            got.extend(shard.to_uint16_native().unwrap());
        }
        assert_eq!(
            got, bits,
            "prefixed-key resolution must reconstruct the source tensor exactly"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Resolution is global, not per-file: when one shard file holds the exact
    /// bare key and another holds a `*.{key}` suffix, the exact key must win
    /// regardless of `read_dir` order — a per-file break could otherwise pick
    /// the suffix from whichever file the OS happened to list first.
    #[test]
    fn load_bf16_tensor_sharded_exact_key_wins_across_files() {
        let dir = std::env::temp_dir().join(format!("st_shard_exact_wins_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        const V: usize = 4;
        const C: usize = 2;
        // Two distinct tensors in two separate files: a prefixed suffix-match
        // and the exact bare key. The loader must return the exact one.
        let suffix_bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x5000 + i).collect();
        let exact_bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x4000 + i).collect();

        // Canonical shard names so both files are in the checkpoint's file set.
        let mut a = HashMap::new();
        a.insert(
            "model.language_model.embed_tokens_per_layer.weight".to_string(),
            MxArray::from_bfloat16(&suffix_bits, &[V as i64, C as i64]).unwrap(),
        );
        save_safetensors(dir.join("model-00001-of-00002.safetensors"), &mut a, None).unwrap();

        let mut b = HashMap::new();
        b.insert(
            "embed_tokens_per_layer.weight".to_string(),
            MxArray::from_bfloat16(&exact_bits, &[V as i64, C as i64]).unwrap(),
        );
        save_safetensors(dir.join("model-00002-of-00002.safetensors"), &mut b, None).unwrap();

        let budget = V * C * 2; // single shard
        let (shards, _) =
            load_bf16_tensor_sharded(&dir, "embed_tokens_per_layer.weight", budget).unwrap();
        let mut got: Vec<u16> = Vec::new();
        for shard in &shards {
            got.extend(shard.to_uint16_native().unwrap());
        }
        assert_eq!(
            got, exact_bits,
            "an exact key in any shard must win over a `*.{{key}}` suffix in another"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `*.{key}` suffix that matches different tensors in different shard
    /// files (with no exact key anywhere) is ambiguous and must be a loud Err,
    /// never a silent pick of whichever file was listed first.
    #[test]
    fn load_bf16_tensor_sharded_ambiguous_suffix_across_files_errors() {
        let dir = std::env::temp_dir().join(format!("st_shard_ambig_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        const V: usize = 4;
        const C: usize = 2;
        let bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x4000 + i).collect();

        // Canonical shard names so both files are in the checkpoint's file set.
        let mut a = HashMap::new();
        a.insert(
            "model.language_model.embed_tokens_per_layer.weight".to_string(),
            MxArray::from_bfloat16(&bits, &[V as i64, C as i64]).unwrap(),
        );
        save_safetensors(dir.join("model-00001-of-00002.safetensors"), &mut a, None).unwrap();

        let mut b = HashMap::new();
        b.insert(
            "vision_model.embed_tokens_per_layer.weight".to_string(),
            MxArray::from_bfloat16(&bits, &[V as i64, C as i64]).unwrap(),
        );
        save_safetensors(dir.join("model-00002-of-00002.safetensors"), &mut b, None).unwrap();

        let budget = V * C * 2;
        match load_bf16_tensor_sharded(&dir, "embed_tokens_per_layer.weight", budget) {
            Ok(_) => panic!("cross-file suffix ambiguity must fail loudly, not pick one"),
            Err(e) => assert!(
                e.reason.contains("ambiguous"),
                "error must name the ambiguity, got: {}",
                e.reason
            ),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Resolution is scoped to the canonical checkpoint files only. A stray,
    /// non-canonically-named `.safetensors` that happens to hold the exact bare
    /// key must NOT win over (or abort) the real prefixed tensor in
    /// `model.safetensors` — it is invisible to the main loader, so it must be
    /// invisible here too.
    #[test]
    fn load_bf16_tensor_sharded_ignores_stray_non_checkpoint_file() {
        let dir = std::env::temp_dir().join(format!("st_shard_stray_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        const V: usize = 4;
        const C: usize = 2;
        let real_bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x4000 + i).collect();
        let stray_bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x5000 + i).collect();

        // The real tensor lives prefixed in the canonical model.safetensors.
        let mut real = HashMap::new();
        real.insert(
            "model.language_model.embed_tokens_per_layer.weight".to_string(),
            MxArray::from_bfloat16(&real_bits, &[V as i64, C as i64]).unwrap(),
        );
        save_safetensors(dir.join("model.safetensors"), &mut real, None).unwrap();

        // A stray file holds the exact bare key with different values; it is not
        // a canonical checkpoint file and must be ignored entirely.
        let mut stray = HashMap::new();
        stray.insert(
            "embed_tokens_per_layer.weight".to_string(),
            MxArray::from_bfloat16(&stray_bits, &[V as i64, C as i64]).unwrap(),
        );
        save_safetensors(dir.join("extra.safetensors"), &mut stray, None).unwrap();

        let budget = V * C * 2;
        let (shards, _) =
            load_bf16_tensor_sharded(&dir, "embed_tokens_per_layer.weight", budget).unwrap();
        let mut got: Vec<u16> = Vec::new();
        for shard in &shards {
            got.extend(shard.to_uint16_native().unwrap());
        }
        assert_eq!(
            got, real_bits,
            "must resolve the real prefixed tensor, not the stray exact-key file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same exact bare key in two canonical shard files is a malformed
    /// checkpoint (a single tensor lives in exactly one shard), so it must fail
    /// loudly rather than silently picking one.
    #[test]
    fn load_bf16_tensor_sharded_duplicate_exact_key_across_files_errors() {
        let dir = std::env::temp_dir().join(format!("st_shard_dupexact_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        const V: usize = 4;
        const C: usize = 2;
        let bits: Vec<u16> = (0..(V * C) as u16).map(|i| 0x4000 + i).collect();

        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            let mut t = HashMap::new();
            t.insert(
                "embed_tokens_per_layer.weight".to_string(),
                MxArray::from_bfloat16(&bits, &[V as i64, C as i64]).unwrap(),
            );
            save_safetensors(dir.join(shard), &mut t, None).unwrap();
        }

        let budget = V * C * 2;
        match load_bf16_tensor_sharded(&dir, "embed_tokens_per_layer.weight", budget) {
            Ok(_) => panic!("duplicate exact key across shards must fail loudly, not pick one"),
            Err(e) => assert!(
                e.reason.contains("exact key"),
                "error must name the duplicate exact key, got: {}",
                e.reason
            ),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    // Out-of-range / mismatch requests must fail (return Err) rather than read
    // garbage — the writer guard relies on this to fall back to array_to_bytes.
    #[test]
    fn test_passthrough_raw_read_rejects_out_of_range() {
        let dir = std::env::temp_dir().join(format!("st_raw_oob_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bf16.safetensors");

        let bits: Vec<u16> = vec![0x3F80, 0xBF80];
        let arr = MxArray::from_bfloat16(&bits, &[2]).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("emb".to_string(), arr);
        save_safetensors(&path, &mut tensors, None).unwrap();

        let st = SafeTensorsFile::load(&path).unwrap();
        let info = st.tensors.get("emb").unwrap();
        let file_offset = (st.data_offset + info.data_offsets[0]) as u64;

        // Ask for far more bytes than the file holds → must fail, not over-read.
        let res = read_passthrough_bytes(&PassthroughSource {
            file_path: path.clone(),
            file_offset,
            byte_len: 1 << 30,
            dtype: DType::BFloat16,
        });
        assert!(res.is_err(), "out-of-range read must return Err");

        std::fs::remove_dir_all(&dir).ok();
    }
}
