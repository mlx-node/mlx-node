/// GGUF Binary Format Parser and Converter
///
/// Reads GGUF model files and converts tensors to MLX SafeTensors format.
/// Supports:
///   * BF16/F16/F32 unquantized tensors, copied through;
///   * Q4_1, repacked into MLX affine quantization — it stores a real
///     per-block minimum, so it keeps its `.biases` companion;
///   * Q4_0/Q8_0, symmetric (`w = d * (q - Z)`): the bias is `-Z * scale`
///     everywhere, so it is not written — `symmetric_zero_point` goes in
///     `config.json` and the loader rebuilds it;
///   * Q6_K/Q4_K/Q5_K, repacked into MLX's LSB-first bitstream with the ggml
///     sub-scales in `.scales`/`.biases`, behind `import_k_quants`
///     (`--gguf-kquant`); see `crate::utils::gguf_kquant`. `.biases` holds
///     ggml's `d`, a SCALE, not an additive bias. With the flag off, Gemma4's
///     Q6_K token embedding keeps the older dequantize-to-BF16 fallback.
///
/// Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::quant_dispatch::SYMMETRIC_ZERO_POINT_KEY;
use crate::utils::gguf_kquant::{KQuantArrays, KQuantFormat, KQuantRepacker, KQuantScales, QK_K};
use crate::utils::safetensors::save_safetensors;

// ── GGUF Constants ──────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION_3: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// ── GGUF Tensor Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    BF16 = 30,
}

impl GgufTensorType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Bytes per element for non-quantized types, bytes per block for quantized
    /// ones. The K-quant sizes are `block_q4_K` / `block_q5_K` / `block_q6_K`
    /// from `crates/mlx-core/vendor/ggml/ggml_kquant_ref.h:41-73`, which the
    /// vendored C static-asserts against ggml's own structs.
    fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18, // block size: 2 byte scale + 16 bytes (32 x 4-bit)
            Self::Q4_1 => 20, // 2 byte scale + 2 byte bias + 16 bytes
            Self::Q8_0 => 34, // 2 byte scale + 32 bytes
            // f16 d + f16 dmin + 12 packed 6-bit (sub-scale, min) pairs +
            // 128 nibble bytes for 256 values.
            Self::Q4K => 144,
            // Q4_K plus a 32-byte plane holding each value's fifth bit.
            Self::Q5K => 176,
            // 128 low-nibble bytes + 64 high-two-bit bytes + 16 signed
            // sub-scales + one f16 super-scale for 256 values.
            Self::Q6K => 210,
        }
    }

    /// Number of elements per block; 1 for the scalar types.
    ///
    /// Deliberately exhaustive — no `_` arm. `data_size()` divides the element
    /// count by this before multiplying by `type_size()`, so a type that fell
    /// through to 1 would take the non-quantized branch, oversize its tensor by
    /// a factor of `block_size`, and shift every later tensor offset in the file
    /// with no error raised anywhere.
    fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Whether the payload is stored as fixed-size blocks. Derived from
    /// `block_size()` so a type cannot answer this and its byte count
    /// inconsistently.
    fn is_quantized(&self) -> bool {
        self.block_size() > 1
    }

    /// Whether `load_quantized_tensor` can repack the blocks into MLX's affine
    /// `(weight, scales, biases)` triplet. False for the K-quants: their
    /// per-16/32-value sub-scales need the K-quant array contract in
    /// `crate::utils::gguf_kquant`, not affine's one f16 scale per block.
    fn is_mlx_affine_quantized(&self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q4_1 | Self::Q8_0)
    }

    /// The repacker format for the ggml K-quants, `None` for everything else.
    ///
    /// Deliberately exhaustive for the same reason as `block_size()`: a K-quant
    /// added here but forgotten in `load_gguf_tensors` would be handed to the
    /// affine or the dense reader and decode to garbage.
    fn k_quant_format(&self) -> Option<KQuantFormat> {
        match self {
            Self::Q4K => Some(KQuantFormat::Q4K),
            Self::Q5K => Some(KQuantFormat::Q5K),
            Self::Q6K => Some(KQuantFormat::Q6K),
            Self::F32 | Self::F16 | Self::BF16 | Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::BF16 => "BF16",
        }
    }
}

// ── GGUF Metadata Value Types ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

// ── Metadata Value ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayString(Vec<String>),
}

impl GgufMetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            Self::Uint64(v) => Some(*v as u32),
            Self::Int64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ── Tensor Info ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    /// Dimensions in GGUF order (reversed from MLX/row-major)
    pub dims: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Get shape in MLX order (reversed from GGUF)
    pub fn mlx_shape(&self) -> Vec<i64> {
        self.dims.iter().rev().map(|&d| d as i64).collect()
    }

    /// Total number of elements
    pub fn num_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    /// Size in bytes of the raw tensor data
    pub fn data_size(&self) -> u64 {
        let n = self.num_elements();
        if self.tensor_type.is_quantized() {
            let block_size = self.tensor_type.block_size() as u64;
            let n_blocks = n / block_size;
            n_blocks * self.tensor_type.type_size() as u64
        } else {
            n * self.tensor_type.type_size() as u64
        }
    }
}

// ── GGUF File ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufMetaValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: u64,
    /// Byte offset where tensor data begins
    pub data_offset: u64,
}

// ── Binary Reader Helpers ───────────────────────────────────────────────────

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> std::io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> std::io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Maximum allocation size for a single string/array read (256 MB).
/// Prevents OOM from crafted GGUF files with huge length fields.
const MAX_GGUF_ALLOC: usize = 256 * 1024 * 1024;

fn read_string(r: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64(r)? as usize;
    if len > MAX_GGUF_ALLOC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("GGUF string length {len} exceeds maximum ({MAX_GGUF_ALLOC})"),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn read_bool(r: &mut impl Read) -> std::io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

// ── Metadata Value Reader ───────────────────────────────────────────────────

fn read_meta_value(r: &mut impl Read, vtype: GgufValueType) -> std::io::Result<GgufMetaValue> {
    match vtype {
        GgufValueType::Uint8 => Ok(GgufMetaValue::Uint8(read_u8(r)?)),
        GgufValueType::Int8 => Ok(GgufMetaValue::Int8(read_i8(r)?)),
        GgufValueType::Uint16 => Ok(GgufMetaValue::Uint16(read_u16(r)?)),
        GgufValueType::Int16 => Ok(GgufMetaValue::Int16(read_i16(r)?)),
        GgufValueType::Uint32 => Ok(GgufMetaValue::Uint32(read_u32(r)?)),
        GgufValueType::Int32 => Ok(GgufMetaValue::Int32(read_i32(r)?)),
        GgufValueType::Float32 => Ok(GgufMetaValue::Float32(read_f32(r)?)),
        GgufValueType::Bool => Ok(GgufMetaValue::Bool(read_bool(r)?)),
        GgufValueType::String => Ok(GgufMetaValue::String(read_string(r)?)),
        GgufValueType::Uint64 => Ok(GgufMetaValue::Uint64(read_u64(r)?)),
        GgufValueType::Int64 => Ok(GgufMetaValue::Int64(read_i64(r)?)),
        GgufValueType::Float64 => Ok(GgufMetaValue::Float64(read_f64(r)?)),
        GgufValueType::Array => read_meta_array(r),
    }
}

fn read_meta_array(r: &mut impl Read) -> std::io::Result<GgufMetaValue> {
    let elem_type = read_u32(r)?;
    let len = read_u64(r)? as usize;
    if len > MAX_GGUF_ALLOC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("GGUF array length {len} exceeds maximum ({MAX_GGUF_ALLOC})"),
        ));
    }

    let elem_vtype = GgufValueType::from_u32(elem_type).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown GGUF array element type: {elem_type}"),
        )
    })?;

    match elem_vtype {
        GgufValueType::Uint32 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_u32(r)?);
            }
            Ok(GgufMetaValue::ArrayU32(v))
        }
        GgufValueType::Int32 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_i32(r)?);
            }
            Ok(GgufMetaValue::ArrayI32(v))
        }
        GgufValueType::Float32 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_f32(r)?);
            }
            Ok(GgufMetaValue::ArrayF32(v))
        }
        GgufValueType::String => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_string(r)?);
            }
            Ok(GgufMetaValue::ArrayString(v))
        }
        _ => {
            // Skip unsupported array types
            for _ in 0..len {
                read_meta_value(r, elem_vtype)?;
            }
            Ok(GgufMetaValue::ArrayU32(Vec::new()))
        }
    }
}

// ── GGUF Parser ─────────────────────────────────────────────────────────────

pub fn parse_gguf<P: AsRef<Path>>(path: P) -> Result<GgufFile> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .map_err(|e| Error::from_reason(format!("Failed to open GGUF file: {e}")))?;
    let mut reader = BufReader::new(file);

    // Read header
    let magic = read_u32(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read GGUF magic: {e}")))?;
    if magic != GGUF_MAGIC {
        return Err(Error::from_reason(format!(
            "Not a GGUF file (magic: 0x{magic:08X}, expected: 0x{GGUF_MAGIC:08X})"
        )));
    }

    let version = read_u32(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read GGUF version: {e}")))?;
    if version < GGUF_VERSION_3 {
        return Err(Error::from_reason(format!(
            "Unsupported GGUF version {version} (only v3+ supported)"
        )));
    }

    let tensor_count = read_u64(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read tensor count: {e}")))?;
    let metadata_kv_count = read_u64(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read metadata count: {e}")))?;

    // Read metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read metadata key: {e}")))?;
        let vtype_u32 = read_u32(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read metadata value type: {e}")))?;
        let vtype = GgufValueType::from_u32(vtype_u32).ok_or_else(|| {
            Error::from_reason(format!("Unknown GGUF metadata value type: {vtype_u32}"))
        })?;
        let value = read_meta_value(&mut reader, vtype).map_err(|e| {
            Error::from_reason(format!("Failed to read metadata value for '{key}': {e}"))
        })?;
        metadata.insert(key, value);
    }

    // Read alignment from metadata (default: 32, minimum: 1 to prevent division by zero)
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
        .max(1);

    // Read tensor info descriptors
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor name: {e}")))?;
        let n_dims = read_u32(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor n_dims: {e}")))?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(
                read_u64(&mut reader)
                    .map_err(|e| Error::from_reason(format!("Failed to read tensor dim: {e}")))?,
            );
        }
        let type_u32 = read_u32(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor type: {e}")))?;
        let tensor_type = GgufTensorType::from_u32(type_u32);
        let offset = read_u64(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor offset: {e}")))?;

        let tensor_type = match tensor_type {
            Some(t) => t,
            None => {
                return Err(Error::from_reason(format!(
                    "Tensor '{}' has unsupported GGUF type {} — only F32(0), F16(1), Q4_0(2), Q4_1(3), Q8_0(8), Q4_K(12), Q5_K(13), Q6_K(14), BF16(30) are recognized. \
                     Other K-quant and IQ formats require dequantization before conversion.",
                    name, type_u32
                )));
            }
        };

        tensors.push(GgufTensorInfo {
            name,
            n_dims,
            dims,
            tensor_type,
            offset,
        });
    }

    // Compute data offset: current position aligned to alignment boundary
    let current_pos = reader
        .stream_position()
        .map_err(|e| Error::from_reason(format!("Failed to get stream position: {e}")))?;
    let data_offset = align_offset(current_pos, alignment);

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensors,
        alignment,
        data_offset,
    })
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

// ── Tensor Loading ──────────────────────────────────────────────────────────

/// Load a single unquantized tensor from the GGUF file
fn load_unquantized_tensor(
    reader: &mut (impl Read + Seek),
    gguf: &GgufFile,
    tensor: &GgufTensorInfo,
) -> Result<MxArray> {
    let abs_offset = gguf.data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
        Error::from_reason(format!("Failed to seek to tensor '{}': {e}", tensor.name))
    })?;

    let n_bytes = tensor.data_size() as usize;
    let mut buf = vec![0u8; n_bytes];
    reader
        .read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("Failed to read tensor '{}': {e}", tensor.name)))?;

    let shape = tensor.mlx_shape();

    match tensor.tensor_type {
        GgufTensorType::F32 => {
            let data: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            MxArray::from_float32(&data, &shape)
        }
        GgufTensorType::F16 => {
            let data: Vec<u16> = buf
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            MxArray::from_float16(&data, &shape)
        }
        GgufTensorType::BF16 => {
            let data: Vec<u16> = buf
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            MxArray::from_bfloat16(&data, &shape)
        }
        _ => Err(Error::from_reason(format!(
            "Unexpected tensor type {} in load_unquantized_tensor",
            tensor.tensor_type.name()
        ))),
    }
}

/// Dequantize a GGUF Q6_K tensor to dense BF16.
///
/// Q6_K stores 256 values in 210 bytes: 128 bytes of low nibbles, 64 bytes of
/// high two-bit planes, 16 signed i8 sub-scales (one per 16 values), and one
/// f16 super-scale. This is intentionally a quality-preserving fallback for the
/// Gemma4 token embedding until mlx-node has native Q6_K gather/tied-head
/// kernels. Streaming one block at a time avoids retaining the additional
/// ~788 MiB packed source alongside the ~1.88 GiB BF16 destination.
fn load_q6k_tensor_bf16(
    reader: &mut (impl Read + Seek),
    gguf: &GgufFile,
    tensor: &GgufTensorInfo,
) -> Result<MxArray> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210;
    const QL_BYTES: usize = 128;
    const QH_OFFSET: usize = 128;
    const SCALES_OFFSET: usize = 192;
    const D_OFFSET: usize = 208;

    let shape = tensor.mlx_shape();
    let last_dim = shape.last().copied().unwrap_or(0);
    if last_dim <= 0 || !(last_dim as usize).is_multiple_of(QK_K) {
        return Err(Error::from_reason(format!(
            "Q6_K tensor '{}' has last dimension {last_dim}; expected a positive multiple of {QK_K}",
            tensor.name
        )));
    }
    let num_elements = tensor.num_elements() as usize;
    if !num_elements.is_multiple_of(QK_K) {
        return Err(Error::from_reason(format!(
            "Q6_K tensor '{}' has {num_elements} elements; expected a multiple of {QK_K}",
            tensor.name
        )));
    }

    let abs_offset = gguf.data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
        Error::from_reason(format!(
            "Failed to seek to Q6_K tensor '{}': {e}",
            tensor.name
        ))
    })?;

    let mut values = Vec::<u16>::new();
    values.try_reserve_exact(num_elements).map_err(|e| {
        Error::from_reason(format!(
            "Failed to allocate BF16 destination for Q6_K tensor '{}': {e}",
            tensor.name
        ))
    })?;
    let mut block = [0u8; BLOCK_BYTES];
    for block_idx in 0..(num_elements / QK_K) {
        reader.read_exact(&mut block).map_err(|e| {
            Error::from_reason(format!(
                "Failed to read Q6_K block {block_idx} from tensor '{}': {e}",
                tensor.name
            ))
        })?;

        let d = half::f16::from_bits(u16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]]))
            .to_f32();

        for value_idx in 0..QK_K {
            // The low nibbles are arranged as two 64-byte planes. Within each
            // plane the low/high nibble halves are themselves split into two
            // 32-value groups. The high two bits use four bit-planes per
            // 128-value half. This matches ggml/gguf's canonical Q6_K layout.
            let group32 = value_idx / 32;
            let lane = value_idx % 32;
            let group_in_half = group32 % 4;
            let ql_index = (group32 / 4) * 64 + (group_in_half % 2) * 32 + lane;
            debug_assert!(ql_index < QL_BYTES);
            let ql_shift = (group_in_half / 2) * 4;
            let low = (block[ql_index] >> ql_shift) & 0x0f;

            let qh_index = QH_OFFSET + (group32 / 4) * 32 + lane;
            let qh_shift = group_in_half * 2;
            let high = (block[qh_index] >> qh_shift) & 0x03;

            let q = i32::from(low | (high << 4)) - 32;
            let sub_scale = f32::from(block[SCALES_OFFSET + value_idx / 16] as i8);
            let value = d * sub_scale * q as f32;
            values.push(half::bf16::from_f32(value).to_bits());
        }
    }

    MxArray::from_bfloat16(&values, &shape)
}

/// The zero point a symmetric ggml block format subtracts from every code.
///
/// ggml stores Q4_0 as `w = d * (q - 8)` and Q8_0 as `w = d * (q - 128)` after
/// the sign-bit flip this importer applies: one f16 scale per 32 weights and no
/// stored offset at all. MLX's affine format is `w = scale * q + bias`, so the
/// repack used to materialize `bias = -Z * scale` — an array whose every entry
/// is a function of the scale beside it, costing 0.5 bpw (681 MB on
/// Gemma-4-12B-QAT) and pushing the import above the GGUF it came from.
///
/// Returning `Some(Z)` here means the tensor's `.biases` companion is left off
/// disk and reconstructed at load by
/// [`expand_symmetric_affine_biases`](crate::engine::persistence::expand_symmetric_affine_biases).
/// `None` means the format stores a real, independently chosen offset (Q4_1's
/// per-block minimum) that no scale can reproduce.
pub fn symmetric_zero_point(ty: GgufTensorType) -> Option<i32> {
    match ty {
        GgufTensorType::Q4_0 => Some(8),
        GgufTensorType::Q8_0 => Some(128),
        GgufTensorType::Q4_1
        | GgufTensorType::F32
        | GgufTensorType::F16
        | GgufTensorType::BF16
        | GgufTensorType::Q4K
        | GgufTensorType::Q5K
        | GgufTensorType::Q6K => None,
    }
}

/// The `.biases` entry the pre-symmetric importer wrote for a symmetric block.
///
/// Kept as the reference the parity gate compares the load-time reconstruction
/// against, so the historical bytes stay reproducible after the writer stopped
/// emitting them.
pub fn derived_symmetric_bias_bits(scale_bits: u16, zero_point: i32) -> u16 {
    let scale = half::f16::from_bits(scale_bits).to_f32();
    half::f16::from_f32(-(zero_point as f32) * scale).to_bits()
}

/// Load a quantized tensor (Q4_0, Q4_1, Q8_0) into its MLX affine companions.
///
/// Q4_1 yields the `(weight, scales, biases)` triplet its stored per-block
/// minimum requires. Q4_0 and Q8_0 yield `(weight, scales)` only: their offset
/// is the constant `-Z * scale`, so it is reconstructed at load instead of
/// stored (see [`symmetric_zero_point`]).
fn load_quantized_tensor(
    reader: &mut (impl Read + Seek),
    gguf: &GgufFile,
    tensor: &GgufTensorInfo,
) -> Result<Vec<(String, MxArray)>> {
    let abs_offset = gguf.data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
        Error::from_reason(format!("Failed to seek to tensor '{}': {e}", tensor.name))
    })?;

    let data_size = tensor.data_size() as usize;
    let mut raw = vec![0u8; data_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| Error::from_reason(format!("Failed to read tensor '{}': {e}", tensor.name)))?;

    let shape = tensor.mlx_shape();
    let num_elements = tensor.num_elements() as usize;
    let block_size: usize = 32;
    let n_blocks = num_elements / block_size;

    // Determine weights_per_byte for packed format
    let weights_per_byte: usize = match tensor.tensor_type {
        GgufTensorType::Q4_0 | GgufTensorType::Q4_1 => 2, // 4-bit: 2 weights per byte
        GgufTensorType::Q8_0 => 1,                        // 8-bit: 1 weight per byte
        _ => unreachable!(),
    };

    // Weight shape: last dim divided by (weights_per_byte * 4) for uint32 packing
    let mut w_shape = shape.clone();
    let last = *w_shape.last().unwrap();
    *w_shape.last_mut().unwrap() = last / (weights_per_byte as i64 * 4);

    // Scales/biases shape: last dim divided by block_size
    let mut sb_shape = shape;
    *sb_shape.last_mut().unwrap() = last / block_size as i64;

    let w_elements: usize = w_shape.iter().map(|&d| d as usize).product();
    let sb_elements: usize = sb_shape.iter().map(|&d| d as usize).product();

    let mut weights_packed = vec![0u32; w_elements];
    let mut scales = vec![0u16; sb_elements]; // f16
    // Only the asymmetric format fills this; the symmetric ones reconstruct
    // their offset from the scale at load and never allocate the array.
    let mut biases: Vec<u16> = Vec::new(); // f16

    let type_size = tensor.tensor_type.type_size();

    match tensor.tensor_type {
        GgufTensorType::Q4_0 => {
            // Block: 2 bytes f16 scale, 16 bytes (32 x 4-bit weights)
            for i in 0..n_blocks {
                let block = &raw[i * type_size..(i + 1) * type_size];
                scales[i] = u16::from_le_bytes([block[0], block[1]]);

                // Unpack 4-bit weights into int8, then repack into uint32
                let mut unpacked = [0i8; 32];
                for j in 0..16 {
                    unpacked[j] = (block[2 + j] & 0x0F) as i8;
                    unpacked[16 + j] = (block[2 + j] >> 4) as i8;
                }
                // Pack 8 values per u32 (4 bits each)
                let base = i * (block_size / (weights_per_byte * 4));
                for k in 0..(block_size / 8) {
                    let mut packed: u32 = 0;
                    for b in 0..8 {
                        packed |= ((unpacked[k * 8 + b] as u8 & 0x0F) as u32) << (b * 4);
                    }
                    weights_packed[base + k] = packed;
                }
            }
        }
        GgufTensorType::Q4_1 => {
            // Block: 2 bytes f16 scale, 2 bytes f16 bias, 16 bytes (32 x 4-bit weights)
            biases = vec![0u16; sb_elements];
            for i in 0..n_blocks {
                let block = &raw[i * type_size..(i + 1) * type_size];
                scales[i] = u16::from_le_bytes([block[0], block[1]]);
                biases[i] = u16::from_le_bytes([block[2], block[3]]);

                let mut unpacked = [0i8; 32];
                for j in 0..16 {
                    unpacked[j] = (block[4 + j] & 0x0F) as i8;
                    unpacked[16 + j] = (block[4 + j] >> 4) as i8;
                }
                let base = i * (block_size / (weights_per_byte * 4));
                for k in 0..(block_size / 8) {
                    let mut packed: u32 = 0;
                    for b in 0..8 {
                        packed |= ((unpacked[k * 8 + b] as u8 & 0x0F) as u32) << (b * 4);
                    }
                    weights_packed[base + k] = packed;
                }
            }
        }
        GgufTensorType::Q8_0 => {
            // Block: 2 bytes f16 scale, 32 bytes (32 x 8-bit signed weights)
            for i in 0..n_blocks {
                let block = &raw[i * type_size..(i + 1) * type_size];
                scales[i] = u16::from_le_bytes([block[0], block[1]]);

                // Convert signed int8 to unsigned (add 128 / flip sign bit) then pack into u32
                let base = i * (block_size / 4); // 8 bits per weight, 4 per u32
                for k in 0..(block_size / 4) {
                    let mut packed: u32 = 0;
                    for b in 0..4 {
                        let signed_val = block[2 + k * 4 + b] as i8;
                        let unsigned_val = (signed_val as u8) ^ 0x80; // flip sign bit
                        packed |= (unsigned_val as u32) << (b * 8);
                    }
                    weights_packed[base + k] = packed;
                }
            }
        }
        _ => unreachable!(),
    }

    // Create MxArray tensors
    let w_i64_shape: Vec<i64> = w_shape.to_vec();
    let sb_i64_shape: Vec<i64> = sb_shape.to_vec();

    let weight_arr = MxArray::from_uint32(&weights_packed, &w_i64_shape)?;
    let scales_arr = MxArray::from_float16(&scales, &sb_i64_shape)?;

    // Strip .weight suffix for prefix, then add .scales/.biases
    let name = &tensor.name;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);

    let mut out = vec![
        (name.clone(), weight_arr),
        (format!("{prefix}.scales"), scales_arr),
    ];
    if symmetric_zero_point(tensor.tensor_type).is_none() {
        out.push((
            format!("{prefix}.biases"),
            MxArray::from_float16(&biases, &sb_i64_shape)?,
        ));
    }
    Ok(out)
}

/// Source bytes a single K-quant read pulls in before repacking them. Large
/// enough to amortize the read syscall across many rows, small enough that the
/// source side never rivals the tensor being built.
const KQUANT_CHUNK_TARGET_BYTES: usize = 4 << 20;

/// Read `rows` rows of `format` super-blocks from the reader's current position
/// and repack them.
///
/// Rows repack independently, so the source is consumed in chunks of at most
/// `chunk_target_bytes` (rounded down to whole rows, never below one) instead of
/// as one buffer. `label` names the tensor for read errors.
fn repack_rows_from_reader(
    reader: &mut impl Read,
    format: KQuantFormat,
    k: usize,
    rows: usize,
    chunk_target_bytes: usize,
    label: &str,
) -> Result<KQuantArrays> {
    let row_bytes = format.row_bytes(k);
    let chunk_rows = (chunk_target_bytes / row_bytes).clamp(1, rows.max(1));
    let mut buf = vec![0u8; chunk_rows * row_bytes];

    let mut repacker = KQuantRepacker::new(format, k, rows)?;
    let mut done = 0usize;
    while done < rows {
        let n = chunk_rows.min(rows - done);
        let chunk = &mut buf[..n * row_bytes];
        reader.read_exact(chunk).map_err(|e| {
            Error::from_reason(format!(
                "Failed to read rows {done}..{} of {label}: {e}",
                done + n
            ))
        })?;
        repacker.push_rows(chunk, n)?;
        done += n;
    }
    Ok(repacker.finish())
}

/// The three output keys one imported K-quant tensor occupies: the packed
/// `.weight` under the source tensor's own name, plus the `.scales` / `.biases`
/// sidecars sharing its prefix.
///
/// `load_kquant_repack` names its arrays with this, and the converter's
/// do-not-dtype-cast set spells out the same three names, so the producer and
/// the consumer cannot drift into disagreeing about the suffixes.
fn k_quant_output_keys(weight_key: &str) -> [String; 3] {
    let prefix = weight_key.strip_suffix(".weight").unwrap_or(weight_key);
    [
        weight_key.to_string(),
        format!("{prefix}.scales"),
        format!("{prefix}.biases"),
    ]
}

/// Import a ggml K-quant tensor as MLX K-quant arrays, without dequantizing.
///
/// Emits the same `(weight, scales, biases)` triplet shape as the affine path,
/// because ~25 sites elsewhere in the loader use a `.scales` / `.biases` pair as
/// the "this tensor is quantized" sentinel. The contents differ: `.weight` is an
/// LSB-first `bits`-wide code stream, `.scales` holds ggml's per-16/32-value
/// sub-scales (signed for Q6_K, `(sc, m)` pairs for Q4_K / Q5_K), and `.biases`
/// holds ggml's `d` — and `dmin` for Q4_K / Q5_K — which are SCALES, not biases.
/// `crate::utils::gguf_kquant` documents the full contract and owns the packing.
///
/// The GGUF payload is read a few megabytes at a time rather than whole, so a
/// multi-hundred-megabyte tensor never has its source bytes and its repacked
/// copy resident together. The destination arrays are still built in full —
/// they are the tensor.
fn load_kquant_repack(
    reader: &mut (impl Read + Seek),
    gguf: &GgufFile,
    tensor: &GgufTensorInfo,
    format: KQuantFormat,
) -> Result<Vec<(String, MxArray)>> {
    let shape = tensor.mlx_shape();
    let last_dim = shape.last().copied().unwrap_or(0);
    if last_dim <= 0 || !(last_dim as usize).is_multiple_of(QK_K) {
        return Err(Error::from_reason(format!(
            "{} tensor '{}' has last dimension {last_dim}; expected a positive multiple of {QK_K}",
            tensor.tensor_type.name(),
            tensor.name
        )));
    }
    let k = last_dim as usize;
    // `num_elements()` is the product of the dims, so dividing by the last one
    // is exact.
    let rows = tensor.num_elements() as usize / k;

    let abs_offset = gguf.data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
        Error::from_reason(format!(
            "Failed to seek to {} tensor '{}': {e}",
            tensor.tensor_type.name(),
            tensor.name
        ))
    })?;

    let arrays = repack_rows_from_reader(
        reader,
        format,
        k,
        rows,
        KQUANT_CHUNK_TARGET_BYTES,
        &format!("{} tensor '{}'", tensor.tensor_type.name(), tensor.name),
    )?;

    let row_shape = |cols: usize| -> Vec<i64> {
        let mut s = shape.clone();
        if let Some(last) = s.last_mut() {
            *last = cols as i64;
        }
        s
    };

    // Destructured so each staging buffer is freed as soon as MLX has copied
    // it, instead of all three outliving the last copy.
    let KQuantArrays {
        weight,
        scales,
        biases,
    } = arrays;
    let weight_arr = MxArray::from_uint32(&weight, &row_shape(format.weight_cols(k)))?;
    drop(weight);
    let scales_shape = row_shape(format.scales_cols(k));
    let scales_arr = match &scales {
        // Q6_K sub-scales are signed; reinterpreting them through uint8 would
        // turn -1 into 255.
        KQuantScales::Signed(v) => MxArray::from_int8(v, &scales_shape)?,
        KQuantScales::Unsigned(v) => MxArray::from_uint8(v, &scales_shape)?,
    };
    drop(scales);
    let biases_arr = MxArray::from_float16(&biases, &row_shape(format.biases_cols(k)))?;
    drop(biases);

    let [weight_key, scales_key, biases_key] = k_quant_output_keys(&tensor.name);
    Ok(vec![
        (weight_key, weight_arr),
        (scales_key, scales_arr),
        (biases_key, biases_arr),
    ])
}

/// What `load_gguf_tensors` does beyond a dtype-preserving read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GgufLoadOptions {
    /// Log progress every 50 tensors.
    pub verbose: bool,
    /// Repack ggml Q4_K / Q5_K / Q6_K blocks into MLX K-quant arrays.
    ///
    /// Off by default, which is the pre-K-quant behaviour: Q6_K is accepted
    /// only as the Gemma4 token embedding and dequantized to BF16, and Q4_K /
    /// Q5_K are rejected.
    pub import_k_quants: bool,
}

/// Load all tensors from a GGUF file
pub fn load_gguf_tensors<P: AsRef<Path>>(
    path: P,
    gguf: &GgufFile,
    options: GgufLoadOptions,
) -> Result<HashMap<String, MxArray>> {
    let file = fs::File::open(path.as_ref())
        .map_err(|e| Error::from_reason(format!("Failed to open GGUF file: {e}")))?;
    let mut reader = BufReader::new(file);
    let mut weights = HashMap::new();

    for (i, tensor) in gguf.tensors.iter().enumerate() {
        if options.verbose && (i % 50 == 0 || i == gguf.tensors.len() - 1) {
            info!(
                "Loading tensor {}/{}: {} ({}, shape {:?})",
                i + 1,
                gguf.tensors.len(),
                tensor.name,
                tensor.tensor_type.name(),
                tensor.mlx_shape()
            );
        }

        if let Some(format) = tensor.tensor_type.k_quant_format() {
            if options.import_k_quants {
                for (name, arr) in load_kquant_repack(&mut reader, gguf, tensor, format)? {
                    weights.insert(name, arr);
                }
            } else if tensor.tensor_type == GgufTensorType::Q6K {
                // The pre-K-quant fallback: one dequantized tensor, in one
                // position, for one architecture.
                let is_gemma4 = gguf
                    .metadata
                    .get("general.architecture")
                    .and_then(GgufMetaValue::as_str)
                    == Some("gemma4");
                if !is_gemma4 || tensor.name != "token_embd.weight" {
                    return Err(Error::from_reason(format!(
                        "Q6_K tensor '{}' is unsupported in this position. mlx-node currently supports Q6_K only for Gemma4 token_embd.weight, dequantized to BF16.",
                        tensor.name
                    )));
                }
                let arr = load_q6k_tensor_bf16(&mut reader, gguf, tensor)?;
                weights.insert(tensor.name.clone(), arr);
            } else {
                return Err(Error::from_reason(format!(
                    "{} tensor '{}' needs the K-quant import, which is not enabled. mlx-node reads {} only when K-quant import is requested.",
                    tensor.tensor_type.name(),
                    tensor.name,
                    tensor.tensor_type.name(),
                )));
            }
        } else if tensor.tensor_type.is_mlx_affine_quantized() {
            let triplet = load_quantized_tensor(&mut reader, gguf, tensor)?;
            for (name, arr) in triplet {
                weights.insert(name, arr);
            }
        } else {
            let arr = load_unquantized_tensor(&mut reader, gguf, tensor)?;
            weights.insert(tensor.name.clone(), arr);
        }
    }

    Ok(weights)
}

// ── Key Remapping ───────────────────────────────────────────────────────────

fn is_gemma4_main_gguf(metadata: &HashMap<String, GgufMetaValue>) -> bool {
    metadata
        .get("general.architecture")
        .and_then(GgufMetaValue::as_str)
        == Some("gemma4")
}

fn is_gemma4_mmproj_gguf(metadata: &HashMap<String, GgufMetaValue>) -> bool {
    metadata
        .get("general.architecture")
        .and_then(GgufMetaValue::as_str)
        == Some("clip")
        && (metadata
            .get("clip.vision.projector_type")
            .and_then(GgufMetaValue::as_str)
            == Some("gemma4uv")
            || metadata
                .get("clip.audio.projector_type")
                .and_then(GgufMetaValue::as_str)
                == Some("gemma4ua"))
}

/// The three names a quantized tensor group ships under. Both GGUF quant
/// importers emit the same trio: `load_quantized_tensor` (affine Q4_0/Q4_1/Q8_0)
/// and `load_kquant_repack` (Q4_K/Q5_K/Q6_K) each write `{base}.weight` plus a
/// `{base}.scales` / `{base}.biases` sidecar pair. Loaders probe the sidecar by
/// name to decide a tensor is quantized (e.g. `embed_tokens.scales`), so a
/// rename that moves only `.weight` strands the sidecars under their old names
/// and the group silently degrades to a bare packed weight.
const QUANT_GROUP_SUFFIXES: [&str; 3] = [".weight", ".scales", ".biases"];

/// Rename `{from_base}` to `{to_base}` for a global tensor, carrying whichever
/// of the canonical quant-group suffixes the key uses.
///
/// The per-layer rules are infix `replace()`s (`.ffn_down.` →
/// `.mlp.down_proj.`), so they already ride along on any suffix. The global
/// tensors (`token_embd`, `output`) are renamed by whole-string match instead,
/// which is why they need this: matching the suffix explicitly keeps a
/// quantized global tensor's `.scales`/`.biases` attached to its renamed
/// `.weight`. Requiring the remainder to be exactly one of the three suffixes
/// (rather than a bare prefix test) keeps neighbours like `output_norm.weight`
/// and `token_embd_extra.weight` untouched.
fn rename_global_quant_group(name: &str, from_base: &str, to_base: &str) -> Option<String> {
    let suffix = name.strip_prefix(from_base)?;
    QUANT_GROUP_SUFFIXES
        .contains(&suffix)
        .then(|| format!("{to_base}{suffix}"))
}

/// Gemma4's GGUF naming has four distinct residual norms. The generic mapper
/// predates that architecture and maps `ffn_norm` and `post_attention_norm` to
/// the same destination, silently losing one tensor when collected into a
/// HashMap. Keep the architecture-specific mapping explicit and complete.
fn gemma4_name_to_hf(name: &str) -> Option<String> {
    if name == "rope_freqs.weight" {
        // RoPE frequencies are derived from config at runtime.
        return None;
    }

    let mut result = name.to_string();
    if result.starts_with("blk.") {
        result = result.replacen("blk.", "model.layers.", 1);
    }

    result = result.replace(".attn_q.", ".self_attn.q_proj.");
    result = result.replace(".attn_k.", ".self_attn.k_proj.");
    result = result.replace(".attn_v.", ".self_attn.v_proj.");
    result = result.replace(".attn_output.", ".self_attn.o_proj.");
    result = result.replace(".attn_q_norm.", ".self_attn.q_norm.");
    result = result.replace(".attn_k_norm.", ".self_attn.k_norm.");
    result = result.replace(".ffn_gate.", ".mlp.gate_proj.");
    result = result.replace(".ffn_down.", ".mlp.down_proj.");
    result = result.replace(".ffn_up.", ".mlp.up_proj.");

    result = result.replace(".attn_norm.", ".input_layernorm.");
    result = result.replace(".post_attention_norm.", ".post_attention_layernorm.");
    result = result.replace(".ffn_norm.", ".pre_feedforward_layernorm.");
    result = result.replace(".post_ffw_norm.", ".post_feedforward_layernorm.");
    result = result.replace(".layer_output_scale.weight", ".layer_scalar");

    // Global tensors rename whole-string, so they carry the quant-group suffix
    // explicitly (a quantized `token_embd`/`output` ships `.scales`/`.biases`
    // beside `.weight`). `output_norm` stays an exact `.weight` match — norms
    // are never quantized, so it has no group to carry.
    if let Some(renamed) = rename_global_quant_group(&result, "token_embd", "model.embed_tokens") {
        result = renamed;
    } else if result == "output_norm.weight" {
        result = "model.norm.weight".to_string();
    } else if let Some(renamed) = rename_global_quant_group(&result, "output", "lm_head") {
        result = renamed;
    }
    Some(result)
}

fn gemma4_mmproj_name_to_hf(name: &str) -> Option<String> {
    // Whole-string matches, so — like `token_embd`/`output` above — a quantized
    // projection needs its sidecars carried across or they strand under their
    // GGUF names. Reachable when an mmproj is the PRIMARY output.
    for (from, to) in [
        (
            "mm.a.input_projection",
            "model.embed_audio.embedding_projection",
        ),
        (
            "mm.input_projection",
            "model.embed_vision.embedding_projection",
        ),
    ] {
        if let Some(renamed) = rename_global_quant_group(name, from, to) {
            return Some(renamed);
        }
    }
    let mapped = match name {
        "v.patch_embd.weight" => "model.vision_embedder.patch_dense.weight",
        "v.patch_embd.bias" => "model.vision_embedder.patch_dense.bias",
        "v.patch_norm.1.weight" => "model.vision_embedder.patch_ln1.weight",
        "v.patch_norm.1.bias" => "model.vision_embedder.patch_ln1.bias",
        "v.patch_norm.2.weight" => "model.vision_embedder.patch_ln2.weight",
        "v.patch_norm.2.bias" => "model.vision_embedder.patch_ln2.bias",
        "v.position_embd.weight" => "model.vision_embedder.pos_embedding",
        "v.patch_norm.3.weight" => "model.vision_embedder.pos_norm.weight",
        "v.patch_norm.3.bias" => "model.vision_embedder.pos_norm.bias",
        _ => return Some(gguf_name_to_hf(name)),
    };
    Some(mapped.to_string())
}

fn gguf_name_to_hf_for_metadata(
    name: &str,
    metadata: &HashMap<String, GgufMetaValue>,
) -> Option<String> {
    if is_gemma4_main_gguf(metadata) {
        gemma4_name_to_hf(name)
    } else if is_gemma4_mmproj_gguf(metadata) {
        gemma4_mmproj_name_to_hf(name)
    } else {
        Some(gguf_name_to_hf(name))
    }
}

/// Remap GGUF tensor name to HuggingFace-style name (standard LLM mapping)
pub fn gguf_name_to_hf(name: &str) -> String {
    // ── Vision encoder (mmproj GGUF) ──────────────────────────────────────
    // Handle vision keys first to avoid LLM mapping conflicts
    // (e.g., vision .attn_qkv. must NOT be mapped to .linear_attn.in_proj_qkv.)

    if name.starts_with("v.blk.") {
        let mut result = name.replacen("v.blk.", "vision_tower.blocks.", 1);
        result = result.replace(".attn_qkv.", ".attn.qkv.");
        result = result.replace(".attn_out.", ".attn.proj.");
        result = result.replace(".ffn_up.", ".mlp.linear_fc1.");
        result = result.replace(".ffn_down.", ".mlp.linear_fc2.");
        result = result.replace(".ln1.", ".norm1.");
        result = result.replace(".ln2.", ".norm2.");
        return result;
    }
    if let Some(suffix) = name.strip_prefix("mm.0.") {
        return format!("vision_tower.merger.linear_fc1.{suffix}");
    }
    if let Some(suffix) = name.strip_prefix("mm.2.") {
        return format!("vision_tower.merger.linear_fc2.{suffix}");
    }
    if name.starts_with("v.post_ln.") {
        return name.replacen("v.post_ln.", "vision_tower.merger.norm.", 1);
    }
    if name.starts_with("v.patch_embd.") {
        return name.replacen("v.patch_embd.", "vision_tower.patch_embed.proj.", 1);
    }
    if name == "v.position_embd.weight" {
        return "vision_tower.pos_embed.weight".to_string();
    }

    // ── LLM mappings ──────────────────────────────────────────────────────

    let mut result = name.to_string();

    // Layer prefix: blk.{n}. → model.layers.{n}.
    if result.starts_with("blk.") {
        result = result.replacen("blk.", "model.layers.", 1);
    }

    // Attention projections
    result = result.replace(".attn_q.", ".self_attn.q_proj.");
    result = result.replace(".attn_k.", ".self_attn.k_proj.");
    result = result.replace(".attn_v.", ".self_attn.v_proj.");
    result = result.replace(".attn_output.", ".self_attn.o_proj.");

    // Attention norms (used by Qwen3.5 full-attention layers)
    result = result.replace(".attn_q_norm.", ".self_attn.q_norm.");
    result = result.replace(".attn_k_norm.", ".self_attn.k_norm.");

    // FFN projections
    result = result.replace(".ffn_gate.", ".mlp.gate_proj.");
    result = result.replace(".ffn_down.", ".mlp.down_proj.");
    result = result.replace(".ffn_up.", ".mlp.up_proj.");

    // Norms
    result = result.replace(".attn_norm.", ".input_layernorm.");
    result = result.replace(".ffn_norm.", ".post_attention_layernorm.");
    result = result.replace(".post_attention_norm.", ".post_attention_layernorm.");

    // Qwen3.5 GatedDeltaNet / linear attention (SSM-like) components
    result = result.replace(".attn_qkv.", ".linear_attn.in_proj_qkv.");
    result = result.replace(".attn_gate.", ".linear_attn.in_proj_z.");
    result = result.replace(".ssm_beta.", ".linear_attn.in_proj_b.");
    result = result.replace(".ssm_alpha.", ".linear_attn.in_proj_a.");
    result = result.replace(".ssm_out.", ".linear_attn.out_proj.");
    result = result.replace(".ssm_conv1d.", ".linear_attn.conv1d.");
    result = result.replace(".ssm_norm.", ".linear_attn.norm.");

    // ssm_dt.bias → linear_attn.dt_bias (no .weight suffix — it's a plain bias)
    if result.contains(".ssm_dt.bias") {
        result = result.replace(".ssm_dt.bias", ".linear_attn.dt_bias");
    }

    // ssm_a → linear_attn.A_log (no .weight suffix)
    if result.ends_with(".ssm_a") {
        result = result.replace(".ssm_a", ".linear_attn.A_log");
    }

    // ── Global layers ───────────────────────────────────────────────────

    // Whole-string renames, so they carry the quant-group suffix explicitly:
    // a quantized `token_embd` / untied `output` ships `.scales`/`.biases`
    // beside `.weight`. `output_norm` stays an exact `.weight` match — norms
    // are never quantized, so it has no group to carry.
    if let Some(renamed) = rename_global_quant_group(&result, "token_embd", "model.embed_tokens") {
        result = renamed;
    } else if result == "output_norm.weight" {
        result = "model.norm.weight".to_string();
    } else if let Some(renamed) = rename_global_quant_group(&result, "output", "lm_head") {
        result = renamed;
    }

    result
}

/// Remap all keys from GGUF names to HuggingFace names. Reject collisions
/// rather than letting `HashMap::collect` silently keep one of two semantic
/// tensors (the failure mode that exposed Gemma4's four-norm mapping gap).
fn remap_keys(
    weights: HashMap<String, MxArray>,
    metadata: &HashMap<String, GgufMetaValue>,
) -> Result<HashMap<String, MxArray>> {
    let mut remapped = HashMap::with_capacity(weights.len());
    let mut sources = HashMap::<String, String>::with_capacity(weights.len());
    for (source, value) in weights {
        let Some(dest) = gguf_name_to_hf_for_metadata(&source, metadata) else {
            continue;
        };
        if let Some(previous) = sources.get(&dest) {
            return Err(Error::from_reason(format!(
                "GGUF key remap collision: source tensors '{previous}' and '{source}' both map to '{dest}'"
            )));
        }
        sources.insert(dest.clone(), source);
        remapped.insert(dest, value);
    }
    Ok(remapped)
}

fn square_side_for_chw(flat_dim: i64, key: &str) -> Result<i64> {
    const CHANNELS: i64 = 3;
    if flat_dim <= 0 || flat_dim % CHANNELS != 0 {
        return Err(Error::from_reason(format!(
            "Gemma4 mmproj tensor '{key}' has flattened dimension {flat_dim}; expected 3 * side * side"
        )));
    }
    let spatial = flat_dim / CHANNELS;
    let side = (spatial as f64).sqrt() as i64;
    if side * side != spatial {
        return Err(Error::from_reason(format!(
            "Gemma4 mmproj tensor '{key}' has flattened dimension {flat_dim}; {spatial} pixels is not square"
        )));
    }
    Ok(side)
}

/// GGUF stores Gemma4's flattened image patch state in CHW order, while the
/// unified MLX/HF checkpoint consumes HWC-flattened values. It also stores the
/// two position tables as `[2, positions, hidden]`, versus
/// `[positions, 2, hidden]` in the checkpoint. Apply only to the positively
/// identified Gemma4 unified mmproj so other CLIP/Qwen layouts stay unchanged.
fn fixup_gemma4_mmproj_layout(
    weights: &mut HashMap<String, MxArray>,
    metadata: &HashMap<String, GgufMetaValue>,
) -> Result<()> {
    if !is_gemma4_mmproj_gguf(metadata) {
        return Ok(());
    }

    let patch_key = "model.vision_embedder.patch_dense.weight";
    if let Some(weight) = weights.remove(patch_key) {
        let shape = weight.shape()?.to_vec();
        if shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "Gemma4 mmproj tensor '{patch_key}' must be 2-D, got {shape:?}"
            )));
        }
        let (out, flat) = (shape[0], shape[1]);
        let side = square_side_for_chw(flat, patch_key)?;
        let transformed = weight
            .reshape(&[out, 3, side, side])?
            .transpose(Some(&[0, 2, 3, 1]))?
            .reshape(&[out, flat])?;
        weights.insert(patch_key.to_string(), transformed);
    }

    for key in [
        "model.vision_embedder.patch_ln1.weight",
        "model.vision_embedder.patch_ln1.bias",
    ] {
        if let Some(value) = weights.remove(key) {
            let shape = value.shape()?.to_vec();
            if shape.len() != 1 {
                return Err(Error::from_reason(format!(
                    "Gemma4 mmproj tensor '{key}' must be 1-D, got {shape:?}"
                )));
            }
            let flat = shape[0];
            let side = square_side_for_chw(flat, key)?;
            let transformed = value
                .reshape(&[3, side, side])?
                .transpose(Some(&[1, 2, 0]))?
                .reshape(&[flat])?;
            weights.insert(key.to_string(), transformed);
        }
    }

    let position_key = "model.vision_embedder.pos_embedding";
    if let Some(position) = weights.remove(position_key) {
        let shape = position.shape()?.to_vec();
        if shape.len() != 3 || shape[0] != 2 {
            return Err(Error::from_reason(format!(
                "Gemma4 mmproj tensor '{position_key}' must have shape [2, positions, hidden], got {shape:?}"
            )));
        }
        weights.insert(
            position_key.to_string(),
            position.transpose(Some(&[1, 0, 2]))?,
        );
    }

    Ok(())
}

/// Post-process weights after remapping to fix shapes/values that differ between GGUF and HF.
///
/// - conv1d.weight: GGUF stores 2D [C, K] → reshape to [C, K, 1] (model expectation)
/// - Norm weights: GGUF stores delta from 1.0 (like unsanitized HF) → add +1.0
///   Applies to input_layernorm, post_attention_layernorm, q_norm, k_norm, final_norm,
///   but NOT linear_attn.norm (stored with final values, no shift needed).
fn fixup_shapes(weights: &mut HashMap<String, MxArray>) -> Result<()> {
    // Fix conv1d shapes
    let conv1d_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("conv1d.weight"))
        .cloned()
        .collect();

    for key in conv1d_keys {
        if let Some(arr) = weights.remove(&key) {
            let ndim = arr.ndim()?;
            if ndim == 2 {
                let c = arr.shape_at(0)?;
                let k = arr.shape_at(1)?;
                let reshaped = arr.reshape(&[c, k, 1])?;
                info!("Reshaped {key}: [{c}, {k}] → [{c}, {k}, 1]");
                weights.insert(key, reshaped);
            } else {
                weights.insert(key, arr);
            }
        }
    }

    // Stack Conv3d patch_embed weights: GGUF stores two 4D temporal slices as
    // proj.weight [out, in, kH, kW] and proj.weight.1 [out, in, kH, kW].
    // MLX Conv3d expects 5D [out, kD, kH, kW, in]. Stack and transpose.
    let w0_key = "vision_tower.patch_embed.proj.weight".to_string();
    let w1_key = "vision_tower.patch_embed.proj.weight.1".to_string();
    if weights.contains_key(&w0_key) && weights.contains_key(&w1_key) {
        let w0 = weights.remove(&w0_key).unwrap();
        let w1 = weights.remove(&w1_key).unwrap();
        // Stack: 2x [out, in, kH, kW] → [out, in, kH, kW, 2]
        let stacked = MxArray::stack(vec![&w0, &w1], Some(4))?;
        // Transpose [0, 4, 2, 3, 1]: [out, in, kH, kW, 2] → [out, 2, kH, kW, in]
        let conv3d = stacked.transpose(Some(&[0, 4, 2, 3, 1]))?;
        info!("Stacked patch_embed.proj.weight + weight.1 into Conv3d 5D");
        weights.insert(w0_key, conv3d);
    }

    // NOTE: GGUF stores actual trained norm weights (not deltas from 1.0).
    // The +1.0 shift is handled by persistence.rs sanitize_weights() when
    // it detects unsanitized HF checkpoints (MTP weights or wrong conv1d axis).
    // We do NOT apply it here — the GGUF values are the correct final values.

    Ok(())
}

/// Re-interleave a 1D array from deinterleaved [even, odd] → interleaved [h0, h1, h2, ...]
fn reinterleave_1d(arr: &MxArray, n_heads: i64) -> Result<MxArray> {
    let half = n_heads / 2;
    // arr = [h0, h2, h4, ..., h1, h3, h5, ...]
    // result = [h0, h1, h2, h3, ...]
    let even = arr.slice(&[0], &[half])?; // first half = even heads
    let odd = arr.slice(&[half], &[n_heads])?; // second half = odd heads
    // Stack [even, odd] → [n_heads/2, 2] → reshape [n_heads]
    let stacked = MxArray::stack(vec![&even, &odd], Some(1))?; // [half, 2]
    stacked.reshape(&[n_heads])
}

/// Re-interleave rows of a 2D array from deinterleaved head order.
/// Groups of head_dim rows: [even_heads, odd_heads] → [h0, h1, h2, ...]
fn reinterleave_rows(arr: &MxArray, n_heads: i64, head_dim: i64) -> Result<MxArray> {
    let cols = arr.shape_at(1)?;
    let half = n_heads / 2;
    // Reshape to [n_heads, head_dim, cols]
    let grouped = arr.reshape(&[n_heads, head_dim, cols])?;
    // grouped = [even_h0, even_h1, ..., odd_h0, odd_h1, ...]
    let even = grouped.slice(&[0, 0, 0], &[half, head_dim, cols])?;
    let odd = grouped.slice(&[half, 0, 0], &[n_heads, head_dim, cols])?;
    // Stack to [half, 2, head_dim, cols]
    let stacked = MxArray::stack(vec![&even, &odd], Some(1))?;
    // Reshape to [n_heads * head_dim, cols]
    stacked.reshape(&[n_heads * head_dim, cols])
}

/// Re-interleave columns of a 2D array from deinterleaved head order.
fn reinterleave_cols(arr: &MxArray, n_heads: i64, head_dim: i64) -> Result<MxArray> {
    // Transpose, reinterleave rows, transpose back
    let transposed = arr.transpose(None)?;
    let fixed = reinterleave_rows(&transposed, n_heads, head_dim)?;
    fixed.transpose(None)
}

/// Fix Qwen3.5 GatedDeltaNet (linear attention) tensors.
///
/// GGUF (llama.cpp) deinterleaves head dimensions for the value/state head axis
/// (linear_num_value_heads). This function re-interleaves them back to HF order.
/// Also converts ssm_a from `-exp(A_log)` back to `A_log`.
fn fixup_qwen35_linear_attn(
    weights: &mut HashMap<String, MxArray>,
    metadata: &HashMap<String, GgufMetaValue>,
) -> Result<()> {
    // Detect Qwen3.5 by checking for linear attention weights
    let has_linear_attn = weights.keys().any(|k| k.contains("linear_attn."));
    if !has_linear_attn {
        return Ok(());
    }

    // Get value head dimensions from GGUF metadata
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // ssm.state_size = value_head_dim (128)
    // ssm.inner_size = total state dimension (n_value_heads * value_head_dim = 4096)
    // ssm.group_count = n_key_heads (16)
    let value_head_dim = metadata
        .get(&format!("{arch}.ssm.state_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as i64;

    let ssm_inner_size = metadata
        .get(&format!("{arch}.ssm.inner_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as i64;

    let n_value_heads = ssm_inner_size / value_head_dim;
    let head_dim = value_head_dim;

    // Q+K dim: first part of qkv that doesn't need reinterleaving
    let qk_dim = weights
        .iter()
        .find(|(k, _)| k.contains("linear_attn.in_proj_qkv.weight"))
        .and_then(|(_, arr)| {
            let total = arr.shape_at(0).ok()?;
            let v_dim = n_value_heads * head_dim;
            Some(total - v_dim)
        })
        .unwrap_or(4096);

    info!(
        "Qwen3.5 linear attention fixup: n_value_heads={}, head_dim={}, qk_dim={}",
        n_value_heads, head_dim, qk_dim
    );

    if n_value_heads < 2 {
        return Ok(());
    }

    // Process each layer's linear attention tensors
    let layer_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.contains("linear_attn."))
        .cloned()
        .collect();

    for key in &layer_keys {
        // A_log: deinterleave + convert from -exp(A_log) back to A_log
        if key.ends_with("linear_attn.A_log") {
            if let Some(arr) = weights.remove(key) {
                let reinterleaved = reinterleave_1d(&arr, n_value_heads)?;
                // Convert: A_log = log(-ssm_a) since GGUF stores -exp(A_log)
                let neg = reinterleaved.negative()?;
                let a_log = neg.log()?;
                info!("Fixed {key}: re-interleaved + log(-x)");
                weights.insert(key.clone(), a_log);
            }
            continue;
        }

        // dt_bias: deinterleave only
        if key.ends_with("linear_attn.dt_bias") {
            if let Some(arr) = weights.remove(key) {
                let fixed = reinterleave_1d(&arr, n_value_heads)?;
                info!("Fixed {key}: re-interleaved");
                weights.insert(key.clone(), fixed);
            }
            continue;
        }

        // in_proj_a, in_proj_b: deinterleave rows (n_value_heads rows of 1 element group)
        if key.ends_with("in_proj_a.weight") || key.ends_with("in_proj_b.weight") {
            if let Some(arr) = weights.remove(key) {
                let fixed = reinterleave_rows(&arr, n_value_heads, 1)?;
                info!("Fixed {key}: row re-interleave ({n_value_heads} heads)");
                weights.insert(key.clone(), fixed);
            }
            continue;
        }

        // in_proj_z: deinterleave rows (n_value_heads heads of head_dim rows)
        if key.ends_with("in_proj_z.weight") {
            if let Some(arr) = weights.remove(key) {
                let fixed = reinterleave_rows(&arr, n_value_heads, head_dim)?;
                info!("Fixed {key}: row re-interleave ({n_value_heads} x {head_dim})");
                weights.insert(key.clone(), fixed);
            }
            continue;
        }

        // out_proj: deinterleave columns (n_value_heads heads of head_dim cols)
        if key.ends_with("out_proj.weight") && key.contains("linear_attn.") {
            if let Some(arr) = weights.remove(key) {
                let fixed = reinterleave_cols(&arr, n_value_heads, head_dim)?;
                info!("Fixed {key}: column re-interleave ({n_value_heads} x {head_dim})");
                weights.insert(key.clone(), fixed);
            }
            continue;
        }

        // in_proj_qkv: deinterleave only the V portion rows
        if key.ends_with("in_proj_qkv.weight") {
            if let Some(arr) = weights.remove(key) {
                let total_rows = arr.shape_at(0)?;
                let v_rows = total_rows - qk_dim;
                if v_rows > 0 && v_rows == n_value_heads * head_dim {
                    let cols = arr.shape_at(1)?;
                    let qk_part = arr.slice(&[0, 0], &[qk_dim, cols])?;
                    let v_part = arr.slice(&[qk_dim, 0], &[total_rows, cols])?;
                    let v_part_2d = v_part.reshape(&[v_rows, cols])?;
                    let v_fixed = reinterleave_rows(&v_part_2d, n_value_heads, head_dim)?;
                    let fixed = MxArray::concatenate(&qk_part, &v_fixed, 0)?;
                    info!("Fixed {key}: V portion row re-interleave (qk={qk_dim}, v={v_rows})");
                    weights.insert(key.clone(), fixed);
                } else {
                    weights.insert(key.clone(), arr);
                }
            }
            continue;
        }

        // conv1d: deinterleave only the V portion rows (same split as qkv)
        if key.ends_with("conv1d.weight") && key.contains("linear_attn.") {
            if let Some(arr) = weights.remove(key) {
                let total_rows = arr.shape_at(0)?;
                let v_rows = total_rows - qk_dim;
                if v_rows > 0 && v_rows == n_value_heads * head_dim {
                    // conv1d is [C, K, 1] after reshape
                    let kernel = arr.shape_at(1)?;
                    let last_dim = if arr.ndim()? == 3 {
                        arr.shape_at(2)?
                    } else {
                        1
                    };
                    // Reshape to [C, K*last_dim] for row reinterleave
                    let flat = arr.reshape(&[total_rows, kernel * last_dim])?;
                    let flat_cols = kernel * last_dim;
                    let qk_part = flat.slice(&[0, 0], &[qk_dim, flat_cols])?;
                    let v_part = flat.slice(&[qk_dim, 0], &[total_rows, flat_cols])?;
                    let v_fixed = reinterleave_rows(&v_part, n_value_heads, head_dim)?;
                    let fixed = MxArray::concatenate(&qk_part, &v_fixed, 0)?;
                    // Reshape back
                    let result = fixed.reshape(&[total_rows, kernel, last_dim])?;
                    info!("Fixed {key}: V portion row re-interleave");
                    weights.insert(key.clone(), result);
                } else {
                    weights.insert(key.clone(), arr);
                }
            }
            continue;
        }
    }

    Ok(())
}

/// Find an imported K-quant tensor that the Qwen3.5 GGUF layout fixup would
/// have to reorder as a dense array.
///
/// K-quant storage is a three-array group: packed codes in `.weight` and the
/// ggml block scales in `.scales` / `.biases`. The dense Qwen3.5 fixup below
/// slices and transposes logical rows or columns of `.weight` only. Applying it
/// to the packed array both uses the wrong dimensions and leaves its sidecars
/// in the old order. Until that transform is implemented at the complete
/// quant-group level, reject exactly those tensors while the GGUF header is
/// still the only part of the file that has been read.
fn qwen35_kquant_dense_fixup_target(gguf: &GgufFile) -> Option<(&GgufTensorInfo, String)> {
    const DENSE_FIXUP_SUFFIXES: &[&str] = &[
        ".linear_attn.in_proj_a.weight",
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.out_proj.weight",
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.conv1d.weight",
    ];

    gguf.tensors.iter().find_map(|tensor| {
        tensor.tensor_type.k_quant_format()?;
        let mapped = gguf_name_to_hf_for_metadata(&tensor.name, &gguf.metadata)?;
        DENSE_FIXUP_SUFFIXES
            .iter()
            .any(|suffix| mapped.ends_with(suffix))
            .then_some((tensor, mapped))
    })
}

/// GGUF has no HuggingFace `model_type`, so require both a known llama.cpp
/// Qwen3.5 architecture tag and the characteristic mixed full-attention/GDN
/// tensor shape before enabling the fixed Unsloth MXFP map. `qwen3` is kept
/// for older Qwen3.5 writers; ordinary Qwen3 fails the weight-shape check.
fn is_qwen35_hybrid_gguf(
    metadata: &HashMap<String, GgufMetaValue>,
    weight_keys: &[String],
) -> bool {
    let is_qwen35_arch = metadata
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .is_some_and(|arch| matches!(arch, "qwen35" | "qwen35moe" | "qwen3"));

    is_qwen35_arch && crate::convert::has_qwen35_hybrid_weight_shape(weight_keys)
}

/// Return the first quantized tensor in an ordinary Qwen3 GGUF.
///
/// The Qwen3 runtime is deliberately dense-only. Source-quantized GGUF tensors
/// would otherwise be preserved as affine/K-quant groups and written with a
/// non-empty quantization block that the loader must reject. Some older
/// Qwen3.5 writers also use `general.architecture = "qwen3"`, so exempt the
/// characteristic full-attention + GatedDeltaNet hybrid shape.
fn plain_qwen3_quantized_source(gguf: &GgufFile) -> Option<&GgufTensorInfo> {
    let arch = gguf
        .metadata
        .get("general.architecture")
        .and_then(GgufMetaValue::as_str);
    if arch != Some("qwen3") {
        return None;
    }

    let weight_keys = gguf
        .tensors
        .iter()
        .filter_map(|tensor| gguf_name_to_hf_for_metadata(&tensor.name, &gguf.metadata))
        .collect::<Vec<_>>();
    if is_qwen35_hybrid_gguf(&gguf.metadata, &weight_keys) {
        return None;
    }

    gguf.tensors
        .iter()
        .find(|tensor| SourceQuantProfile::for_gguf_type(tensor.tensor_type).is_some())
}

// ── Config Extraction ───────────────────────────────────────────────────────

/// Extract HuggingFace-compatible config.json fields from GGUF metadata
pub fn extract_config(metadata: &HashMap<String, GgufMetaValue>) -> serde_json::Value {
    let mut config = serde_json::Map::new();

    // Detect architecture name (e.g., "qwen3" from "qwen3.embedding_length")
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string();

    // Model type
    if let Some(name) = metadata.get("general.name").and_then(|v| v.as_str()) {
        config.insert(
            "_name_or_path".to_string(),
            serde_json::Value::String(name.to_string()),
        );
    }
    config.insert(
        "model_type".to_string(),
        serde_json::Value::String(arch.clone()),
    );

    // Map GGUF keys to HF config keys
    let mappings: &[(&str, &str)] = &[
        ("embedding_length", "hidden_size"),
        ("block_count", "num_hidden_layers"),
        ("feed_forward_length", "intermediate_size"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
        ("context_length", "max_position_embeddings"),
        ("rope.freq_base", "rope_theta"),
        ("attention.layer_norm_rms_epsilon", "rms_norm_eps"),
    ];

    for &(gguf_suffix, hf_key) in mappings {
        let gguf_key = format!("{arch}.{gguf_suffix}");
        if let Some(val) = metadata.get(&gguf_key) {
            match val {
                GgufMetaValue::Uint32(v) => {
                    config.insert(hf_key.to_string(), serde_json::Value::Number((*v).into()));
                }
                GgufMetaValue::Int32(v) => {
                    config.insert(hf_key.to_string(), serde_json::Value::Number((*v).into()));
                }
                GgufMetaValue::Uint64(v) => {
                    config.insert(hf_key.to_string(), serde_json::Value::Number((*v).into()));
                }
                GgufMetaValue::Float32(v) => {
                    if let Some(n) = serde_json::Number::from_f64(*v as f64) {
                        config.insert(hf_key.to_string(), serde_json::Value::Number(n));
                    }
                }
                _ => {}
            }
        }
    }

    // Vocab size from tokenizer metadata
    if let Some(val) = metadata.get(&format!("{arch}.vocab_size")) {
        if let Some(n) = val.as_u32() {
            config.insert(
                "vocab_size".to_string(),
                serde_json::Value::Number(n.into()),
            );
        }
    } else if let Some(GgufMetaValue::ArrayString(tokens)) = metadata.get("tokenizer.ggml.tokens") {
        config.insert(
            "vocab_size".to_string(),
            serde_json::Value::Number(tokens.len().into()),
        );
    }

    // Head dimension. The Qwen3 family decouples head_dim from
    // hidden_size / num_attention_heads: Qwen3-0.6B has 16 query heads yet a
    // head_dim of 128 (q_proj is [2048, 1024], not [1024, 1024]). GGUF records
    // this as `{arch}.attention.key_length`, so for the Qwen3 family honor it
    // as authoritative — deriving `hidden / heads` = 64 made every Qwen3 GGUF
    // fail to load with `q_norm expected [64], got [128]`.
    //
    // This preference is scoped to the Qwen3 family (matching
    // `is_qwen35_hybrid_gguf`'s arch set) precisely because a single top-level
    // head_dim is only meaningful when every attention layer shares it. Gemma4
    // does NOT: its sliding-window layers use head_dim 256 and its global
    // layers 512, so its GGUF reports `key_length=512` (global) alongside
    // `key_length_swa=256`. Gemma4's native loader derives per-layer-type head
    // dims itself and ignores this top-level field, so it must stay on the
    // legacy `hidden / heads` derivation — both to keep the field byte-identical
    // with existing Gemma4 K-quant conversions and because neither GGUF value is
    // the whole story there.
    let is_qwen_family = matches!(arch.as_str(), "qwen3" | "qwen35" | "qwen35moe");
    // A key_length of 0 is not a head dim; fall through to the derivation
    // rather than writing a config that divides by it downstream. Same guard
    // the `heads > 0` arm below applies to the derived value.
    let key_length = if is_qwen_family {
        metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.as_u32())
            .filter(|&k| k > 0)
            .map(u64::from)
    } else {
        None
    };
    let derived_head_dim = match (
        config.get("hidden_size").and_then(|v| v.as_u64()),
        config.get("num_attention_heads").and_then(|v| v.as_u64()),
    ) {
        (Some(hidden), Some(heads)) if heads > 0 => Some(hidden / heads),
        _ => None,
    };
    if let Some(head_dim) = key_length.or(derived_head_dim) {
        config.insert(
            "head_dim".to_string(),
            serde_json::Value::Number(head_dim.into()),
        );
    }

    serde_json::Value::Object(config)
}

/// What a `quantization` config entry says about one tensor: the
/// `(bits, group_size, mode)` triple plus the symmetric zero point, if any.
///
/// Every field participates in every comparison. Bits alone cannot tell a
/// Q4_0 tensor from a Q4_K one — both are 4-bit — yet they decode through
/// different kernels with different group sizes, so a bits-only equality test
/// reports an Unsloth Dynamic GGUF as uniformly affine and mislabels every
/// K-quant tensor in it. The zero point is in the identity for the same reason
/// one step down: Q4_0 and Q4_1 are both 4-bit affine at group 32 and differ
/// only in whether the offset is derived or stored.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct SourceQuantProfile {
    bits: i32,
    group_size: i32,
    mode: &'static str,
    /// Set when the source block format is symmetric and the converter left the
    /// derived `.biases` companion off disk. Part of the identity, so a
    /// symmetric Q4_0 tensor and an asymmetric Q4_1 tensor no longer collapse
    /// onto one profile and a file holding both is described per tensor.
    symmetric_zero_point: Option<i32>,
}

impl SourceQuantProfile {
    /// How the converter writes a tensor of this GGUF type, or `None` when the
    /// tensor lands as a dense float array.
    ///
    /// Deliberately exhaustive — no `_` arm — for the same reason
    /// `GgufTensorType::block_size` is: a quantized type that fell through to
    /// `None` here would be written with no `quantization` entry describing it
    /// and decoded with whatever geometry the runtime happens to default to.
    fn for_gguf_type(ty: GgufTensorType) -> Option<Self> {
        match ty {
            GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16 => None,
            // `load_quantized_tensor` repacks these into MLX affine groups of
            // 32 without changing their block geometry. Q4_0 and Q8_0 are
            // symmetric, so their profile also records the zero point the
            // loader needs to rebuild the omitted `.biases`.
            GgufTensorType::Q4_0 => Some(Self::affine(4).symmetric(ty)),
            GgufTensorType::Q4_1 => Some(Self::affine(4)),
            GgufTensorType::Q8_0 => Some(Self::affine(8).symmetric(ty)),
            // `load_kquant_repack` keeps ggml's geometry verbatim, so the
            // triple is read off the repacker format rather than restated
            // here; `k_quant_format` stays the only type -> format mapping.
            GgufTensorType::Q4K | GgufTensorType::Q5K | GgufTensorType::Q6K => {
                ty.k_quant_format().map(Self::k_quant)
            }
        }
    }

    const fn affine(bits: i32) -> Self {
        Self {
            bits,
            group_size: 32,
            mode: "affine",
            symmetric_zero_point: None,
        }
    }

    /// Record the zero point `ty`'s blocks subtract, if it has one.
    fn symmetric(self, ty: GgufTensorType) -> Self {
        Self {
            symmetric_zero_point: symmetric_zero_point(ty),
            ..self
        }
    }

    fn k_quant(format: KQuantFormat) -> Self {
        Self {
            bits: format.bits() as i32,
            group_size: format.group_size() as i32,
            mode: format.mlx_mode(),
            symmetric_zero_point: None,
        }
    }

    /// Whether a tensor with this profile has to be named explicitly in the
    /// config instead of relying on the top-level triple. Affine at group 32 is
    /// the only shape a generic default can reproduce; the K-quant modes carry
    /// ggml geometry no default supplies.
    fn requires_explicit_entry(self) -> bool {
        self.mode != "affine"
    }

    fn to_json(self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("bits".to_string(), serde_json::json!(self.bits));
        obj.insert("group_size".to_string(), serde_json::json!(self.group_size));
        obj.insert("mode".to_string(), serde_json::json!(self.mode));
        if let Some(zero_point) = self.symmetric_zero_point {
            obj.insert(
                SYMMETRIC_ZERO_POINT_KEY.to_string(),
                serde_json::json!(zero_point),
            );
        }
        serde_json::Value::Object(obj)
    }

    fn describe(self) -> String {
        match self.symmetric_zero_point {
            Some(zero_point) => format!(
                "{}-bit {} (group_size {}, symmetric zero point {zero_point})",
                self.bits, self.mode, self.group_size
            ),
            None => format!(
                "{}-bit {} (group_size {})",
                self.bits, self.mode, self.group_size
            ),
        }
    }
}

/// Describe tensors that were already quantized in the GGUF source.
///
/// Q4_0/Q4_1/Q8_0 go through `load_quantized_tensor` and — when the K-quant
/// import is on — Q4_K/Q5_K/Q6_K go through `load_kquant_repack`. Neither path
/// changes the source block geometry, so that geometry has to reach the output
/// config even when the caller never asked for a `--quantize` pass; otherwise
/// Gemma4 falls back to the runtime's generic group-size default (64) and
/// decodes exact packed bytes with the wrong shape.
///
/// `import_k_quants` mirrors `GgufLoadOptions`: with it off the loader rejected
/// Q4_K/Q5_K outright and dequantized the Gemma4 Q6_K embedding to BF16, so no
/// K-quant tensor reaches the output for this to describe.
/// Whether this Gemma4 checkpoint takes V from the key projection on its
/// global layers.
///
/// Gemma4 interleaves sliding-window and global (full-attention) layers, and on
/// the global ones V is not a projection at all: the layer reuses the key
/// projection output, so no `attn_v` tensor is ever written. llama.cpp encodes
/// this by marking `wv` as `TENSOR_NOT_REQUIRED` and falling back to
/// `V := Kcur` when it is absent; `Gemma4Attention` spells the same thing as
/// `attention_k_eq_v`, which drops `v_proj` and takes V from the raw key
/// projection before K's norm and RoPE.
///
/// No GGUF metadata key records it, so the only witness is the tensor list
/// itself. Without this the loader asks for a `v_proj` the publisher never
/// wrote and the conversion dies on `layers.5.self_attn.v_proj` — which is what
/// blocks Google's QAT GGUFs, the only form those checkpoints ship in.
///
/// The signal is deliberately a *mix*: some decoder layers carrying `attn_v`
/// and others not. A checkpoint where every layer has V is an ordinary one, and
/// a checkpoint where none does is not the interleaved pattern this describes —
/// neither should silently flip the flag.
///
/// The return value is the full `layer_types` array rather than a bare bool,
/// because the flag alone is not enough. `attention_k_eq_v` only takes effect
/// on layers the config calls `full_attention`, and when `layer_types` is
/// absent the loader synthesizes one from `sliding_window_pattern`, which
/// defaults to 5 and puts the global layers at indices 4, 9, 14 … The real file
/// has a period of 6, and guessing 5 silently mistypes every layer.
///
/// `gemma4.attention.sliding_window_pattern` would say so directly, but it is an
/// array of GGUF element type 7 (bool), which [`read_meta_array`] has no variant
/// for: the unsupported-type arm skips the payload and yields an empty
/// `ArrayU32`. It is in any case only a redundant second witness. The reliable
/// one is `gemma4.attention.head_count_kv`, an `ArrayI32` with one entry per
/// layer whose minimum marks the global layers — the same array
/// [`apply_gemma4_attention_geometry`] already reads.
///
/// The header is cross-checked before the flag is flipped, because the tensor
/// list alone is fail-open: a checkpoint truncated mid-download, or corrupted,
/// is missing an `attn_v` too, and reading that as K=V suppresses the loader's
/// missing-weight check (`models/gemma4/persistence.rs`, `v_proj required when
/// this layer does not use k_eq_v`) and feeds attention the keys as values. So
/// where the header says something, it has to say the same thing:
///
/// * the `blk.N` indices run `0..N` with no gaps, and `gemma4.block_count`, when
///   present, equals `N`,
/// * `head_count_kv`, when it is an ARRAY, has one entry per layer, and
/// * the layers at that array's minimum are precisely the layers with no
///   `attn_v`.
///
/// A disagreement is a named error rather than a silent `None`: the tensor list
/// is a partial-V mixture either way, so the load cannot succeed, and reporting
/// the disagreement beats relocating the failure to `layers.N.self_attn.v_proj`.
///
/// A header that says NOTHING is a different case and is not an error. Absence
/// of a witness is not evidence of corruption, and the types measured off the
/// tensor list are right whatever the header does or does not repeat; erroring
/// here would also reject a file whose config the caller supplies with
/// `--config-dir`, where the header is never consulted at all.
///
/// That tolerance is only safe because the caller refuses the case it creates.
/// A header with no per-layer counts leaves
/// [`apply_gemma4_attention_geometry`] unable to state
/// `num_global_key_value_heads`, and a synthesized config would then describe
/// the global layers with the sliding layers' KV head count — which nothing
/// downstream catches, because the quantized load path discards `out_features`
/// and the mismatch only surfaces as an abort inside `reshape` at the first
/// token. So `convert_gguf_to_safetensors` requires both counts before it
/// writes anything, and this helper is left free to answer the question it can
/// actually answer.
fn gemma4_layer_types_from_missing_v(gguf: &GgufFile) -> Result<Option<Vec<String>>> {
    let mut layers = std::collections::BTreeSet::new();
    for tensor in &gguf.tensors {
        if let Some(rest) = tensor.name.strip_prefix("blk.")
            && let Some((idx, _)) = rest.split_once('.')
            && let Ok(n) = idx.parse::<u32>()
        {
            layers.insert(n);
        }
    }

    let missing_v: std::collections::BTreeSet<u32> = layers
        .iter()
        .copied()
        .filter(|layer| {
            let prefix = format!("blk.{layer}.attn_v");
            !gguf.tensors.iter().any(|t| t.name.starts_with(&prefix))
        })
        .collect();

    // Not the interleave: every layer projects V (an ordinary checkpoint), or
    // none does (a V-less file is not the sliding/global mix this describes).
    // Neither may flip the flag, and neither is an error.
    if missing_v.is_empty() || missing_v.len() == layers.len() {
        return Ok(None);
    }

    // Both the emitted `layer_types` and the `head_count_kv` comparison below
    // are positional — entry `i` describes block `i` — so they only mean
    // anything if the blocks run `0..N` with no gaps. `gemma4.block_count`, when
    // the file states it, must agree with that count; when it is absent there is
    // simply nothing to compare against, and refusing on that basis would reject
    // a file whose own tensor list is perfectly self-consistent.
    let block_count = gguf
        .metadata
        .get("gemma4.block_count")
        .and_then(GgufMetaValue::as_u32);
    if !layers.iter().copied().eq(0..layers.len() as u32)
        || block_count.is_some_and(|n| n as usize != layers.len())
    {
        return Err(Error::from_reason(format!(
            "Gemma4 GGUF omits 'attn_v' on {} of {} decoder blocks, but its blocks are not \
             exactly 0..gemma4.block_count ({block_count:?}) — the tensor list is truncated or \
             renumbered, refusing to read it as the key-equals-value attention layout",
            missing_v.len(),
            layers.len(),
        )));
    }

    let types: Vec<String> = layers
        .iter()
        .map(|layer| {
            if missing_v.contains(layer) {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect();

    // `head_count_kv` only carries per-layer information when it is an ARRAY. A
    // scalar states one count for the whole model and an absent key states
    // nothing at all; neither can mark which layers are global, so neither
    // contradicts the tensor list. Return the measured answer instead of
    // erroring — see the doc comment for why absence is not corruption.
    if !gemma4_kv_head_counts_are_per_layer(&gguf.metadata) {
        return Ok(Some(types));
    }

    let kv_per_layer = gemma4_kv_head_counts(&gguf.metadata);
    if kv_per_layer.len() != layers.len() {
        return Err(Error::from_reason(format!(
            "Gemma4 GGUF omits 'attn_v' on {} of {} decoder blocks, but \
             'gemma4.attention.head_count_kv' holds {} entries instead of one per layer — \
             nothing corroborates which layers are global, refusing to read it as the \
             key-equals-value attention layout",
            missing_v.len(),
            layers.len(),
            kv_per_layer.len(),
        )));
    }

    let fewest_kv_heads = kv_per_layer
        .iter()
        .copied()
        .min()
        .expect("length was just checked against a non-empty layer set");
    let global_by_kv: std::collections::BTreeSet<u32> = kv_per_layer
        .iter()
        .enumerate()
        .filter(|&(_, &heads)| heads == fewest_kv_heads)
        .map(|(idx, _)| idx as u32)
        .collect();
    if global_by_kv != missing_v {
        return Err(Error::from_reason(format!(
            "Gemma4 GGUF omits 'attn_v' on layers {missing_v:?}, but \
             'gemma4.attention.head_count_kv' puts its {fewest_kv_heads}-head layers at \
             {global_by_kv:?} — the tensor list and the header disagree about which layers are \
             global, refusing to read it as the key-equals-value attention layout"
        )));
    }

    Ok(Some(types))
}

/// Whether `gemma4.attention.head_count_kv` is spelled as an array, the only
/// form that says anything per layer.
///
/// [`gemma4_kv_head_counts`] flattens a scalar into a one-element vector so the
/// geometry helper can still use it, which makes a scalar indistinguishable from
/// a one-entry array by length alone. The cross-check needs the distinction: a
/// scalar is silence, not a contradicting witness.
fn gemma4_kv_head_counts_are_per_layer(metadata: &HashMap<String, GgufMetaValue>) -> bool {
    matches!(
        metadata.get("gemma4.attention.head_count_kv"),
        Some(GgufMetaValue::ArrayI32(_) | GgufMetaValue::ArrayU32(_))
    )
}

/// Whether a config already states `key`, looked up the way the Gemma4 loader
/// looks it up: `text_config` first, then the top level (`get_config_bool` and
/// `parse_layer_types` in `models/gemma4/persistence.rs`). Checking both means a
/// nested statement is honored even though the converter writes flat keys — a
/// top-level write would be shadowed by `text_config` and silently do nothing.
fn gemma4_config_states(config: &serde_json::Value, key: &str) -> bool {
    config
        .get("text_config")
        .and_then(|text_config| text_config.get(key))
        .is_some()
        || config.get(key).is_some()
}

/// The per-layer KV head counts a Gemma4 GGUF declares, empty when the key is
/// absent.
///
/// Gemma4 varies the KV head count per layer, so `head_count_kv` is an *array*
/// rather than the scalar the generic metadata mapping handles. Shared by
/// [`gemma4_layer_types_from_missing_v`] and
/// [`apply_gemma4_attention_geometry`] so the array the layer types are checked
/// against is literally the array the geometry is built from.
fn gemma4_kv_head_counts(metadata: &HashMap<String, GgufMetaValue>) -> Vec<u32> {
    match metadata.get("gemma4.attention.head_count_kv") {
        Some(GgufMetaValue::ArrayI32(v)) => v.iter().map(|&n| n as u32).collect(),
        Some(GgufMetaValue::ArrayU32(v)) => v.clone(),
        Some(scalar) => scalar.as_u32().into_iter().collect(),
        None => Vec::new(),
    }
}

/// Whether `head_count_kv` can give [`apply_gemma4_attention_geometry`] BOTH the
/// sliding and the global KV head count for `layer_types` — literally the pair
/// of lookups that helper's `pick` makes, so the two cannot disagree about what
/// counts as "stated".
fn gemma4_kv_counts_cover_both_layer_types(
    metadata: &HashMap<String, GgufMetaValue>,
    layer_types: &[String],
) -> bool {
    let kv_per_layer = gemma4_kv_head_counts(metadata);
    [false, true].into_iter().all(|global| {
        layer_types
            .iter()
            .position(|t| (t == "full_attention") == global)
            .is_some_and(|idx| idx < kv_per_layer.len())
    })
}

/// How `gemma4.attention.head_count_kv` is spelled, for the error raised when it
/// cannot supply both counts.
fn describe_gemma4_kv_head_counts(metadata: &HashMap<String, GgufMetaValue>) -> String {
    match metadata.get("gemma4.attention.head_count_kv") {
        None => "the key is absent".to_string(),
        Some(GgufMetaValue::ArrayI32(_) | GgufMetaValue::ArrayU32(_)) => {
            format!("it holds {} entries", gemma4_kv_head_counts(metadata).len())
        }
        Some(_) => "it is a single scalar covering the whole model".to_string(),
    }
}

/// Fill in the Gemma4 attention geometry that the generic metadata mapping
/// cannot express.
///
/// Gemma4 decouples the head dimension from `hidden_size / num_attention_heads`
/// and uses a different one per layer type — 512 on global layers and 256 on
/// sliding ones for the 12B, against a derived value of 240 that matches
/// neither. It also varies the KV head count per layer, so `head_count_kv` is
/// an *array* in the GGUF rather than a scalar, and the generic mapping (which
/// handles scalars only) drops it entirely, leaving the loader on its default
/// of 2.
///
/// `layer_types` says which entry of that array is which: the sliding count
/// becomes `num_key_value_heads` and the global one
/// `num_global_key_value_heads`, mirroring how `Gemma4Config::effective_*`
/// reads them back.
fn apply_gemma4_attention_geometry(
    config: &mut serde_json::Map<String, serde_json::Value>,
    metadata: &HashMap<String, GgufMetaValue>,
    layer_types: Option<&[String]>,
) {
    let mut put = |key: &str, value: u32| {
        config.insert(key.to_string(), serde_json::Value::Number(value.into()));
    };

    let global_head_dim = metadata
        .get("gemma4.attention.key_length")
        .and_then(|v| v.as_u32());
    let sliding_head_dim = metadata
        .get("gemma4.attention.key_length_swa")
        .and_then(|v| v.as_u32())
        .or(global_head_dim);

    // `head_dim` is the sliding one; global layers read `global_head_dim`.
    if let Some(hd) = sliding_head_dim {
        put("head_dim", hd);
    }
    if let Some(hd) = global_head_dim {
        put("global_head_dim", hd);
    }
    if let Some(w) = metadata
        .get("gemma4.attention.sliding_window")
        .and_then(|v| v.as_u32())
    {
        put("sliding_window", w);
    }

    let kv_per_layer = gemma4_kv_head_counts(metadata);
    if kv_per_layer.is_empty() {
        return;
    }

    let pick = |global: bool| -> Option<u32> {
        let types = layer_types?;
        let idx = types
            .iter()
            .position(|t| (t == "full_attention") == global)?;
        kv_per_layer.get(idx).copied()
    };

    if let Some(kv) = pick(false).or_else(|| kv_per_layer.first().copied()) {
        put("num_key_value_heads", kv);
    }
    if let Some(kv) = pick(true) {
        put("num_global_key_value_heads", kv);
    }
}

fn preserved_source_quantization(
    gguf: &GgufFile,
    import_k_quants: bool,
) -> Result<Option<serde_json::Value>> {
    let mut profiles = std::collections::BTreeMap::<String, SourceQuantProfile>::new();
    let mut profile_counts = HashMap::<SourceQuantProfile, usize>::new();

    for tensor in &gguf.tensors {
        if !import_k_quants && tensor.tensor_type.k_quant_format().is_some() {
            continue;
        }
        let Some(profile) = SourceQuantProfile::for_gguf_type(tensor.tensor_type) else {
            continue;
        };
        let Some(mapped) = gguf_name_to_hf_for_metadata(&tensor.name, &gguf.metadata) else {
            continue;
        };
        let prefix = mapped.strip_suffix(".weight").unwrap_or(&mapped);
        let prefix = super::normalize_override_key(prefix);
        if let Some(previous) = profiles.insert(prefix.clone(), profile)
            && previous != profile
        {
            return Err(Error::from_reason(format!(
                "GGUF source quantization collision: '{prefix}' resolves to both {} and {} tensors",
                previous.describe(),
                profile.describe()
            )));
        }
        *profile_counts.entry(profile).or_default() += 1;
    }

    if profiles.is_empty() {
        return Ok(None);
    }

    let default_profile = profile_counts
        .into_iter()
        .max_by_key(|&(profile, count)| (count, profile))
        .map(|(profile, _)| profile)
        .expect("non-empty profiles imply a profile count");
    let mixed = profiles.values().any(|&profile| profile != default_profile);

    let mut quant = serde_json::Map::new();
    quant.insert("bits".to_string(), serde_json::json!(default_profile.bits));
    quant.insert(
        "group_size".to_string(),
        serde_json::json!(default_profile.group_size),
    );
    quant.insert("mode".to_string(), serde_json::json!(default_profile.mode));
    if let Some(zero_point) = default_profile.symmetric_zero_point {
        quant.insert(
            SYMMETRIC_ZERO_POINT_KEY.to_string(),
            serde_json::json!(zero_point),
        );
    }
    for (prefix, profile) in profiles {
        // In a mixed file every source-quantized tensor gets an entry, so the
        // top-level triple is only a fallback and cannot misdecode the odd ones
        // out. K-quant tensors are named even in a file that looks uniform:
        // Unsloth Dynamic GGUFs mix formats by construction, and a reader that
        // consulted only the top-level triple would decode the tensors it does
        // not cover with the wrong kernel and group size.
        if mixed || profile.requires_explicit_entry() {
            quant.insert(prefix, profile.to_json());
        }
    }
    Ok(Some(serde_json::Value::Object(quant)))
}

// ── Dtype Conversion ────────────────────────────────────────────────────────

fn convert_tensor_dtype(arr: MxArray, target: DType) -> Result<MxArray> {
    let current = arr.dtype()?;
    if current == target {
        return Ok(arr);
    }
    arr.astype(target)
}

// ── NAPI Export ──────────────────────────────────────────────────────────────

#[napi(object)]
pub struct GgufConversionOptions {
    /// Path to the GGUF file
    pub input_path: String,

    /// Output directory for converted SafeTensors model
    pub output_dir: String,

    /// Optional directory containing the authoritative HuggingFace config and
    /// tokenizer/processor assets. GGUF metadata is not rich enough to recreate
    /// unified Gemma4 config fields such as head_dim, layer_types, vision, and
    /// audio configuration exactly.
    pub config_source_dir: Option<String>,

    /// Target dtype: "float32", "float16", "bfloat16" (default: keep original)
    pub dtype: Option<String>,

    /// Enable verbose logging
    pub verbose: Option<bool>,

    /// Enable quantization of converted weights
    pub quantize: Option<bool>,

    /// Quantization bits (default: 4)
    pub quant_bits: Option<i32>,

    /// Quantization group size (default: 64)
    pub quant_group_size: Option<i32>,

    /// Quantization mode: "affine" or "mxfp8"
    pub quant_mode: Option<String>,

    /// Quantization recipe for per-layer mixed-bit quantization.
    /// Options: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth
    pub quant_recipe: Option<String>,

    /// Path to an imatrix GGUF file for AWQ-style pre-scaling.
    /// Improves quantization quality by amplifying important weight channels.
    pub imatrix_path: Option<String>,

    /// Output filename (default: "model.safetensors").
    /// Useful for saving vision weights separately (e.g., "vision.safetensors").
    pub output_filename: Option<String>,

    /// When true, remap LLM weight keys for VLM compatibility:
    /// "model.X" → "language_model.model.X", "lm_head.X" → "language_model.lm_head.X"
    /// This makes the safetensors compatible with mlx-vlm.
    pub vlm_key_prefix: Option<bool>,

    /// Upgrade quantization to micro-scaling FP (mxfp4 / mxfp8).
    /// When true, applies after the recipe predicate: eligible 8-bit affine
    /// decisions become mxfp8 and 4-bit become mxfp4. Kept affine (not upgraded):
    /// affine-only loaders (lm_head, embed_tokens, router.proj,
    /// embedding_projection) at their recipe bits, MoE router gates (8-bit affine),
    /// and the recipe-pinned attention/GDN projections (o_proj / out_proj /
    /// in_proj_a / in_proj_b, 8-bit affine). Requires `quant_mode = "affine"`.
    /// Forces `group_size = 32` for upgraded layers.
    pub quant_mxfp: Option<bool>,

    /// Import ggml Q4_K / Q5_K / Q6_K tensors as MLX K-quant arrays instead of
    /// rejecting them (default: false). The blocks are repacked, never
    /// dequantized, so the output keeps the source file's weights and byte size.
    /// With this off, Q6_K remains the Gemma4 token-embedding BF16 fallback and
    /// Q4_K / Q5_K are an error.
    pub import_k_quants: Option<bool>,
}

#[napi(object)]
pub struct GgufConversionResult {
    pub num_tensors: i32,
    pub num_parameters: i64,
    pub output_path: String,
    pub tensor_names: Vec<String>,
    pub source_format: String,
}

fn apply_gguf_awq_prescaling(
    weights: &mut HashMap<String, MxArray>,
    imatrix: &crate::utils::imatrix::ImatrixData,
) -> Result<()> {
    crate::convert::reject_awq_for_prequantized_body(weights)?;
    let num_layers = crate::convert::infer_num_layers_from_weights(weights);
    crate::convert::apply_awq_prescaling(weights, imatrix, 0.5, num_layers)?;
    Ok(())
}

#[napi]
pub async fn convert_gguf_to_safetensors(
    options: GgufConversionOptions,
) -> Result<GgufConversionResult> {
    let input_path = PathBuf::from(&options.input_path);
    let output_dir = PathBuf::from(&options.output_dir);
    let config_source_dir = options.config_source_dir.as_deref().map(PathBuf::from);
    let verbose = options.verbose.unwrap_or(false);

    if !input_path.exists() {
        return Err(Error::from_reason(format!(
            "GGUF file not found: {}",
            input_path.display()
        )));
    }
    if let Some(ref dir) = config_source_dir
        && !dir.is_dir()
    {
        return Err(Error::from_reason(format!(
            "Config source directory not found or not a directory: {}",
            dir.display()
        )));
    }

    // Match the SafeTensors entry point's early policy gate. Plain affine
    // Unsloth cannot run without AWQ calibration; fixed-map selectors defer
    // the decision until GGUF architecture + remapped hybrid shape are known.
    if let Some(ref recipe) = options.quant_recipe {
        crate::convert::validate_unsloth_imatrix_selector(
            recipe,
            options.imatrix_path.as_deref(),
            options.quant_mxfp.unwrap_or(false),
            options.quant_mode.as_deref().unwrap_or("affine"),
        )
        .map_err(Error::from_reason)?;
    }

    // The nvidia recipe is documented + validated only for qwen3_5 /
    // qwen3_5_moe and is a data-free port with a fixed format map. Reject an
    // unsupported model type + the flags that would silently alter or
    // contradict the map BEFORE the imatrix AWQ pre-scaling (below) can rewrite
    // weights and BEFORE the recipe predicate is built/wrapped with
    // `apply_mxfp_upgrade`. Shares one validator with the safetensors entry
    // point (`convert_model_inner`) so both paths reject the same combinations
    // with identical messages.
    //
    // The GGUF path has no HF `model_type` in scope here: `GgufConversionOptions`
    // carries none, the CLI never forwards `--model-type` to it, and the arch is
    // only inferred from GGUF metadata later (as e.g. `qwen3`, never the exact
    // `qwen3_5`/`qwen3_5_moe`). Passing `None` therefore rejects nvidia-on-GGUF
    // via the model-type gate — the correct outcome, since the recipe is a
    // faithful full-precision→mxfp port and re-quantizing an already-lossy GGUF
    // was never a supported path.
    if options.quant_recipe.as_deref() == Some("nvidia") {
        crate::convert::validate_nvidia_recipe_options(
            None,  // config_family: no HF model_type in GGUF scope → rejected
            None,  // requested_model_type: CLI never forwards --model-type here
            false, // is_moe: unused once config_family None rejects upfront
            options.imatrix_path.as_deref(),
            options.quant_mxfp.unwrap_or(false),
            options.quant_bits,
            options.quant_group_size,
        )
        .map_err(Error::from_reason)?;
    }

    // Serialize all conversions process-wide before touching MLX's default
    // device + stream. Then route every MLX op through CPU for the duration
    // of this call. See `crate::convert::convert_mutex` and
    // `crate::convert::CpuConvertGuard` for the full rationale — same
    // reasoning applies here for GGUF→SafeTensors conversion of huge MoE
    // checkpoints.
    let _convert_lock = crate::convert::convert_mutex().lock().await;
    let _stream_guard = crate::convert::CpuConvertGuard::enter_cpu();

    // Parse GGUF header and metadata
    info!("Parsing GGUF file: {}", input_path.display());
    let gguf = parse_gguf(&input_path)?;

    info!(
        "GGUF v{}: {} tensors, {} metadata keys",
        gguf.version,
        gguf.tensor_count,
        gguf.metadata.len()
    );

    if verbose {
        if let Some(arch) = gguf.metadata.get("general.architecture") {
            info!("Architecture: {:?}", arch);
        }
        if let Some(name) = gguf.metadata.get("general.name") {
            info!("Model name: {:?}", name);
        }
    }

    // Log tensor type distribution
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for t in &gguf.tensors {
        *type_counts.entry(t.tensor_type.name()).or_insert(0) += 1;
    }
    for (ttype, count) in &type_counts {
        info!("  {ttype}: {count} tensors");
    }

    let import_k_quants = options.import_k_quants.unwrap_or(false);

    // Plain Qwen3 has no quantized inference runtime. Preserving Q4_0/Q4_1/
    // Q8_0 or K-quant source groups would successfully write an artifact whose
    // loader rejects its quantization metadata. Reject from the header instead;
    // true Qwen3.5 hybrids remain distinguished by their mixed attention/GDN
    // tensor shape even when an older writer labels the architecture `qwen3`.
    if let Some(tensor) = plain_qwen3_quantized_source(&gguf) {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' uses {}, but the plain Qwen3 runtime is dense-only and cannot consume preserved quantized weights. Use a floating-point Qwen3 source or a Qwen3.5 hybrid checkpoint. Rejected from the GGUF header before tensor data was loaded.",
            tensor.name,
            tensor.tensor_type.name(),
        )));
    }

    // Qwen3.5 GGUF stores GatedDeltaNet head axes in llama.cpp order, so the
    // conversion must reinterleave several linear-attention projections. That
    // transform currently operates on dense arrays. A K-quant projection is a
    // packed `.weight` plus `.scales` / `.biases`; reshaping only `.weight`
    // previously crossed the MLX FFI boundary with impossible packed geometry
    // and aborted the process with an uncaught foreign exception. Fail from the
    // header instead of loading or mutating any tensor payload.
    if import_k_quants && let Some((tensor, mapped)) = qwen35_kquant_dense_fixup_target(&gguf) {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' ({}) maps to Qwen3.5 linear-attention tensor '{}', which requires a head-axis reinterleave during conversion. --gguf-kquant cannot safely apply that dense transform to packed codes without also reordering the group's .scales/.biases, so this source cannot currently be imported losslessly. Use a floating-point source or a preconverted checkpoint. Rejected from the GGUF header before tensor data was loaded.",
            tensor.name,
            tensor.tensor_type.name(),
            mapped,
        )));
    }

    // K-quant import repacks ggml's Q4_K/Q5_K/Q6_K blocks bit-for-bit and never
    // dequantizes them, so any path that would re-quantize the model — an
    // explicit `--quantize`, a `--q-recipe`, the `--q-mxfp` upgrade, or
    // `--imatrix-path` AWQ pre-scaling — both contradicts the import and forces
    // whole-model float residency. Reject the combination upfront, before
    // loading gigabytes of weights, when the source actually carries K-quant
    // tensors. (With the import off, Q6_K stays the Gemma4 embedding BF16
    // fallback and re-quantizing it is still supported, so the guard is scoped
    // to the import path.)
    if import_k_quants
        && gguf
            .tensors
            .iter()
            .any(|t| t.tensor_type.k_quant_format().is_some())
    {
        let requantizing = options.quantize.unwrap_or(false)
            || options.quant_recipe.is_some()
            || options.quant_mxfp.unwrap_or(false)
            || options.imatrix_path.is_some();
        if requantizing {
            return Err(Error::from_reason(
                "K-quant GGUF import cannot be combined with re-quantization \
                 (--quantize / --q-recipe / --q-mxfp / --imatrix-path): ggml \
                 Q4_K/Q5_K/Q6_K blocks are imported bit-for-bit and never \
                 dequantized. Import them as-is, or drop --gguf-kquant to convert \
                 a dequantized copy.",
            ));
        }
    }

    // Symmetric ggml blocks (`d * (q - Z)`: Q4_0, Q8_0) repack losslessly into
    // MLX affine groups whose `.biases` companion is DERIVED from the scale at
    // load rather than stored, which only works because the emitted config
    // carries `symmetric_zero_point`. `--quantize` skips every already-packed
    // group, so the body is unchanged, but the config write then takes the
    // freshly-quantized branch and emits a `build_quantization_object` that
    // knows nothing about the marker — a checkpoint that converts and then
    // cannot be loaded.
    //
    // Carrying the marker forward instead is not an option: with `--quantize`
    // the output is a MIXTURE of skipped symmetric groups (bias derived) and
    // newly quantized float tensors (bias stored), and a single top-level claim
    // would falsely describe the latter. Reject upfront, mirroring the K-quant
    // guard above and the already-quantized gemma-QAT reject in
    // `convert::convert_model_inner`.
    //
    // Not mirrored in the CLI: unlike `--gguf-kquant`, this condition is read
    // off the file's tensor types rather than the flags, and nothing on the TS
    // side parses GGUF. The check still runs before a single weight byte is
    // read — `parse_gguf` above reads only the header, metadata, and tensor
    // descriptors.
    if options.quantize.unwrap_or(false)
        && let Some(tensor) = gguf
            .tensors
            .iter()
            .find(|t| symmetric_zero_point(t.tensor_type).is_some())
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' is already quantized ({}); convert without --quantize \
             (--q-mxfp requires it). Symmetric ggml blocks are repacked losslessly to \
             MLX affine groups whose '.biases' companion is derived from the scale \
             instead of being stored, and --quantize skips them and then writes a \
             quantization block describing only the tensors it did quantize — \
             dropping the '{SYMMETRIC_ZERO_POINT_KEY}' that derivation needs.",
            tensor.name,
            tensor.tensor_type.name(),
        )));
    }

    // The output file name decides whether this conversion owns config.json, and
    // two checks below turn on that: the secondary-output reject immediately
    // after, and the Gemma4 layout detection. Both belong here, ahead of the
    // save — `save_safetensors` opens the destination with `File::create`, which
    // truncates any existing file the instant it runs, so a rejection raised
    // after that point has already destroyed a readable checkpoint.
    let safetensors_filename = options
        .output_filename
        .as_deref()
        .unwrap_or("model.safetensors");
    let is_primary_model = safetensors_filename == "model.safetensors";

    // A secondary output (`--mmproj` writes `vision.safetensors`) shares the
    // output directory with the primary conversion and deliberately writes no
    // config.json — the primary one owns it. So a tensor whose decode geometry
    // can ONLY be expressed in config.json has nowhere to record it: a symmetric
    // ggml block loses the `symmetric_zero_point` its derived `.biases` is
    // rebuilt from, and an imported K-quant loses the mode/bits/group_size that
    // spell out ggml's geometry. Either way the bytes land packed correctly and
    // are then decoded as the runtime's generic affine default.
    //
    // K-quant tensors are exempt when the import is off, matching
    // `preserved_source_quantization`: without the import no K-quant array
    // reaches the output for a config entry to describe.
    //
    // Header-only, like the two guards above — `parse_gguf` has read the header,
    // metadata and tensor descriptors and nothing else.
    if !is_primary_model
        && let Some((tensor, profile)) = gguf.tensors.iter().find_map(|t| {
            if !import_k_quants && t.tensor_type.k_quant_format().is_some() {
                return None;
            }
            let profile = SourceQuantProfile::for_gguf_type(t.tensor_type)?;
            (profile.symmetric_zero_point.is_some() || profile.requires_explicit_entry())
                .then_some((t, profile))
        })
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' is {}, which loads as {} only when config.json says so — but this \
             conversion writes '{safetensors_filename}', a secondary output that writes no \
             config.json (the primary 'model.safetensors' conversion owns it). The packed bytes \
             would be written and then decoded with the runtime's default geometry. Convert this \
             file as the primary output, or use a source whose tensors are float or plain \
             group-32 affine.",
            tensor.name,
            tensor.tensor_type.name(),
            profile.describe(),
        )));
    }

    // Measured off the tensor list and cross-checked against the header, both of
    // which `parse_gguf` has already read, so the whole decision is available
    // before the destructive save. Only the primary output can act on it: the
    // fields it produces are config.json fields.
    let gemma4_layer_types = if is_primary_model && is_gemma4_main_gguf(&gguf.metadata) {
        gemma4_layer_types_from_missing_v(&gguf)?
    } else {
        None
    };

    // Where config.json and the runtime assets are read from: an explicit
    // `--config-dir` is authoritative, otherwise the directory beside the GGUF.
    // Resolved here rather than at the config write below because the guard that
    // follows has to know whether the config will be SYNTHESIZED, and it has to
    // know it before `save_safetensors` truncates the destination.
    let gguf_dir = input_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let asset_dir = config_source_dir.clone().unwrap_or(gguf_dir);
    let src_config = asset_dir.join("config.json");
    let synthesized_config = !src_config.exists();

    // A synthesized config gets its KV head counts from
    // `apply_gemma4_attention_geometry`, which can only tell the sliding count
    // from the global one when `head_count_kv` is an array with an entry per
    // layer. A scalar leaves `num_global_key_value_heads` unwritten and the
    // loader reuses the sliding count for the global layers; an absent key
    // leaves both unwritten and the loader falls back to 2. Detecting the K=V
    // layout is proof that those two counts DIFFER, so either fallback builds
    // global K/V projections at the wrong width. Nothing downstream catches it —
    // the quantized load path discards `out_features`, so the first token
    // reaches `keys.reshape(...)` and MLX aborts the process across the
    // `extern "C-unwind"` boundary instead of returning an error. An
    // authoritative config states the counts itself and needs no header.
    if let Some(types) = &gemma4_layer_types
        && synthesized_config
        && !gemma4_kv_counts_cover_both_layer_types(&gguf.metadata, types)
    {
        return Err(Error::from_reason(format!(
            "Gemma4 GGUF omits 'attn_v' on {} of its {} decoder blocks, so its global layers \
             have a different KV head count from its sliding ones — but \
             'gemma4.attention.head_count_kv' does not state both ({}), and there is no \
             config.json to take them from, so the converted config would describe the global \
             layers with the sliding count. Supply the model's config.json with --config-dir.",
            types.iter().filter(|t| *t == "full_attention").count(),
            types.len(),
            describe_gemma4_kv_head_counts(&gguf.metadata),
        )));
    }

    // Output keys an optional global `--dtype` request must leave alone,
    // spelled out rather than inferred from a suffix or a dtype.
    //
    // Without the K-quant import, Q6_K is a BF16-only dense fallback: it is a
    // genuine float array, so nothing else in the cast loop would spare it.
    // With the import on, the same tensors arrive as a packed uint32 `.weight`
    // plus `.scales` / `.biases` sidecars whose exact integer and f16 bit
    // patterns *are* the weights. The loop's generic `.scales` / `.biases` and
    // uint32 skips happen to cover those three today, which makes them a trap
    // keyed on suffix spelling; naming the keys here through the same helper
    // the importer names them with means changing either the suffixes or the
    // array dtypes cannot quietly start casting them.
    let preserve_dtype_keys: std::collections::HashSet<String> = gguf
        .tensors
        .iter()
        .filter(|t| t.tensor_type.k_quant_format().is_some())
        .filter_map(|t| gguf_name_to_hf_for_metadata(&t.name, &gguf.metadata))
        .flat_map(|key| -> Vec<String> {
            if import_k_quants {
                k_quant_output_keys(&key).to_vec()
            } else {
                // The pre-import fallback dequantizes the Gemma4 Q6_K embedding
                // to BF16 and rejects the other two, so there is one key.
                vec![key]
            }
        })
        .collect();

    // Load tensors
    info!("Loading tensors...");
    let mut weights = load_gguf_tensors(
        &input_path,
        &gguf,
        GgufLoadOptions {
            verbose,
            import_k_quants,
        },
    )?;
    info!("Loaded {} tensors", weights.len());

    // Remap keys from GGUF to HF naming
    info!("Remapping tensor names to HuggingFace format...");
    weights = remap_keys(weights, &gguf.metadata)?;

    // Unified Gemma4 mmproj tensors need name-aware CHW→HWC and position-table
    // permutations after remapping, before the generic shape fixups run.
    fixup_gemma4_mmproj_layout(&mut weights, &gguf.metadata)?;

    // Fix shapes that differ between GGUF and HF format
    fixup_shapes(&mut weights)?;

    // Fix Qwen3.5 linear attention head deinterleaving and A_log conversion
    fixup_qwen35_linear_attn(&mut weights, &gguf.metadata)?;

    // Optional dtype conversion (only for non-quantized weights)
    if let Some(dtype_str) = &options.dtype {
        let target_dtype = match dtype_str.as_str() {
            "float32" | "f32" => DType::Float32,
            "float16" | "f16" => DType::Float16,
            "bfloat16" | "bf16" => DType::BFloat16,
            other => {
                return Err(Error::from_reason(format!(
                    "Unsupported dtype: {other}. Use float32, float16, or bfloat16"
                )));
            }
        };

        info!("Converting to {dtype_str}...");
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            if preserve_dtype_keys.contains(&key) {
                continue;
            }
            // Skip quantized weight triplets
            if key.ends_with(".scales") || key.ends_with(".biases") {
                continue;
            }
            // Skip uint32 packed quantized weights
            if let Some(arr) = weights.get(&key)
                && arr.dtype()? == DType::Uint32
            {
                continue;
            }
            if let Some(arr) = weights.remove(&key) {
                let converted = convert_tensor_dtype(arr, target_dtype)?;
                weights.insert(key, converted);
            }
        }
    }

    // Apply AWQ pre-scaling if imatrix provided. GGUF Q4/Q8 tensors have
    // already been repacked to Uint32 + sidecars above, so the shared guard
    // must run before AWQ can multiply packed integers or fold norms twice.
    if let Some(ref imatrix_path) = options.imatrix_path {
        let imatrix = crate::utils::imatrix::parse_imatrix(imatrix_path)?;
        apply_gguf_awq_prescaling(&mut weights, &imatrix)?;
    }

    // Optional quantization
    let mut per_layer_overrides: HashMap<String, serde_json::Value> = HashMap::new();
    let do_quantize = options.quantize.unwrap_or(false);
    let quant_mode_str = options
        .quant_mode
        .as_deref()
        .unwrap_or("affine")
        .to_string();
    let quant_mxfp = options.quant_mxfp.unwrap_or(false);

    // Validate quant_mode before it reaches FFI
    if do_quantize
        && quant_mode_str != "affine"
        && quant_mode_str != "mxfp8"
        && quant_mode_str != "mxfp4"
        && quant_mode_str != "nvfp4"
    {
        return Err(Error::from_reason(format!(
            "Invalid quant_mode '{}': must be 'affine', 'mxfp8', 'mxfp4', or 'nvfp4'",
            quant_mode_str
        )));
    }

    // --q-mxfp orthogonality: requires affine baseline.
    if quant_mxfp && !do_quantize {
        return Err(Error::from_reason(
            "--q-mxfp requires --quantize to be enabled".to_string(),
        ));
    }
    if quant_mxfp && quant_mode_str != "affine" {
        return Err(Error::from_reason(format!(
            "--q-mxfp requires --q-mode affine (default), got '{}'. \
             --q-mxfp orthogonally upgrades affine decisions to mxfp4/mxfp8.",
            quant_mode_str
        )));
    }

    let quant_bits = options.quant_bits.unwrap_or(match quant_mode_str.as_str() {
        "mxfp4" => 4,
        "mxfp8" => 8,
        "nvfp4" => 4,
        _ => 4,
    });
    let quant_group_size = options
        .quant_group_size
        .unwrap_or(match quant_mode_str.as_str() {
            "mxfp4" | "mxfp8" => 32,
            "nvfp4" => 16,
            _ => 64,
        });

    // MXFP modes have strict bits/group_size invariants enforced by the MLX
    // backend. Surface the failure here (with a clear message) rather than
    // letting it bubble up as a confusing FFI error mid-conversion.
    if do_quantize && quant_mode_str == "mxfp4" && (quant_bits != 4 || quant_group_size != 32) {
        return Err(Error::from_reason(format!(
            "mxfp4 requires bits=4 and group_size=32 (got bits={quant_bits}, group_size={quant_group_size})"
        )));
    }
    if do_quantize && quant_mode_str == "mxfp8" && (quant_bits != 8 || quant_group_size != 32) {
        return Err(Error::from_reason(format!(
            "mxfp8 requires bits=8 and group_size=32 (got bits={quant_bits}, group_size={quant_group_size})"
        )));
    }
    if do_quantize && quant_mode_str == "nvfp4" {
        crate::convert::validate_nvfp4_invariants(quant_bits, quant_group_size)
            .map_err(Error::from_reason)?;
    }

    if do_quantize
        && options.quant_recipe.is_some()
        && quant_mode_str != "affine"
        && quant_mode_str != "nvfp4"
    {
        return Err(Error::from_reason(format!(
            "--q-recipe is compatible with --q-mode affine or nvfp4 only; for mxfp4/mxfp8 use --q-mxfp instead. Got '{}'.",
            quant_mode_str
        )));
    }
    // Restrict --q-mode nvfp4 + --q-recipe to recipes that have model-aware
    // tensor-class exclusions for NVFP4-sensitive layers. See
    // [`crate::convert::validate_nvfp4_recipe`] for the full rationale.
    if do_quantize
        && quant_mode_str == "nvfp4"
        && let Some(ref recipe) = options.quant_recipe
    {
        crate::convert::validate_nvfp4_recipe(recipe).map_err(Error::from_reason)?;
    }

    // Effective mode/group_size recorded in config.json. The no-recipe path
    // updates these when --q-mxfp upgrades the global mode to mxfp4/mxfp8.
    let mut quant_mode_effective = quant_mode_str.clone();
    let mut quant_group_size_effective = quant_group_size;

    if do_quantize {
        if let Some(ref recipe) = options.quant_recipe {
            info!(
                "Quantizing weights: {quant_bits}-bit {quant_mode_str} (group_size={quant_group_size}, recipe={recipe})..."
            );
            let weight_keys: Vec<String> = weights.keys().cloned().collect();
            // See convert.rs: under --q-mode nvfp4 the global gs=16 is illegal
            // for the affine decisions recipes emit (lm_head, AWQ-corrected
            // attn/SSM projections). Pass a recipe-affine-appropriate gs.
            let recipe_gs = if quant_mode_str == "nvfp4" {
                64
            } else {
                quant_group_size
            };
            // Mirror the SafeTensors selector: verified Qwen hybrids use the
            // official class map for both translated MXFP and DGX NVFP4;
            // plain affine and ambiguous/non-Qwen inputs retain Dynamic 2.0.
            let is_qwen35_hybrid = is_qwen35_hybrid_gguf(&gguf.metadata, &weight_keys);
            let official_unsloth_kind = crate::convert::select_official_unsloth_recipe(
                recipe,
                quant_mxfp,
                &quant_mode_str,
                is_qwen35_hybrid,
            );
            crate::convert::validate_unsloth_imatrix_after_selection(
                recipe,
                options.imatrix_path.as_deref(),
                official_unsloth_kind,
            )
            .map_err(Error::from_reason)?;
            if recipe == "unsloth" && options.imatrix_path.is_none() {
                warn!("{}", crate::convert::UNSLOTH_NO_IMATRIX_WARNING);
            }
            let predicate = match official_unsloth_kind {
                Some(kind) => crate::convert::build_official_unsloth_recipe(&weight_keys, kind),
                None => crate::convert::build_predicate_for_recipe(
                    recipe,
                    &weight_keys,
                    quant_bits,
                    recipe_gs,
                )
                .map_err(Error::from_reason)?,
            };
            let predicate: Box<dyn Fn(&str) -> crate::convert::QuantDecision + Send + Sync> =
                if quant_mxfp && official_unsloth_kind.is_none() {
                    crate::convert::apply_mxfp_upgrade(predicate, quant_bits)
                } else if quant_mode_str == "nvfp4" && official_unsloth_kind.is_none() {
                    // Recipe + --q-mode nvfp4: promote 4-bit recipe decisions
                    // to NVFP4 (group_size=16). Mutually exclusive with
                    // quant_mxfp, since --q-mxfp requires --q-mode affine.
                    crate::convert::apply_nvfp4_upgrade(predicate)
                } else {
                    predicate
                };
            per_layer_overrides = crate::convert::quantize_weights_with_recipe_pub(
                &mut weights,
                quant_bits,
                quant_group_size,
                &quant_mode_str,
                &*predicate,
                // GGUF conversion never targets lfm2 here; keep the embedding
                // bf16 (the legacy embedding-skip behavior).
                false,
            )?;
        } else {
            // No recipe. NVFP4 cannot land here — the legacy `quantize_weights`
            // path uniformly applies the global mode/bits/group_size to every
            // quantizable tensor, which for NVFP4 silently corrupts the
            // sensitivity-critical tensors (linear_attn.out_proj, down_proj,
            // etc.). See `convert::NVFP4_NO_RECIPE_ERROR` for the full
            // rationale.
            if quant_mode_str == "nvfp4" {
                return Err(Error::from_reason(
                    crate::convert::NVFP4_NO_RECIPE_ERROR.to_string(),
                ));
            }
            // --q-mxfp overrides the global mode + group_size so the
            // legacy quantize path emits mxfp4/mxfp8 weights.
            let (effective_mode, effective_gs) = if quant_mxfp {
                match quant_bits {
                    8 => ("mxfp8".to_string(), 32),
                    4 => ("mxfp4".to_string(), 32),
                    _ => {
                        return Err(Error::from_reason(format!(
                            "--q-mxfp without a recipe requires --q-bits 4 or 8 (got {})",
                            quant_bits
                        )));
                    }
                }
            } else {
                (quant_mode_str.clone(), quant_group_size)
            };
            info!(
                "Quantizing weights: {quant_bits}-bit {effective_mode} (group_size={effective_gs})..."
            );
            // Capture per-layer overrides emitted by the legacy path (router-gate
            // upgrades, lm_head / router.proj / embed_tokens affine downgrades).
            // These must round-trip into config.json so the loader dispatches
            // each layer to the correct builder.
            per_layer_overrides = crate::convert::quantize_weights_pub(
                &mut weights,
                quant_bits,
                effective_gs,
                &effective_mode,
                // GGUF conversion never targets lfm2 here; keep the embedding bf16.
                false,
            )?;
            if !per_layer_overrides.is_empty() {
                info!(
                    "No-recipe quantization emitted {} per-layer overrides (router-gate / affine-only keys): {:?}",
                    per_layer_overrides.len(),
                    per_layer_overrides.keys().collect::<Vec<_>>()
                );
            }
            quant_mode_effective = effective_mode;
            quant_group_size_effective = effective_gs;
        }
    }

    // Remap LLM keys for VLM compatibility when requested
    if options.vlm_key_prefix.unwrap_or(false) {
        let keys: Vec<String> = weights.keys().cloned().collect();
        for key in keys {
            if key.starts_with("model.") || key.starts_with("lm_head.") {
                let new_key = format!("language_model.{}", key);
                if let Some(val) = weights.remove(&key) {
                    weights.insert(new_key, val);
                }
            }
        }
        info!("Remapped LLM keys with language_model prefix for VLM compatibility");
    }

    // Create output directory
    fs::create_dir_all(&output_dir)
        .map_err(|e| Error::from_reason(format!("Failed to create output directory: {e}")))?;

    // Count parameters
    let num_parameters: i64 = weights
        .values()
        .map(|arr| arr.size().unwrap_or(0) as i64)
        .sum();

    // Save SafeTensors
    let safetensors_path = output_dir.join(safetensors_filename);
    info!("Saving to {}", safetensors_path.display());
    // Capture tensor names BEFORE `save_safetensors` — it drains `weights`
    // as it streams each tensor to disk so MLX-allocated backing buffers
    // can be released immediately on large MoE checkpoints. Reading
    // `weights.keys()` after the save would return an empty list and the
    // GgufConversionResult would report num_tensors = 0 to JS callers.
    let tensor_names: Vec<String> = weights.keys().cloned().collect();
    // Add "format: mlx" metadata so loaders (e.g., mlx-vlm) know weights are
    // already in MLX layout and skip sanitize (which would double-apply +1.0 to norms).
    let st_metadata = serde_json::json!({ "format": "mlx" });
    save_safetensors(&safetensors_path, &mut weights, Some(st_metadata))?;

    // Only write config.json and tokenizer files for the primary model file
    // (`is_primary_model`, resolved before the save). Secondary files (e.g.,
    // vision.safetensors for mmproj) should not overwrite the config written by
    // the primary LLM conversion.
    if is_primary_model {
        // `asset_dir` / `src_config` / `synthesized_config` were resolved before
        // the save (an explicit `--config-dir` is authoritative — needed by
        // unified Gemma4, whose full text/vision/audio config cannot be rebuilt
        // from GGUF metadata — otherwise the alongside-GGUF lookup and the
        // metadata-extraction fallback).
        let config_path = output_dir.join("config.json");
        if config_source_dir.is_some() && !src_config.exists() {
            return Err(Error::from_reason(format!(
                "Authoritative config.json not found in config source directory: {}",
                asset_dir.display()
            )));
        }

        // Load or extract config, then inject quantization metadata if needed
        let mut config_json: serde_json::Value = if src_config.exists() {
            let data = fs::read_to_string(&src_config)
                .map_err(|e| Error::from_reason(format!("Failed to read config.json: {e}")))?;
            serde_json::from_str(&data)
                .map_err(|e| Error::from_reason(format!("Failed to parse config.json: {e}")))?
        } else {
            extract_config(&gguf.metadata)
        };

        // Measured off the tensor list and cross-checked against the header, so
        // it also applies to a hand-supplied config.json: a config for a GGUF
        // whose global layers carry no `attn_v` but that says nothing about the
        // layout is describing weights that do not exist, and the weights are
        // the artifact. A config that DOES state either field is the stronger
        // source and keeps it — writing over an explicit statement would make
        // `--config-dir` unable to correct a mislabelled checkpoint.
        if is_gemma4_main_gguf(&gguf.metadata) {
            if let Some(types) = &gemma4_layer_types {
                if !gemma4_config_states(&config_json, "attention_k_eq_v") {
                    config_json["attention_k_eq_v"] = serde_json::Value::Bool(true);
                }
                if !gemma4_config_states(&config_json, "layer_types") {
                    config_json["layer_types"] = serde_json::Value::Array(
                        types
                            .iter()
                            .cloned()
                            .map(serde_json::Value::String)
                            .collect(),
                    );
                }
            }
            // Only when we synthesized the config: an authoritative config.json
            // already states this geometry, and the GGUF is the weaker source.
            if synthesized_config && let Some(obj) = config_json.as_object_mut() {
                apply_gemma4_attention_geometry(obj, &gguf.metadata, gemma4_layer_types.as_deref());
            }
        }

        if do_quantize {
            let quant_obj = crate::convert::build_quantization_object(
                quant_bits,
                Some(quant_group_size_effective),
                &quant_mode_effective,
                &per_layer_overrides,
                /* is_privacy_filter */ false,
                /* skip_mtp */ false,
            );
            config_json["quantization"] = quant_obj.clone();
            config_json["quantization_config"] = quant_obj;
        } else if let Some(quant_obj) = preserved_source_quantization(&gguf, import_k_quants)? {
            // Source Q4_0/Q4_1/Q8_0 tensors were losslessly repacked into MLX
            // affine groups of 32, and imported K-quants keep ggml's own
            // geometry. Record that even without an extra quantize request;
            // otherwise Gemma4 defaults to group_size 64 and decodes the exact
            // packed bytes with the wrong geometry.
            config_json["quantization"] = quant_obj.clone();
            config_json["quantization_config"] = quant_obj;
        }

        let config_str = serde_json::to_string_pretty(&config_json)
            .map_err(|e| Error::from_reason(format!("Failed to serialize config: {e}")))?;
        fs::write(&config_path, &config_str)
            .map_err(|e| Error::from_reason(format!("Failed to write config.json: {e}")))?;
        if do_quantize {
            info!("Wrote config.json with quantization metadata");
        } else if preserved_source_quantization(&gguf, import_k_quants)?.is_some() {
            info!("Wrote config.json with preserved GGUF source quantization metadata");
        } else if src_config.exists() {
            info!("Copied config.json from source directory");
        } else {
            info!("Wrote config.json (extracted from GGUF metadata)");
        }

        // Copy the same runtime assets as the SafeTensors converter. In
        // particular unified Gemma4 requires its external chat template and
        // processor config in addition to tokenizer.json.
        let asset_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "generation_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "processor_config.json",
            "viterbi_calibration.json",
        ];
        for filename in &asset_files {
            let src = asset_dir.join(filename);
            if src.exists() {
                let dst = output_dir.join(filename);
                if src == dst {
                    continue;
                }
                if config_source_dir.is_some() {
                    fs::copy(&src, &dst).map_err(|e| {
                        Error::from_reason(format!("Failed to copy {filename}: {e}"))
                    })?;
                    info!("Copied {filename}");
                } else if let Err(e) = fs::copy(&src, &dst) {
                    warn!("Failed to copy {filename}: {e}");
                } else {
                    info!("Copied {filename}");
                }
            } else if matches!(*filename, "tokenizer.json" | "tokenizer_config.json") {
                warn!(
                    "{filename} not found in asset source directory {}. You may need to download it separately.",
                    asset_dir.display()
                );
            }
        }
    } else {
        info!(
            "Skipping config.json and tokenizer writes for secondary output file: {safetensors_filename}"
        );
    }

    // Build source format description
    let source_format = type_counts
        .iter()
        .map(|(t, c)| format!("{t}({c})"))
        .collect::<Vec<_>>()
        .join(", ");

    Ok(GgufConversionResult {
        num_tensors: tensor_names.len() as i32,
        num_parameters,
        output_path: output_dir.to_string_lossy().to_string(),
        tensor_names,
        source_format,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gguf_kquant::repack_kquant;

    fn build_minimal_gguf(
        metadata: &[(&str, GgufMetaValue)],
        tensors: &[(&str, &[u64], GgufTensorType, &[u8])],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&GGUF_VERSION_3.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // Metadata count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        // Write metadata KV pairs
        for (key, value) in metadata {
            // Key string: len(u64) + bytes
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());

            match value {
                GgufMetaValue::Uint32(v) => {
                    buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                GgufMetaValue::String(s) => {
                    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
                    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    buf.extend_from_slice(s.as_bytes());
                }
                GgufMetaValue::Float32(v) => {
                    buf.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                GgufMetaValue::ArrayI32(values) => {
                    buf.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
                    buf.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
                    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for v in values {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                _ => panic!("Unsupported metadata type in test helper"),
            }
        }

        // Compute tensor data: collect offsets
        let alignment: u64 = GGUF_DEFAULT_ALIGNMENT;
        let mut tensor_data_parts: Vec<&[u8]> = Vec::new();
        let mut tensor_offsets: Vec<u64> = Vec::new();
        let mut current_offset: u64 = 0;

        for (_, _, _, data) in tensors {
            // Align offset
            let remainder = current_offset % alignment;
            if remainder != 0 {
                current_offset += alignment - remainder;
            }
            tensor_offsets.push(current_offset);
            tensor_data_parts.push(data);
            current_offset += data.len() as u64;
        }

        // Write tensor info descriptors
        for (i, (name, dims, ttype, _)) in tensors.iter().enumerate() {
            // Name string
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            // n_dims
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            // dims
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            // type
            buf.extend_from_slice(&(*ttype as u32).to_le_bytes());
            // offset
            buf.extend_from_slice(&tensor_offsets[i].to_le_bytes());
        }

        // Pad to alignment for data section
        let remainder = buf.len() as u64 % alignment;
        if remainder != 0 {
            let padding = alignment - remainder;
            buf.extend(vec![0u8; padding as usize]);
        }

        // Write tensor data
        let data_start = buf.len();
        for (i, data) in tensor_data_parts.iter().enumerate() {
            let target_offset = data_start + tensor_offsets[i] as usize;
            while buf.len() < target_offset {
                buf.push(0);
            }
            buf.extend_from_slice(data);
        }

        buf
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let data = build_minimal_gguf(
            &[
                (
                    "general.architecture",
                    GgufMetaValue::String("llama".to_string()),
                ),
                ("llama.block_count", GgufMetaValue::Uint32(32)),
            ],
            &[],
        );

        let tmp = std::env::temp_dir().join("test_minimal.gguf");
        fs::write(&tmp, &data).unwrap();

        let gguf = parse_gguf(&tmp).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensor_count, 0);
        assert_eq!(gguf.metadata.len(), 2);

        if let Some(GgufMetaValue::String(arch)) = gguf.metadata.get("general.architecture") {
            assert_eq!(arch, "llama");
        } else {
            panic!("Missing architecture metadata");
        }

        if let Some(GgufMetaValue::Uint32(count)) = gguf.metadata.get("llama.block_count") {
            assert_eq!(*count, 32);
        } else {
            panic!("Missing block_count metadata");
        }

        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_parse_gguf_with_f32_tensor() {
        // 2x3 F32 tensor = 24 bytes
        let tensor_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // GGUF dims are reversed: for a [2, 3] MLX shape, GGUF stores [3, 2]
        let data = build_minimal_gguf(
            &[],
            &[("test.weight", &[3, 2], GgufTensorType::F32, &tensor_data)],
        );

        let tmp = std::env::temp_dir().join("test_f32_tensor.gguf");
        fs::write(&tmp, &data).unwrap();

        let gguf = parse_gguf(&tmp).unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "test.weight");
        assert_eq!(gguf.tensors[0].tensor_type, GgufTensorType::F32);
        // MLX shape should be reversed
        assert_eq!(gguf.tensors[0].mlx_shape(), vec![2, 3]);

        let weights = load_gguf_tensors(&tmp, &gguf, GgufLoadOptions::default()).unwrap();
        assert_eq!(weights.len(), 1);

        let arr = weights.get("test.weight").unwrap();
        assert_eq!(arr.ndim().unwrap(), 2);
        assert_eq!(arr.size().unwrap(), 6);
        assert_eq!(arr.shape_at(0).unwrap(), 2);
        assert_eq!(arr.shape_at(1).unwrap(), 3);

        let values = arr.to_float32().unwrap();
        let values_vec: Vec<f32> = values.iter().copied().collect();
        assert_eq!(values_vec, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        fs::remove_file(&tmp).ok();
    }

    fn pack_q6k_block(q: &[i8; 256], scales: &[i8; 16], d: f32) -> [u8; 210] {
        let mut block = [0u8; 210];
        for (value_idx, &signed) in q.iter().enumerate() {
            assert!((-32..=31).contains(&signed));
            let encoded = (i32::from(signed) + 32) as u8;
            let group32 = value_idx / 32;
            let lane = value_idx % 32;
            let group_in_half = group32 % 4;
            let ql_index = (group32 / 4) * 64 + (group_in_half % 2) * 32 + lane;
            let ql_shift = (group_in_half / 2) * 4;
            block[ql_index] |= (encoded & 0x0f) << ql_shift;

            let qh_index = 128 + (group32 / 4) * 32 + lane;
            let qh_shift = group_in_half * 2;
            block[qh_index] |= ((encoded >> 4) & 0x03) << qh_shift;
        }
        for (i, &scale) in scales.iter().enumerate() {
            block[192 + i] = scale as u8;
        }
        block[208..210].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block
    }

    /// Inverse of `gguf_kquant::get_scale_min_k4` (:173): pack eight 6-bit
    /// `(sc, m)` pairs into the 12 bytes ggml stores them in. `j < 4` lands the
    /// whole 6-bit value in the low six bits of one byte; `j >= 4` splits it into
    /// a low nibble and a two-bit field stolen from the top of an earlier byte.
    fn pack_scales_min_k4(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for j in 0..4 {
            q[j] |= sc[j] & 63;
            q[j + 4] |= m[j] & 63;
        }
        for j in 4..8 {
            q[j + 4] |= (sc[j] & 0x0f) | ((m[j] & 0x0f) << 4);
            q[j - 4] |= ((sc[j] >> 4) & 0x03) << 6;
            q[j] |= ((m[j] >> 4) & 0x03) << 6;
        }
        q
    }

    /// Inverse of `gguf_kquant::q4k_code` (:276): lay 256 four-bit codes into a
    /// Q4_K super-block. Byte `bi` of `qs` holds the low nibble of value
    /// `64*(bi/32)+(bi%32)` and the high nibble of that value plus 32.
    fn pack_q4k_block(
        codes: &[u8; 256],
        sc: &[u8; 8],
        m: &[u8; 8],
        d: f32,
        dmin: f32,
    ) -> [u8; 144] {
        let mut block = [0u8; 144];
        block[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block[4..16].copy_from_slice(&pack_scales_min_k4(sc, m));
        for bi in 0..128 {
            let half = bi / 32;
            let lane = bi % 32;
            let lo = codes[64 * half + lane] & 0x0f;
            let hi = codes[64 * half + 32 + lane] & 0x0f;
            block[16 + bi] = lo | (hi << 4);
        }
        block
    }

    /// Inverse of `gguf_kquant::q5k_code` (:288): Q4_K's low-nibble layout plus a
    /// fifth bit plane in `qh`, indexed by lane with bit `2*(v/64) + (v%64>=32)`.
    fn pack_q5k_block(
        codes: &[u8; 256],
        sc: &[u8; 8],
        m: &[u8; 8],
        d: f32,
        dmin: f32,
    ) -> [u8; 176] {
        let mut block = [0u8; 176];
        block[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        block[4..16].copy_from_slice(&pack_scales_min_k4(sc, m));
        for (v, &code) in codes.iter().enumerate() {
            let lane = v % 32;
            let bit = 2 * (v / 64) + usize::from(v % 64 >= 32);
            block[16 + lane] |= ((code >> 4) & 1) << bit;
        }
        for bi in 0..128 {
            let half = bi / 32;
            let lane = bi % 32;
            let lo = codes[64 * half + lane] & 0x0f;
            let hi = codes[64 * half + 32 + lane] & 0x0f;
            block[48 + bi] = lo | (hi << 4);
        }
        block
    }

    /// Recover `count` codes from MLX's LSB-first `bits`-wide `.weight` stream —
    /// the inverse of `gguf_kquant::BitPacker`. Codes straddle uint32 words for
    /// bits 5 and 6, so a bit cursor is required; `packed |= code << (i*bits)`
    /// would only unpack the 4-bit case.
    fn unpack_lsb_codes(weight: &[u32], bits: usize, count: usize) -> Vec<u32> {
        let mask = (1u64 << bits) - 1;
        let mut codes = Vec::with_capacity(count);
        let mut acc = 0u64;
        let mut nbits = 0u32;
        let mut wi = 0usize;
        for _ in 0..count {
            while nbits < bits as u32 {
                acc |= u64::from(weight[wi]) << nbits;
                wi += 1;
                nbits += 32;
            }
            codes.push((acc & mask) as u32);
            acc >>= bits;
            nbits -= bits as u32;
        }
        codes
    }

    /// Import a single hand-built K-quant super-block through the production
    /// loader with the K-quant import on, returning its `.weight`/`.scales`/
    /// `.biases` triple.
    fn load_single_kquant_block(ty: GgufTensorType, block: &[u8]) -> HashMap<String, MxArray> {
        let data = build_minimal_gguf(&[], &[("blk.0.ffn_down.weight", &[256, 1], ty, block)]);
        let tmp = std::env::temp_dir().join(format!(
            "mlx-node-kquant-roundtrip-{}-{}.gguf",
            ty.name(),
            std::process::id()
        ));
        fs::write(&tmp, data).unwrap();
        let gguf = parse_gguf(&tmp).unwrap();
        let weights = load_gguf_tensors(
            &tmp,
            &gguf,
            GgufLoadOptions {
                verbose: false,
                import_k_quants: true,
            },
        )
        .unwrap();
        fs::remove_file(&tmp).ok();
        weights
    }

    #[test]
    fn q4k_import_round_trips_known_codes_scales_and_biases_bit_exact() {
        // Known-answer round-trip: hand-build a super-block, then assert the
        // repacked arrays reproduce the exact codes and scale bit patterns fed
        // in — independent of the repacker that produced them. Codes vary with
        // the super-block half so a low/high nibble swap cannot pass.
        let mut codes = [0u8; 256];
        for (v, c) in codes.iter_mut().enumerate() {
            *c = ((v + 3 * (v / 32)) % 16) as u8;
        }
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            sc[j] = ((3 * j + 7) & 63) as u8; // exercises the j>=4 split field
            m[j] = ((61 - 5 * j) & 63) as u8;
        }
        let block = pack_q4k_block(&codes, &sc, &m, 0.5, -0.25);
        let weights = load_single_kquant_block(GgufTensorType::Q4K, &block);

        let weight = weights.get("blk.0.ffn_down.weight").unwrap();
        assert_eq!(weight.dtype().unwrap(), DType::Uint32);
        let packed: Vec<u32> = weight.to_uint32().unwrap().iter().copied().collect();
        assert_eq!(
            unpack_lsb_codes(&packed, 4, 256),
            codes.iter().map(|&c| u32::from(c)).collect::<Vec<_>>()
        );

        let scales = weights.get("blk.0.ffn_down.scales").unwrap();
        assert_eq!(scales.dtype().unwrap(), DType::Uint8);
        let want_scales: Vec<u8> = (0..8).flat_map(|j| [sc[j], m[j]]).collect();
        assert_eq!(scales.to_uint8().unwrap(), want_scales);

        let biases = weights.get("blk.0.ffn_down.biases").unwrap();
        assert_eq!(biases.dtype().unwrap(), DType::Float16);
        assert_eq!(
            biases.to_uint16_native().unwrap(),
            vec![
                half::f16::from_f32(0.5).to_bits(),
                half::f16::from_f32(-0.25).to_bits(),
            ]
        );
    }

    #[test]
    fn q5k_import_round_trips_known_codes_scales_and_biases_bit_exact() {
        // Same known-answer round-trip for Q5_K. Codes span the full 5-bit range
        // so the `qh` fifth-bit plane is exercised, not just the `qs` nibbles.
        let mut codes = [0u8; 256];
        for (v, c) in codes.iter_mut().enumerate() {
            *c = ((v + 3 * (v / 32)) % 32) as u8;
        }
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            sc[j] = ((5 * j + 9) & 63) as u8;
            m[j] = ((58 - 4 * j) & 63) as u8;
        }
        let block = pack_q5k_block(&codes, &sc, &m, 0.125, -0.75);
        let weights = load_single_kquant_block(GgufTensorType::Q5K, &block);

        let weight = weights.get("blk.0.ffn_down.weight").unwrap();
        assert_eq!(weight.dtype().unwrap(), DType::Uint32);
        let packed: Vec<u32> = weight.to_uint32().unwrap().iter().copied().collect();
        assert_eq!(
            unpack_lsb_codes(&packed, 5, 256),
            codes.iter().map(|&c| u32::from(c)).collect::<Vec<_>>()
        );

        let scales = weights.get("blk.0.ffn_down.scales").unwrap();
        assert_eq!(scales.dtype().unwrap(), DType::Uint8);
        let want_scales: Vec<u8> = (0..8).flat_map(|j| [sc[j], m[j]]).collect();
        assert_eq!(scales.to_uint8().unwrap(), want_scales);

        let biases = weights.get("blk.0.ffn_down.biases").unwrap();
        assert_eq!(biases.dtype().unwrap(), DType::Float16);
        assert_eq!(
            biases.to_uint16_native().unwrap(),
            vec![
                half::f16::from_f32(0.125).to_bits(),
                half::f16::from_f32(-0.75).to_bits(),
            ]
        );
    }

    #[test]
    fn q6k_import_round_trips_known_codes_scales_and_biases_bit_exact() {
        // Q6_K keeps ggml's signed sub-scales, so `.scales` stays int8 and the
        // packed code is the ggml value plus 32. Assert the codes survive the
        // ql/qh de-swizzle and the signed scales keep their sign.
        let mut q = [0i8; 256];
        for (v, val) in q.iter_mut().enumerate() {
            *val = (((v + 3 * (v / 16)) % 64) as i32 - 32) as i8;
        }
        let mut scales = [0i8; 16];
        for (j, s) in scales.iter_mut().enumerate() {
            *s = (j as i32 * 4 - 30) as i8;
        }
        let block = pack_q6k_block(&q, &scales, 0.375);
        let weights = load_single_kquant_block(GgufTensorType::Q6K, &block);

        let weight = weights.get("blk.0.ffn_down.weight").unwrap();
        assert_eq!(weight.dtype().unwrap(), DType::Uint32);
        let packed: Vec<u32> = weight.to_uint32().unwrap().iter().copied().collect();
        assert_eq!(
            unpack_lsb_codes(&packed, 6, 256),
            q.iter()
                .map(|&c| (i32::from(c) + 32) as u32)
                .collect::<Vec<_>>()
        );

        let got_scales = weights.get("blk.0.ffn_down.scales").unwrap();
        assert_eq!(got_scales.dtype().unwrap(), DType::Int8);
        assert_eq!(got_scales.to_int8().unwrap(), scales.to_vec());

        let biases = weights.get("blk.0.ffn_down.biases").unwrap();
        assert_eq!(biases.dtype().unwrap(), DType::Float16);
        assert_eq!(
            biases.to_uint16_native().unwrap(),
            vec![half::f16::from_f32(0.375).to_bits()]
        );
    }

    #[test]
    fn gemma4_q6k_token_embedding_dequantizes_to_bf16() {
        let mut q = [0i8; 256];
        for (i, value) in q.iter_mut().enumerate() {
            *value = (i as i32 % 64 - 32) as i8;
        }
        let mut scales = [0i8; 16];
        for (i, scale) in scales.iter_mut().enumerate() {
            *scale = i as i8 - 8;
        }
        let block = pack_q6k_block(&q, &scales, 0.5);
        let data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".to_string()),
            )],
            &[("token_embd.weight", &[256, 1], GgufTensorType::Q6K, &block)],
        );
        let tmp = std::env::temp_dir().join(format!(
            "mlx-node-gemma4-q6k-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::write(&tmp, data).unwrap();

        let gguf = parse_gguf(&tmp).unwrap();
        assert_eq!(gguf.tensors[0].tensor_type, GgufTensorType::Q6K);
        assert_eq!(gguf.tensors[0].data_size(), 210);
        let weights = load_gguf_tensors(&tmp, &gguf, GgufLoadOptions::default()).unwrap();
        let embedding = weights.get("token_embd.weight").unwrap();
        assert_eq!(embedding.dtype().unwrap(), DType::BFloat16);
        assert_eq!(embedding.shape().unwrap().as_ref(), &[1, 256]);
        let got = embedding.to_float32().unwrap();
        for i in 0..256 {
            let exact = 0.5 * f32::from(scales[i / 16]) * f32::from(q[i]);
            let expected = half::bf16::from_f32(exact).to_f32();
            assert_eq!(got[i], expected, "Q6_K mismatch at value {i}");
        }

        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn q6k_rejects_non_gemma_embedding_position() {
        let block = pack_q6k_block(&[0; 256], &[1; 16], 1.0);
        let data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".to_string()),
            )],
            &[(
                "blk.0.attn_q.weight",
                &[256, 1],
                GgufTensorType::Q6K,
                &block,
            )],
        );
        let tmp =
            std::env::temp_dir().join(format!("mlx-node-invalid-q6k-{}.gguf", std::process::id()));
        fs::write(&tmp, data).unwrap();
        let gguf = parse_gguf(&tmp).unwrap();
        let err = match load_gguf_tensors(&tmp, &gguf, GgufLoadOptions::default()) {
            Ok(_) => panic!("unsupported Q6_K placement unexpectedly loaded"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains("only for Gemma4 token_embd.weight"));
        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn k_quant_geometry_agrees_with_the_repacker() {
        // The tensor-type table and the repacker each carry their own copy of
        // the ggml block geometry. Cross-check them so the two cannot drift.
        for (format, ty) in [
            (KQuantFormat::Q4K, GgufTensorType::Q4K),
            (KQuantFormat::Q5K, GgufTensorType::Q5K),
            (KQuantFormat::Q6K, GgufTensorType::Q6K),
        ] {
            assert_eq!(GgufTensorType::from_u32(format.gguf_type()), Some(ty));
            assert_eq!(ty.type_size(), format.block_bytes(), "{}", ty.name());
            assert_eq!(ty.block_size(), QK_K, "{}", ty.name());
            assert!(ty.is_quantized(), "{}", ty.name());
            // K-quants carry per-16/32-value sub-scales, so they must never be
            // handed to the affine `(weight, scales, biases)` repacker.
            assert!(!ty.is_mlx_affine_quantized(), "{}", ty.name());
        }
    }

    #[test]
    fn data_size_counts_k_quant_super_blocks() {
        // 8 rows of 4096 values = 128 super-blocks. Were `block_size()` to fall
        // through to 1, `data_size()` would take the non-quantized branch and
        // return 32768 * type_size, putting every later tensor in the file at
        // the wrong offset with no error.
        for (ty, block_bytes) in [
            (GgufTensorType::Q4K, 144),
            (GgufTensorType::Q5K, 176),
            (GgufTensorType::Q6K, 210),
        ] {
            let tensor = GgufTensorInfo {
                name: "blk.0.ffn_down.weight".to_string(),
                n_dims: 2,
                dims: vec![4096, 8],
                tensor_type: ty,
                offset: 0,
            };
            assert_eq!(tensor.num_elements(), 32768);
            assert_eq!(tensor.data_size(), 128 * block_bytes, "{}", ty.name());
        }
    }

    /// Deterministic block bytes: fixed-seed LCG, no wall clock and no `rand`.
    /// Every byte pattern is a legal K-quant block, so no packer is needed here
    /// — the repacker is what these tests compare against, and it is itself
    /// gated bitwise against ggml by `tests/kquant_ggml_parity.rs`.
    fn lcg_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    fn k_quant_cases() -> [(KQuantFormat, GgufTensorType); 3] {
        [
            (KQuantFormat::Q4K, GgufTensorType::Q4K),
            (KQuantFormat::Q5K, GgufTensorType::Q5K),
            (KQuantFormat::Q6K, GgufTensorType::Q6K),
        ]
    }

    #[test]
    fn k_quant_import_emits_the_mlx_array_contract() {
        let (rows, k) = (3usize, 512usize);
        for (format, ty) in k_quant_cases() {
            let payload = lcg_bytes(u64::from(format.gguf_type()), rows * format.row_bytes(k));
            // GGUF dims are reversed, so an MLX [rows, k] tensor is stored [k, rows].
            let data = build_minimal_gguf(
                &[],
                &[(
                    "blk.0.ffn_down.weight",
                    &[k as u64, rows as u64],
                    ty,
                    &payload,
                )],
            );
            let tmp = std::env::temp_dir().join(format!(
                "mlx-node-kquant-import-{}-{}.gguf",
                ty.name(),
                std::process::id()
            ));
            fs::write(&tmp, data).unwrap();
            let gguf = parse_gguf(&tmp).unwrap();
            let weights = load_gguf_tensors(
                &tmp,
                &gguf,
                GgufLoadOptions {
                    verbose: false,
                    import_k_quants: true,
                },
            )
            .unwrap();
            fs::remove_file(&tmp).ok();

            assert_eq!(weights.len(), 3, "{}", ty.name());
            let expected = repack_kquant(format, &payload, rows, k).unwrap();

            let weight = weights.get("blk.0.ffn_down.weight").unwrap();
            assert_eq!(weight.dtype().unwrap(), DType::Uint32, "{}", ty.name());
            assert_eq!(
                weight.shape().unwrap().as_ref(),
                &[rows as i64, format.weight_cols(k) as i64],
                "{}",
                ty.name()
            );
            assert_eq!(
                weight
                    .to_uint32()
                    .unwrap()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                expected.weight,
                "{} .weight",
                ty.name()
            );

            let scales = weights.get("blk.0.ffn_down.scales").unwrap();
            assert_eq!(
                scales.shape().unwrap().as_ref(),
                &[rows as i64, format.scales_cols(k) as i64],
                "{}",
                ty.name()
            );
            match &expected.scales {
                // Q6_K sub-scales are signed. Landing them in a uint8 array
                // would read -1 back as 255 and every negative-scaled group
                // would decode with the wrong sign.
                KQuantScales::Signed(want) => {
                    assert_eq!(scales.dtype().unwrap(), DType::Int8, "{}", ty.name());
                    assert_eq!(&scales.to_int8().unwrap(), want, "{} .scales", ty.name());
                }
                KQuantScales::Unsigned(want) => {
                    assert_eq!(scales.dtype().unwrap(), DType::Uint8, "{}", ty.name());
                    assert_eq!(&scales.to_uint8().unwrap(), want, "{} .scales", ty.name());
                }
            }

            // `.biases` carries ggml's `d` (and `dmin`), which are SCALES. The
            // f16 bit patterns must survive verbatim — no float round-trip.
            let biases = weights.get("blk.0.ffn_down.biases").unwrap();
            assert_eq!(biases.dtype().unwrap(), DType::Float16, "{}", ty.name());
            assert_eq!(
                biases.shape().unwrap().as_ref(),
                &[rows as i64, format.biases_cols(k) as i64],
                "{}",
                ty.name()
            );
            assert_eq!(
                biases.to_uint16_native().unwrap(),
                expected.biases,
                "{} .biases",
                ty.name()
            );
        }
    }

    #[test]
    fn k_quant_import_is_chunk_size_invariant() {
        // A 4 MiB chunk target needs a multi-megabyte tensor to split, so drive
        // the read loop directly at chunk sizes that do split: an off-by-one in
        // the loop would otherwise only ever show up on a real model.
        let (rows, k) = (7usize, 256usize);
        for (format, _) in k_quant_cases() {
            let row_bytes = format.row_bytes(k);
            let payload = lcg_bytes(u64::from(format.gguf_type()) + 99, rows * row_bytes);
            let whole = repack_kquant(format, &payload, rows, k).unwrap();
            for chunk_target in [0, 1, row_bytes, 3 * row_bytes, usize::MAX] {
                let mut cursor = std::io::Cursor::new(&payload);
                let got = repack_rows_from_reader(
                    &mut cursor,
                    format,
                    k,
                    rows,
                    chunk_target,
                    "test tensor",
                )
                .unwrap();
                assert_eq!(
                    got,
                    whole,
                    "{} at chunk target {chunk_target}",
                    format.mlx_mode()
                );
                // Every byte of the payload was consumed, and no more.
                assert_eq!(cursor.position() as usize, payload.len());
            }
        }
    }

    #[test]
    fn k_quant_import_off_rejects_q4k_and_q5k() {
        for (format, ty) in [
            (KQuantFormat::Q4K, GgufTensorType::Q4K),
            (KQuantFormat::Q5K, GgufTensorType::Q5K),
        ] {
            let payload = vec![0u8; format.block_bytes()];
            let data =
                build_minimal_gguf(&[], &[("blk.0.ffn_down.weight", &[256, 1], ty, &payload)]);
            let tmp = std::env::temp_dir().join(format!(
                "mlx-node-kquant-off-{}-{}.gguf",
                ty.name(),
                std::process::id()
            ));
            fs::write(&tmp, data).unwrap();
            let gguf = parse_gguf(&tmp).unwrap();
            let err = match load_gguf_tensors(&tmp, &gguf, GgufLoadOptions::default()) {
                Ok(_) => panic!("{} loaded with the K-quant import off", ty.name()),
                Err(err) => err,
            };
            fs::remove_file(&tmp).ok();
            let text = format!("{err}");
            assert!(text.contains(ty.name()), "{text}");
            assert!(text.contains("K-quant import"), "{text}");
        }
    }

    #[test]
    fn q6k_import_repacks_the_gemma4_embedding_instead_of_dequantizing_it() {
        // Same fixture as `gemma4_q6k_token_embedding_dequantizes_to_bf16`,
        // which pins the flag-off BF16 fallback. With the import on the tensor
        // stays quantized and gains its `.scales` / `.biases`.
        let block = pack_q6k_block(&[0; 256], &[1; 16], 1.0);
        let data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".to_string()),
            )],
            &[("token_embd.weight", &[256, 1], GgufTensorType::Q6K, &block)],
        );
        let tmp =
            std::env::temp_dir().join(format!("mlx-node-q6k-import-{}.gguf", std::process::id()));
        fs::write(&tmp, data).unwrap();
        let gguf = parse_gguf(&tmp).unwrap();
        let weights = load_gguf_tensors(
            &tmp,
            &gguf,
            GgufLoadOptions {
                verbose: false,
                import_k_quants: true,
            },
        )
        .unwrap();
        fs::remove_file(&tmp).ok();

        let weight = weights.get("token_embd.weight").unwrap();
        assert_eq!(weight.dtype().unwrap(), DType::Uint32);
        assert_eq!(weight.shape().unwrap().as_ref(), &[1, 48]);
        assert_eq!(
            weights.get("token_embd.scales").unwrap().dtype().unwrap(),
            DType::Int8
        );
        assert_eq!(
            weights.get("token_embd.biases").unwrap().dtype().unwrap(),
            DType::Float16
        );
    }

    /// The user-visible bug: a K-quant `token_embd` must survive the key remap as
    /// a complete group so the loader's `embed_tokens.scales` probe fires and the
    /// embedding takes the PACKED quantized branch. Previously only `.weight` was
    /// renamed, so the probe missed, the dense fallback received a uint32
    /// `embed_tokens.weight`, and the load died in `ensure_dense_weight_floating`.
    ///
    /// Asserts at the remap + probe boundary (a full model load needs a real
    /// checkpoint): post-remap the trio is `model.embed_tokens.*`, and after the
    /// `model.` strip the loader applies (`gemma4/persistence.rs`) the probe key
    /// `embed_tokens.scales` is present with its K-quant dtypes intact.
    #[test]
    fn kquant_token_embd_survives_the_remap_as_a_packed_group() {
        let block = pack_q6k_block(&[0; 256], &[1; 16], 1.0);
        let data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".to_string()),
            )],
            &[("token_embd.weight", &[256, 1], GgufTensorType::Q6K, &block)],
        );
        let tmp =
            std::env::temp_dir().join(format!("mlx-node-q6k-remap-{}.gguf", std::process::id()));
        fs::write(&tmp, data).unwrap();
        let gguf = parse_gguf(&tmp).unwrap();
        let weights = load_gguf_tensors(
            &tmp,
            &gguf,
            GgufLoadOptions {
                verbose: false,
                import_k_quants: true,
            },
        )
        .unwrap();
        fs::remove_file(&tmp).ok();

        let remapped = remap_keys(weights, &gguf.metadata).expect("remap must not collide");

        // The whole group lands under the HF embedding name.
        assert_eq!(
            remapped
                .get("model.embed_tokens.weight")
                .expect("packed weight must remap")
                .dtype()
                .unwrap(),
            DType::Uint32
        );
        assert_eq!(
            remapped
                .get("model.embed_tokens.scales")
                .expect("`.scales` sidecar must remap alongside `.weight`")
                .dtype()
                .unwrap(),
            DType::Int8
        );
        assert_eq!(
            remapped
                .get("model.embed_tokens.biases")
                .expect("`.biases` sidecar must remap alongside `.weight`")
                .dtype()
                .unwrap(),
            DType::Float16
        );
        // No sidecar was left stranded under its GGUF name.
        assert!(
            !remapped.contains_key("token_embd.scales")
                && !remapped.contains_key("token_embd.biases"),
            "sidecars must not survive under their GGUF names"
        );

        // The loader strips `model.` and probes `{base}.scales` to choose the
        // packed quantized branch over the dense fallback.
        let probed: HashMap<String, ()> = remapped
            .keys()
            .map(|k| (k.strip_prefix("model.").unwrap_or(k).to_string(), ()))
            .collect();
        assert!(
            probed.contains_key("embed_tokens.scales"),
            "the packed-embedding probe key must exist after the `model.` strip"
        );
    }

    #[test]
    fn k_quant_import_rejects_a_last_dim_that_is_not_a_super_block_multiple() {
        // A K-quant row is a whole number of 256-value super-blocks. Reject a
        // tensor that is not, rather than repacking a truncated final block.
        let tensor = GgufTensorInfo {
            name: "blk.0.ffn_down.weight".to_string(),
            n_dims: 2,
            dims: vec![128, 4],
            tensor_type: GgufTensorType::Q4K,
            offset: 0,
        };
        let gguf = GgufFile {
            version: GGUF_VERSION_3,
            tensor_count: 1,
            metadata: HashMap::new(),
            tensors: vec![tensor.clone()],
            alignment: GGUF_DEFAULT_ALIGNMENT,
            data_offset: 0,
        };
        let mut cursor = std::io::Cursor::new(vec![0u8; 4 * 144]);
        let err = match load_kquant_repack(&mut cursor, &gguf, &tensor, KQuantFormat::Q4K) {
            Ok(_) => panic!("a 128-wide Q4_K row unexpectedly imported"),
            Err(err) => err,
        };
        let text = format!("{err}");
        assert!(text.contains("last dimension 128"), "{text}");
        assert!(text.contains("multiple of 256"), "{text}");
    }

    #[test]
    fn k_quant_import_names_its_arrays_through_the_shared_helper() {
        // The converter's do-not-dtype-cast set spells out the same three keys
        // via `k_quant_output_keys`. If the importer ever names its arrays by
        // hand again the two drift apart and a `--dtype` request starts casting
        // packed weights; this fails first.
        let (rows, k) = (2usize, 256usize);
        for (format, ty) in k_quant_cases() {
            let payload = lcg_bytes(
                u64::from(format.gguf_type()) + 7,
                rows * format.row_bytes(k),
            );
            let data = build_minimal_gguf(
                &[],
                &[(
                    "blk.0.ffn_down.weight",
                    &[k as u64, rows as u64],
                    ty,
                    &payload,
                )],
            );
            let tmp = std::env::temp_dir().join(format!(
                "mlx-node-kquant-keys-{}-{}.gguf",
                ty.name(),
                std::process::id()
            ));
            fs::write(&tmp, data).unwrap();
            let gguf = parse_gguf(&tmp).unwrap();
            let weights = load_gguf_tensors(
                &tmp,
                &gguf,
                GgufLoadOptions {
                    verbose: false,
                    import_k_quants: true,
                },
            )
            .unwrap();
            fs::remove_file(&tmp).ok();

            let mut got: Vec<String> = weights.into_keys().collect();
            got.sort();
            let mut want = k_quant_output_keys("blk.0.ffn_down.weight").to_vec();
            want.sort();
            assert_eq!(got, want, "{}", ty.name());
        }
    }

    #[tokio::test]
    async fn k_quant_import_survives_a_global_dtype_request() {
        // `--dtype` must not touch an imported K-quant triplet: the packed
        // uint32 codes and the int8 / f16 bit patterns beside them are the
        // weights. A float cast would silently rewrite all three.
        let (rows, k) = (2usize, 256usize);
        let format = KQuantFormat::Q6K;
        let payload = lcg_bytes(0xC0FFEE, rows * format.row_bytes(k));
        let norm = 3.25f32.to_le_bytes();
        let data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("llama".to_string()),
            )],
            &[
                (
                    "blk.0.ffn_down.weight",
                    &[k as u64, rows as u64],
                    GgufTensorType::Q6K,
                    &payload,
                ),
                ("output_norm.weight", &[1], GgufTensorType::F32, &norm),
            ],
        );
        let root = std::env::temp_dir().join(format!(
            "mlx-node-kquant-dtype-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("model.gguf");
        fs::write(&input, data).unwrap();
        let output = root.join("output");

        convert_gguf_to_safetensors(GgufConversionOptions {
            input_path: input.to_string_lossy().into_owned(),
            output_dir: output.to_string_lossy().into_owned(),
            config_source_dir: None,
            dtype: Some("float32".to_string()),
            verbose: Some(false),
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: None,
            vlm_key_prefix: Some(false),
            quant_mxfp: Some(false),
            import_k_quants: Some(true),
        })
        .await
        .unwrap();

        let safetensors =
            crate::utils::safetensors::SafeTensorsFile::load(output.join("model.safetensors"))
                .unwrap();
        use crate::utils::safetensors::SafeTensorDType;
        let dtype_of = |key: &str| safetensors.tensors[key].dtype.clone();
        assert!(matches!(
            dtype_of("model.layers.0.mlp.down_proj.weight"),
            SafeTensorDType::U32
        ));
        assert!(matches!(
            dtype_of("model.layers.0.mlp.down_proj.scales"),
            SafeTensorDType::I8
        ));
        assert!(matches!(
            dtype_of("model.layers.0.mlp.down_proj.biases"),
            SafeTensorDType::F16
        ));
        // The dense tensor did take the requested cast, so the skip above is
        // targeted rather than the dtype pass having been a no-op.
        assert!(matches!(
            dtype_of("model.norm.weight"),
            SafeTensorDType::F32
        ));

        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(
            config["quantization"]["language_model.model.layers.0.mlp.down_proj"],
            serde_json::json!({ "bits": 6, "group_size": 16, "mode": "q6k" })
        );

        fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn qwen35_kquant_linear_attention_rejects_from_header() {
        // Qwen3.5's `attn_qkv` maps to `linear_attn.in_proj_qkv`, one of the
        // projections whose value-head rows must be reinterleaved. A packed
        // Q5_K array cannot go through the dense-only fixup independently of
        // its scales/biases. Pin the public conversion boundary so this is a
        // recoverable, precise error instead of an MLX foreign exception.
        let format = KQuantFormat::Q5K;
        let (rows, k) = (2usize, QK_K);
        let payload = lcg_bytes(0x3515, rows * format.row_bytes(k));
        let data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("qwen35".to_string()),
            )],
            &[(
                "blk.0.attn_qkv.weight",
                &[k as u64, rows as u64],
                GgufTensorType::Q5K,
                &payload,
            )],
        );
        let root = std::env::temp_dir().join(format!(
            "mlx-node-qwen35-kquant-linear-attn-reject-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("model.gguf");
        fs::write(&input, data).unwrap();
        let output = root.join("output");

        let result = convert_gguf_to_safetensors(GgufConversionOptions {
            input_path: input.to_string_lossy().into_owned(),
            output_dir: output.to_string_lossy().into_owned(),
            config_source_dir: None,
            dtype: None,
            verbose: Some(false),
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: None,
            vlm_key_prefix: Some(false),
            quant_mxfp: Some(false),
            import_k_quants: Some(true),
        })
        .await;
        let err = match result {
            Err(err) => err,
            Ok(_) => panic!("packed Qwen3.5 linear attention must fail before dense fixup"),
        };

        let text = err.to_string();
        assert!(text.contains("blk.0.attn_qkv.weight"), "{text}");
        assert!(text.contains("Q5_K"), "{text}");
        assert!(text.contains("linear_attn.in_proj_qkv.weight"), "{text}");
        assert!(text.contains(".scales/.biases"), "{text}");
        assert!(text.contains("before tensor data was loaded"), "{text}");
        assert!(
            !output.exists(),
            "header-only rejection must not create or truncate output"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn plain_qwen3_quantized_source_has_no_runtime_but_hybrid_and_float_remain_allowed() {
        let mut plain = source_quant_fixture(&[("blk.0.attn_q.weight", GgufTensorType::Q4_0)]);
        plain.metadata.insert(
            "general.architecture".to_string(),
            GgufMetaValue::String("qwen3".to_string()),
        );
        let rejected = plain_qwen3_quantized_source(&plain)
            .expect("source-quantized plain Qwen3 must fail closed");
        assert_eq!(rejected.name, "blk.0.attn_q.weight");
        assert_eq!(rejected.tensor_type, GgufTensorType::Q4_0);

        let mut floating = source_quant_fixture(&[("blk.0.attn_q.weight", GgufTensorType::BF16)]);
        floating.metadata = plain.metadata.clone();
        assert!(
            plain_qwen3_quantized_source(&floating).is_none(),
            "floating Qwen3 conversion must remain available"
        );

        // Old writers can label Qwen3.5 as `qwen3`. Its characteristic full
        // attention + GDN shape must win over the raw architecture string.
        let hybrid_keys = qwen35_hybrid_test_keys();
        let hybrid_tensors = hybrid_keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                (
                    key.as_str(),
                    if index == 0 {
                        GgufTensorType::Q4_0
                    } else {
                        GgufTensorType::BF16
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut hybrid = source_quant_fixture(&hybrid_tensors);
        hybrid.metadata = plain.metadata;
        assert!(
            plain_qwen3_quantized_source(&hybrid).is_none(),
            "Qwen3.5 hybrid checkpoints must remain on their quantized runtime"
        );
    }

    #[test]
    fn q4_0_repack_keeps_payload_and_scale_exact() {
        let scale_bits = half::f16::from_f32(0.25).to_bits();
        let mut block = [0u8; 18];
        block[..2].copy_from_slice(&scale_bits.to_le_bytes());
        for (i, byte) in block[2..].iter_mut().enumerate() {
            *byte = ((15 - i) as u8) << 4 | i as u8;
        }
        let data = build_minimal_gguf(
            &[],
            &[("test.weight", &[32, 1], GgufTensorType::Q4_0, &block)],
        );
        let tmp =
            std::env::temp_dir().join(format!("mlx-node-q4-repack-{}.gguf", std::process::id()));
        fs::write(&tmp, data).unwrap();
        let gguf = parse_gguf(&tmp).unwrap();
        let weights = load_gguf_tensors(&tmp, &gguf, GgufLoadOptions::default()).unwrap();

        let packed = weights.get("test.weight").unwrap().to_uint32().unwrap();
        let mut unpacked = [0u8; 32];
        for i in 0..16 {
            unpacked[i] = block[2 + i] & 0x0f;
            unpacked[16 + i] = block[2 + i] >> 4;
        }
        let mut expected = [0u32; 4];
        for (word, out) in expected.iter_mut().enumerate() {
            for lane in 0..8 {
                *out |= u32::from(unpacked[word * 8 + lane]) << (lane * 4);
            }
        }
        assert_eq!(packed.iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(
            weights
                .get("test.scales")
                .unwrap()
                .to_uint16_native()
                .unwrap(),
            vec![scale_bits]
        );
        // Q4_0 is symmetric: its offset is the constant -8 * scale, so the
        // import writes no `.biases` companion and the loader rebuilds it.
        assert!(
            !weights.contains_key("test.biases"),
            "Q4_0 must not write a derived .biases companion"
        );
        assert_eq!(
            derived_symmetric_bias_bits(scale_bits, 8),
            half::f16::from_f32(-2.0).to_bits()
        );
        fs::remove_file(&tmp).ok();
    }

    /// Load one 32-weight block of `ty` through the production import path.
    fn load_single_affine_block(
        ty: GgufTensorType,
        block: &[u8],
        label: &str,
    ) -> HashMap<String, MxArray> {
        let data = build_minimal_gguf(&[], &[("test.weight", &[32, 1], ty, block)]);
        let tmp = std::env::temp_dir().join(format!(
            "mlx-node-affine-{label}-{}.gguf",
            std::process::id()
        ));
        fs::write(&tmp, data).unwrap();
        let gguf = parse_gguf(&tmp).unwrap();
        let weights = load_gguf_tensors(&tmp, &gguf, GgufLoadOptions::default()).unwrap();
        fs::remove_file(&tmp).ok();
        weights
    }

    #[test]
    fn q8_0_import_omits_the_derived_bias() {
        let scale_bits = half::f16::from_f32(0.5).to_bits();
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&scale_bits.to_le_bytes());
        for (i, byte) in block[2..].iter_mut().enumerate() {
            *byte = (i as i8).wrapping_sub(16) as u8;
        }
        let weights = load_single_affine_block(GgufTensorType::Q8_0, &block, "q8");

        assert_eq!(
            weights
                .get("test.scales")
                .unwrap()
                .to_uint16_native()
                .unwrap(),
            vec![scale_bits]
        );
        assert!(
            !weights.contains_key("test.biases"),
            "Q8_0 is `d * (q - 128)` — its offset is derived, not stored"
        );
        assert_eq!(
            derived_symmetric_bias_bits(scale_bits, 128),
            half::f16::from_f32(-64.0).to_bits()
        );
    }

    #[test]
    fn q4_1_import_keeps_its_own_stored_minimum() {
        // Q4_1 is the asymmetric sibling: its second f16 field is a minimum
        // chosen per block, unrelated to the scale. Deriving it would destroy
        // every weight in the tensor, so the import must keep storing it.
        let scale_bits = half::f16::from_f32(0.25).to_bits();
        let min_bits = half::f16::from_f32(-1.75).to_bits();
        let mut block = [0u8; 20];
        block[..2].copy_from_slice(&scale_bits.to_le_bytes());
        block[2..4].copy_from_slice(&min_bits.to_le_bytes());
        for (i, byte) in block[4..].iter_mut().enumerate() {
            *byte = ((15 - i) as u8) << 4 | i as u8;
        }
        let weights = load_single_affine_block(GgufTensorType::Q4_1, &block, "q41");

        let biases = weights
            .get("test.biases")
            .expect("Q4_1 stores a real minimum and must keep its .biases")
            .to_uint16_native()
            .unwrap();
        assert_eq!(biases, vec![min_bits]);
        assert_ne!(
            biases[0],
            derived_symmetric_bias_bits(scale_bits, 8),
            "the stored minimum must not coincide with the symmetric derivation, \
             or this fixture cannot tell the two apart"
        );
    }

    #[test]
    fn symmetric_zero_point_is_pinned_per_gguf_type() {
        assert_eq!(symmetric_zero_point(GgufTensorType::Q4_0), Some(8));
        assert_eq!(symmetric_zero_point(GgufTensorType::Q8_0), Some(128));
        assert_eq!(symmetric_zero_point(GgufTensorType::Q4_1), None);
        for ty in [
            GgufTensorType::Q4K,
            GgufTensorType::Q5K,
            GgufTensorType::Q6K,
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::BF16,
        ] {
            assert_eq!(symmetric_zero_point(ty), None, "{}", ty.name());
        }

        let profile = |ty| SourceQuantProfile::for_gguf_type(ty).unwrap();
        assert_eq!(profile(GgufTensorType::Q4_0).symmetric_zero_point, Some(8));
        assert_eq!(
            profile(GgufTensorType::Q8_0).symmetric_zero_point,
            Some(128)
        );
        assert_eq!(profile(GgufTensorType::Q4_1).symmetric_zero_point, None);
        // The split is what lets a file holding both name them separately.
        assert_ne!(profile(GgufTensorType::Q4_0), profile(GgufTensorType::Q4_1));
    }

    #[test]
    fn mixed_q4_0_and_q4_1_source_metadata_separates_them() {
        let gguf = source_quant_fixture(&[
            ("blk.0.attn_q.weight", GgufTensorType::Q4_0),
            ("blk.0.attn_k.weight", GgufTensorType::Q4_0),
            ("blk.0.ffn_down.weight", GgufTensorType::Q4_1),
        ]);
        let quant = preserved_source_quantization(&gguf, false)
            .unwrap()
            .unwrap();

        assert_eq!(quant[SYMMETRIC_ZERO_POINT_KEY], 8);
        assert_eq!(
            quant["language_model.model.layers.0.self_attn.q_proj"],
            serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "affine",
                SYMMETRIC_ZERO_POINT_KEY: 8,
            })
        );
        assert_eq!(
            quant["language_model.model.layers.0.mlp.down_proj"],
            serde_json::json!({ "bits": 4, "group_size": 32, "mode": "affine" }),
            "the Q4_1 tensor must be named and must carry no zero point"
        );
    }

    #[test]
    fn gguf_awq_rejects_repacked_body_before_mutating_weight_or_norm() {
        let weight_key = "model.layers.0.mlp.gate_proj.weight";
        let scale_key = "model.layers.0.mlp.gate_proj.scales";
        let norm_key = "model.layers.0.post_attention_layernorm.weight";
        let mut weights = HashMap::from([
            (
                weight_key.to_string(),
                MxArray::from_uint32(&[0x1234_5678, 0x9abc_def0], &[1, 2]).unwrap(),
            ),
            (
                scale_key.to_string(),
                MxArray::ones(&[1, 1], Some(DType::Float16)).unwrap(),
            ),
            (
                norm_key.to_string(),
                MxArray::from_float32(&[1.25, 0.75], &[2])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap(),
            ),
        ]);
        let packed_before = weights[weight_key].to_uint32().unwrap().to_vec();
        let norm_before = weights[norm_key].to_uint16_native().unwrap();
        let imatrix = crate::utils::imatrix::ImatrixData {
            importance: HashMap::from([(weight_key.to_string(), vec![1.0, 4.0])]),
            chunk_count: 1,
            chunk_size: 2,
        };

        let err = apply_gguf_awq_prescaling(&mut weights, &imatrix)
            .expect_err("GGUF AWQ must reject repacked Q4/Q8 storage before mutation");
        let message = err.reason.to_string();
        assert!(message.contains("--imatrix-path"), "{message}");
        assert!(
            message.contains("already-quantized model body"),
            "{message}"
        );
        assert!(message.contains("before mutating tensors"), "{message}");
        assert!(message.contains(scale_key), "{message}");
        assert_eq!(
            weights[weight_key].to_uint32().unwrap().to_vec(),
            packed_before,
            "packed GGUF payload must remain bit-exact"
        );
        assert_eq!(
            weights[norm_key].to_uint16_native().unwrap(),
            norm_before,
            "the norm must not receive a partial AWQ fold"
        );
    }

    fn gemma4_metadata() -> HashMap<String, GgufMetaValue> {
        HashMap::from([(
            "general.architecture".to_string(),
            GgufMetaValue::String("gemma4".to_string()),
        )])
    }

    fn gemma4_mmproj_metadata() -> HashMap<String, GgufMetaValue> {
        HashMap::from([
            (
                "general.architecture".to_string(),
                GgufMetaValue::String("clip".to_string()),
            ),
            (
                "clip.vision.projector_type".to_string(),
                GgufMetaValue::String("gemma4uv".to_string()),
            ),
        ])
    }

    #[test]
    fn gemma4_mapping_keeps_four_norms_distinct_and_drops_rope_cache() {
        let metadata = gemma4_metadata();
        let cases = [
            (
                "blk.7.attn_norm.weight",
                Some("model.layers.7.input_layernorm.weight"),
            ),
            (
                "blk.7.post_attention_norm.weight",
                Some("model.layers.7.post_attention_layernorm.weight"),
            ),
            (
                "blk.7.ffn_norm.weight",
                Some("model.layers.7.pre_feedforward_layernorm.weight"),
            ),
            (
                "blk.7.post_ffw_norm.weight",
                Some("model.layers.7.post_feedforward_layernorm.weight"),
            ),
            (
                "blk.7.layer_output_scale.weight",
                Some("model.layers.7.layer_scalar"),
            ),
            ("rope_freqs.weight", None),
        ];
        for (source, expected) in cases {
            assert_eq!(
                gguf_name_to_hf_for_metadata(source, &metadata).as_deref(),
                expected,
                "mapping mismatch for {source}"
            );
        }
    }

    #[test]
    fn gemma4_mapping_rejects_destination_collisions() {
        let weights = HashMap::from([
            (
                "blk.0.ffn_norm.weight".to_string(),
                MxArray::from_float32(&[1.0], &[1]).unwrap(),
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight".to_string(),
                MxArray::from_float32(&[2.0], &[1]).unwrap(),
            ),
        ]);
        let err = match remap_keys(weights, &gemma4_metadata()) {
            Ok(_) => panic!("colliding Gemma4 keys unexpectedly remapped"),
            Err(err) => err,
        };
        let message = format!("{err}");
        assert!(message.contains("GGUF key remap collision"));
        assert!(message.contains("blk.0.ffn_norm.weight"));
        assert!(message.contains("model.layers.0.pre_feedforward_layernorm.weight"));
        assert!(message.contains("model.layers.0.pre_feedforward_layernorm.weight'"));
    }

    #[test]
    fn gemma4_mmproj_mapping_and_layout_match_unified_checkpoint() {
        let metadata = gemma4_mmproj_metadata();
        let raw = HashMap::from([
            (
                "v.patch_embd.weight".to_string(),
                MxArray::from_float32(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[1, 12])
                    .unwrap(),
            ),
            (
                "v.patch_norm.1.weight".to_string(),
                MxArray::from_float32(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[12])
                    .unwrap(),
            ),
            (
                "v.position_embd.weight".to_string(),
                MxArray::from_float32(&[0.0, 1.0, 2.0, 3.0], &[2, 2, 1]).unwrap(),
            ),
            (
                "mm.input_projection.weight".to_string(),
                MxArray::from_float32(&[5.0], &[1, 1]).unwrap(),
            ),
            (
                "mm.a.input_projection.weight".to_string(),
                MxArray::from_float32(&[6.0], &[1, 1]).unwrap(),
            ),
        ]);

        let mut weights = remap_keys(raw, &metadata).unwrap();
        fixup_gemma4_mmproj_layout(&mut weights, &metadata).unwrap();

        let patch = weights
            .get("model.vision_embedder.patch_dense.weight")
            .unwrap();
        assert_eq!(patch.shape().unwrap().as_ref(), &[1, 12]);
        assert_eq!(
            patch
                .to_float32()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
        );

        assert_eq!(
            weights
                .get("model.vision_embedder.patch_ln1.weight")
                .unwrap()
                .to_float32()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
        );

        let position = weights.get("model.vision_embedder.pos_embedding").unwrap();
        assert_eq!(position.shape().unwrap().as_ref(), &[2, 2, 1]);
        assert_eq!(
            position
                .to_float32()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![0.0, 2.0, 1.0, 3.0]
        );
        assert!(weights.contains_key("model.embed_vision.embedding_projection.weight"));
        assert!(weights.contains_key("model.embed_audio.embedding_projection.weight"));
    }

    #[test]
    fn q4_source_metadata_is_affine_four_bit_group_32_without_requantizing() {
        let gguf = GgufFile {
            version: GGUF_VERSION_3,
            tensor_count: 2,
            metadata: gemma4_metadata(),
            tensors: vec![
                GgufTensorInfo {
                    name: "blk.0.attn_q.weight".to_string(),
                    n_dims: 2,
                    dims: vec![32, 1],
                    tensor_type: GgufTensorType::Q4_0,
                    offset: 0,
                },
                GgufTensorInfo {
                    name: "output_norm.weight".to_string(),
                    n_dims: 1,
                    dims: vec![1],
                    tensor_type: GgufTensorType::F32,
                    offset: 32,
                },
            ],
            alignment: GGUF_DEFAULT_ALIGNMENT,
            data_offset: 0,
        };
        let quant = preserved_source_quantization(&gguf, false)
            .unwrap()
            .unwrap();
        assert_eq!(quant["bits"], 4);
        assert_eq!(quant["group_size"], 32);
        assert_eq!(quant["mode"], "affine");
        // Q4_0 blocks are `d * (q - 8)`, so the config carries the zero point
        // the loader needs to rebuild the omitted `.biases`.
        assert_eq!(quant[SYMMETRIC_ZERO_POINT_KEY], 8);
        // A file of one profile still needs no per-tensor entries.
        assert_eq!(quant.as_object().unwrap().len(), 4);
    }

    /// Tensor descriptors only — `preserved_source_quantization` reads names
    /// and types, never payload bytes.
    fn source_quant_fixture(tensors: &[(&str, GgufTensorType)]) -> GgufFile {
        GgufFile {
            version: GGUF_VERSION_3,
            tensor_count: tensors.len() as u64,
            metadata: HashMap::new(),
            tensors: tensors
                .iter()
                .enumerate()
                .map(|(i, (name, tensor_type))| GgufTensorInfo {
                    name: (*name).to_string(),
                    n_dims: 2,
                    dims: vec![256, 1],
                    tensor_type: *tensor_type,
                    offset: i as u64 * 256,
                })
                .collect(),
            alignment: GGUF_DEFAULT_ALIGNMENT,
            data_offset: 0,
        }
    }

    /// The layers a healthy Gemma4 checkpoint of `n` blocks runs as global: KV
    /// heads drop to 1 on every sixth layer, matching
    /// `gemma-4-12b-it-qat-q4_0.gguf`, whose `head_count_kv` is
    /// `[8,8,8,8,8,1, 8,8,8,8,8,1, …]`.
    fn healthy_global_layers(n: u32) -> Vec<u32> {
        (0..n).filter(|layer| (layer + 1) % 6 == 0).collect()
    }

    /// Layer names for `n` decoder layers, where every layer listed in
    /// `global` is written without an `attn_v` tensor, paired with the header a
    /// healthy `n`-block Gemma4 checkpoint carries.
    ///
    /// The tensor list and the header are supplied independently on purpose:
    /// truncation and corruption remove tensors, they do not rewrite the
    /// header, so a damaged checkpoint is exactly a `global` that disagrees
    /// with `gemma4.attention.head_count_kv`. Tests that need a damaged header
    /// instead mutate `metadata` on the returned file.
    fn kv_layout_fixture(n: u32, global: &[u32]) -> GgufFile {
        let mut names = Vec::new();
        for layer in 0..n {
            names.push(format!("blk.{layer}.attn_q.weight"));
            names.push(format!("blk.{layer}.attn_k.weight"));
            if !global.contains(&layer) {
                names.push(format!("blk.{layer}.attn_v.weight"));
            }
            names.push(format!("blk.{layer}.attn_output.weight"));
        }
        let refs: Vec<(&str, GgufTensorType)> = names
            .iter()
            .map(|n| (n.as_str(), GgufTensorType::Q4_0))
            .collect();
        let mut gguf = source_quant_fixture(&refs);
        let healthy = healthy_global_layers(n);
        let head_count_kv: Vec<i32> = (0..n)
            .map(|layer| if healthy.contains(&layer) { 1 } else { 8 })
            .collect();
        gguf.metadata = HashMap::from([
            (
                "general.architecture".to_string(),
                GgufMetaValue::String("gemma4".to_string()),
            ),
            ("gemma4.block_count".to_string(), GgufMetaValue::Uint32(n)),
            (
                "gemma4.attention.head_count_kv".to_string(),
                GgufMetaValue::ArrayI32(head_count_kv),
            ),
        ]);
        gguf
    }

    /// The shape of Google's `gemma-4-12b-it-qat-q4_0-gguf`: 48 layers, every
    /// sixth one written without `attn_v` because it takes V from the key
    /// projection.
    ///
    /// The positions are the point, not just the count. `attention_k_eq_v` only
    /// applies to layers the config calls `full_attention`, and the loader's
    /// fallback period of 5 puts those at 4, 9, 14 … — so with the flag set but
    /// the array missing, layer 5 reads as sliding, V is demanded anyway, and
    /// the load dies on `layers.5.self_attn.v_proj`. That is the exact failure
    /// this fixture reproduces.
    #[test]
    fn gemma4_missing_attn_v_yields_layer_types_at_the_real_period() {
        let global = healthy_global_layers(48);
        assert_eq!(global.len(), 8, "gemma4 publishes 8 global layers of 48");

        let types = gemma4_layer_types_from_missing_v(&kv_layout_fixture(48, &global))
            .expect("a corroborated checkpoint is not an error")
            .expect("a mixed checkpoint yields layer_types");
        assert_eq!(types.len(), 48);

        let full: Vec<usize> = types
            .iter()
            .enumerate()
            .filter(|(_, t)| *t == "full_attention")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            full,
            vec![5, 11, 17, 23, 29, 35, 41, 47],
            "period 6, not the loader's default of 5"
        );
    }

    /// The mutation that matters in the other direction: a detector that always
    /// fires would drop `v_proj` on an ordinary checkpoint and feed attention
    /// the keys as values — silent corruption rather than a load failure. A
    /// checkpoint whose layers all carry `attn_v` must yield nothing, and so
    /// must the degenerate all-missing case, which is not the interleave this
    /// describes. Neither is an error: they are simply not this layout.
    #[test]
    fn gemma4_kv_detection_needs_a_genuine_mix() {
        assert!(
            gemma4_layer_types_from_missing_v(&kv_layout_fixture(48, &[]))
                .unwrap()
                .is_none(),
            "every layer has attn_v: this is an ordinary checkpoint"
        );
        let all: Vec<u32> = (0..48).collect();
        assert!(
            gemma4_layer_types_from_missing_v(&kv_layout_fixture(48, &all))
                .unwrap()
                .is_none(),
            "no layer has attn_v: not the sliding/global interleave"
        );
    }

    /// A partial-V tensor list the header does not corroborate is rejected by
    /// name instead of being read as the key-equals-value layout.
    ///
    /// Both cases are damage, not a layout: one `attn_v` lost to a truncated
    /// download, and a contiguous run lost to a corrupt one. The old acceptance
    /// test was `with_v && without_v`, which is true for every mixture, so both
    /// were reclassified as K=V — and that flag suppresses the loader's
    /// `Missing required weight: layers.N.self_attn.v_proj` check on exactly the
    /// layers whose weights went missing.
    ///
    /// Mutation this catches: drop the `head_count_kv` cross-check (or weaken it
    /// to a count/cadence comparison) and both calls return `Ok(Some(..))` at
    /// the damaged period, failing the `unwrap_err`.
    #[test]
    fn gemma4_kv_detection_rejects_a_tensor_list_the_header_contradicts() {
        let stray = gemma4_layer_types_from_missing_v(&kv_layout_fixture(48, &[37]))
            .expect_err("a single stray missing attn_v is damage, not a layout");
        let stray = format!("{stray}");
        assert!(stray.contains("37"), "{stray}");
        assert!(stray.contains("head_count_kv"), "{stray}");

        let block: Vec<u32> = (15..35).collect();
        let contiguous = gemma4_layer_types_from_missing_v(&kv_layout_fixture(48, &block))
            .expect_err("a contiguous V-less run is not the sliding/global cadence");
        let contiguous = format!("{contiguous}");
        assert!(contiguous.contains("head_count_kv"), "{contiguous}");
    }

    /// The two structural preconditions the set comparison rests on. Without
    /// them the comparison is between a set of layer indices and a set of
    /// positions in an array that need not describe the same layers.
    ///
    /// Mutation this catches: drop either length/range check and the calls
    /// return `Ok(..)` instead of erroring — a 47-entry header would silently
    /// corroborate a 48-block file, and a renumbered tensor list would index the
    /// array at the wrong layers.
    #[test]
    fn gemma4_kv_detection_requires_a_header_that_covers_every_block() {
        let global = healthy_global_layers(48);

        let mut short_header = kv_layout_fixture(48, &global);
        short_header.metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufMetaValue::ArrayI32(vec![8; 47]),
        );
        let err = format!(
            "{}",
            gemma4_layer_types_from_missing_v(&short_header)
                .expect_err("a header with fewer entries than layers corroborates nothing")
        );
        assert!(err.contains("47 entries"), "{err}");

        let mut wrong_block_count = kv_layout_fixture(48, &global);
        wrong_block_count
            .metadata
            .insert("gemma4.block_count".to_string(), GgufMetaValue::Uint32(47));
        let err = format!(
            "{}",
            gemma4_layer_types_from_missing_v(&wrong_block_count)
                .expect_err("a block count that disagrees with the tensor list is damage")
        );
        assert!(err.contains("block_count"), "{err}");
    }

    /// A header that says nothing is silence, not a contradiction: the measured
    /// layout stands.
    ///
    /// `head_count_kv` is optional and may be written as a scalar, and
    /// `block_count` may be missing outright. Erroring on any of those rejects a
    /// partial-V checkpoint whose own tensor list is self-consistent — and
    /// `apply_gemma4_attention_geometry`, which reads the very same array,
    /// degrades gracefully on exactly this input, so refusing here also made the
    /// two disagree about how much header they demand. The mistyping this used
    /// to guard against still fails loudly downstream, on the 256-vs-512
    /// `head_dim` split every published Gemma-4 carries.
    ///
    /// Mutation this catches: restore `kv_per_layer.len() != layers.len()` as an
    /// unconditional reject (or require `block_count` to be present) and each of
    /// these three calls returns `Err`, failing the `expect`.
    #[test]
    fn gemma4_kv_detection_accepts_a_header_that_corroborates_nothing() {
        let global = healthy_global_layers(48);
        let expected: Vec<usize> = global.iter().map(|&layer| layer as usize).collect();

        let full_attention = |file: &GgufFile, why: &str| -> Vec<usize> {
            let types = gemma4_layer_types_from_missing_v(file)
                .unwrap_or_else(|e| panic!("{why}: {e}"))
                .unwrap_or_else(|| panic!("{why}: a mixed checkpoint yields layer_types"));
            types
                .iter()
                .enumerate()
                .filter(|(_, t)| *t == "full_attention")
                .map(|(i, _)| i)
                .collect()
        };

        let mut absent = kv_layout_fixture(48, &global);
        absent.metadata.remove("gemma4.attention.head_count_kv");
        assert_eq!(
            full_attention(&absent, "an absent head_count_kv corroborates nothing"),
            expected
        );

        let mut scalar = kv_layout_fixture(48, &global);
        scalar.metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufMetaValue::Uint32(8),
        );
        assert_eq!(
            full_attention(&scalar, "a scalar head_count_kv says nothing per layer"),
            expected
        );

        let mut no_block_count = kv_layout_fixture(48, &global);
        no_block_count.metadata.remove("gemma4.block_count");
        assert_eq!(
            full_attention(
                &no_block_count,
                "an absent block_count corroborates nothing"
            ),
            expected
        );
    }

    /// The positional premise the whole comparison rests on: entry `i` of
    /// `head_count_kv` describes block `i`, and `layer_types[i]` is written for
    /// block `i`. A tensor list that does not run `0..N` breaks both, and
    /// tolerating an absent `block_count` must not tolerate that.
    ///
    /// Mutation this catches: replace the contiguity test with a bare
    /// `block_count.is_some_and(...)` check and the renumbered list is accepted,
    /// mislabelling every layer by the offset.
    #[test]
    fn gemma4_kv_detection_rejects_blocks_that_do_not_start_at_zero() {
        let global = healthy_global_layers(48);
        let mut renumbered = kv_layout_fixture(48, &global);
        for tensor in &mut renumbered.tensors {
            if let Some(rest) = tensor.name.strip_prefix("blk.")
                && let Some((idx, tail)) = rest.split_once('.')
                && let Ok(n) = idx.parse::<u32>()
            {
                tensor.name = format!("blk.{}.{tail}", n + 1);
            }
        }
        renumbered.metadata.remove("gemma4.attention.head_count_kv");
        renumbered.metadata.remove("gemma4.block_count");
        let err = format!(
            "{}",
            gemma4_layer_types_from_missing_v(&renumbered)
                .expect_err("blocks that do not start at 0 break the positional mapping")
        );
        assert!(err.contains("truncated or renumbered"), "{err}");
    }

    /// An authoritative config that already states the layout keeps it; one that
    /// says nothing still gets the measured answer.
    ///
    /// `text_config` is checked as well as the top level because that is where
    /// `parse_layer_types` looks first — a top-level write beside a
    /// `text_config.layer_types` is dead weight the loader never reads.
    ///
    /// Mutation this catches: reverting the call site to an unconditional write
    /// makes the first two assertions fail.
    #[test]
    fn an_authoritative_gemma4_config_keeps_the_layout_it_states() {
        let stated = serde_json::json!({
            "attention_k_eq_v": false,
            "text_config": { "layer_types": ["full_attention"] },
        });
        assert!(gemma4_config_states(&stated, "attention_k_eq_v"));
        assert!(gemma4_config_states(&stated, "layer_types"));

        let nested_flag = serde_json::json!({
            "text_config": { "attention_k_eq_v": true },
        });
        assert!(gemma4_config_states(&nested_flag, "attention_k_eq_v"));

        let silent = serde_json::json!({ "text_config": { "head_dim": 256 } });
        assert!(!gemma4_config_states(&silent, "attention_k_eq_v"));
        assert!(!gemma4_config_states(&silent, "layer_types"));
    }

    #[test]
    fn mixed_q4_0_and_q4_k_source_metadata_names_every_tensor() {
        // Q4_0 and Q4_K are both 4-bit and decode completely differently.
        // Comparing bit-widths alone reports this file as uniform, emits no
        // per-tensor entries, and leaves the Q4_K tensor labelled 4-bit affine
        // group 32 — wrong kernel, wrong group size, no error anywhere.
        let gguf = source_quant_fixture(&[
            ("blk.0.attn_q.weight", GgufTensorType::Q4_0),
            ("blk.0.attn_k.weight", GgufTensorType::Q4_0),
            ("blk.0.ffn_down.weight", GgufTensorType::Q4K),
        ]);
        let quant = preserved_source_quantization(&gguf, true).unwrap().unwrap();

        // The modal profile stays the top-level fallback.
        assert_eq!(quant["bits"], 4);
        assert_eq!(quant["group_size"], 32);
        assert_eq!(quant["mode"], "affine");

        assert_eq!(
            quant["language_model.model.layers.0.mlp.down_proj"],
            serde_json::json!({ "bits": 4, "group_size": 32, "mode": "q4k" })
        );
        assert_eq!(
            quant["language_model.model.layers.0.self_attn.q_proj"],
            serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "affine",
                SYMMETRIC_ZERO_POINT_KEY: 8,
            })
        );
        assert_eq!(
            quant["language_model.model.layers.0.self_attn.k_proj"],
            serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "affine",
                SYMMETRIC_ZERO_POINT_KEY: 8,
            })
        );
    }

    #[test]
    fn k_quant_source_metadata_is_named_even_in_a_uniform_file() {
        // Unsloth Dynamic GGUFs are mixed by construction, so a K-quant tensor
        // is named whether or not this particular file happens to look uniform.
        for (format, tensor_type) in k_quant_cases() {
            let gguf = source_quant_fixture(&[("blk.0.ffn_down.weight", tensor_type)]);
            let quant = preserved_source_quantization(&gguf, true).unwrap().unwrap();
            let expected = serde_json::json!({
                "bits": format.bits(),
                "group_size": format.group_size(),
                "mode": format.mlx_mode(),
            });
            assert_eq!(quant["bits"], expected["bits"], "{}", tensor_type.name());
            assert_eq!(
                quant["group_size"],
                expected["group_size"],
                "{}",
                tensor_type.name()
            );
            assert_eq!(quant["mode"], expected["mode"], "{}", tensor_type.name());
            assert_eq!(
                quant["language_model.model.layers.0.mlp.down_proj"],
                expected,
                "{}",
                tensor_type.name()
            );
        }
    }

    #[test]
    fn k_quant_source_metadata_is_absent_without_the_import() {
        // With the import off the loader rejected Q4_K / Q5_K and dequantized
        // the Gemma4 Q6_K embedding to BF16, so no K-quant tensor survives to
        // be described.
        for (_, tensor_type) in k_quant_cases() {
            let gguf = source_quant_fixture(&[("blk.0.ffn_down.weight", tensor_type)]);
            assert!(
                preserved_source_quantization(&gguf, false)
                    .unwrap()
                    .is_none(),
                "{}",
                tensor_type.name()
            );
        }
    }

    #[test]
    fn source_quantization_collision_names_the_full_triple() {
        let gguf = source_quant_fixture(&[
            ("blk.0.ffn_down.weight", GgufTensorType::Q4_0),
            ("blk.0.ffn_down.weight", GgufTensorType::Q4K),
        ]);
        let message = preserved_source_quantization(&gguf, true)
            .unwrap_err()
            .reason
            .clone();
        assert!(
            message.contains("4-bit affine (group_size 32, symmetric zero point 8)"),
            "{message}"
        );
        assert!(message.contains("4-bit q4k (group_size 32)"), "{message}");
    }

    #[tokio::test]
    async fn conversion_uses_authoritative_config_assets_and_preserves_source_dtypes() {
        let unique = format!(
            "mlx-node-gguf-config-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).unwrap();

        let mut q4 = [0u8; 18];
        q4[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        q4[2..].fill(0x87);
        let norm = 3.25f32.to_le_bytes();
        let gguf_data = build_minimal_gguf(
            &[(
                "general.architecture",
                GgufMetaValue::String("gemma4".to_string()),
            )],
            &[
                ("blk.0.attn_q.weight", &[32, 1], GgufTensorType::Q4_0, &q4),
                ("output_norm.weight", &[1], GgufTensorType::F32, &norm),
            ],
        );
        let input = root.join("model.gguf");
        fs::write(&input, gguf_data).unwrap();

        let source_config = serde_json::json!({
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "head_dim": 256,
                "layer_types": ["full_attention", "sliding_attention"]
            },
            "conversion_test_sentinel": "preserve-me"
        });
        fs::write(
            source.join("config.json"),
            serde_json::to_vec_pretty(&source_config).unwrap(),
        )
        .unwrap();
        let assets = [
            ("tokenizer.json", "tokenizer-exact"),
            ("tokenizer_config.json", "tokenizer-config-exact"),
            ("chat_template.jinja", "chat-template-exact"),
            ("processor_config.json", "processor-config-exact"),
        ];
        for (name, contents) in assets {
            fs::write(source.join(name), contents).unwrap();
        }

        convert_gguf_to_safetensors(GgufConversionOptions {
            input_path: input.to_string_lossy().into_owned(),
            output_dir: output.to_string_lossy().into_owned(),
            config_source_dir: Some(source.to_string_lossy().into_owned()),
            dtype: None,
            verbose: Some(false),
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: None,
            vlm_key_prefix: Some(false),
            quant_mxfp: Some(false),
            import_k_quants: Some(false),
        })
        .await
        .unwrap();

        let mut expected_config = source_config;
        // The source tensors are ggml Q4_0 (`d * (q - 8)`), so the block carries
        // the zero point that stands in for the `.biases` array the import no
        // longer writes.
        let expected_quant = serde_json::json!({
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            SYMMETRIC_ZERO_POINT_KEY: 8,
        });
        expected_config["quantization"] = expected_quant.clone();
        expected_config["quantization_config"] = expected_quant;
        let converted_config: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(converted_config, expected_config);
        for (name, contents) in assets {
            assert_eq!(fs::read_to_string(output.join(name)).unwrap(), contents);
        }

        let safetensors =
            crate::utils::safetensors::SafeTensorsFile::load(output.join("model.safetensors"))
                .unwrap();
        assert!(matches!(
            safetensors.tensors["model.norm.weight"].dtype,
            crate::utils::safetensors::SafeTensorDType::F32
        ));
        assert!(matches!(
            safetensors.tensors["model.layers.0.self_attn.q_proj.weight"].dtype,
            crate::utils::safetensors::SafeTensorDType::U32
        ));

        fs::remove_dir_all(root).ok();
    }

    /// A miniature of Google's QAT layout on disk: `layers` decoder blocks in
    /// Q4_0, `global` written without `attn_v`, and the header that corroborates
    /// it — `block_count` plus a `head_count_kv` array whose minimum sits on
    /// exactly those layers.
    fn write_gemma4_kv_layout_gguf(root: &Path, layers: u32, global: &[u32]) -> PathBuf {
        write_gemma4_kv_layout_gguf_with_header(root, layers, global, global)
    }

    /// Writes a gemma4 GGUF whose tensor list and `head_count_kv` header can be
    /// made to disagree: `tensor_global` picks the layers that omit `attn_v`,
    /// `header_global` picks the layers the header marks as low-KV. Passing the
    /// same set for both produces a self-consistent file.
    fn write_gemma4_kv_layout_gguf_with_header(
        root: &Path,
        layers: u32,
        tensor_global: &[u32],
        header_global: &[u32],
    ) -> PathBuf {
        let head_count_kv: Vec<i32> = (0..layers)
            .map(|layer| if header_global.contains(&layer) { 1 } else { 8 })
            .collect();
        write_gemma4_kv_layout_gguf_with_kv_meta(
            root,
            layers,
            tensor_global,
            Some(GgufMetaValue::ArrayI32(head_count_kv)),
        )
    }

    /// The same file with `gemma4.attention.head_count_kv` spelled however the
    /// caller likes — `None` omits the key entirely, a scalar states one count
    /// for the whole model. Both are legal GGUF and neither says which layers
    /// are global.
    fn write_gemma4_kv_layout_gguf_with_kv_meta(
        root: &Path,
        layers: u32,
        tensor_global: &[u32],
        head_count_kv: Option<GgufMetaValue>,
    ) -> PathBuf {
        let global = tensor_global;
        let mut q4 = [0u8; 18];
        q4[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        q4[2..].fill(0x87);

        let mut names = Vec::new();
        for layer in 0..layers {
            names.push(format!("blk.{layer}.attn_q.weight"));
            names.push(format!("blk.{layer}.attn_k.weight"));
            if !global.contains(&layer) {
                names.push(format!("blk.{layer}.attn_v.weight"));
            }
            names.push(format!("blk.{layer}.attn_output.weight"));
        }
        let tensors: Vec<(&str, &[u64], GgufTensorType, &[u8])> = names
            .iter()
            .map(|name| {
                (
                    name.as_str(),
                    &[32u64, 1][..],
                    GgufTensorType::Q4_0,
                    &q4[..],
                )
            })
            .collect();

        let mut metadata = vec![
            (
                "general.architecture",
                GgufMetaValue::String("gemma4".to_string()),
            ),
            ("gemma4.block_count", GgufMetaValue::Uint32(layers)),
        ];
        if let Some(value) = head_count_kv {
            metadata.push(("gemma4.attention.head_count_kv", value));
        }
        let data = build_minimal_gguf(&metadata, &tensors);
        let input = root.join("model.gguf");
        fs::write(&input, data).unwrap();
        input
    }

    /// End-to-end at the call site: an authoritative `--config-dir` config that
    /// already states the attention layout keeps its own answer, and one that
    /// says nothing gets the measured one.
    ///
    /// The helper-level test above cannot see this — the call site could go back
    /// to writing both keys unconditionally and `gemma4_config_states` would
    /// still behave. `layer_types` is stated under `text_config` because that is
    /// where `parse_layer_types` reads it first, so a top-level write beside it
    /// is not merely redundant: it is silently ignored while claiming to have
    /// corrected the config.
    ///
    /// `save_safetensors` opens the destination with `File::create`, which
    /// truncates it before the first byte is written, so a conversion that can
    /// still fail afterwards destroys a previously good output on its way to
    /// reporting the error. The gemma4 layout check reads nothing but tensor
    /// names and metadata, both of which `parse_gguf` has already produced, so
    /// it belongs with the other guards that run before any weight is loaded.
    ///
    /// Mutation this catches: move `gemma4_layer_types_from_missing_v` back
    /// below `save_safetensors` and the prior `model.safetensors` comes back
    /// truncated or replaced, so the byte comparison fails. The conversion
    /// still returns `Err` either way — the error is not what is being pinned,
    /// the survival of the destination is.
    #[tokio::test]
    async fn a_contradicted_gemma4_layout_is_rejected_before_the_output_is_touched() {
        let unique = format!(
            "mlx-node-gguf-preflight-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let output = root.join("out");
        fs::create_dir_all(&output).unwrap();

        // Tensors omit `attn_v` on layer 3; the header marks layer 5 as the
        // low-KV one. Exactly the disagreement the cross-check exists to catch.
        let input = write_gemma4_kv_layout_gguf_with_header(&root, 6, &[3], &[5]);

        let existing = output.join("model.safetensors");
        let sentinel = b"a previously converted model that must survive a failed re-convert";
        fs::write(&existing, sentinel).unwrap();

        let converted = convert_gguf_to_safetensors(GgufConversionOptions {
            input_path: input.to_string_lossy().into_owned(),
            output_dir: output.to_string_lossy().into_owned(),
            config_source_dir: None,
            dtype: None,
            verbose: Some(false),
            quantize: Some(false),
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            output_filename: None,
            vlm_key_prefix: Some(false),
            quant_mxfp: Some(false),
            import_k_quants: Some(false),
        })
        .await;
        let Err(err) = converted else {
            panic!("a header that contradicts the tensor list must be rejected");
        };
        assert!(
            err.reason.contains("head_count_kv"),
            "the error must name the header it disagreed with: {}",
            err.reason
        );

        assert_eq!(
            fs::read(&existing).unwrap(),
            sentinel,
            "the destination was written before the layout was validated"
        );

        fs::remove_dir_all(&root).ok();
    }

    /// Detecting the K=V layout proves the sliding and the global layers have
    /// DIFFERENT KV head counts. A synthesized config can only state the global
    /// one when `head_count_kv` is a per-layer array: a scalar leaves
    /// `num_global_key_value_heads` unwritten and the loader reuses the sliding
    /// count, an absent key leaves both unwritten and it falls back to 2. Either
    /// way the conversion and the load both SUCCEED — the quantized path
    /// discards `out_features`, so nothing checks the width — and the process
    /// aborts inside `mlx_array_reshape` at the first token, across an
    /// `extern "C-unwind"` boundary that turns the error into a crash. So the
    /// converter refuses, names the header, and points at `--config-dir`, which
    /// supplies the counts and converts.
    ///
    /// `apply_gemma4_attention_geometry` has no other test: the whole failure
    /// mode is that it degrades quietly.
    ///
    /// Mutations this catches: (1) drop the guard — both conversions return Ok
    /// and the sentinel is gone; (2) put the guard at the geometry call site
    /// instead, which runs after `save_safetensors` — the Err still comes back
    /// but the sentinel assertion fails, exactly the property the pre-write test
    /// above pins; (3) weaken it to `head_count_kv` merely being present — the
    /// scalar case stops failing.
    #[tokio::test]
    async fn a_synthesized_gemma4_config_needs_both_kv_head_counts() {
        for (label, header) in [
            ("an absent", None),
            ("a scalar", Some(GgufMetaValue::Uint32(8))),
        ] {
            let unique = format!(
                "mlx-node-gguf-kv-counts-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let root = std::env::temp_dir().join(unique);
            let output = root.join("out");
            fs::create_dir_all(&output).unwrap();

            // Layer 5 omits `attn_v` — the same partial-V tensor list the
            // helper writes for the self-consistent case, with only the header
            // changed.
            let input = write_gemma4_kv_layout_gguf_with_kv_meta(&root, 6, &[5], header);

            let existing = output.join("model.safetensors");
            let sentinel = b"a previously converted model that must survive a failed re-convert";
            fs::write(&existing, sentinel).unwrap();

            let options =
                |output_dir: &Path, config_source_dir: Option<String>| GgufConversionOptions {
                    input_path: input.to_string_lossy().into_owned(),
                    output_dir: output_dir.to_string_lossy().into_owned(),
                    config_source_dir,
                    dtype: None,
                    verbose: Some(false),
                    quantize: Some(false),
                    quant_bits: None,
                    quant_group_size: None,
                    quant_mode: None,
                    quant_recipe: None,
                    imatrix_path: None,
                    output_filename: None,
                    vlm_key_prefix: Some(false),
                    quant_mxfp: Some(false),
                    import_k_quants: Some(false),
                };

            let Err(err) = convert_gguf_to_safetensors(options(&output, None)).await else {
                panic!("{label} head_count_kv cannot describe the global layers of a K=V GGUF");
            };
            assert!(
                err.reason.contains("gemma4.attention.head_count_kv"),
                "the error must name the header it needs: {}",
                err.reason
            );
            assert!(
                err.reason.contains("--config-dir"),
                "the error must name the escape hatch: {}",
                err.reason
            );
            assert_eq!(
                fs::read(&existing).unwrap(),
                sentinel,
                "the destination was written before the KV head counts were checked"
            );

            // The escape hatch: an authoritative config states the geometry
            // itself, so the header is never consulted and the same file
            // converts.
            let source = root.join("source");
            fs::create_dir_all(&source).unwrap();
            fs::write(
                source.join("config.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "model_type": "gemma4",
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 1,
                }))
                .unwrap(),
            )
            .unwrap();
            let with_config = root.join("with-config");
            fs::create_dir_all(&with_config).unwrap();
            convert_gguf_to_safetensors(options(
                &with_config,
                Some(source.to_string_lossy().into_owned()),
            ))
            .await
            .unwrap_or_else(|e| panic!("--config-dir must still convert {label}: {}", e.reason));
            assert!(with_config.join("model.safetensors").is_file());

            fs::remove_dir_all(&root).ok();
        }
    }

    /// Mutation this catches: drop either `gemma4_config_states` check at the
    /// call site and the first block's assertions fail — `attention_k_eq_v`
    /// flips to `true` and a top-level `layer_types` appears.
    #[tokio::test]
    async fn an_authoritative_config_outranks_the_measured_gemma4_layout() {
        let unique = format!(
            "mlx-node-gguf-k-eq-v-scope-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let source = root.join("source");
        fs::create_dir_all(&source).unwrap();
        let input = write_gemma4_kv_layout_gguf(&root, 6, &[5]);

        let convert_with = |config: serde_json::Value, output: PathBuf| {
            fs::write(
                source.join("config.json"),
                serde_json::to_vec_pretty(&config).unwrap(),
            )
            .unwrap();
            let input = input.clone();
            let source = source.clone();
            async move {
                convert_gguf_to_safetensors(GgufConversionOptions {
                    input_path: input.to_string_lossy().into_owned(),
                    output_dir: output.to_string_lossy().into_owned(),
                    config_source_dir: Some(source.to_string_lossy().into_owned()),
                    dtype: None,
                    verbose: Some(false),
                    quantize: Some(false),
                    quant_bits: None,
                    quant_group_size: None,
                    quant_mode: None,
                    quant_recipe: None,
                    imatrix_path: None,
                    output_filename: None,
                    vlm_key_prefix: Some(false),
                    quant_mxfp: Some(false),
                    import_k_quants: Some(false),
                })
                .await
                .unwrap();
                let written: serde_json::Value =
                    serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
                written
            }
        };

        let stated = convert_with(
            serde_json::json!({
                "model_type": "gemma4",
                "attention_k_eq_v": false,
                "text_config": {
                    "layer_types": vec!["sliding_attention"; 6],
                },
            }),
            root.join("stated"),
        )
        .await;
        assert_eq!(stated["attention_k_eq_v"], serde_json::json!(false));
        assert_eq!(
            stated["text_config"]["layer_types"],
            serde_json::json!(vec!["sliding_attention"; 6])
        );
        assert!(
            stated.get("layer_types").is_none(),
            "a top-level layer_types beside text_config.layer_types is never read: {stated}"
        );

        let silent = convert_with(
            serde_json::json!({ "model_type": "gemma4" }),
            root.join("silent"),
        )
        .await;
        assert_eq!(silent["attention_k_eq_v"], serde_json::json!(true));
        assert_eq!(
            silent["layer_types"],
            serde_json::json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ])
        );

        fs::remove_dir_all(&root).ok();
    }

    /// `--quantize` on a GGUF that already carries symmetric ggml blocks is
    /// rejected at convert time, not at load time.
    ///
    /// Q4_0/Q8_0 groups arrive already packed, so `quantize_weights_inner` skips
    /// every one of them and records no per-layer override. The config write
    /// then takes the `do_quantize` branch, emits a fresh
    /// `build_quantization_object`, and never reaches the arm that would carry
    /// `preserved_source_quantization` forward — so the `symmetric_zero_point`
    /// that `expand_symmetric_affine_biases` needs to rebuild the omitted
    /// `.biases` is dropped. Before this guard both conversions succeeded and
    /// produced a checkpoint no loader can read.
    ///
    /// Both fixtures are genuine mixtures — a packed symmetric attention
    /// projection beside a float FFN tensor the quantizer really does quantize —
    /// which is why keeping the top-level marker is not an option: it would
    /// claim a derived bias for the freshly quantized tensor that stores a real
    /// one.
    ///
    /// Mutation this catches: delete the guard, or narrow it to K-quants only,
    /// and both conversions succeed again — the `expect_err` fails.
    #[tokio::test]
    async fn quantize_is_rejected_for_a_symmetric_gguf_source() {
        let mut q4 = [0u8; 18];
        q4[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        q4[2..].fill(0x87);
        let mut q8 = [0u8; 34];
        q8[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        q8[2..].fill(0x11);

        let cases: [(GgufTensorType, &[u8], i32); 2] = [
            (GgufTensorType::Q4_0, &q4, 4),
            (GgufTensorType::Q8_0, &q8, 8),
        ];
        for (tensor_type, block, bits) in cases {
            // A float tensor the quantizer really does quantize, so the output
            // would be a mixture rather than an all-skipped body.
            let ffn = vec![0u8; 64 * 4];
            let norm = 3.25f32.to_le_bytes();
            let data = build_minimal_gguf(
                &[(
                    "general.architecture",
                    GgufMetaValue::String("llama".to_string()),
                )],
                &[
                    ("blk.0.attn_q.weight", &[32, 1], tensor_type, block),
                    ("blk.0.ffn_down.weight", &[64, 1], GgufTensorType::F32, &ffn),
                    ("output_norm.weight", &[1], GgufTensorType::F32, &norm),
                ],
            );
            let root = std::env::temp_dir().join(format!(
                "mlx-node-gguf-symmetric-requantize-{}-{}-{}",
                tensor_type.name(),
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&root).unwrap();
            let input = root.join("model.gguf");
            fs::write(&input, data).unwrap();

            let outcome = convert_gguf_to_safetensors(GgufConversionOptions {
                input_path: input.to_string_lossy().into_owned(),
                output_dir: root.join("output").to_string_lossy().into_owned(),
                config_source_dir: None,
                dtype: None,
                verbose: Some(false),
                quantize: Some(true),
                quant_bits: Some(bits),
                quant_group_size: Some(32),
                quant_mode: None,
                quant_recipe: None,
                imatrix_path: None,
                output_filename: None,
                vlm_key_prefix: Some(false),
                quant_mxfp: Some(false),
                import_k_quants: Some(false),
            })
            .await;
            fs::remove_dir_all(&root).ok();

            let Err(err) = outcome else {
                panic!(
                    "--quantize on a {} GGUF must be rejected",
                    tensor_type.name()
                );
            };
            let message = format!("{err}");
            assert!(message.contains("blk.0.attn_q.weight"), "{message}");
            assert!(message.contains(tensor_type.name()), "{message}");
            assert!(message.contains("without --quantize"), "{message}");
            assert!(message.contains(SYMMETRIC_ZERO_POINT_KEY), "{message}");
        }
    }

    /// The guard is scoped to the formats whose `.biases` is derived. Q4_1
    /// stores a real per-block minimum no scale can reproduce, so it carries no
    /// `symmetric_zero_point` and is not what this rejects.
    ///
    /// Mutation this catches: widening the guard to
    /// `is_mlx_affine_quantized()` (or to any already-quantized tensor) starts
    /// rejecting Q4_1 too and this assertion fails.
    #[test]
    fn the_symmetric_requantize_guard_does_not_cover_q4_1() {
        assert!(symmetric_zero_point(GgufTensorType::Q4_0).is_some());
        assert!(symmetric_zero_point(GgufTensorType::Q8_0).is_some());
        assert!(symmetric_zero_point(GgufTensorType::Q4_1).is_none());
    }

    /// A float-only source cannot trip the secondary-output guard, because a
    /// float tensor has no quantization metadata to record in the first place.
    ///
    /// This is the blast radius of that guard on the only mmproj Google ships
    /// for the QAT checkpoint: `mmproj-gemma-4-12b-it-qat-q4_0.gguf` holds nine
    /// F32 tensors and two BF16 ones and nothing else, so neither predicate can
    /// fire on it.
    #[test]
    fn float_gguf_types_carry_no_source_quantization_profile() {
        for ty in [
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::BF16,
        ] {
            assert!(
                SourceQuantProfile::for_gguf_type(ty).is_none(),
                "{}",
                ty.name()
            );
        }
    }

    /// A secondary output writes no config.json, so a source carrying tensors
    /// that can only be decoded with a config entry is rejected upfront.
    ///
    /// `--mmproj` runs a second conversion into `vision.safetensors`, and only
    /// the primary `model.safetensors` conversion writes config.json. A
    /// symmetric ggml block landing there loses the `symmetric_zero_point` its
    /// derived `.biases` is rebuilt from; an imported K-quant loses the
    /// mode/bits/group_size that spell out ggml's geometry. Either way the
    /// packed bytes are written correctly and then decoded as generic affine.
    ///
    /// Mutation this catches: drop the guard, or narrow it to one of the two
    /// predicates, and the corresponding `expect_err` fails. The last two cases
    /// are the scope controls — widening the guard to every output breaks the
    /// primary case, and dropping its `import_k_quants` exemption breaks the
    /// dequantized-Q6_K case.
    #[tokio::test]
    async fn a_secondary_output_rejects_tensors_that_need_a_config_entry() {
        let mut q4 = [0u8; 18];
        q4[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        q4[2..].fill(0x87);
        let q6k = pack_q6k_block(&[0; 256], &[1; 16], 1.0);

        fn run(
            name: &str,
            dims: &[u64],
            tensor_type: GgufTensorType,
            payload: &[u8],
            output_filename: Option<&str>,
            import_k_quants: bool,
        ) -> (std::path::PathBuf, GgufConversionOptions) {
            let norm = 3.25f32.to_le_bytes();
            let data = build_minimal_gguf(
                &[(
                    "general.architecture",
                    GgufMetaValue::String("gemma4".to_string()),
                )],
                &[
                    (name, dims, tensor_type, payload),
                    ("output_norm.weight", &[1], GgufTensorType::F32, &norm),
                ],
            );
            let root = std::env::temp_dir().join(format!(
                "mlx-node-gguf-secondary-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&root).unwrap();
            let input = root.join("model.gguf");
            fs::write(&input, data).unwrap();
            let options = GgufConversionOptions {
                input_path: input.to_string_lossy().into_owned(),
                output_dir: root.join("output").to_string_lossy().into_owned(),
                config_source_dir: None,
                dtype: None,
                verbose: Some(false),
                quantize: Some(false),
                quant_bits: None,
                quant_group_size: None,
                quant_mode: None,
                quant_recipe: None,
                imatrix_path: None,
                output_filename: output_filename.map(str::to_string),
                vlm_key_prefix: Some(false),
                quant_mxfp: Some(false),
                import_k_quants: Some(import_k_quants),
            };
            (root, options)
        }

        // A symmetric ggml block in a secondary output has nowhere to record
        // `symmetric_zero_point`.
        let (root, options) = run(
            "blk.0.attn_q.weight",
            &[32, 1],
            GgufTensorType::Q4_0,
            &q4,
            Some("vision.safetensors"),
            false,
        );
        let outcome = convert_gguf_to_safetensors(options).await;
        fs::remove_dir_all(&root).ok();
        let Err(err) = outcome else {
            panic!("a symmetric block in a secondary output must be rejected");
        };
        let message = format!("{err}");
        assert!(message.contains("blk.0.attn_q.weight"), "{message}");
        assert!(message.contains("Q4_0"), "{message}");
        assert!(message.contains("vision.safetensors"), "{message}");
        assert!(message.contains("config.json"), "{message}");

        // An imported K-quant has nowhere to record its ggml geometry.
        let (root, options) = run(
            "token_embd.weight",
            &[256, 1],
            GgufTensorType::Q6K,
            &q6k,
            Some("vision.safetensors"),
            true,
        );
        let outcome = convert_gguf_to_safetensors(options).await;
        fs::remove_dir_all(&root).ok();
        let Err(err) = outcome else {
            panic!("an imported K-quant in a secondary output must be rejected");
        };
        let message = format!("{err}");
        assert!(message.contains("token_embd.weight"), "{message}");
        assert!(message.contains("Q6_K"), "{message}");
        assert!(message.contains("q6k"), "{message}");

        // Scope control: the primary output records both, so it stays allowed.
        let (root, options) = run(
            "blk.0.attn_q.weight",
            &[32, 1],
            GgufTensorType::Q4_0,
            &q4,
            None,
            false,
        );
        let outcome = convert_gguf_to_safetensors(options).await;
        fs::remove_dir_all(&root).ok();
        outcome.expect("the primary output writes config.json and keeps working");

        // Scope control: without the import, Q6_K is dequantized to BF16 and
        // needs no config entry — so a secondary output may carry it.
        let (root, options) = run(
            "token_embd.weight",
            &[256, 1],
            GgufTensorType::Q6K,
            &q6k,
            Some("vision.safetensors"),
            false,
        );
        let outcome = convert_gguf_to_safetensors(options).await;
        fs::remove_dir_all(&root).ok();
        outcome.expect("a dequantized Q6_K embedding needs no config entry");
    }

    #[test]
    fn test_key_remapping_standard() {
        // Standard LLM tensor names
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_q.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.31.attn_k.weight"),
            "model.layers.31.self_attn.k_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_v.weight"),
            "model.layers.0.self_attn.v_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_output.weight"),
            "model.layers.0.self_attn.o_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ffn_gate.weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ffn_down.weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ffn_up.weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_norm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ffn_norm.weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            gguf_name_to_hf("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(gguf_name_to_hf("output_norm.weight"), "model.norm.weight");
        assert_eq!(gguf_name_to_hf("output.weight"), "lm_head.weight");
    }

    /// A quantized global tensor ships as a `.weight`/`.scales`/`.biases` trio
    /// (both the affine `load_quantized_tensor` repack and the K-quant
    /// `load_kquant_repack` emit all three). The global renames are whole-string
    /// matches, so before this they moved only `.weight` and stranded the
    /// sidecars under their GGUF names — the loaders probe `embed_tokens.scales`
    /// / `lm_head.scales` to decide a tensor is quantized, so the group silently
    /// degraded to a bare packed weight that then failed the dense dtype guard.
    #[test]
    fn global_renames_carry_the_whole_quant_group() {
        for (gguf, hf) in [("token_embd", "model.embed_tokens"), ("output", "lm_head")] {
            for suffix in [".weight", ".scales", ".biases"] {
                assert_eq!(
                    gguf_name_to_hf(&format!("{gguf}{suffix}")),
                    format!("{hf}{suffix}"),
                    "generic mapper must carry {suffix} for {gguf}"
                );
                assert_eq!(
                    gemma4_name_to_hf(&format!("{gguf}{suffix}")).unwrap(),
                    format!("{hf}{suffix}"),
                    "gemma4 mapper must carry {suffix} for {gguf}"
                );
            }
        }
    }

    /// The quant-group carry must not over-generalize: `output_norm` is a norm
    /// (never quantized) and only its `.weight` is renamed, and a neighbour that
    /// merely shares the `token_embd` / `output` prefix is left alone.
    #[test]
    fn global_renames_do_not_over_generalize() {
        // output_norm.weight keeps its own destination, NOT lm_head.*.
        assert_eq!(gguf_name_to_hf("output_norm.weight"), "model.norm.weight");
        assert_eq!(
            gemma4_name_to_hf("output_norm.weight").unwrap(),
            "model.norm.weight"
        );
        // A non-canonical suffix on a global tensor is not a quant-group member
        // and must pass through unrenamed.
        assert_eq!(gguf_name_to_hf("token_embd.bias"), "token_embd.bias");
        assert_eq!(gguf_name_to_hf("output.bias"), "output.bias");
        // A longer base that merely starts with the same prefix is untouched.
        assert_eq!(
            gguf_name_to_hf("token_embd_extra.weight"),
            "token_embd_extra.weight"
        );
        assert_eq!(
            rename_global_quant_group("output_norm.weight", "output", "lm_head"),
            None,
            "prefix-only match must not rename"
        );
    }

    /// A tied model (only `token_embd.*`) and an untied one (`token_embd.*` plus
    /// `output.*`) must both remap unambiguously now that three keys map where
    /// one did before: the six destinations are pairwise distinct, so
    /// `remap_keys`' injectivity check has nothing to trip on.
    #[test]
    fn tied_and_untied_quant_groups_remap_without_collision() {
        let mut seen = std::collections::HashSet::new();
        for base in ["token_embd", "output"] {
            for suffix in [".weight", ".scales", ".biases"] {
                let dest = gguf_name_to_hf(&format!("{base}{suffix}"));
                assert!(
                    seen.insert(dest.clone()),
                    "duplicate remap destination '{dest}' would collide in remap_keys"
                );
            }
        }
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn test_key_remapping_qwen35() {
        // Qwen3.5 full-attention layers
        assert_eq!(
            gguf_name_to_hf("blk.3.attn_q_norm.weight"),
            "model.layers.3.self_attn.q_norm.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.3.attn_k_norm.weight"),
            "model.layers.3.self_attn.k_norm.weight"
        );

        // Qwen3.5 post_attention_norm (different from ffn_norm)
        assert_eq!(
            gguf_name_to_hf("blk.0.post_attention_norm.weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );

        // Qwen3.5 GatedDeltaNet / linear attention
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_qkv.weight"),
            "model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_gate.weight"),
            "model.layers.0.linear_attn.in_proj_z.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_beta.weight"),
            "model.layers.0.linear_attn.in_proj_b.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_alpha.weight"),
            "model.layers.0.linear_attn.in_proj_a.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_out.weight"),
            "model.layers.0.linear_attn.out_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_conv1d.weight"),
            "model.layers.0.linear_attn.conv1d.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_norm.weight"),
            "model.layers.0.linear_attn.norm.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_dt.bias"),
            "model.layers.0.linear_attn.dt_bias"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ssm_a"),
            "model.layers.0.linear_attn.A_log"
        );
    }

    fn qwen35_hybrid_test_keys() -> Vec<String> {
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.linear_attn.in_proj_qkv.weight",
            "model.layers.1.linear_attn.in_proj_z.weight",
            "model.layers.1.linear_attn.out_proj.weight",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    #[test]
    fn official_unsloth_gguf_gate_requires_qwen35_arch_and_hybrid_shape() {
        let keys = qwen35_hybrid_test_keys();
        for arch in ["qwen35", "qwen35moe", "qwen3"] {
            let metadata = HashMap::from([(
                "general.architecture".to_string(),
                GgufMetaValue::String(arch.to_string()),
            )]);
            let is_qwen35_hybrid = is_qwen35_hybrid_gguf(&metadata, &keys);
            assert!(
                is_qwen35_hybrid,
                "{arch} plus the hybrid shape must select the fixed map"
            );
            assert_eq!(
                crate::convert::select_official_unsloth_recipe(
                    "unsloth",
                    true,
                    "affine",
                    is_qwen35_hybrid,
                ),
                Some(crate::convert::OfficialUnslothRecipeKind::QwenMxfp)
            );
            assert_eq!(
                crate::convert::select_official_unsloth_recipe(
                    "unsloth",
                    false,
                    "nvfp4",
                    is_qwen35_hybrid,
                ),
                Some(crate::convert::OfficialUnslothRecipeKind::QwenNvfp4)
            );
        }

        for arch in ["gemma", "llama", "qwen3next"] {
            let metadata = HashMap::from([(
                "general.architecture".to_string(),
                GgufMetaValue::String(arch.to_string()),
            )]);
            assert!(
                !is_qwen35_hybrid_gguf(&metadata, &keys),
                "non-Qwen3.5 arch {arch} must preserve the legacy upgrader"
            );
        }

        let metadata = HashMap::from([(
            "general.architecture".to_string(),
            GgufMetaValue::String("qwen35".to_string()),
        )]);
        assert!(!is_qwen35_hybrid_gguf(
            &metadata,
            &["model.layers.0.self_attn.q_proj.weight".to_string()]
        ));
    }

    #[test]
    fn test_config_extraction() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetaValue::String("qwen3".to_string()),
        );
        metadata.insert(
            "general.name".to_string(),
            GgufMetaValue::String("Qwen3-0.6B".to_string()),
        );
        metadata.insert(
            "qwen3.embedding_length".to_string(),
            GgufMetaValue::Uint32(1024),
        );
        metadata.insert("qwen3.block_count".to_string(), GgufMetaValue::Uint32(28));
        metadata.insert(
            "qwen3.feed_forward_length".to_string(),
            GgufMetaValue::Uint32(3072),
        );
        metadata.insert(
            "qwen3.attention.head_count".to_string(),
            GgufMetaValue::Uint32(16),
        );
        metadata.insert(
            "qwen3.attention.head_count_kv".to_string(),
            GgufMetaValue::Uint32(8),
        );
        metadata.insert(
            "qwen3.attention.layer_norm_rms_epsilon".to_string(),
            GgufMetaValue::Float32(1e-6),
        );

        let config = extract_config(&metadata);
        assert_eq!(config["model_type"], "qwen3");
        assert_eq!(config["hidden_size"], 1024);
        assert_eq!(config["num_hidden_layers"], 28);
        assert_eq!(config["intermediate_size"], 3072);
        assert_eq!(config["num_attention_heads"], 16);
        assert_eq!(config["num_key_value_heads"], 8);
        assert_eq!(config["head_dim"], 64); // 1024/16 — no key_length present
    }

    /// Qwen3 decouples head_dim from hidden_size / num_attention_heads:
    /// Qwen3-0.6B has 16 query heads but head_dim 128 (q_proj is [2048, 1024]).
    /// GGUF records this as `attention.key_length`, which must win over the
    /// `hidden / heads` derivation — otherwise every Qwen3 GGUF fails to load
    /// with `q_norm expected [64], got [128]`.
    #[test]
    fn test_config_extraction_honors_key_length_head_dim() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetaValue::String("qwen3".to_string()),
        );
        metadata.insert(
            "qwen3.embedding_length".to_string(),
            GgufMetaValue::Uint32(1024),
        );
        metadata.insert(
            "qwen3.attention.head_count".to_string(),
            GgufMetaValue::Uint32(16),
        );
        metadata.insert(
            "qwen3.attention.head_count_kv".to_string(),
            GgufMetaValue::Uint32(8),
        );
        // The decoupled head_dim: 128, not 1024/16 = 64.
        metadata.insert(
            "qwen3.attention.key_length".to_string(),
            GgufMetaValue::Uint32(128),
        );

        let config = extract_config(&metadata);
        assert_eq!(config["head_dim"], 128);
    }

    /// Gemma4 must NOT adopt `attention.key_length` as its top-level head_dim:
    /// its sliding-window layers use head_dim 256 and its global layers 512, so
    /// the GGUF reports key_length=512 alongside key_length_swa=256 and neither
    /// is the single answer. The native Gemma4 loader derives per-layer-type
    /// dims itself and ignores this field, which stays `hidden / heads` = 240
    /// for 12B. This locks in byte-identity with existing Gemma4 K-quant
    /// conversions (the reference artifact carries head_dim 240, not 512).
    #[test]
    fn test_config_extraction_gemma4_ignores_key_length() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetaValue::String("gemma4".to_string()),
        );
        metadata.insert(
            "gemma4.embedding_length".to_string(),
            GgufMetaValue::Uint32(3840),
        );
        metadata.insert(
            "gemma4.attention.head_count".to_string(),
            GgufMetaValue::Uint32(16),
        );
        metadata.insert(
            "gemma4.attention.key_length".to_string(),
            GgufMetaValue::Uint32(512),
        );

        let config = extract_config(&metadata);
        assert_eq!(config["head_dim"], 240); // 3840/16 — key_length ignored
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(63, 32), 64);
        assert_eq!(align_offset(64, 32), 64);
    }

    #[test]
    fn test_tensor_info_shapes() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![768, 1024], // GGUF order: [cols, rows]
            tensor_type: GgufTensorType::BF16,
            offset: 0,
        };

        // MLX shape should be reversed
        assert_eq!(info.mlx_shape(), vec![1024, 768]);
        assert_eq!(info.num_elements(), 786432);
        assert_eq!(info.data_size(), 786432 * 2); // BF16 = 2 bytes
    }
}
