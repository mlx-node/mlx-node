/// GGUF Binary Format Parser and Converter
///
/// Reads GGUF model files and converts tensors to MLX SafeTensors format.
/// Supports BF16/F16/F32 unquantized tensors and Q4_0/Q4_1/Q8_0 quantized tensors.
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
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Bytes per element for non-quantized types. Quantized types return block size info.
    fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18, // block size: 2 byte scale + 16 bytes (32 x 4-bit)
            Self::Q4_1 => 20, // 2 byte scale + 2 byte bias + 16 bytes
            Self::Q8_0 => 34, // 2 byte scale + 32 bytes
        }
    }

    /// Number of elements per block for quantized types
    fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            _ => 1,
        }
    }

    fn is_quantized(&self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q4_1 | Self::Q8_0)
    }

    fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q8_0 => "Q8_0",
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
                    "Tensor '{}' has unsupported GGUF type {} — only F32(0), F16(1), Q4_0(2), Q4_1(3), Q8_0(8), BF16(30) are supported. \
                     K-quant formats (Q4_K, Q5_K, Q6_K, etc.) require dequantization before conversion.",
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

/// Load a quantized tensor (Q4_0, Q4_1, Q8_0) and produce (weight, scales, biases) triplet
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
    let mut biases = vec![0u16; sb_elements]; // f16

    let type_size = tensor.tensor_type.type_size();

    match tensor.tensor_type {
        GgufTensorType::Q4_0 => {
            // Block: 2 bytes f16 scale, 16 bytes (32 x 4-bit weights)
            for i in 0..n_blocks {
                let block = &raw[i * type_size..(i + 1) * type_size];
                let scale_bytes = u16::from_le_bytes([block[0], block[1]]);
                scales[i] = scale_bytes;
                // bias = -8 * scale
                let scale_f32 = half::f16::from_bits(scale_bytes).to_f32();
                biases[i] = half::f16::from_f32(-8.0 * scale_f32).to_bits();

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
                let scale_bytes = u16::from_le_bytes([block[0], block[1]]);
                scales[i] = scale_bytes;
                // bias = -128 * scale
                let scale_f32 = half::f16::from_bits(scale_bytes).to_f32();
                biases[i] = half::f16::from_f32(-128.0 * scale_f32).to_bits();

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
    let biases_arr = MxArray::from_float16(&biases, &sb_i64_shape)?;

    // Strip .weight suffix for prefix, then add .scales/.biases
    let name = &tensor.name;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);

    Ok(vec![
        (name.clone(), weight_arr),
        (format!("{prefix}.scales"), scales_arr),
        (format!("{prefix}.biases"), biases_arr),
    ])
}

/// Load all tensors from a GGUF file
pub fn load_gguf_tensors<P: AsRef<Path>>(
    path: P,
    gguf: &GgufFile,
    verbose: bool,
) -> Result<HashMap<String, MxArray>> {
    let file = fs::File::open(path.as_ref())
        .map_err(|e| Error::from_reason(format!("Failed to open GGUF file: {e}")))?;
    let mut reader = BufReader::new(file);
    let mut weights = HashMap::new();

    for (i, tensor) in gguf.tensors.iter().enumerate() {
        if verbose && (i % 50 == 0 || i == gguf.tensors.len() - 1) {
            info!(
                "Loading tensor {}/{}: {} ({}, shape {:?})",
                i + 1,
                gguf.tensors.len(),
                tensor.name,
                tensor.tensor_type.name(),
                tensor.mlx_shape()
            );
        }

        if tensor.tensor_type.is_quantized() {
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

    if result == "token_embd.weight" {
        result = "model.embed_tokens.weight".to_string();
    } else if result == "output_norm.weight" {
        result = "model.norm.weight".to_string();
    } else if result == "output.weight" {
        result = "lm_head.weight".to_string();
    }

    result
}

/// Remap all keys in a weight map from GGUF names to HuggingFace names
fn remap_keys(weights: HashMap<String, MxArray>) -> HashMap<String, MxArray> {
    weights
        .into_iter()
        .map(|(k, v)| (gguf_name_to_hf(&k), v))
        .collect()
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

    // Head dimension (may be derived)
    if let (Some(hidden), Some(heads)) = (
        config.get("hidden_size").and_then(|v| v.as_u64()),
        config.get("num_attention_heads").and_then(|v| v.as_u64()),
    ) && heads > 0
    {
        let head_dim = hidden / heads;
        config.insert(
            "head_dim".to_string(),
            serde_json::Value::Number(head_dim.into()),
        );
    }

    serde_json::Value::Object(config)
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
}

#[napi(object)]
pub struct GgufConversionResult {
    pub num_tensors: i32,
    pub num_parameters: i64,
    pub output_path: String,
    pub tensor_names: Vec<String>,
    pub source_format: String,
}

#[napi]
pub async fn convert_gguf_to_safetensors(
    options: GgufConversionOptions,
) -> Result<GgufConversionResult> {
    let input_path = PathBuf::from(&options.input_path);
    let output_dir = PathBuf::from(&options.output_dir);
    let verbose = options.verbose.unwrap_or(false);

    if !input_path.exists() {
        return Err(Error::from_reason(format!(
            "GGUF file not found: {}",
            input_path.display()
        )));
    }

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

    // Load tensors
    info!("Loading tensors...");
    let mut weights = load_gguf_tensors(&input_path, &gguf, verbose)?;
    info!("Loaded {} tensors", weights.len());

    // Remap keys from GGUF to HF naming
    info!("Remapping tensor names to HuggingFace format...");
    weights = remap_keys(weights);

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

    // Apply AWQ pre-scaling if imatrix provided
    if let Some(ref imatrix_path) = options.imatrix_path {
        let imatrix = crate::utils::imatrix::parse_imatrix(imatrix_path)?;
        let num_layers = crate::convert::infer_num_layers_from_weights(&weights);
        crate::convert::apply_awq_prescaling(&mut weights, &imatrix, 0.5, num_layers)?;
    }

    // Optional quantization
    let mut per_layer_overrides: HashMap<String, serde_json::Value> = HashMap::new();
    let do_quantize = options.quantize.unwrap_or(false);
    let quant_mode_str = options
        .quant_mode
        .as_deref()
        .unwrap_or("affine")
        .to_string();
    let quant_bits = options
        .quant_bits
        .unwrap_or(if quant_mode_str == "mxfp8" { 8 } else { 4 });
    let quant_group_size = options
        .quant_group_size
        .unwrap_or(if quant_mode_str == "mxfp8" { 32 } else { 64 });

    if do_quantize {
        if let Some(ref recipe) = options.quant_recipe {
            info!(
                "Quantizing weights: {quant_bits}-bit {quant_mode_str} (group_size={quant_group_size}, recipe={recipe})..."
            );
            let weight_keys: Vec<String> = weights.keys().cloned().collect();
            let predicate = crate::convert::build_predicate_for_recipe(
                recipe,
                &weight_keys,
                quant_bits,
                quant_group_size,
            )
            .map_err(Error::from_reason)?;
            per_layer_overrides = crate::convert::quantize_weights_with_recipe_pub(
                &mut weights,
                quant_bits,
                quant_group_size,
                &quant_mode_str,
                &*predicate,
            )?;
        } else {
            info!(
                "Quantizing weights: {quant_bits}-bit {quant_mode_str} (group_size={quant_group_size})..."
            );
            crate::convert::quantize_weights_pub(
                &mut weights,
                quant_bits,
                quant_group_size,
                &quant_mode_str,
            )?;
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
    let safetensors_filename = options
        .output_filename
        .as_deref()
        .unwrap_or("model.safetensors");
    let safetensors_path = output_dir.join(safetensors_filename);
    info!("Saving to {}", safetensors_path.display());
    // Add "format: mlx" metadata so loaders (e.g., mlx-vlm) know weights are
    // already in MLX layout and skip sanitize (which would double-apply +1.0 to norms).
    let st_metadata = serde_json::json!({ "format": "mlx" });
    save_safetensors(&safetensors_path, &weights, Some(st_metadata))?;

    // Only write config.json and tokenizer files for the primary model file.
    // Secondary files (e.g., vision.safetensors for mmproj) should not overwrite
    // the config written by the primary LLM conversion.
    let is_primary_model = safetensors_filename == "model.safetensors";

    if is_primary_model {
        // config.json: prefer real config from alongside the GGUF file, fallback to extraction
        let gguf_dir = input_path.parent().unwrap_or(Path::new("."));
        let config_path = output_dir.join("config.json");
        let src_config = gguf_dir.join("config.json");

        // Load or extract config, then inject quantization metadata if needed
        let mut config_json: serde_json::Value = if src_config.exists() {
            let data = fs::read_to_string(&src_config)
                .map_err(|e| Error::from_reason(format!("Failed to read config.json: {e}")))?;
            serde_json::from_str(&data)
                .map_err(|e| Error::from_reason(format!("Failed to parse config.json: {e}")))?
        } else {
            extract_config(&gguf.metadata)
        };

        if do_quantize {
            let mut quant_obj = serde_json::json!({
                "group_size": quant_group_size,
                "bits": quant_bits,
                "mode": quant_mode_str,
            });
            if let Some(obj) = quant_obj.as_object_mut() {
                for (path, override_val) in &per_layer_overrides {
                    obj.insert(super::normalize_override_key(path), override_val.clone());
                }
            }
            config_json["quantization"] = quant_obj.clone();
            config_json["quantization_config"] = quant_obj;
        }

        let config_str = serde_json::to_string_pretty(&config_json)
            .map_err(|e| Error::from_reason(format!("Failed to serialize config: {e}")))?;
        fs::write(&config_path, &config_str)
            .map_err(|e| Error::from_reason(format!("Failed to write config.json: {e}")))?;
        if do_quantize {
            info!("Wrote config.json with quantization metadata");
        } else if src_config.exists() {
            info!("Copied config.json from source directory");
        } else {
            info!("Wrote config.json (extracted from GGUF metadata)");
        }

        // Try to copy tokenizer files from alongside the GGUF file
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
        for filename in &tokenizer_files {
            let src = gguf_dir.join(filename);
            if src.exists() {
                let dst = output_dir.join(filename);
                if let Err(e) = fs::copy(&src, &dst) {
                    warn!("Failed to copy {filename}: {e}");
                } else {
                    info!("Copied {filename}");
                }
            } else {
                warn!(
                    "{filename} not found alongside GGUF file. You may need to download it separately."
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

    let tensor_names: Vec<String> = weights.keys().cloned().collect();

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

        let weights = load_gguf_tensors(&tmp, &gguf, false).unwrap();
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
        assert_eq!(config["head_dim"], 64); // 1024/16
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
