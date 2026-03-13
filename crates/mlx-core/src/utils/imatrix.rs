/// imatrix GGUF Parser
///
/// Parses Unsloth imatrix GGUF files containing per-channel importance scores
/// computed from calibration data. These scores are used for AWQ-style pre-scaling
/// to improve quantization quality.
///
/// File format: standard GGUF v3 with tensor pairs:
///   - `{name}.in_sum2` (F32, shape [input_channels]) — sum of squared activations
///   - `{name}.counts` (F32, shape [1]) — number of calibration tokens
///
/// Importance = in_sum2 / counts (per-channel average squared activation magnitude)
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use napi::bindgen_prelude::*;
use tracing::info;

use super::gguf::gguf_name_to_hf;

// Reuse GGUF constants
const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION_3: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// GGUF value type IDs we need for metadata parsing
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;

/// Parsed imatrix data with per-weight importance scores.
pub struct ImatrixData {
    /// Per-weight importance scores: HF key → Vec<f32> (normalized per-channel importance).
    /// Keys use HuggingFace naming (e.g., "model.layers.0.mlp.gate_proj.weight").
    pub importance: HashMap<String, Vec<f32>>,
    pub chunk_count: u32,
    pub chunk_size: u32,
}

/// Tensor info from the imatrix GGUF file header
struct ImatrixTensorInfo {
    name: String,
    n_elements: u64,
    offset: u64,
}

// ── Binary Reader Helpers ───────────────────────────────────────────────────

fn read_u32_le(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32_le(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_gguf_string(r: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Skip a GGUF metadata value (we only need a few specific keys)
fn skip_meta_value(r: &mut (impl Read + Seek), vtype: u32) -> std::io::Result<()> {
    match vtype {
        0 | 7 => {
            // Uint8, Bool
            r.seek(SeekFrom::Current(1))?;
        }
        1 => {
            // Int8
            r.seek(SeekFrom::Current(1))?;
        }
        2 | 3 => {
            // Uint16, Int16
            r.seek(SeekFrom::Current(2))?;
        }
        4..=6 => {
            // Uint32, Int32, Float32
            r.seek(SeekFrom::Current(4))?;
        }
        10..=12 => {
            // Uint64, Int64, Float64
            r.seek(SeekFrom::Current(8))?;
        }
        8 => {
            // String
            let len = read_u64_le(r)? as i64;
            r.seek(SeekFrom::Current(len))?;
        }
        9 => {
            // Array
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            for _ in 0..count {
                skip_meta_value(r, elem_type)?;
            }
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown GGUF value type: {vtype}"),
            ));
        }
    }
    Ok(())
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

/// Parse an imatrix GGUF file, returning per-weight importance scores.
///
/// Keys in the returned HashMap use HuggingFace naming convention
/// (e.g., "model.layers.0.mlp.gate_proj.weight"), converted via `gguf_name_to_hf()`.
pub fn parse_imatrix<P: AsRef<Path>>(path: P) -> Result<ImatrixData> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .map_err(|e| Error::from_reason(format!("Failed to open imatrix file: {e}")))?;
    let mut reader = BufReader::new(file);

    // Read GGUF header
    let magic = read_u32_le(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read imatrix magic: {e}")))?;
    if magic != GGUF_MAGIC {
        return Err(Error::from_reason(format!(
            "Not a GGUF file (magic: 0x{magic:08X}, expected: 0x{GGUF_MAGIC:08X})"
        )));
    }

    let version = read_u32_le(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read imatrix version: {e}")))?;
    if version < GGUF_VERSION_3 {
        return Err(Error::from_reason(format!(
            "Unsupported GGUF version {version} (only v3+ supported)"
        )));
    }

    let tensor_count = read_u64_le(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read tensor count: {e}")))?;
    let metadata_kv_count = read_u64_le(&mut reader)
        .map_err(|e| Error::from_reason(format!("Failed to read metadata count: {e}")))?;

    // Read metadata — extract imatrix-specific fields
    let mut chunk_count: u32 = 0;
    let mut chunk_size: u32 = 0;
    let mut alignment: u64 = GGUF_DEFAULT_ALIGNMENT;

    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read metadata key: {e}")))?;
        let vtype = read_u32_le(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read value type: {e}")))?;

        match key.as_str() {
            "imatrix.chunk_count" => {
                chunk_count = match vtype {
                    GGUF_TYPE_UINT32 => read_u32_le(&mut reader).unwrap_or(0),
                    GGUF_TYPE_INT32 => read_u32_le(&mut reader).unwrap_or(0),
                    GGUF_TYPE_UINT64 => read_u64_le(&mut reader).unwrap_or(0) as u32,
                    GGUF_TYPE_INT64 => read_u64_le(&mut reader).unwrap_or(0) as u32,
                    _ => {
                        skip_meta_value(&mut reader, vtype).map_err(|e| {
                            Error::from_reason(format!("Failed to skip metadata: {e}"))
                        })?;
                        0
                    }
                };
            }
            "imatrix.chunk_size" => {
                chunk_size = match vtype {
                    GGUF_TYPE_UINT32 => read_u32_le(&mut reader).unwrap_or(0),
                    GGUF_TYPE_INT32 => read_u32_le(&mut reader).unwrap_or(0),
                    GGUF_TYPE_UINT64 => read_u64_le(&mut reader).unwrap_or(0) as u32,
                    GGUF_TYPE_INT64 => read_u64_le(&mut reader).unwrap_or(0) as u32,
                    _ => {
                        skip_meta_value(&mut reader, vtype).map_err(|e| {
                            Error::from_reason(format!("Failed to skip metadata: {e}"))
                        })?;
                        0
                    }
                };
            }
            "general.alignment" => {
                alignment = match vtype {
                    GGUF_TYPE_UINT32 => read_u32_le(&mut reader).unwrap_or(32) as u64,
                    GGUF_TYPE_INT32 => read_u32_le(&mut reader).unwrap_or(32) as u64,
                    GGUF_TYPE_UINT64 => read_u64_le(&mut reader).unwrap_or(32),
                    GGUF_TYPE_INT64 => read_u64_le(&mut reader).unwrap_or(32) as u64,
                    _ => {
                        skip_meta_value(&mut reader, vtype).map_err(|e| {
                            Error::from_reason(format!("Failed to skip metadata: {e}"))
                        })?;
                        GGUF_DEFAULT_ALIGNMENT
                    }
                };
            }
            _ => {
                skip_meta_value(&mut reader, vtype).map_err(|e| {
                    Error::from_reason(format!("Failed to skip metadata value for '{key}': {e}"))
                })?;
            }
        }
    }

    // Read tensor info descriptors
    let mut tensor_infos: Vec<ImatrixTensorInfo> = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor name: {e}")))?;
        let n_dims = read_u32_le(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor n_dims: {e}")))?;

        let mut n_elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = read_u64_le(&mut reader)
                .map_err(|e| Error::from_reason(format!("Failed to read tensor dim: {e}")))?;
            n_elements *= dim;
        }

        let type_u32 = read_u32_le(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor type: {e}")))?;
        // GGUF tensor type F32 = 0. Imatrix data is always F32.
        if type_u32 != 0 {
            return Err(Error::from_reason(format!(
                "imatrix tensor '{name}' has unexpected type {type_u32} (expected F32=0)"
            )));
        }

        let offset = read_u64_le(&mut reader)
            .map_err(|e| Error::from_reason(format!("Failed to read tensor offset: {e}")))?;

        tensor_infos.push(ImatrixTensorInfo {
            name,
            n_elements,
            offset,
        });
    }

    // Compute data offset (aligned)
    let current_pos = reader
        .stream_position()
        .map_err(|e| Error::from_reason(format!("Failed to get stream position: {e}")))?;
    let data_offset = align_offset(current_pos, alignment);

    // Read tensor data and build importance map
    // Tensors come in pairs: {name}.in_sum2 and {name}.counts
    let mut sum2_map: HashMap<String, Vec<f32>> = HashMap::new();
    let mut counts_map: HashMap<String, f32> = HashMap::new();

    for ti in &tensor_infos {
        let abs_offset = data_offset + ti.offset;
        reader.seek(SeekFrom::Start(abs_offset)).map_err(|e| {
            Error::from_reason(format!("Failed to seek to tensor '{}': {e}", ti.name))
        })?;

        let mut data = vec![0f32; ti.n_elements as usize];
        for val in &mut data {
            *val = read_f32_le(&mut reader).map_err(|e| {
                Error::from_reason(format!("Failed to read tensor '{}' data: {e}", ti.name))
            })?;
        }

        if ti.name.ends_with(".in_sum2") {
            let base = ti.name.strip_suffix(".in_sum2").unwrap().to_string();
            sum2_map.insert(base, data);
        } else if ti.name.ends_with(".counts") {
            let base = ti.name.strip_suffix(".counts").unwrap().to_string();
            counts_map.insert(base, data[0]);
        }
    }

    // Compute importance = in_sum2 / counts for each weight, keyed by HF name
    let mut importance: HashMap<String, Vec<f32>> = HashMap::new();
    for (gguf_name, sum2) in &sum2_map {
        let counts = match counts_map.get(gguf_name) {
            Some(&c) if c > 0.0 => c,
            _ => continue,
        };

        let imp: Vec<f32> = sum2.iter().map(|&s| s / counts).collect();

        // Convert GGUF name to HF name:
        // imatrix tensors are named like "blk.0.ffn_gate.weight"
        // gguf_name_to_hf expects the name without the ".weight" suffix already stripped?
        // Actually gguf_name_to_hf maps "blk.0.ffn_gate.weight" -> "model.layers.0.mlp.gate_proj.weight"
        let hf_name = gguf_name_to_hf(gguf_name);
        importance.insert(hf_name, imp);
    }

    info!(
        "Parsed imatrix: {} weights, chunk_count={}, chunk_size={}",
        importance.len(),
        chunk_count,
        chunk_size
    );

    Ok(ImatrixData {
        importance,
        chunk_count,
        chunk_size,
    })
}
