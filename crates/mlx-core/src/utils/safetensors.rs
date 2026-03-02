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
            SafeTensorDType::U32 => {
                // Load as uint32 - packed quantized weights
                let u32_data = bytes_to_u32(&buffer);
                MxArray::from_uint32(&u32_data, &shape)
            }
            _ => Err(Error::from_reason(format!(
                "Unsupported dtype for tensor {}: {:?}. Supported: F32, F16, BF16, I32, U8, U32, F8_E4M3",
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

/// Save tensors to SafeTensors format.
///
/// Uses a two-pass streaming approach to avoid materializing all tensor bytes
/// in memory at once — critical for large models (e.g. 27B params / 52 GB).
///
/// Pass 1: Compute byte sizes from array metadata (no evaluation), build header.
/// Pass 2: Write header, then stream each tensor's bytes to disk one at a time.
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, MxArray>,
    metadata: Option<serde_json::Value>,
) -> Result<()> {
    use std::io::{BufWriter, Write};

    // Sort tensor names for deterministic output
    let mut tensor_names: Vec<String> = tensors.keys().cloned().collect();
    tensor_names.sort();

    // --- Pass 1: Build header from metadata only (no tensor evaluation) ---
    let mut header = serde_json::Map::new();
    let mut current_offset = 0usize;

    for name in &tensor_names {
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

    // Stream each tensor: materialize → write → drop
    for name in &tensor_names {
        let array = tensors.get(name).unwrap();
        let bytes = array_to_bytes(array)?;
        writer
            .write_all(&bytes)
            .map_err(|e| Error::from_reason(format!("Failed to write tensor {}: {}", name, e)))?;
        // `bytes` is dropped here, freeing memory before the next tensor
    }

    writer
        .flush()
        .map_err(|e| Error::from_reason(format!("Failed to flush file: {}", e)))?;

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
}
