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
pub enum SafeTensorDType {
    F32,  // float32
    F16,  // float16
    BF16, // bfloat16
    I32,  // int32
    I64,  // int64
    U8,   // uint8
    I8,   // int8
    F64,  // float64
    BOOL, // boolean
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
            SafeTensorDType::I8 => 1,
            SafeTensorDType::F64 => 8,
            SafeTensorDType::BOOL => 1,
        }
    }

    /// Convert to MLX DType if supported
    pub fn to_mlx_dtype(&self) -> Option<DType> {
        match self {
            SafeTensorDType::F32 => Some(DType::Float32),
            SafeTensorDType::F16 => Some(DType::Float16),
            SafeTensorDType::BF16 => Some(DType::BFloat16),
            SafeTensorDType::I32 => Some(DType::Int32),
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
                // Load as f32 then convert to f16 to preserve original dtype
                let float_data = f16_bytes_to_f32(&buffer);
                let f32_array = MxArray::from_float32(&float_data, &shape)?;
                f32_array.astype(DType::Float16)
            }
            SafeTensorDType::BF16 => {
                // Load as f32 then convert to bf16 for better performance
                let float_data = bf16_bytes_to_f32(&buffer);
                let f32_array = MxArray::from_float32(&float_data, &shape)?;
                // Convert to bf16 to match original dtype
                f32_array.astype(DType::BFloat16)
            }
            SafeTensorDType::I32 => {
                // Convert bytes to i32 array
                let int_data = bytes_to_i32(&buffer);
                MxArray::from_int32(&int_data, &shape)
            }
            _ => Err(Error::from_reason(format!(
                "Unsupported dtype for tensor {}: {:?}. Supported: F32, F16, BF16, I32",
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

/// Convert f16 bytes to f32 (IEEE 754 half precision to single precision)
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert bf16 bytes to f32 (bfloat16 to single precision)
/// bfloat16 is just f32 with the lower 16 bits truncated
fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            // BF16 is just the upper 16 bits of f32
            let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            // Shift left 16 bits to get f32 representation
            let f32_bits = (bf16_bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

// ============================================================================
// SafeTensors Writer
// ============================================================================

/// Save tensors to SafeTensors format
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, MxArray>,
    metadata: Option<serde_json::Value>,
) -> Result<()> {
    use std::io::Write;

    // Build header with tensor metadata
    let mut header = serde_json::Map::new();
    let mut current_offset = 0usize;
    let mut tensor_data_vec: Vec<(String, Vec<u8>)> = Vec::new();

    // Sort tensor names for deterministic output
    let mut tensor_names: Vec<String> = tensors.keys().cloned().collect();
    tensor_names.sort();

    for name in tensor_names {
        let array = tensors.get(&name).unwrap();

        // Get array metadata
        let shape = array.shape()?;
        let shape_vec: Vec<usize> = shape.as_ref().iter().map(|&x| x as usize).collect();
        let dtype = array.dtype()?;

        // Convert array to bytes
        let bytes = array_to_bytes(array)?;
        let byte_size = bytes.len();

        // Create tensor info
        let tensor_info = serde_json::json!({
            "dtype": dtype_to_safetensor_str(dtype),
            "shape": shape_vec,
            "data_offsets": [current_offset, current_offset + byte_size]
        });

        header.insert(name.clone(), tensor_info);
        tensor_data_vec.push((name, bytes));
        current_offset += byte_size;
    }

    // Add metadata if provided
    if let Some(meta) = metadata {
        header.insert("__metadata__".to_string(), meta);
    }

    // Serialize header to JSON
    let header_json = serde_json::to_string(&header)
        .map_err(|e| Error::from_reason(format!("Failed to serialize header: {}", e)))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Write to file
    let mut file = File::create(path.as_ref())
        .map_err(|e| Error::from_reason(format!("Failed to create file: {}", e)))?;

    // Write header length (8 bytes, little-endian)
    file.write_all(&header_len.to_le_bytes())
        .map_err(|e| Error::from_reason(format!("Failed to write header length: {}", e)))?;

    // Write header JSON
    file.write_all(header_bytes)
        .map_err(|e| Error::from_reason(format!("Failed to write header: {}", e)))?;

    // Write tensor data
    for (_name, bytes) in tensor_data_vec {
        file.write_all(&bytes)
            .map_err(|e| Error::from_reason(format!("Failed to write tensor data: {}", e)))?;
    }

    file.flush()
        .map_err(|e| Error::from_reason(format!("Failed to flush file: {}", e)))?;

    Ok(())
}

/// Convert MxArray to bytes for SafeTensors format
fn array_to_bytes(array: &MxArray) -> Result<Vec<u8>> {
    let dtype = array.dtype()?;

    match dtype {
        DType::Float32 => {
            let data = array.to_float32()?;
            Ok(f32_to_bytes(&data))
        }
        DType::Float16 => {
            // Convert to f16 bytes
            let data = array.to_float32()?; // Get as f32 first
            Ok(f32_to_f16_bytes(&data))
        }
        DType::BFloat16 => {
            // Convert to bf16 bytes
            let data = array.to_float32()?;
            Ok(f32_to_bf16_bytes(&data))
        }
        DType::Int32 => {
            let data = array.to_int32()?;
            Ok(i32_to_bytes(&data))
        }
        DType::Uint32 => {
            let data = array.to_uint32()?;
            Ok(data.iter().flat_map(|&x| x.to_le_bytes()).collect())
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

/// Convert f32 array to f16 bytes
fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&x| {
            let f16_val = half::f16::from_f32(x);
            f16_val.to_le_bytes()
        })
        .collect()
}

/// Convert f32 array to bf16 bytes
fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&x| {
            // BF16 is just the upper 16 bits of f32
            let f32_bits = x.to_bits();
            let bf16_bits = (f32_bits >> 16) as u16;
            bf16_bits.to_le_bytes()
        })
        .collect()
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
