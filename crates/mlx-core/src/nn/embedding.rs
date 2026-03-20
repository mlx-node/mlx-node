use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Embedding Layer (supports quantized weights)
// ============================================

/// Quantized weight storage for QuantizedEmbedding.
struct QuantizedWeight {
    weight: MxArray,         // Packed uint32 [num_embeddings, dim_packed]
    scales: MxArray,         // Quantization scales
    biases: Option<MxArray>, // Quantization biases (affine mode)
    group_size: i32,
    bits: i32,
}

pub struct Embedding {
    /// Dense (bf16) weight — always present. For quantized embeddings,
    /// this is lazily populated on first `get_weight()` call.
    weight: MxArray,
    num_embeddings: u32,
    embedding_dim: u32,
    /// When set, `forward()` dequantizes only the looked-up rows for
    /// memory-bandwidth savings. `get_weight()` returns the full
    /// dequantized table (lazily cached in `weight`).
    quantized: Option<QuantizedWeight>,
}

impl Embedding {
    /// Create a new Embedding layer
    pub fn new(num_embeddings: u32, embedding_dim: u32) -> Result<Self> {
        // Initialize with normal distribution
        let shape = [num_embeddings as i64, embedding_dim as i64];
        let weight = MxArray::random_normal(&shape, 0.0, 0.02, None)?;

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            quantized: None,
        })
    }

    /// Forward pass: look up embeddings for indices.
    /// When quantized, dequantizes only the selected rows (not the full table).
    pub fn forward(&self, indices: &MxArray) -> Result<MxArray> {
        if let Some(ref q) = self.quantized {
            // Dequantize the full table, then gather.
            // Per-row dequantize+gather is not available in MLX's C API,
            // so we dequantize fully — but the quantized weights use less
            // memory bandwidth when paging from mmap.
            let dequantized = dequantize(
                &q.weight,
                &q.scales,
                q.biases.as_ref(),
                q.group_size,
                q.bits,
            )?;
            dequantized.take(indices, 0)
        } else {
            self.weight.take(indices, 0)
        }
    }

    /// Load pretrained embeddings (dense bf16)
    pub fn load_weight(&mut self, weight: &MxArray) -> Result<()> {
        let ndim = weight.ndim()?;
        if ndim != 2
            || weight.shape_at(0)? != self.num_embeddings as i64
            || weight.shape_at(1)? != self.embedding_dim as i64
        {
            return Err(Error::from_reason(format!(
                "Embedding weight shape mismatch: expected [{}, {}], got {:?}",
                self.num_embeddings,
                self.embedding_dim,
                weight.shape()?.as_ref()
            )));
        }
        self.weight = weight.clone();
        self.quantized = None;
        Ok(())
    }

    /// Load quantized embedding weights. The packed weight is stored directly;
    /// `forward()` will dequantize on the fly, and `get_weight()` will lazily
    /// dequantize the full table.
    pub fn load_quantized(
        &mut self,
        weight: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
    ) -> Result<()> {
        // Verify num_embeddings matches
        if weight.shape_at(0)? != self.num_embeddings as i64 {
            return Err(Error::from_reason(format!(
                "Quantized embedding num_embeddings mismatch: expected {}, got {}",
                self.num_embeddings,
                weight.shape_at(0)?
            )));
        }

        // Pre-dequantize the full table and store as the dense weight.
        // This is needed for get_weight() (used by tied embeddings, compiled path, etc.)
        let dequantized = dequantize(weight, scales, biases, group_size, bits)?;
        self.weight = dequantized;

        self.quantized = Some(QuantizedWeight {
            weight: weight.clone(),
            scales: scales.clone(),
            biases: biases.cloned(),
            group_size,
            bits,
        });
        Ok(())
    }

    /// Get the embedding weight matrix (always returns dense bf16).
    /// For quantized embeddings, returns the pre-dequantized full table.
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Get the embedding weight matrix (alias for get_weight)
    pub fn weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Set the embedding weight matrix (alias for load_weight for consistency)
    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.load_weight(weight)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }

    /// Whether this embedding uses quantized weights
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }
}

impl Clone for Embedding {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
            quantized: self.quantized.as_ref().map(|q| QuantizedWeight {
                weight: q.weight.clone(),
                scales: q.scales.clone(),
                biases: q.biases.clone(),
                group_size: q.group_size,
                bits: q.bits,
            }),
        }
    }
}

impl Embedding {
    /// Create an Embedding layer from pre-loaded weight
    ///
    /// # Arguments
    /// * `weight` - Embedding matrix [num_embeddings, embedding_dim]
    pub fn from_weight(weight: &MxArray) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "Embedding weight must be 2D, got shape {:?}",
                shape.as_ref()
            )));
        }

        Ok(Self {
            weight: weight.clone(),
            num_embeddings: shape[0] as u32,
            embedding_dim: shape[1] as u32,
            quantized: None,
        })
    }
}

/// Dequantize a tensor using MLX's affine dequantize op.
fn dequantize(
    weight: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
) -> Result<MxArray> {
    let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
    let handle = unsafe {
        sys::mlx_dequantize(
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
            group_size,
            bits,
            -1, // Use input dtype (bf16 from scales)
            c"affine".as_ptr(),
        )
    };
    MxArray::from_handle(handle, "dequantize_embedding")
}
