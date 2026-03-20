use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Embedding Layer (supports quantized weights)
// ============================================

pub struct Embedding {
    /// Dense (bf16) weight — always present. For quantized embeddings,
    /// `load_quantized()` pre-dequantizes the full table into this field.
    weight: MxArray,
    num_embeddings: u32,
    embedding_dim: u32,
    /// True when weights were loaded via `load_quantized()`.
    /// The packed weight/scales/biases are NOT retained — only the
    /// pre-dequantized dense table in `weight` is kept to avoid
    /// doubling memory for large vocab tables (248K × 4096 = ~2GB).
    is_quantized_flag: bool,
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
            is_quantized_flag: false,
        })
    }

    /// Forward pass: look up embeddings for indices.
    /// Always uses the dense weight (pre-dequantized for quantized embeddings).
    pub fn forward(&self, indices: &MxArray) -> Result<MxArray> {
        self.weight.take(indices, 0)
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
        self.is_quantized_flag = false;
        Ok(())
    }

    /// Load quantized embedding weights.
    /// Pre-dequantizes the full table into `self.weight` — the packed
    /// weight/scales/biases are NOT retained to save memory.
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
        self.is_quantized_flag = true;
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
        self.is_quantized_flag
    }
}

impl Clone for Embedding {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
            is_quantized_flag: self.is_quantized_flag,
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
            is_quantized_flag: false,
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
