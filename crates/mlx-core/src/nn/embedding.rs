use crate::array::MxArray;
use napi::bindgen_prelude::*;

// ============================================
// Embedding Layer
// ============================================

pub struct Embedding {
    weight: MxArray,
    num_embeddings: u32,
    embedding_dim: u32,
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
        })
    }

    /// Forward pass: look up embeddings for indices
    pub fn forward(&self, indices: &MxArray) -> Result<MxArray> {
        // Use take operation to gather embeddings
        self.weight.take(indices, 0)
    }

    /// Load pretrained embeddings
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
        // Clone the Arc reference (no need to copy the underlying MLX array)
        self.weight = weight.clone();
        Ok(())
    }

    /// Get the embedding weight matrix
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
}

impl Clone for Embedding {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
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
        })
    }
}
