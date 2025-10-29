use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================
// Embedding Layer
// ============================================

#[napi]
pub struct Embedding {
    weight: MxArray,
    num_embeddings: u32,
    embedding_dim: u32,
}

#[napi]
impl Embedding {
    /// Create a new Embedding layer
    #[napi(constructor)]
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
    #[napi]
    pub fn forward(&self, indices: &MxArray) -> Result<MxArray> {
        // Use take operation to gather embeddings
        self.weight.take(indices, 0)
    }

    /// Load pretrained embeddings
    #[napi]
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
    #[napi]
    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    /// Set the embedding weight matrix (alias for load_weight for consistency)
    #[napi]
    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.load_weight(weight)
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
