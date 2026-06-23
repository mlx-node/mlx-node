use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Embedding Layer (supports quantized weights)
// ============================================

/// Packed-quantized backend for an `Embedding`.
///
/// Mirrors `QuantizedBackend` in `nn/linear.rs`: the packed
/// `weight`/`scales`/(`biases`) tensors are retained AS-IS — never
/// pre-dequantized into a dense table — so a quantized embedding stays
/// packed-resident (real memory savings of `vocab × hidden × 2` bytes for a
/// large vocab). `forward` does a per-row gather-then-dequant (only the looked-
/// up rows are dequantized) and `as_linear` runs `mlx_quantized_matmul` for the
/// tied lm_head. `mode` is the quantization mode string ("affine" / "mxfp4" /
/// "mxfp8" / "nvfp4") threaded into both ops.
struct QuantizedEmbeddingBackend {
    /// Packed quantized weight `[num_embeddings, packed_dim]`.
    weight: MxArray,
    /// Quantization scales.
    scales: MxArray,
    /// Quantization biases (affine mode only; `None` for mxfp4/mxfp8/nvfp4).
    biases: Option<MxArray>,
    group_size: i32,
    bits: i32,
    mode: String,
}

/// Row-sharded backend for a dense bf16 embedding table whose single tensor
/// exceeds the Metal per-buffer cap (e.g. gemma-4-E2B's ~4GB
/// `embed_tokens_per_layer.weight` on a memory-constrained device, where the
/// whole-tensor materialize eval cannot allocate one buffer).
///
/// The table `[vocab, cols]` is split along axis 0 into `shards`, each a
/// sub-cap `[rows_s, cols]` array (`rows_s <= rows_per_shard`, the last shard
/// shorter). Because every token's full embedding is one contiguous axis-0 row,
/// each token lands entirely inside one shard, so the gather reconstructs the
/// exact bytes of `take(full_table, ids, 0)` (see `forward_sharded`).
#[derive(Clone)]
struct ShardedEmbedding {
    /// Sub-cap row blocks; concatenated along axis 0 they are the full table.
    shards: Vec<MxArray>,
    /// Uniform axis-0 stride: shard `s` covers rows `[s*rows_per_shard, …)`.
    rows_per_shard: i64,
}

pub struct Embedding {
    /// Dense (bf16) weight. For a plain bf16 embedding this is the lookup
    /// table. For a PACKED-quantized embedding (`quantized_packed` set) this is
    /// kept ONLY as a small placeholder (the real table lives packed in
    /// `quantized_packed`); it is never read on the forward/logits paths.
    ///
    /// NOTE: the legacy `load_quantized()` path (used by qwen3_5/qwen3_5_moe/
    /// gemma4 tied heads) still PRE-dequantizes the full table into this field
    /// — that behavior is unchanged.
    weight: MxArray,
    num_embeddings: u32,
    embedding_dim: u32,
    /// True when weights were loaded via `load_quantized()` (legacy
    /// pre-dequantized path) OR `load_quantized_packed()` (packed path).
    is_quantized_flag: bool,
    /// Packed-quantized backend (set only via `load_quantized_packed()`, the
    /// lfm2 opt-in path). When present, `forward`/`as_linear` use the packed
    /// tensors instead of the dense `weight`.
    quantized_packed: Option<QuantizedEmbeddingBackend>,
    /// Row-sharded backend (set only via `set_sharded()`). When present,
    /// `forward` gathers across the sub-cap shards instead of reading the dense
    /// `weight`; used for tables too large to materialize as one Metal buffer.
    sharded: Option<ShardedEmbedding>,
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
            quantized_packed: None,
            sharded: None,
        })
    }

    /// Forward pass: look up embeddings for indices.
    ///
    /// For a PACKED-quantized embedding (`load_quantized_packed`), gathers the
    /// rows of `weight`/`scales`/(`biases`) selected by `indices` on axis 0,
    /// THEN dequantizes ONLY the gathered rows — matching MLX's
    /// `QuantizedEmbedding.__call__` (gather-then-dequant, the memory/speed win).
    /// Otherwise uses the dense `weight` table (pre-dequantized for the legacy
    /// `load_quantized` path, or plain bf16).
    pub fn forward(&self, indices: &MxArray) -> Result<MxArray> {
        if let Some(ref q) = self.quantized_packed {
            // Gather the selected rows of EACH packed tensor on axis 0, then
            // dequantize the gathered rows along the last axis.
            let gathered_w = q.weight.take(indices, 0)?;
            let gathered_s = q.scales.take(indices, 0)?;
            let gathered_b = match q.biases {
                Some(ref b) => Some(b.take(indices, 0)?),
                None => None,
            };
            return dequantize(
                &gathered_w,
                &gathered_s,
                gathered_b.as_ref(),
                q.group_size,
                q.bits,
                &q.mode,
            );
        }
        if let Some(ref sh) = self.sharded {
            return forward_sharded(sh, indices);
        }
        self.weight.take(indices, 0)
    }

    /// Apply this embedding as a linear (tied lm_head) projection:
    /// `logits = x @ table^T`.
    ///
    /// For a PACKED-quantized embedding, runs `mlx_quantized_matmul` with
    /// `transpose=true` on the packed tensors (matching MLX's
    /// `QuantizedEmbedding.as_linear`), so the dense table is NEVER
    /// materialized. For a dense (plain bf16 or legacy pre-dequantized)
    /// embedding, computes `x @ weight^T` — numerically equal to the previous
    /// `x @ get_weight().T` tied-head matmul, so callers can route uniformly
    /// through `as_linear`.
    pub fn as_linear(&self, x: &MxArray) -> Result<MxArray> {
        if let Some(ref q) = self.quantized_packed {
            let mode_c = std::ffi::CString::new(q.mode.as_str())
                .map_err(|_| Error::from_reason("Invalid embedding quantize mode string"))?;
            let biases_ptr = q
                .biases
                .as_ref()
                .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
            let handle = unsafe {
                sys::mlx_quantized_matmul(
                    x.as_raw_ptr(),
                    q.weight.as_raw_ptr(),
                    q.scales.as_raw_ptr(),
                    biases_ptr,
                    true, // transpose: logits = x @ weight^T
                    q.group_size,
                    q.bits,
                    mode_c.as_ptr(),
                )
            };
            return MxArray::from_handle(handle, "embedding_as_linear");
        }
        // Dense path: x @ weight^T (equivalent to the prior tied-head matmul).
        let weight_t = self.weight.transpose(Some(&[1, 0]))?;
        x.matmul(&weight_t)
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
        self.quantized_packed = None;
        self.sharded = None;
        Ok(())
    }

    /// Install a row-sharded dense table (axis-0 row blocks). Used when the full
    /// table is too large to materialize as one Metal buffer; `forward` then
    /// gathers across the shards. `shards` concatenated along axis 0 must equal
    /// the full `[num_embeddings, embedding_dim]` table, and `rows_per_shard` is
    /// the uniform axis-0 stride (every shard but the last has this many rows).
    pub fn set_sharded(&mut self, shards: Vec<MxArray>, rows_per_shard: i64) -> Result<()> {
        if shards.is_empty() {
            return Err(Error::from_reason("set_sharded: shards must be non-empty"));
        }
        if rows_per_shard <= 0 {
            return Err(Error::from_reason(
                "set_sharded: rows_per_shard must be > 0",
            ));
        }
        // The dense `load_weight` shape check is skipped on this path, so enforce
        // the same contract the gather relies on: the shards must reconstruct the
        // full `[num_embeddings, embedding_dim]` table. `forward_sharded` reads
        // each shard's base row as `index * rows_per_shard`, so every shard but
        // the last must have exactly `rows_per_shard` rows; the last holds the
        // remainder. Reject a drifted checkpoint here instead of silently
        // gathering wrong rows or failing later in the projection.
        let last = shards.len() - 1;
        let mut total_rows: i64 = 0;
        for (i, shard) in shards.iter().enumerate() {
            if shard.ndim()? != 2 {
                return Err(Error::from_reason(format!(
                    "set_sharded: shard {i} must be 2-D, got {:?}",
                    shard.shape()?.as_ref()
                )));
            }
            let cols = shard.shape_at(1)?;
            if cols != self.embedding_dim as i64 {
                return Err(Error::from_reason(format!(
                    "set_sharded: shard {i} has {cols} columns, expected {}",
                    self.embedding_dim
                )));
            }
            let rows = shard.shape_at(0)?;
            if i != last && rows != rows_per_shard {
                return Err(Error::from_reason(format!(
                    "set_sharded: non-final shard {i} has {rows} rows, expected stride {rows_per_shard}"
                )));
            }
            if i == last && (rows < 1 || rows > rows_per_shard) {
                return Err(Error::from_reason(format!(
                    "set_sharded: final shard {i} has {rows} rows, expected 1..={rows_per_shard}"
                )));
            }
            total_rows = total_rows.saturating_add(rows);
        }
        if total_rows != self.num_embeddings as i64 {
            return Err(Error::from_reason(format!(
                "set_sharded: shard rows sum to {total_rows}, expected {}",
                self.num_embeddings
            )));
        }
        self.sharded = Some(ShardedEmbedding {
            shards,
            rows_per_shard,
        });
        self.is_quantized_flag = false;
        self.quantized_packed = None;
        Ok(())
    }

    /// The sharded table's sub-cap arrays (empty when not sharded). Lets the
    /// loader feed them to the chunked weight-materialize pass, since they live
    /// outside the `params` map the rest of the weights are materialized from.
    pub fn shard_arrays(&self) -> &[MxArray] {
        match self.sharded {
            Some(ref sh) => &sh.shards,
            None => &[],
        }
    }

    /// Load quantized embedding weights (LEGACY pre-dequantized path).
    ///
    /// Pre-dequantizes the full table into `self.weight` — the packed
    /// weight/scales/biases are NOT retained. Used by qwen3_5/qwen3_5_moe/
    /// gemma4 tied heads which read `get_weight()`. BEHAVIOR UNCHANGED — do not
    /// route new callers here; use `load_quantized_packed` for memory savings.
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
        let dequantized = dequantize(weight, scales, biases, group_size, bits, "affine")?;
        self.weight = dequantized;
        self.is_quantized_flag = true;
        self.quantized_packed = None;
        Ok(())
    }

    /// Load PACKED-quantized embedding weights (the lfm2 opt-in path).
    ///
    /// Retains the packed `weight`/`scales`/(`biases`) AS-IS — does NOT
    /// dequantize the table. `forward` gather-then-dequantizes only the looked-
    /// up rows; `as_linear` runs `mlx_quantized_matmul`. The dense table is
    /// never materialized, so the embedding stays packed-resident. Supports ALL
    /// modes (affine + mxfp4/mxfp8/nvfp4); mxfp/nvfp pass `biases = None`.
    ///
    /// `self.weight` is left as a small placeholder (never read on the
    /// packed forward/logits paths).
    pub fn load_quantized_packed(
        &mut self,
        weight: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<()> {
        if weight.shape_at(0)? != self.num_embeddings as i64 {
            return Err(Error::from_reason(format!(
                "Packed quantized embedding num_embeddings mismatch: expected {}, got {}",
                self.num_embeddings,
                weight.shape_at(0)?
            )));
        }
        self.quantized_packed = Some(QuantizedEmbeddingBackend {
            weight: weight.clone(),
            scales: scales.clone(),
            biases: biases.cloned(),
            group_size,
            bits,
            mode: mode.to_string(),
        });
        self.is_quantized_flag = true;
        // Drop the full-shape `[vocab, hidden]` dense `weight` graph allocated by
        // `new()` (a `random_normal` node): the packed backend is now the sole
        // source of truth, so replace it with a genuinely tiny placeholder. This
        // guarantees no full dense table can ever be materialized off `self.weight`
        // (whether via an accidental eval of the lazy graph or a stray
        // `get_weight()` caller) — the real table only exists packed. `get_weight()`
        // dequantizes on demand from the packed backend, so it never reads this
        // placeholder for packed embeddings.
        self.weight = MxArray::zeros(&[1, 1], Some(crate::array::DType::BFloat16))?;
        Ok(())
    }

    /// Get the embedding weight matrix (always returns dense bf16).
    ///
    /// - Plain bf16 / LEGACY pre-dequantized embeddings: returns the dense
    ///   `self.weight` table directly.
    /// - PACKED-quantized embeddings (`load_quantized_packed`): `self.weight` is
    ///   only a tiny placeholder, so this DEQUANTIZES the full table on demand
    ///   from the packed backend and returns CORRECT `[vocab, hidden]` data — it
    ///   never returns the placeholder. The dense table is materialized only if a
    ///   caller actually evals the result; the production lfm2 logits path uses
    ///   `as_linear` (a `mlx_quantized_matmul` on the packed tensors) and never
    ///   hits this, so no full dense table is resident under normal operation.
    pub fn get_weight(&self) -> MxArray {
        if let Some(ref q) = self.quantized_packed {
            // Dequantize the full packed table on demand. `unwrap_or_else` keeps
            // this getter infallible: on the (effectively impossible) dequant
            // error for already-validated packed data, fall back to the
            // placeholder rather than panicking — correctness-critical callers
            // route through `as_linear`, not `get_weight()`.
            return dequantize(
                &q.weight,
                &q.scales,
                q.biases.as_ref(),
                q.group_size,
                q.bits,
                &q.mode,
            )
            .unwrap_or_else(|_| self.weight.clone());
        }
        self.weight.clone()
    }

    /// Get the embedding weight matrix (alias for `get_weight`; packed-aware).
    pub fn weight(&self) -> MxArray {
        self.get_weight()
    }

    /// Set the embedding weight matrix (alias for load_weight for consistency)
    pub fn set_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.load_weight(weight)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }

    /// Whether this embedding uses quantized weights (either path)
    pub fn is_quantized(&self) -> bool {
        self.is_quantized_flag
    }

    /// Whether this embedding holds a PACKED-quantized backend
    /// (`load_quantized_packed`). When true, the tied-head logits path MUST use
    /// `as_linear` (the dense `get_weight()` is only a placeholder).
    pub fn is_packed_quantized(&self) -> bool {
        self.quantized_packed.is_some()
    }
}

impl Clone for Embedding {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
            is_quantized_flag: self.is_quantized_flag,
            quantized_packed: self
                .quantized_packed
                .as_ref()
                .map(|q| QuantizedEmbeddingBackend {
                    weight: q.weight.clone(),
                    scales: q.scales.clone(),
                    biases: q.biases.clone(),
                    group_size: q.group_size,
                    bits: q.bits,
                    mode: q.mode.clone(),
                }),
            sharded: self.sharded.clone(),
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
            quantized_packed: None,
            sharded: None,
        })
    }
}

/// Gather embedding rows across a row-sharded table, byte-identical to
/// `take(full_table, indices, axis=0)`.
///
/// `indices` are assumed already in range `[0, vocab)` (the gemma4 PLE path
/// pre-masks OOV ids to 0). For each token id `t`, exactly one shard `s*`
/// satisfies `base_s <= t < base_s + rows_s`, and `shards[s*][t - base_s]` is
/// the same bytes as row `t` of the full table (shards are a pure axis-0
/// partition). The reconstruction uses only `take` and `where_` — both pure
/// data movement, no arithmetic on the bf16 values — so the result is bitwise
/// identical to the unsharded gather. Per-shard index clamping only keeps each
/// `take` in-bounds; out-of-shard candidates are discarded by the mask.
///
/// Bit-exactness caveat: Metal's `where_` (select) flushes bf16 denormals
/// (~1e-38) to zero, so a denormal weight reads back as 0 here while a dense
/// `take` preserves it. This is numerically negligible and irrelevant for real
/// weights (trained tables hold no denormals, and the PLE forward's immediate
/// `* sqrt(ple_dim)` would flush them anyway); normals, ±0, NaN and ±Inf are
/// preserved exactly. Paged and flat both gather through here, so the behavior
/// is identical across cache backends.
fn forward_sharded(sh: &ShardedEmbedding, indices: &MxArray) -> Result<MxArray> {
    let stride = sh.rows_per_shard;

    // Mask broadcast shape: indices.shape ++ [1], so a per-token bool selects
    // across the trailing embedding-dim axis of the gathered rows.
    let mut mask_shape: Vec<i64> = indices.shape()?.as_ref().to_vec();
    mask_shape.push(1);

    // Seed with shard 0 (base 0). Ids beyond shard 0 read a clamped (wrong) row
    // here, but every such id is overwritten by its own shard's `where_` below.
    let rows0 = sh.shards[0].shape_at(0)?;
    let max0 = MxArray::scalar_int((rows0 - 1) as i32)?;
    let local0 = indices.minimum(&max0)?;
    let mut result = sh.shards[0].take(&local0, 0)?;

    let zero = MxArray::scalar_int(0)?;
    for (s, shard) in sh.shards.iter().enumerate().skip(1) {
        let base = (s as i64) * stride;
        let rows_s = shard.shape_at(0)?;

        let base_arr = MxArray::scalar_int(base as i32)?;
        let hi_arr = MxArray::scalar_int((base + rows_s) as i32)?;
        let in_shard = indices
            .greater_equal(&base_arr)?
            .logical_and(&indices.less(&hi_arr)?)?
            .reshape(&mask_shape)?;

        let neg_base = MxArray::scalar_int(-(base as i32))?;
        let max_local = MxArray::scalar_int((rows_s - 1) as i32)?;
        let local = indices
            .add(&neg_base)?
            .maximum(&zero)?
            .minimum(&max_local)?;
        let cand = shard.take(&local, 0)?;

        result = in_shard.where_(&cand, &result)?;
    }
    Ok(result)
}

/// Dequantize a tensor using MLX's dequantize op, threading the quant `mode`.
fn dequantize(
    weight: &MxArray,
    scales: &MxArray,
    biases: Option<&MxArray>,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Result<MxArray> {
    let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
    let mode_c = std::ffi::CString::new(mode)
        .map_err(|_| Error::from_reason("Invalid embedding dequantize mode string"))?;
    let handle = unsafe {
        sys::mlx_dequantize(
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
            group_size,
            bits,
            -1, // Use input dtype (bf16 from scales)
            mode_c.as_ptr(),
        )
    };
    MxArray::from_handle(handle, "dequantize_embedding")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    /// Affine-quantize a 2D bf16 weight via `mlx_quantize`, returning
    /// `(packed_weight, scales, biases)`.
    fn quantize_affine(
        weight: &MxArray,
        group_size: i32,
        bits: i32,
    ) -> (MxArray, MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize affine failed");
        assert!(!out_b.is_null(), "affine quantize must return biases");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
            MxArray::from_handle(out_b, "b").expect("b"),
        )
    }

    /// MXFP8-quantize a 2D bf16 weight, returning `(packed_weight, scales)`
    /// (mxfp8 has no biases).
    fn quantize_mxfp8(weight: &MxArray) -> (MxArray, MxArray) {
        let mut out_q: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            sys::mlx_quantize(
                weight.as_raw_ptr(),
                32, // mxfp8 group_size
                8,  // mxfp8 bits
                c"mxfp8".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize mxfp8 failed");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
        )
    }

    fn to_f32_vec(a: &MxArray) -> Vec<f32> {
        a.astype(DType::Float32)
            .expect("astype f32")
            .to_float32()
            .expect("to_float32")
            .to_vec()
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// PACKED affine embedding: `forward(indices)` (gather-then-dequant) must
    /// equal dequantize-full-table-THEN-gather within quant tolerance.
    #[test]
    fn packed_affine_forward_matches_dequant_full_then_gather() {
        let vocab = 8i64;
        let hidden = 64i64; // one affine group of 64
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let dense = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs, qb) = quantize_affine(&dense, 64, 4);

        let mut emb = Embedding::new(vocab as u32, hidden as u32).expect("new");
        emb.load_quantized_packed(&qw, &qs, Some(&qb), 64, 4, "affine")
            .expect("load packed affine");
        assert!(emb.is_packed_quantized());

        let idx = MxArray::from_int32(&[0, 3, 7, 1], &[4]).expect("idx");

        // gather-then-dequant (our forward)
        let got = emb.forward(&idx).expect("forward");
        // dequant-full-table-then-gather (reference)
        let full = dequantize(&qw, &qs, Some(&qb), 64, 4, "affine").expect("dequant full");
        let ref_rows = full.take(&idx, 0).expect("gather ref");

        let max_err = max_abs_err(&to_f32_vec(&got), &to_f32_vec(&ref_rows));
        assert!(
            max_err < 1e-3,
            "packed affine forward must match dequant-then-gather: max_err={max_err}"
        );
    }

    /// Gemma4 untied `embed_tokens` footprint optimization: loading the quantized
    /// table via `load_quantized_packed` (gather-then-dequant, ~168 MiB packed)
    /// instead of the legacy dense `load_quantized` (dequant-whole-table-then-
    /// gather, ~1.34 GiB bf16) must be BYTE-IDENTICAL at the lookup — at gemma4's
    /// 2-bit affine `embed_tokens` shape class. Unlike
    /// `packed_affine_forward_matches_dequant_full_then_gather` (max_err < 1e-3),
    /// this asserts EXACT equality: affine dequant is per-element with no
    /// cross-row reduction, so gather-then-dequant equals dequant-then-gather
    /// bit-for-bit. Guards the `persistence.rs` untied embed_tokens packed-load
    /// change.
    #[test]
    fn packed_affine_2bit_lookup_byte_identical_to_legacy_dense() {
        let vocab = 16i64;
        let hidden = 64i64; // one affine group of 64 at 2-bit (gemma4 embed_tokens class)
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let dense = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs, qb) = quantize_affine(&dense, 64, 2);

        let idx = MxArray::from_int32(&[0, 5, 15, 3, 8], &[5]).expect("idx");

        // Legacy dense path: load_quantized dequantizes the whole table, then forward gathers.
        let mut emb_dense = Embedding::new(vocab as u32, hidden as u32).expect("new dense");
        emb_dense
            .load_quantized(&qw, &qs, Some(&qb), 64, 2)
            .expect("load dense");
        let out_dense = emb_dense.forward(&idx).expect("forward dense");

        // Packed path: load_quantized_packed keeps it packed; forward gathers rows then dequantizes.
        let mut emb_packed = Embedding::new(vocab as u32, hidden as u32).expect("new packed");
        emb_packed
            .load_quantized_packed(&qw, &qs, Some(&qb), 64, 2, "affine")
            .expect("load packed");
        assert!(emb_packed.is_packed_quantized());
        let out_packed = emb_packed.forward(&idx).expect("forward packed");

        let a = to_f32_vec(&out_dense);
        let b = to_f32_vec(&out_packed);
        assert_eq!(a.len(), b.len(), "lookup shape mismatch");
        let max_err = max_abs_err(&a, &b);
        assert_eq!(
            max_err, 0.0,
            "packed 2-bit affine lookup must be byte-identical to legacy dense: max_err={max_err}"
        );
    }

    /// PACKED embedding must NOT keep the full `[vocab, hidden]` dense table from
    /// `new()` resident: `load_quantized_packed` replaces `self.weight` with a
    /// tiny placeholder, and `get_weight()` dequantizes the FULL table on demand
    /// from the packed backend — returning correct `[vocab, hidden]` data, never
    /// the placeholder or the stale random init. (Regression for the Codex [high]
    /// finding: the packed path previously left the full random table installed.)
    #[test]
    fn packed_get_weight_dequantizes_not_placeholder() {
        let vocab = 8i64;
        let hidden = 64i64; // one affine group of 64
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
        let dense = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs, qb) = quantize_affine(&dense, 64, 4);

        let mut emb = Embedding::new(vocab as u32, hidden as u32).expect("new");
        emb.load_quantized_packed(&qw, &qs, Some(&qb), 64, 4, "affine")
            .expect("load packed affine");
        assert!(emb.is_packed_quantized());

        // The on-disk placeholder is tiny — the full dense table is NOT resident.
        assert_eq!(
            emb.weight.shape().expect("placeholder shape").as_ref(),
            &[1, 1],
            "packed load must replace self.weight with a [1,1] placeholder"
        );

        // get_weight() must return the CORRECT full table (shape + values), via
        // on-demand dequant — NOT the [1,1] placeholder, NOT stale random init.
        let got = emb.get_weight();
        assert_eq!(
            got.shape().expect("get_weight shape").as_ref(),
            &[vocab, hidden],
            "packed get_weight() must return the full [vocab, hidden] table"
        );
        let full = dequantize(&qw, &qs, Some(&qb), 64, 4, "affine").expect("dequant full");
        let max_err = max_abs_err(&to_f32_vec(&got), &to_f32_vec(&full));
        assert!(
            max_err < 1e-6,
            "packed get_weight() must equal the on-demand dequantized table: max_err={max_err}"
        );
    }

    /// PACKED mxfp8 embedding: same gather-then-dequant equivalence (mode-aware,
    /// no biases).
    #[test]
    fn packed_mxfp8_forward_matches_dequant_full_then_gather() {
        let vocab = 8i64;
        let hidden = 64i64; // two mxfp8 groups of 32
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.03).collect();
        let dense = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs) = quantize_mxfp8(&dense);

        let mut emb = Embedding::new(vocab as u32, hidden as u32).expect("new");
        emb.load_quantized_packed(&qw, &qs, None, 32, 8, "mxfp8")
            .expect("load packed mxfp8");

        let idx = MxArray::from_int32(&[2, 5, 0], &[3]).expect("idx");
        let got = emb.forward(&idx).expect("forward");
        let full = dequantize(&qw, &qs, None, 32, 8, "mxfp8").expect("dequant full");
        let ref_rows = full.take(&idx, 0).expect("gather ref");

        let max_err = max_abs_err(&to_f32_vec(&got), &to_f32_vec(&ref_rows));
        assert!(
            max_err < 1e-3,
            "packed mxfp8 forward must match dequant-then-gather: max_err={max_err}"
        );
    }

    /// PACKED affine embedding: `as_linear(x)` must match `x @ dequant(table)^T`
    /// within quantized_matmul tolerance (the tied-head path).
    #[test]
    fn packed_affine_as_linear_matches_dense_matmul() {
        let vocab = 8i64;
        let hidden = 64i64;
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        let dense = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs, qb) = quantize_affine(&dense, 64, 4);

        let mut emb = Embedding::new(vocab as u32, hidden as u32).expect("new");
        emb.load_quantized_packed(&qw, &qs, Some(&qb), 64, 4, "affine")
            .expect("load packed affine");

        // x: [batch=2, hidden]
        let xdata: Vec<f32> = (0..(2 * hidden))
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let x = MxArray::from_float32(&xdata, &[2, hidden])
            .expect("x")
            .astype(DType::BFloat16)
            .expect("x bf16");

        let got = emb.as_linear(&x).expect("as_linear"); // [2, vocab]
        // Reference: x @ dequant(table)^T
        let full = dequantize(&qw, &qs, Some(&qb), 64, 4, "affine").expect("dequant full");
        let full_t = full.transpose(Some(&[1, 0])).expect("transpose");
        let ref_logits = x.matmul(&full_t).expect("ref matmul");

        let max_err = max_abs_err(&to_f32_vec(&got), &to_f32_vec(&ref_logits));
        // quantized_matmul vs dense-dequant matmul differ only by kernel
        // rounding; a loose ceiling still catches a wrong transpose/mode.
        assert!(
            max_err < 5e-2,
            "packed affine as_linear must match x @ dequant(table)^T: max_err={max_err}"
        );
    }

    /// DENSE embedding: `as_linear(x)` must EQUAL `x @ get_weight()^T` exactly
    /// (same op graph). This is the tied-head equivalence the lfm2 logits path
    /// relies on whenever the embedding is plain bf16.
    #[test]
    fn dense_as_linear_equals_get_weight_matmul() {
        let vocab = 8i64;
        let hidden = 16i64;
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let table = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("table")
            .astype(DType::BFloat16)
            .expect("bf16");

        let mut emb = Embedding::new(vocab as u32, hidden as u32).expect("new");
        emb.load_weight(&table).expect("load dense");
        assert!(!emb.is_packed_quantized());

        let xdata: Vec<f32> = (0..(3 * hidden))
            .map(|i| ((i % 4) as f32 - 1.5) * 0.2)
            .collect();
        let x = MxArray::from_float32(&xdata, &[3, hidden])
            .expect("x")
            .astype(DType::BFloat16)
            .expect("x bf16");

        let got = emb.as_linear(&x).expect("as_linear");
        // Previous tied-head computation: x @ get_weight()^T.
        let w = emb.get_weight();
        let w_t = w.transpose(Some(&[1, 0])).expect("transpose");
        let prev = x.matmul(&w_t).expect("prev matmul");

        let got_v = to_f32_vec(&got);
        let prev_v = to_f32_vec(&prev);
        assert_eq!(
            got_v, prev_v,
            "dense as_linear must EQUAL x @ get_weight()^T"
        );
    }

    /// Sharded gather must be BYTE-IDENTICAL to a dense `take(weight, ids, 0)`,
    /// including for special bf16 bit patterns (NaN / ±0 / ±Inf) that any
    /// value-domain op (e.g. a mask-multiply-sum) would perturb. Stresses shard
    /// boundaries, repeats, out-of-order ids, and a short final shard.
    ///
    /// bf16 denormals are intentionally excluded: Metal's `where_` flushes them
    /// to zero (see `forward_sharded`), they don't occur in real weights, and
    /// the PLE forward's `* sqrt(ple_dim)` would flush them anyway.
    #[test]
    fn sharded_forward_matches_dense_take_byte_identical() {
        const V: usize = 7;
        const C: usize = 3;
        // Row-major [V, C] bf16 bit patterns; rows 3 and 5 carry NaN/±Inf/-0 so
        // a bitwise mistake (e.g. mask-multiply-sum) would show up.
        let w: Vec<u16> = vec![
            0x3f00, 0x3f80, 0x4000, // row 0: 0.5, 1.0, 2.0
            0x4040, 0x40a0, 0xc080, // row 1
            0x4110, 0x4150, 0x4190, // row 2
            0x7fc0, 0x7f80, 0xff80, // row 3: NaN, +Inf, -Inf
            0x41f0, 0x4210, 0x4230, // row 4
            0x8000, 0x3e80, 0xbf80, // row 5: -0, 0.25, -1.0
            0x4310, 0x4330, 0x4350, // row 6
        ];
        let full = MxArray::from_bfloat16(&w, &[V as i64, C as i64]).expect("full");

        // Shard along axis 0 with stride 2: rows [0,1] [2,3] [4,5] [6].
        let rows_per_shard = 2usize;
        let mut shards = Vec::new();
        let mut base = 0usize;
        while base < V {
            let rows_s = (V - base).min(rows_per_shard);
            let slice = &w[base * C..(base + rows_s) * C];
            shards.push(MxArray::from_bfloat16(slice, &[rows_s as i64, C as i64]).expect("shard"));
            base += rows_s;
        }
        assert_eq!(shards.len(), 4, "expected 4 shards (last short)");

        let mut emb = Embedding::new(V as u32, C as u32).expect("emb");
        emb.set_sharded(shards, rows_per_shard as i64)
            .expect("set_sharded");

        // Cover every shard, both boundary rows of each, repeats, reverse order,
        // and the special-pattern rows 3 and 5.
        let ids_i32: Vec<i32> = vec![0, 6, 3, 5, 2, 1, 4, 6, 0, 3, 5];
        let ids = MxArray::from_int32(&ids_i32, &[ids_i32.len() as i64]).expect("ids");

        let dense = full.take(&ids, 0).expect("dense take");
        let got = emb.forward(&ids).expect("sharded forward");

        let dense_u16 = dense.to_uint16_native().expect("dense bytes");
        let got_u16 = got.to_uint16_native().expect("sharded bytes");
        assert_eq!(
            got_u16, dense_u16,
            "sharded gather must be byte-identical to dense take"
        );
    }

    /// `set_sharded` must reject shards that don't reconstruct the configured
    /// `[num_embeddings, embedding_dim]` table — the dense `load_weight` shape
    /// check is skipped on the sharded path, so a drifted checkpoint would
    /// otherwise install silently and gather wrong rows.
    #[test]
    fn set_sharded_rejects_shape_mismatch() {
        let z =
            |rows: i64, cols: i64| MxArray::zeros(&[rows, cols], Some(DType::BFloat16)).expect("z");

        // Row count short of num_embeddings (2 + 2 = 4, expected 7).
        let mut emb = Embedding::new(7, 3).expect("emb");
        let err = emb.set_sharded(vec![z(2, 3), z(2, 3)], 2).unwrap_err();
        assert!(
            err.reason.contains("rows sum to 4"),
            "row-coverage mismatch must error, got: {}",
            err.reason
        );

        // Wrong column width (4 != embedding_dim 3).
        let mut emb = Embedding::new(4, 3).expect("emb");
        let err = emb.set_sharded(vec![z(2, 4), z(2, 4)], 2).unwrap_err();
        assert!(
            err.reason.contains("columns, expected 3"),
            "column mismatch must error, got: {}",
            err.reason
        );

        // Non-final shard not equal to the stride breaks the base-row math.
        let mut emb = Embedding::new(5, 3).expect("emb");
        let err = emb.set_sharded(vec![z(3, 3), z(2, 3)], 2).unwrap_err();
        assert!(
            err.reason.contains("non-final shard 0 has 3 rows"),
            "non-final stride mismatch must error, got: {}",
            err.reason
        );

        // The well-formed [2,2,1] partition of 5 rows still passes.
        let mut emb = Embedding::new(5, 3).expect("emb");
        emb.set_sharded(vec![z(2, 3), z(2, 3), z(1, 3)], 2)
            .expect("well-formed shards must install");
    }
}
