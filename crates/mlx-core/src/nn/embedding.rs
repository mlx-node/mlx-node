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
        Ok(())
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
        })
    }
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
}
