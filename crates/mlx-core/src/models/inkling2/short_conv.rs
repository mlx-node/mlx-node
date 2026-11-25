use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};

/// Per-visit rolling causal history for Inkling2 ShortConv (`k = 4` -> `k - 1 = 3` rows).
///
/// Stores the last `k - 1` **pre-convolution** rows in oldest-to-newest order:
/// index `0` = `t - (k - 1)` (oldest), index `k - 2` = `t - 1` (newest).
#[derive(Clone)]
pub struct VisitShortConvState {
    pub k: MxArray,
    pub v: MxArray,
    pub attn_post: MxArray,
    pub mlp_post: MxArray,
}

impl VisitShortConvState {
    pub fn zeros(
        history_len: usize,
        kv_dim: usize,
        hidden_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let h = history_len as i64;
        let kv = kv_dim as i64;
        let d = hidden_size as i64;
        Ok(Self {
            k: MxArray::zeros(&[h, kv], Some(dtype))?,
            v: MxArray::zeros(&[h, kv], Some(dtype))?,
            attn_post: MxArray::zeros(&[h, d], Some(dtype))?,
            mlp_post: MxArray::zeros(&[h, d], Some(dtype))?,
        })
    }

    pub(crate) fn promote_cuda_device(&mut self) -> Result<usize> {
        let mut promoted = 0usize;
        promoted += self.k.promote_cuda_device()?;
        promoted += self.v.promote_cuda_device()?;
        promoted += self.attn_post.promote_cuda_device()?;
        promoted += self.mlp_post.promote_cuda_device()?;
        Ok(promoted)
    }
}

/// Collection of per-visit ShortConv states across all logical visits (48 for Inkling-2-Mini).
#[derive(Clone)]
pub struct Inkling2ShortConvState {
    pub visits: Vec<VisitShortConvState>,
    pub kernel_size: usize,
}

impl Inkling2ShortConvState {
    pub fn new(
        num_visits: usize,
        kernel_size: usize,
        kv_dim: usize,
        hidden_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        if kernel_size < 2 {
            return Err(Error::from_reason(format!(
                "Inkling2ShortConvState: kernel_size must be >= 2, got {kernel_size}"
            )));
        }
        let history_len = kernel_size - 1;
        let mut visits = Vec::with_capacity(num_visits);
        for _ in 0..num_visits {
            visits.push(VisitShortConvState::zeros(
                history_len,
                kv_dim,
                hidden_size,
                dtype,
            )?);
        }
        Ok(Self {
            visits,
            kernel_size,
        })
    }

    pub(crate) fn promote_cuda_device(&mut self) -> Result<usize> {
        let mut promoted = 0usize;
        for visit in &mut self.visits {
            promoted += visit.promote_cuda_device()?;
        }
        Ok(promoted)
    }
}

/// Static depthwise causal ShortConv (`k = 4`, residual `+ x[t]`, scale `(k + 1)^(-1/2)`).
///
/// - `x`: `[T, C]` (BF16 or F32)
/// - `weight`: `[C, K]` in **oldest-to-newest** order (`weight[:, 0]` = lag `K-1`, `weight[:, K-1]` = lag `0`).
/// - `history`: mutable `[K - 1, C]` rolling pre-conv state (oldest at row `0`, newest at row `K - 2`).
///
/// Accumulates in FP32 and returns output in `x`'s dtype (BF16).
pub fn static_short_conv(x: &MxArray, weight: &MxArray, history: &mut MxArray) -> Result<MxArray> {
    let x_shape = x.shape()?;
    let w_shape = weight.shape()?;
    if x_shape.len() != 2 || w_shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "static_short_conv: expected 2D x [T, C] and weight [C, K], got {:?} and {:?}",
            x_shape.as_ref(),
            w_shape.as_ref()
        )));
    }
    let t = x_shape[0];
    let c = x_shape[1];
    let k = w_shape[1] as usize;
    if w_shape[0] != c || k < 2 {
        return Err(Error::from_reason(format!(
            "static_short_conv: incompatible weight {:?} for channels {c}",
            w_shape.as_ref()
        )));
    }

    let out_dtype = x.dtype()?;
    let hist_len = (k - 1) as i64;
    let hist_cast = if history.dtype()? != out_dtype {
        history.astype(out_dtype)?
    } else {
        history.clone()
    };

    // `[K - 1 + T, C]`
    let full = MxArray::concatenate(&hist_cast, x, 0)?;
    // Update rolling state to last `K - 1` pre-conv rows.
    *history = full.slice(&[t, 0], &[t + hist_len, c])?;

    let full_f32 = full.astype(DType::Float32)?;
    let w_f32 = if weight.dtype()? == DType::Float32 {
        weight.clone()
    } else {
        weight.astype(DType::Float32)?
    };

    // Residual term: `lag 0` = `x[t]` = `full[K - 1 .. K - 1 + T]`
    let mut acc = full_f32.slice(&[hist_len, 0], &[hist_len + t, c])?;
    for j in 0..k {
        let j_i64 = j as i64;
        // j=0 is oldest tap (lag K-1, starts at 0); j=K-1 is newest tap (lag 0, starts at K-1).
        let x_tap = full_f32.slice(&[j_i64, 0], &[j_i64 + t, c])?;
        let w_tap = w_f32
            .slice(&[0, j_i64], &[c, j_i64 + 1])?
            .reshape(&[1, c])?;
        let term = x_tap.mul(&w_tap)?;
        acc = acc.add(&term)?;
    }

    let scale = 1.0_f64 / ((k as f64) + 1.0).sqrt();
    let scaled = acc.mul_scalar(scale)?;
    if out_dtype != DType::Float32 {
        scaled.astype(out_dtype)
    } else {
        Ok(scaled)
    }
}

/// Dynamic depthwise causal ShortConv for Global layers' K and V streams.
///
/// - `x`: `[T, C]` where `C = num_kv_heads * head_dim`
/// - `static_weight`: `[C, K]` in **oldest-to-newest** order (GGUF layout: `w[:, 0]` = lag `K-1`, `w[:, K-1]` = lag `0`)
/// - `dyn_taps`: `[T, num_kv_heads, K]` in **newest-to-oldest** order (`dyn[:, :, 0]` = lag `0`, `dyn[:, :, K-1]` = lag `K-1`)
/// - `history`: mutable `[K - 1, C]` rolling pre-conv state (oldest at row `0`, newest at row `K - 2`).
///
/// Formula (`scale = (2*K + 1)^(-1/2) = 1/3` for `K = 4`):
/// `y[t, c] = (x[t, c] + sum_{lag=0..K} (w[c, K - 1 - lag] + dyn[t, head, lag]) * x[t - lag, c]) * scale`
pub fn dynamic_short_conv(
    x: &MxArray,
    static_weight: &MxArray,
    dyn_taps: &MxArray,
    history: &mut MxArray,
) -> Result<MxArray> {
    let x_shape = x.shape()?;
    let w_shape = static_weight.shape()?;
    let d_shape = dyn_taps.shape()?;
    if x_shape.len() != 2 || w_shape.len() != 2 || d_shape.len() != 3 {
        return Err(Error::from_reason(format!(
            "dynamic_short_conv: invalid shapes x={:?}, w={:?}, dyn={:?}",
            x_shape.as_ref(),
            w_shape.as_ref(),
            d_shape.as_ref()
        )));
    }
    let t = x_shape[0];
    let c = x_shape[1];
    let k = w_shape[1] as usize;
    let n_kv = d_shape[1];
    if w_shape[0] != c || d_shape[0] != t || d_shape[2] as usize != k || c % n_kv != 0 {
        return Err(Error::from_reason(format!(
            "dynamic_short_conv: incompatible shapes x={:?}, w={:?}, dyn={:?}",
            x_shape.as_ref(),
            w_shape.as_ref(),
            d_shape.as_ref()
        )));
    }
    let head_dim = c / n_kv;
    let out_dtype = x.dtype()?;
    let hist_len = (k - 1) as i64;
    let hist_cast = if history.dtype()? != out_dtype {
        history.astype(out_dtype)?
    } else {
        history.clone()
    };

    let full = MxArray::concatenate(&hist_cast, x, 0)?;
    *history = full.slice(&[t, 0], &[t + hist_len, c])?;

    let full_f32 = full.astype(DType::Float32)?;
    let w_f32 = if static_weight.dtype()? == DType::Float32 {
        static_weight.clone()
    } else {
        static_weight.astype(DType::Float32)?
    };
    let dyn_f32 = if dyn_taps.dtype()? == DType::Float32 {
        dyn_taps.clone()
    } else {
        dyn_taps.astype(DType::Float32)?
    };

    // Residual term: `lag 0` = `x[t]`
    let mut acc = full_f32.slice(&[hist_len, 0], &[hist_len + t, c])?;
    for lag in 0..k {
        let lag_i64 = lag as i64;
        // `full` start offset for `x[t - lag]` is `(K - 1) - lag`.
        let start = hist_len - lag_i64;
        let x_tap = full_f32.slice(&[start, 0], &[start + t, c])?;

        // Static weight is oldest-to-newest, so lag `lag` is column `K - 1 - lag`.
        let w_col = hist_len - lag_i64;
        let w_tap = w_f32
            .slice(&[0, w_col], &[c, w_col + 1])?
            .reshape(&[1, c])?;

        // Dynamic taps are newest-to-oldest (`dyn[:, :, lag]`), broadcast per KV head across `head_dim`.
        let d_tap = dyn_f32
            .slice(&[0, 0, lag_i64], &[t, n_kv, lag_i64 + 1])?
            .reshape(&[t, n_kv, 1])?
            .broadcast_to(&[t, n_kv, head_dim])?
            .reshape(&[t, c])?;

        let eff_w = w_tap.add(&d_tap)?;
        let term = x_tap.mul(&eff_w)?;
        acc = acc.add(&term)?;
    }

    let scale = 1.0_f64 / ((2 * k + 1) as f64).sqrt();
    let scaled = acc.mul_scalar(scale)?;
    if out_dtype != DType::Float32 {
        scaled.astype(out_dtype)
    } else {
        Ok(scaled)
    }
}

/// Fused Metal `k = 4` causal ShortConv (static when `dyn_taps` is `None`, dynamic global when
/// `Some`), updating `history` in place. Falls back to the pure-MLX implementation on non-Metal
/// devices or when `kernel_size != 4`.
pub fn fused_short_conv_k4(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    history: &mut MxArray,
) -> Result<MxArray> {
    #[cfg(target_os = "macos")]
    {
        if unsafe { mlx_sys::mlx_metal_is_available() } && weight.shape_at(1)? == 4 {
            let (y, hist_out) = super::kernels::inkling2_sconv_k4(x, weight, history, dyn_taps)?;
            *history = hist_out;
            return Ok(y);
        }
    }
    match dyn_taps {
        Some(dyn_t) => dynamic_short_conv(x, weight, dyn_t, history),
        None => static_short_conv(x, weight, history),
    }
}

/// Per-stream rollback tape for one ShortConv (`k = 4`) stream during speculative verification.
///
/// Captures `hist_before` (`[K-1, C]` BF16) and the `T` pre-convolution candidate rows `rows`
/// (`[T, C]` BF16) written during target verify. On commit of `keep` tokens (`0 <= keep <= T`),
/// [`Self::commit_prefix`] reconstructs the exact `[K-1, C]` rolling history after `keep` tokens
/// with zero corruption from rejected tail tokens!
#[derive(Clone)]
pub struct StreamShortConvTape {
    pub hist_before: MxArray,
    pub rows: MxArray,
}

impl StreamShortConvTape {
    /// Compute the exact `[history_len, C]` rolling ShortConv history after committing `keep`
    /// tokens (`0 <= keep <= rows.shape()[0]`).
    pub fn commit_prefix(&self, keep: usize) -> Result<MxArray> {
        let hist_len = self.hist_before.shape_at(0)? as usize;
        let total_rows = self.rows.shape_at(0)? as usize;
        let c = self.rows.shape_at(1)?;
        if keep > total_rows {
            return Err(Error::from_reason(format!(
                "StreamShortConvTape::commit_prefix: keep={keep} exceeds verified rows={total_rows}"
            )));
        }
        if keep == 0 {
            return Ok(self.hist_before.clone());
        }
        if keep >= hist_len {
            let start = (keep - hist_len) as i64;
            return self.rows.slice(&[start, 0], &[keep as i64, c]);
        }
        let hist_tail = self
            .hist_before
            .slice(&[keep as i64, 0], &[hist_len as i64, c])?;
        let rows_head = self.rows.slice(&[0, 0], &[keep as i64, c])?;
        let hist_cast = if hist_tail.dtype()? != rows_head.dtype()? {
            hist_tail.astype(rows_head.dtype()?)?
        } else {
            hist_tail
        };
        MxArray::concatenate(&hist_cast, &rows_head, 0)
    }

    /// Compute the exact `[history_len, C]` rolling ShortConv history after accepting an
    /// arbitrary tree path `accepted_nodes` (ordered oldest-to-newest along the winning branch).
    ///
    /// When `is_contiguous` is `true` (meaning `accepted_nodes == [0, 1, ..., keep - 1]`),
    /// delegates to the zero-gather [`Self::commit_prefix`] fast path.
    pub fn commit_tree(&self, accepted_nodes: &[u32], is_contiguous: bool) -> Result<MxArray> {
        if is_contiguous {
            return self.commit_prefix(accepted_nodes.len());
        }
        let hist_len = self.hist_before.shape_at(0)? as usize;
        let total_rows = self.rows.shape_at(0)? as usize;
        let c = self.rows.shape_at(1)?;
        let keep = accepted_nodes.len();
        if keep == 0 {
            return Ok(self.hist_before.clone());
        }
        for &node in accepted_nodes {
            if (node as usize) >= total_rows {
                return Err(Error::from_reason(format!(
                    "StreamShortConvTape::commit_tree: node index {node} out of range (rows={total_rows})"
                )));
            }
        }
        if keep >= hist_len {
            let tail = &accepted_nodes[keep - hist_len..keep];
            let mut idx_i32 = Vec::with_capacity(hist_len);
            for &node in tail {
                idx_i32.push(i32::try_from(node).map_err(|_| {
                    Error::from_reason(format!(
                        "StreamShortConvTape::commit_tree: node index {node} exceeds i32::MAX"
                    ))
                })?);
            }
            let idx = MxArray::from_int32(&idx_i32, &[hist_len as i64])?;
            return self.rows.take(&idx, 0);
        }
        let hist_tail = self
            .hist_before
            .slice(&[keep as i64, 0], &[hist_len as i64, c])?;
        let mut idx_i32 = Vec::with_capacity(keep);
        for &node in accepted_nodes {
            idx_i32.push(i32::try_from(node).map_err(|_| {
                Error::from_reason(format!(
                    "StreamShortConvTape::commit_tree: node index {node} exceeds i32::MAX"
                ))
            })?);
        }
        let idx = MxArray::from_int32(&idx_i32, &[keep as i64])?;
        let rows_sel = self.rows.take(&idx, 0)?;
        let hist_cast = if hist_tail.dtype()? != rows_sel.dtype()? {
            hist_tail.astype(rows_sel.dtype()?)?
        } else {
            hist_tail
        };
        MxArray::concatenate(&hist_cast, &rows_sel, 0)
    }

    /// Compute the exact `[history_len, C]` rolling ShortConv history after committing a
    /// strictly-increasing speculative verify-tree path `accepted_rows` (root-to-accepted-node).
    pub fn commit_path(&self, accepted_rows: &[usize]) -> Result<MxArray> {
        let hist_len = self.hist_before.shape_at(0)? as usize;
        let total_rows = self.rows.shape_at(0)? as usize;
        let c = self.rows.shape_at(1)?;
        for (d, &row) in accepted_rows.iter().enumerate() {
            if row >= total_rows {
                return Err(Error::from_reason(format!(
                    "StreamShortConvTape::commit_path: accepted_rows[{d}]={row} >= verified rows={total_rows}"
                )));
            }
            if d > 0 && row <= accepted_rows[d - 1] {
                return Err(Error::from_reason(format!(
                    "StreamShortConvTape::commit_path: accepted_rows must be strictly increasing (got {} then {row})",
                    accepted_rows[d - 1]
                )));
            }
        }
        let keep = accepted_rows.len();
        if keep == 0 {
            return Ok(self.hist_before.clone());
        }
        if accepted_rows.iter().enumerate().all(|(d, &row)| row == d) {
            return self.commit_prefix(keep);
        }
        if keep >= hist_len {
            let tail: Vec<i32> = accepted_rows[keep - hist_len..]
                .iter()
                .map(|&r| r as i32)
                .collect();
            let idx = MxArray::from_int32(&tail, &[hist_len as i64])?;
            return self.rows.take(&idx, 0);
        }
        let hist_tail = self
            .hist_before
            .slice(&[keep as i64, 0], &[hist_len as i64, c])?;
        let indices: Vec<i32> = accepted_rows.iter().map(|&r| r as i32).collect();
        let idx = MxArray::from_int32(&indices, &[keep as i64])?;
        let rows_path = self.rows.take(&idx, 0)?;
        let hist_cast = if hist_tail.dtype()? != rows_path.dtype()? {
            hist_tail.astype(rows_path.dtype()?)?
        } else {
            hist_tail
        };
        MxArray::concatenate(&hist_cast, &rows_path, 0)
    }
}

/// Per-visit ShortConv rollback tape across all 4 streams (`k`, `v`, `attn_post`, `mlp_post`).
#[derive(Clone)]
pub struct VisitShortConvTape {
    pub k: Option<StreamShortConvTape>,
    pub v: Option<StreamShortConvTape>,
    pub attn_post: Option<StreamShortConvTape>,
    pub mlp_post: Option<StreamShortConvTape>,
}

impl VisitShortConvTape {
    pub fn empty() -> Self {
        Self {
            k: None,
            v: None,
            attn_post: None,
            mlp_post: None,
        }
    }

    pub fn commit_prefix(&self, keep: usize, dst: &mut VisitShortConvState) -> Result<()> {
        if let Some(ref t) = self.k {
            dst.k = t.commit_prefix(keep)?;
        }
        if let Some(ref t) = self.v {
            dst.v = t.commit_prefix(keep)?;
        }
        if let Some(ref t) = self.attn_post {
            dst.attn_post = t.commit_prefix(keep)?;
        }
        if let Some(ref t) = self.mlp_post {
            dst.mlp_post = t.commit_prefix(keep)?;
        }
        Ok(())
    }

    pub fn commit_tree(
        &self,
        accepted_nodes: &[u32],
        is_contiguous: bool,
        dst: &mut VisitShortConvState,
    ) -> Result<()> {
        if let Some(ref t) = self.k {
            dst.k = t.commit_tree(accepted_nodes, is_contiguous)?;
        }
        if let Some(ref t) = self.v {
            dst.v = t.commit_tree(accepted_nodes, is_contiguous)?;
        }
        if let Some(ref t) = self.attn_post {
            dst.attn_post = t.commit_tree(accepted_nodes, is_contiguous)?;
        }
        if let Some(ref t) = self.mlp_post {
            dst.mlp_post = t.commit_tree(accepted_nodes, is_contiguous)?;
        }
        Ok(())
    }

    pub fn commit_path(
        &self,
        accepted_rows: &[usize],
        dst: &mut VisitShortConvState,
    ) -> Result<()> {
        if let Some(ref t) = self.k {
            dst.k = t.commit_path(accepted_rows)?;
        }
        if let Some(ref t) = self.v {
            dst.v = t.commit_path(accepted_rows)?;
        }
        if let Some(ref t) = self.attn_post {
            dst.attn_post = t.commit_path(accepted_rows)?;
        }
        if let Some(ref t) = self.mlp_post {
            dst.mlp_post = t.commit_path(accepted_rows)?;
        }
        Ok(())
    }
}

/// Complete 48-visit ShortConv rollback tape for one speculative verify cycle.
#[derive(Clone)]
pub struct Inkling2ShortConvTape {
    pub visits: Vec<VisitShortConvTape>,
}

/// Batch-commit a homogeneous channel-width group of `k=4` ShortConv rollback tapes on CUDA
/// (`<= 96` streams per single kernel launch -> guarantees contiguous `[3, C]` BF16 outputs with
/// zero `copy_g_nd` or per-stream gathers). Returns `Ok(false)` off CUDA or on non-uniform /
/// non-BF16 inputs so callers fall back cleanly per stream.
fn try_commit_stream_group_cuda(
    streams: &mut [(&StreamShortConvTape, &mut MxArray)],
    nodes_arr: Option<&MxArray>,
    accepted_nodes: Option<&[u32]>,
    contiguous_keep: usize,
) -> Result<bool> {
    if !crate::kernels::inkling2::available() || streams.is_empty() {
        return Ok(false);
    }
    let first_h = streams[0].0.hist_before.shape()?;
    let first_r = streams[0].0.rows.shape()?;
    if first_h.len() != 2
        || first_r.len() != 2
        || first_h[0] != 3
        || first_h[1] < 1
        || first_r[0] < 1
        || first_r[1] != first_h[1]
    {
        return Ok(false);
    }
    let expected_c = first_h[1];
    let expected_t = first_r[0];
    let expected_t_usize = expected_t as usize;

    // Validate bounds BEFORE launching CUDA so invalid indices error identically to the CPU path.
    if let Some(nodes) = accepted_nodes {
        for &node in nodes {
            if (node as usize) >= expected_t_usize {
                return Err(Error::from_reason(format!(
                    "StreamShortConvTape::commit_tree: node index {node} out of range (rows={expected_t_usize})"
                )));
            }
        }
    } else if contiguous_keep > expected_t_usize {
        return Err(Error::from_reason(format!(
            "StreamShortConvTape::commit_prefix: keep={contiguous_keep} exceeds verified rows={expected_t_usize}"
        )));
    }

    for (tape, _) in streams.iter() {
        let hs = tape.hist_before.shape()?;
        let rs = tape.rows.shape()?;
        if hs.len() != 2
            || rs.len() != 2
            || hs[0] != 3
            || hs[1] != expected_c
            || rs[0] != expected_t
            || rs[1] != expected_c
            || tape.hist_before.dtype()? != DType::BFloat16
            || tape.rows.dtype()? != DType::BFloat16
        {
            return Ok(false);
        }
    }

    const MAX_CHUNK: usize = 96;
    let mut new_hists: Vec<MxArray> = Vec::with_capacity(streams.len());
    for chunk in streams.chunks(MAX_CHUNK) {
        let h_refs: Vec<&MxArray> = chunk.iter().map(|(t, _)| &t.hist_before).collect();
        let r_refs: Vec<&MxArray> = chunk.iter().map(|(t, _)| &t.rows).collect();
        let Some(outs) = crate::kernels::inkling2::commit_short_conv_tapes_cuda(
            &h_refs,
            &r_refs,
            nodes_arr,
            contiguous_keep,
        )?
        else {
            return Ok(false);
        };
        if outs.len() != chunk.len() {
            return Ok(false);
        }
        new_hists.extend(outs);
    }

    for ((_, dst), hist) in streams.iter_mut().zip(new_hists) {
        **dst = hist;
    }
    Ok(true)
}

impl Inkling2ShortConvTape {
    pub fn new(num_visits: usize) -> Self {
        Self {
            visits: (0..num_visits)
                .map(|_| VisitShortConvTape::empty())
                .collect(),
        }
    }

    /// Commit ONLY the accepted prefix (`0..=verify_rows`) into `state` across all 48 visits.
    ///
    /// On CUDA (`kernel_size == 4`), batches all `k` + `v` (`C=1024`, 96 streams) into 1 launch
    /// and all `attn_post` + `mlp_post` (`C=2304`, 96 streams) into 1 launch — guaranteeing
    /// contiguous `[3, C]` histories with zero `copy_g_nd` on the next step.
    pub fn commit_prefix(&self, keep: usize, state: &mut Inkling2ShortConvState) -> Result<()> {
        if self.visits.len() != state.visits.len() {
            return Err(Error::from_reason(format!(
                "Inkling2ShortConvTape visit count mismatch: tape={} state={}",
                self.visits.len(),
                state.visits.len()
            )));
        }
        if crate::kernels::inkling2::available() && state.kernel_size == 4 {
            let mut kv_streams: Vec<(&StreamShortConvTape, &mut MxArray)> =
                Vec::with_capacity(self.visits.len() * 2);
            let mut hidden_streams: Vec<(&StreamShortConvTape, &mut MxArray)> =
                Vec::with_capacity(self.visits.len() * 2);
            for (tape_v, state_v) in self.visits.iter().zip(state.visits.iter_mut()) {
                if let Some(ref t) = tape_v.k {
                    kv_streams.push((t, &mut state_v.k));
                }
                if let Some(ref t) = tape_v.v {
                    kv_streams.push((t, &mut state_v.v));
                }
                if let Some(ref t) = tape_v.attn_post {
                    hidden_streams.push((t, &mut state_v.attn_post));
                }
                if let Some(ref t) = tape_v.mlp_post {
                    hidden_streams.push((t, &mut state_v.mlp_post));
                }
            }
            if !kv_streams.is_empty()
                && !try_commit_stream_group_cuda(&mut kv_streams, None, None, keep)?
            {
                for (t, dst) in kv_streams {
                    *dst = t.commit_prefix(keep)?;
                }
            }
            if !hidden_streams.is_empty()
                && !try_commit_stream_group_cuda(&mut hidden_streams, None, None, keep)?
            {
                for (t, dst) in hidden_streams {
                    *dst = t.commit_prefix(keep)?;
                }
            }
            return Ok(());
        }
        for (tape_v, state_v) in self.visits.iter().zip(state.visits.iter_mut()) {
            tape_v.commit_prefix(keep, state_v)?;
        }
        Ok(())
    }

    /// Commit an arbitrary accepted tree path `accepted_nodes` into `state` across all 48 visits.
    ///
    /// On CUDA (`kernel_size == 4`), batches all 96 `k`/`v` streams into 1 launch and all 96
    /// `attn_post`/`mlp_post` streams into 1 launch (both contiguous prefix & non-contiguous
    /// tree branches), eliminating 196 gather kernels + 40 `copy_g_nd` copies.
    pub fn commit_tree(
        &self,
        accepted_nodes: &[u32],
        is_contiguous: bool,
        state: &mut Inkling2ShortConvState,
    ) -> Result<()> {
        if self.visits.len() != state.visits.len() {
            return Err(Error::from_reason(format!(
                "Inkling2ShortConvTape visit count mismatch: tape={} state={}",
                self.visits.len(),
                state.visits.len()
            )));
        }
        if is_contiguous || accepted_nodes.is_empty() {
            return self.commit_prefix(accepted_nodes.len(), state);
        }
        if crate::kernels::inkling2::available() && state.kernel_size == 4 {
            let mut idx_i32 = Vec::with_capacity(accepted_nodes.len());
            for &node in accepted_nodes {
                idx_i32.push(i32::try_from(node).map_err(|_| {
                    Error::from_reason(format!(
                        "StreamShortConvTape::commit_tree: node index {node} exceeds i32::MAX"
                    ))
                })?);
            }
            let nodes_arr = MxArray::from_int32(&idx_i32, &[idx_i32.len() as i64])?;
            let mut kv_streams: Vec<(&StreamShortConvTape, &mut MxArray)> =
                Vec::with_capacity(self.visits.len() * 2);
            let mut hidden_streams: Vec<(&StreamShortConvTape, &mut MxArray)> =
                Vec::with_capacity(self.visits.len() * 2);
            for (tape_v, state_v) in self.visits.iter().zip(state.visits.iter_mut()) {
                if let Some(ref t) = tape_v.k {
                    kv_streams.push((t, &mut state_v.k));
                }
                if let Some(ref t) = tape_v.v {
                    kv_streams.push((t, &mut state_v.v));
                }
                if let Some(ref t) = tape_v.attn_post {
                    hidden_streams.push((t, &mut state_v.attn_post));
                }
                if let Some(ref t) = tape_v.mlp_post {
                    hidden_streams.push((t, &mut state_v.mlp_post));
                }
            }
            if !kv_streams.is_empty()
                && !try_commit_stream_group_cuda(
                    &mut kv_streams,
                    Some(&nodes_arr),
                    Some(accepted_nodes),
                    0,
                )?
            {
                for (t, dst) in kv_streams {
                    *dst = t.commit_tree(accepted_nodes, false)?;
                }
            }
            if !hidden_streams.is_empty()
                && !try_commit_stream_group_cuda(
                    &mut hidden_streams,
                    Some(&nodes_arr),
                    Some(accepted_nodes),
                    0,
                )?
            {
                for (t, dst) in hidden_streams {
                    *dst = t.commit_tree(accepted_nodes, false)?;
                }
            }
            return Ok(());
        }
        for (tape_v, state_v) in self.visits.iter().zip(state.visits.iter_mut()) {
            tape_v.commit_tree(accepted_nodes, is_contiguous, state_v)?;
        }
        Ok(())
    }

    /// Commit ONLY the accepted tree path (`accepted_rows`, strictly increasing) into `state`
    /// across all 48 visits.
    pub fn commit_path(
        &self,
        accepted_rows: &[usize],
        state: &mut Inkling2ShortConvState,
    ) -> Result<()> {
        if self.visits.len() != state.visits.len() {
            return Err(Error::from_reason(format!(
                "Inkling2ShortConvTape visit count mismatch: tape={} state={}",
                self.visits.len(),
                state.visits.len()
            )));
        }
        for (tape_v, state_v) in self.visits.iter().zip(state.visits.iter_mut()) {
            tape_v.commit_path(accepted_rows, state_v)?;
        }
        Ok(())
    }
}

/// Apply fused ShortConv over a packed multi-sequence batch partitioned by `cu_seqlens_q`
/// (`[0, len_0, len_0 + len_1, ...]`), updating each sequence's history independently.
///
/// For the common single-sequence case (`histories.len() == 1`), calls [`fused_short_conv_k4`]
/// directly with zero slice/concat overhead.
pub fn fused_short_conv_ragged(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
) -> Result<MxArray> {
    fused_short_conv_ragged_taped(x, weight, dyn_taps, histories, cu_seqlens_q, None)
}

/// Same as [`fused_short_conv_ragged`], additionally recording each sequence's pre-conv rows and
/// pre-verify history into `tapes` (one `Option<&mut Option<StreamShortConvTape>>` per sequence)
/// when speculative rollback is requested.
pub fn fused_short_conv_ragged_taped(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
) -> Result<MxArray> {
    fused_short_conv_ragged_residual_taped(
        x,
        weight,
        dyn_taps,
        None,
        None,
        histories,
        cu_seqlens_q,
        None,
        tapes,
    )
}

#[cfg(target_os = "macos")]
fn record_ragged_tapes(
    x: &MxArray,
    histories: &[&mut MxArray],
    cu_seqlens_q: &[usize],
    channels: i64,
    tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
) -> Result<()> {
    let Some(tape_slots) = tapes else {
        return Ok(());
    };
    if histories.len() == 1 {
        *tape_slots[0] = Some(StreamShortConvTape {
            hist_before: histories[0].clone(),
            rows: x.clone(),
        });
    } else {
        for (seq_idx, hist) in histories.iter().enumerate() {
            let start = cu_seqlens_q[seq_idx] as i64;
            let end = cu_seqlens_q[seq_idx + 1] as i64;
            *tape_slots[seq_idx] = Some(StreamShortConvTape {
                hist_before: (*hist).clone(),
                rows: x.slice(&[start, 0], &[end, channels])?,
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn prepare_cu_seqlens(
    num_seqs: usize,
    cu_seqlens_q: &[usize],
    cu_seqlens_arr: Option<&MxArray>,
) -> Result<Option<MxArray>> {
    if num_seqs <= 1 {
        return Ok(None);
    }
    match cu_seqlens_arr {
        Some(arr) => Ok(Some(arr.clone())),
        None => {
            let vals: Vec<i32> = cu_seqlens_q.iter().map(|&v| v as i32).collect();
            Ok(Some(MxArray::from_int32(&vals, &[vals.len() as i64])?))
        }
    }
}

#[cfg(target_os = "macos")]
fn write_back_ragged_histories(
    histories: &mut [&mut MxArray],
    hist_out: MxArray,
    channels: i64,
) -> Result<()> {
    if histories.len() == 1 {
        *histories[0] = hist_out;
    } else {
        for (seq_idx, hist) in histories.iter_mut().enumerate() {
            let s_i64 = seq_idx as i64;
            **hist = hist_out
                .slice(&[s_i64, 0, 0], &[s_i64 + 1, 3, channels])?
                .squeeze(Some(&[0]))?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ragged_history<R>(
    histories: &mut [&mut MxArray],
    channels: i64,
    dispatch: impl FnOnce(&[&MxArray], bool) -> Result<(R, MxArray)>,
) -> Result<R> {
    let (res, hist_out) = if histories.len() <= 4 {
        let h_refs: Vec<&MxArray> = histories.iter().map(|h| &**h).collect();
        dispatch(&h_refs, false)?
    } else {
        let h_refs: Vec<&MxArray> = histories.iter().map(|h| &**h).collect();
        let packed = MxArray::stack(h_refs, Some(0))?;
        dispatch(&[&packed], true)?
    };
    write_back_ragged_histories(histories, hist_out, channels)?;
    Ok(res)
}

#[cfg(target_os = "macos")]
fn dispatch_ragged_kv_histories<R>(
    k_histories: &mut [&mut MxArray],
    v_histories: &mut [&mut MxArray],
    channels: i64,
    dispatch: impl FnOnce(&[&MxArray], &[&MxArray], bool) -> Result<(R, MxArray, MxArray)>,
) -> Result<R> {
    let (res, kh_out, vh_out) = if k_histories.len() <= 4 {
        let kh_refs: Vec<&MxArray> = k_histories.iter().map(|h| &**h).collect();
        let vh_refs: Vec<&MxArray> = v_histories.iter().map(|h| &**h).collect();
        dispatch(&kh_refs, &vh_refs, false)?
    } else {
        let kh_refs: Vec<&MxArray> = k_histories.iter().map(|h| &**h).collect();
        let vh_refs: Vec<&MxArray> = v_histories.iter().map(|h| &**h).collect();
        let kh_p = MxArray::stack(kh_refs, Some(0))?;
        let vh_p = MxArray::stack(vh_refs, Some(0))?;
        dispatch(&[&kh_p], &[&vh_p], true)?
    };
    write_back_ragged_histories(k_histories, kh_out, channels)?;
    write_back_ragged_histories(v_histories, vh_out, channels)?;
    Ok(res)
}

/// Single-dispatch ragged causal `k=4` ShortConv across `N >= 1` packed sequences with optional
/// FP32 residual epilogue (`res_a` + optional `res_b` + `conv_out`), optional speculative rollback
/// tape capture (`hist_before`, pre-conv `rows`), and optional verify `tree_rows`.
#[allow(clippy::too_many_arguments)]
pub fn fused_short_conv_ragged_residual_taped_tree(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    res_a: Option<&MxArray>,
    res_b: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] cu_seqlens_arr: Option<&MxArray>,
    mut tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    tree_rows: Option<&MxArray>,
) -> Result<MxArray> {
    let num_seqs = histories.len();
    if num_seqs == 0 || cu_seqlens_q.len() != num_seqs + 1 {
        return Err(Error::from_reason(format!(
            "fused_short_conv_ragged: invalid seq count {num_seqs} vs cu_seqlens_q len {}",
            cu_seqlens_q.len()
        )));
    }
    let c = x.shape_at(1)?;
    #[cfg(target_os = "macos")]
    let all_non_empty = (0..num_seqs).all(|s| cu_seqlens_q[s + 1] > cu_seqlens_q[s]);

    #[cfg(target_os = "macos")]
    if unsafe { mlx_sys::mlx_metal_is_available() } && weight.shape_at(1)? == 4 && all_non_empty {
        record_ragged_tapes(x, histories, cu_seqlens_q, c, tapes)?;
        if num_seqs == 1 && res_a.is_none() && res_b.is_none() {
            let (y, hist_out) = super::kernels::inkling2_sconv_k4_tree(
                x,
                weight,
                histories[0],
                dyn_taps,
                tree_rows,
            )?;
            *histories[0] = hist_out;
            return Ok(y);
        }
        let cu_arr = prepare_cu_seqlens(num_seqs, cu_seqlens_q, cu_seqlens_arr)?;
        return dispatch_ragged_history(histories, c, |h_refs, stacked| {
            super::kernels::inkling2_sconv_k4_ragged_tree(
                x,
                weight,
                dyn_taps,
                cu_arr.as_ref(),
                res_a,
                res_b,
                h_refs,
                num_seqs,
                stacked,
                tree_rows,
            )
        });
    }

    if tree_rows.is_some() {
        return Err(Error::from_reason(
            "fused_short_conv_ragged_residual_taped_tree: tree_rows requires Metal k=4 kernel",
        ));
    }

    // Fallback (non-Metal, kernel_size != 4, or empty sequences)
    let act_dtype = x.dtype()?;
    let y = if num_seqs == 1 {
        if let Some(ref mut tape_slots) = tapes {
            *tape_slots[0] = Some(StreamShortConvTape {
                hist_before: histories[0].clone(),
                rows: x.clone(),
            });
        }
        fused_short_conv_k4(x, weight, dyn_taps, histories[0])?
    } else {
        let dyn_dims = match dyn_taps {
            Some(d) => {
                let s = d.shape()?;
                Some((s[1], s[2]))
            }
            None => None,
        };
        let mut outputs = Vec::with_capacity(num_seqs);
        for (seq_idx, hist) in histories.iter_mut().enumerate() {
            let start = cu_seqlens_q[seq_idx] as i64;
            let end = cu_seqlens_q[seq_idx + 1] as i64;
            if end <= start {
                continue;
            }
            let x_seq = x.slice(&[start, 0], &[end, c])?;
            if let Some(ref mut tape_slots) = tapes {
                *tape_slots[seq_idx] = Some(StreamShortConvTape {
                    hist_before: hist.clone(),
                    rows: x_seq.clone(),
                });
            }
            let d_seq = match (dyn_taps, dyn_dims) {
                (Some(d), Some((n_kv, k))) => Some(d.slice(&[start, 0, 0], &[end, n_kv, k])?),
                _ => None,
            };
            let y_seq = fused_short_conv_k4(&x_seq, weight, d_seq.as_ref(), hist)?;
            outputs.push(y_seq);
        }
        let refs: Vec<&MxArray> = outputs.iter().collect();
        MxArray::concatenate_many(refs, Some(0))?
    };
    match (res_a, res_b) {
        (None, None) => Ok(y),
        (Some(a), None) => a
            .astype(DType::Float32)?
            .add(&y.astype(DType::Float32)?)?
            .astype(act_dtype),
        (None, Some(b)) => b
            .astype(DType::Float32)?
            .add(&y.astype(DType::Float32)?)?
            .astype(act_dtype),
        (Some(a), Some(b)) => a
            .astype(DType::Float32)?
            .add(&b.astype(DType::Float32)?)?
            .add(&y.astype(DType::Float32)?)?
            .astype(act_dtype),
    }
}

/// Single-dispatch ragged causal `k=4` ShortConv across `N >= 1` packed sequences with optional
/// FP32 residual epilogue (`res_a` + optional `res_b` + `conv_out`) and optional speculative
/// rollback tape capture (`hist_before`, pre-conv `rows`).
#[allow(clippy::too_many_arguments)]
pub fn fused_short_conv_ragged_residual_taped(
    x: &MxArray,
    weight: &MxArray,
    dyn_taps: Option<&MxArray>,
    res_a: Option<&MxArray>,
    res_b: Option<&MxArray>,
    histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    cu_seqlens_arr: Option<&MxArray>,
    tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
) -> Result<MxArray> {
    fused_short_conv_ragged_residual_taped_tree(
        x,
        weight,
        dyn_taps,
        res_a,
        res_b,
        histories,
        cu_seqlens_q,
        cu_seqlens_arr,
        tapes,
        None,
    )
}

/// Single-dispatch fused K+V causal `k=4` ShortConv + K L2-clip + V weightless RMSNorm across
/// `N >= 1` packed sequences with optional speculative verify `tree_rows`.
#[allow(clippy::too_many_arguments)]
pub fn fused_short_conv_kv_prep_taped_tree(
    k_raw: &MxArray,
    v_raw: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    k_histories: &mut [&mut MxArray],
    v_histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] cu_seqlens_arr: Option<&MxArray>,
    k_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    v_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    num_kv_heads: usize,
    head_dim: usize,
    qk_thresh: f32,
    v_eps: f32,
    tree_rows: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let num_seqs = k_histories.len();
    if num_seqs == 0 || v_histories.len() != num_seqs || cu_seqlens_q.len() != num_seqs + 1 {
        return Err(Error::from_reason(
            "fused_short_conv_kv_prep_taped: invalid sequence count",
        ));
    }
    #[cfg(target_os = "macos")]
    let kv_dim = (num_kv_heads * head_dim) as i64;
    #[cfg(target_os = "macos")]
    let all_non_empty = (0..num_seqs).all(|s| cu_seqlens_q[s + 1] > cu_seqlens_q[s]);

    #[cfg(target_os = "macos")]
    if unsafe { mlx_sys::mlx_metal_is_available() }
        && k_weight.shape_at(1)? == 4
        && v_weight.shape_at(1)? == 4
        && all_non_empty
    {
        record_ragged_tapes(k_raw, k_histories, cu_seqlens_q, kv_dim, k_tapes)?;
        record_ragged_tapes(v_raw, v_histories, cu_seqlens_q, kv_dim, v_tapes)?;
        let cu_arr = prepare_cu_seqlens(num_seqs, cu_seqlens_q, cu_seqlens_arr)?;
        return dispatch_ragged_kv_histories(
            k_histories,
            v_histories,
            kv_dim,
            |kh, vh, stacked| {
                let (k_cur, v_cur, kh_out, vh_out) =
                    super::kernels::inkling2_sconv_k4_kv_prep_tree(
                        k_raw,
                        v_raw,
                        k_weight,
                        v_weight,
                        dyn_k,
                        dyn_v,
                        cu_arr.as_ref(),
                        kh,
                        vh,
                        num_seqs,
                        stacked,
                        num_kv_heads,
                        head_dim,
                        qk_thresh,
                        v_eps,
                        1e-6,
                        tree_rows,
                    )?;
                Ok(((k_cur, v_cur), kh_out, vh_out))
            },
        );
    }

    if tree_rows.is_some() {
        return Err(Error::from_reason(
            "fused_short_conv_kv_prep_taped_tree: tree_rows requires Metal k=4 kernel",
        ));
    }

    // Fallback
    let t_i64 = k_raw.shape()?[0];
    let act_dtype = k_raw.dtype()?;
    let k_conv_out =
        fused_short_conv_ragged_taped(k_raw, k_weight, dyn_k, k_histories, cu_seqlens_q, k_tapes)?;
    let v_conv_out =
        fused_short_conv_ragged_taped(v_raw, v_weight, dyn_v, v_histories, cu_seqlens_q, v_tapes)?;
    let k_3d = k_conv_out.reshape(&[t_i64, num_kv_heads as i64, head_dim as i64])?;
    let v_3d = v_conv_out.reshape(&[t_i64, num_kv_heads as i64, head_dim as i64])?;
    let k_cur = super::attention::l2_clip(&k_3d, None, f64::from(qk_thresh), act_dtype)?;
    let v_cur = crate::nn::normalization::rms_norm_unscaled(&v_3d.astype(DType::Float32)?, v_eps)?
        .astype(act_dtype)?;
    Ok((k_cur, v_cur))
}

/// Single-dispatch fused K+V causal `k=4` ShortConv + K L2-clip + V weightless RMSNorm across
/// `N >= 1` packed sequences with zero DRAM roundtrip between ShortConv and K/V norm/clip.
#[allow(clippy::too_many_arguments)]
pub fn fused_short_conv_kv_prep_taped(
    k_raw: &MxArray,
    v_raw: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    k_histories: &mut [&mut MxArray],
    v_histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] cu_seqlens_arr: Option<&MxArray>,
    k_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    v_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    num_kv_heads: usize,
    head_dim: usize,
    qk_thresh: f32,
    v_eps: f32,
) -> Result<(MxArray, MxArray)> {
    fused_short_conv_kv_prep_taped_tree(
        k_raw,
        v_raw,
        k_weight,
        v_weight,
        dyn_k,
        dyn_v,
        k_histories,
        v_histories,
        cu_seqlens_q,
        cu_seqlens_arr,
        k_tapes,
        v_tapes,
        num_kv_heads,
        head_dim,
        qk_thresh,
        v_eps,
        None,
    )
}

/// Fused K+V ShortConv + K L2-clip + V RMSNorm + in-place PagedKVWrite (INT8-KTM & BF16)
/// with optional speculative verify `tree_rows`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fused_short_conv_kv_prep_paged_taped_tree(
    k_raw: &MxArray,
    v_raw: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    k_histories: &mut [&mut MxArray],
    v_histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    cu_seqlens_arr: Option<&MxArray>,
    k_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    v_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    k_pool: &MxArray,
    v_pool: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    slot_mapping: &MxArray,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    kv_mode: u32,
    qk_thresh: f32,
    v_eps: f32,
    tree_rows: Option<&MxArray>,
) -> Result<Option<(MxArray, MxArray, MxArray, MxArray)>> {
    let num_seqs = k_histories.len();
    if num_seqs == 0 || v_histories.len() != num_seqs || cu_seqlens_q.len() != num_seqs + 1 {
        return Err(Error::from_reason(
            "fused_short_conv_kv_prep_paged_taped: invalid sequence count",
        ));
    }
    #[cfg(target_os = "macos")]
    let kv_dim = (num_kv_heads * head_dim) as i64;
    #[cfg(target_os = "macos")]
    let all_non_empty = (0..num_seqs).all(|s| cu_seqlens_q[s + 1] > cu_seqlens_q[s]);
    if !unsafe { mlx_sys::mlx_metal_is_available() }
        || k_raw.dtype()? != DType::BFloat16
        || v_raw.dtype()? != DType::BFloat16
        || k_weight.shape_at(1)? != 4
        || v_weight.shape_at(1)? != 4
        || !all_non_empty
    {
        return Ok(None);
    }

    record_ragged_tapes(k_raw, k_histories, cu_seqlens_q, kv_dim, k_tapes)?;
    record_ragged_tapes(v_raw, v_histories, cu_seqlens_q, kv_dim, v_tapes)?;
    let cu_arr = prepare_cu_seqlens(num_seqs, cu_seqlens_q, cu_seqlens_arr)?;

    let out = dispatch_ragged_kv_histories(k_histories, v_histories, kv_dim, |kh, vh, stacked| {
        let (kp_out, vp_out, ks_out, vs_out, kh_out, vh_out) =
            super::kernels::inkling2_sconv_k4_kv_prep_paged_tree(
                k_raw,
                v_raw,
                k_weight,
                v_weight,
                dyn_k,
                dyn_v,
                cu_arr.as_ref(),
                kh,
                vh,
                k_pool,
                v_pool,
                k_scale,
                v_scale,
                slot_mapping,
                num_seqs,
                stacked,
                num_kv_heads,
                head_dim,
                block_size,
                kv_mode,
                qk_thresh,
                v_eps,
                1e-6,
                tree_rows,
            )?;
        Ok(((kp_out, vp_out, ks_out, vs_out), kh_out, vh_out))
    })?;
    Ok(Some(out))
}

/// Fused K+V ShortConv + K L2-clip + V RMSNorm + in-place PagedKVWrite (INT8-KTM & BF16)
/// for `N >= 1` packed sequences with zero intermediate `[T, H_kv, D]` DRAM roundtrip.
///
/// Returns `Ok(Some((k_pool', v_pool', k_scale', v_scale')))` when the fused Metal kernel
/// executed, or `Ok(None)` when the caller should fall back to separate `kv_prep` + `PagedKVWrite`.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn fused_short_conv_kv_prep_paged_taped(
    k_raw: &MxArray,
    v_raw: &MxArray,
    k_weight: &MxArray,
    v_weight: &MxArray,
    dyn_k: Option<&MxArray>,
    dyn_v: Option<&MxArray>,
    k_histories: &mut [&mut MxArray],
    v_histories: &mut [&mut MxArray],
    cu_seqlens_q: &[usize],
    cu_seqlens_arr: Option<&MxArray>,
    k_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    v_tapes: Option<&mut [&mut Option<StreamShortConvTape>]>,
    k_pool: &MxArray,
    v_pool: &MxArray,
    k_scale: &MxArray,
    v_scale: &MxArray,
    slot_mapping: &MxArray,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    kv_mode: u32,
    qk_thresh: f32,
    v_eps: f32,
) -> Result<Option<(MxArray, MxArray, MxArray, MxArray)>> {
    fused_short_conv_kv_prep_paged_taped_tree(
        k_raw,
        v_raw,
        k_weight,
        v_weight,
        dyn_k,
        dyn_v,
        k_histories,
        v_histories,
        cu_seqlens_q,
        cu_seqlens_arr,
        k_tapes,
        v_tapes,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        slot_mapping,
        num_kv_heads,
        head_dim,
        block_size,
        kv_mode,
        qk_thresh,
        v_eps,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_short_conv_prefill_matches_step_by_step_decode() {
        let t = 5;
        let c = 4;
        let k = 4;
        let x_vals: Vec<f32> = (0..(t * c)).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // Oldest-to-newest weights
        let w_vals: Vec<f32> = (0..(c * k)).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();

        let x = MxArray::from_float32(&x_vals, &[t as i64, c as i64]).unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, k as i64]).unwrap();

        // Prefill all T tokens at once
        let mut hist_pre =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let out_pre = static_short_conv(&x, &w, &mut hist_pre).unwrap();
        let pre_vec = out_pre.to_float32().unwrap();

        // Step-by-step decode 1 token at a time
        let mut hist_dec =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let mut dec_vec = Vec::new();
        for step in 0..t {
            let x_step = x
                .slice(&[step as i64, 0], &[(step + 1) as i64, c as i64])
                .unwrap();
            let y_step = static_short_conv(&x_step, &w, &mut hist_dec).unwrap();
            dec_vec.extend_from_slice(&y_step.to_float32().unwrap());
        }

        for (a, b) in pre_vec.iter().zip(dec_vec.iter()) {
            assert!((a - b).abs() < 1e-5, "prefill {a} != decode {b}");
        }

        // Verify exact closed-form value at t=0 (all history is zero):
        // y[0, ch] = (x[0, ch] + w[ch, 3] * x[0, ch]) / sqrt(5)
        let inv_sqrt5 = 1.0_f32 / 5.0_f32.sqrt();
        for ch in 0..c {
            let expected = (x_vals[ch] + w_vals[ch * k + 3] * x_vals[ch]) * inv_sqrt5;
            assert!(
                (pre_vec[ch] - expected).abs() < 1e-5,
                "ch={ch}: got {} expected {expected}",
                pre_vec[ch]
            );
        }
    }

    #[test]
    fn dynamic_short_conv_tap_order_and_prefill_decode_parity() {
        let t = 5;
        let n_kv = 2;
        let head_dim = 2;
        let c = n_kv * head_dim;
        let k = 4;

        let x_vals: Vec<f32> = (0..(t * c)).map(|i| (i as f32 + 1.0) * 0.15).collect();
        let w_vals: Vec<f32> = (0..(c * k))
            .map(|i| ((i % 5) as f32 - 2.0) * 0.25)
            .collect();
        let dyn_vals: Vec<f32> = (0..(t * n_kv * k))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
            .collect();

        let x = MxArray::from_float32(&x_vals, &[t as i64, c as i64]).unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, k as i64]).unwrap();
        let dyn_taps =
            MxArray::from_float32(&dyn_vals, &[t as i64, n_kv as i64, k as i64]).unwrap();

        let mut hist_pre =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let out_pre = dynamic_short_conv(&x, &w, &dyn_taps, &mut hist_pre).unwrap();
        let pre_vec = out_pre.to_float32().unwrap();

        let mut hist_dec =
            MxArray::zeros(&[(k - 1) as i64, c as i64], Some(DType::Float32)).unwrap();
        let mut dec_vec = Vec::new();
        for step in 0..t {
            let x_step = x
                .slice(&[step as i64, 0], &[(step + 1) as i64, c as i64])
                .unwrap();
            let d_step = dyn_taps
                .slice(
                    &[step as i64, 0, 0],
                    &[(step + 1) as i64, n_kv as i64, k as i64],
                )
                .unwrap();
            let y_step = dynamic_short_conv(&x_step, &w, &d_step, &mut hist_dec).unwrap();
            dec_vec.extend_from_slice(&y_step.to_float32().unwrap());
        }

        for (a, b) in pre_vec.iter().zip(dec_vec.iter()) {
            assert!((a - b).abs() < 1e-5, "prefill {a} != decode {b}");
        }

        // Verify closed-form at t=3 (full 4 lags present):
        // lag l uses w[ch, 3 - l] + dyn[3, ch / head_dim, l]
        for ch in 0..c {
            let h = ch / head_dim;
            let mut sum = x_vals[3 * c + ch];
            for lag in 0..k {
                let w_l = w_vals[ch * k + (k - 1 - lag)];
                let d_l = dyn_vals[(3 * n_kv + h) * k + lag];
                let x_l = x_vals[(3 - lag) * c + ch];
                sum += (w_l + d_l) * x_l;
            }
            let expected = sum / 3.0;
            assert!(
                (pre_vec[3 * c + ch] - expected).abs() < 1e-5,
                "t=3 ch={ch}: got {} expected {expected}",
                pre_vec[3 * c + ch]
            );
        }
    }

    #[test]
    fn short_conv_tape_commit_prefix_exact_for_all_keep_lengths() {
        let c = 8usize;
        let k = 4usize;
        let hist_len = k - 1;
        let verify_rows = 8usize;

        let init_hist_vals: Vec<f32> = (0..(hist_len * c))
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        let x_vals: Vec<f32> = (0..(verify_rows * c))
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let w_vals: Vec<f32> = (0..(c * k)).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();

        let init_hist = MxArray::from_float32(&init_hist_vals, &[hist_len as i64, c as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let x = MxArray::from_float32(&x_vals, &[verify_rows as i64, c as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, k as i64]).unwrap();

        let mut verify_hist = init_hist.clone();
        let mut tape_slot: Option<StreamShortConvTape> = None;
        let _y = fused_short_conv_ragged_taped(
            &x,
            &w,
            None,
            &mut [&mut verify_hist],
            &[0, verify_rows],
            Some(&mut [&mut tape_slot]),
        )
        .unwrap();
        let tape = tape_slot.expect("tape recorded");

        // Compare `tape.commit_prefix(keep)` against step-by-step exact history for EVERY keep in 0..=8
        for keep in 0..=verify_rows {
            let mut ref_hist = init_hist.clone();
            for step in 0..keep {
                let x_row = x
                    .slice(&[step as i64, 0], &[(step + 1) as i64, c as i64])
                    .unwrap();
                let _ = static_short_conv(&x_row, &w, &mut ref_hist).unwrap();
            }
            let got = tape
                .commit_prefix(keep)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let exp = ref_hist
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(got.len(), exp.len());
            for (idx, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "keep={keep} idx={idx}: tape {a} != ref {b}"
                );
            }
        }
    }

    #[test]
    fn test_short_conv_tape_commit_tree_matches_branch_execution() {
        let c = 6usize;
        let hist_len = 3usize;
        let verify_rows = 8usize;

        let init_hist_vals: Vec<f32> = (0..(hist_len * c))
            .map(|i| -10.0 - i as f32 * 0.25)
            .collect();
        let x_vals: Vec<f32> = (0..(verify_rows * c))
            .map(|i| 1.0 + i as f32 * 0.5)
            .collect();
        let w_vals: Vec<f32> = (0..(c * 4)).map(|i| (i % 5) as f32 * 0.1).collect();

        let init_hist = MxArray::from_float32(&init_hist_vals, &[hist_len as i64, c as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let rows = MxArray::from_float32(&x_vals, &[verify_rows as i64, c as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let w = MxArray::from_float32(&w_vals, &[c as i64, 4]).unwrap();

        let tape = StreamShortConvTape {
            hist_before: init_hist.clone(),
            rows: rows.clone(),
        };

        let cases: &[(&[u32], bool)] = &[
            (&[], false),
            (&[0, 1, 2], true),
            (&[0, 1, 2, 3], true),
            (&[0], false),
            (&[0, 3], false),
            (&[0, 2, 5], false),
            (&[0, 2, 5, 7], false),
            (&[1, 4, 6, 7], false),
        ];

        for &(path, is_contiguous) in cases {
            let mut ref_hist = init_hist.clone();
            for &node in path {
                let row = rows
                    .slice(&[node as i64, 0], &[node as i64 + 1, c as i64])
                    .unwrap();
                let _ = static_short_conv(&row, &w, &mut ref_hist).unwrap();
            }
            let got = tape
                .commit_tree(path, is_contiguous)
                .unwrap()
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            let exp = ref_hist
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap();
            assert_eq!(got.len(), exp.len());
            for (idx, (a, b)) in got.iter().zip(exp.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "path={path:?} idx={idx}: got {a} != exp {b}"
                );
            }
        }

        assert!(tape.commit_tree(&[0, 8], false).is_err());
    }

    #[test]
    fn test_full_48_visit_tape_batched_commit_prefix_and_tree_parity() {
        let num_visits = 48usize;
        let kv_dim = 16usize;
        let hidden = 32usize;
        let verify_rows = 8usize;

        let mut tape = Inkling2ShortConvTape::new(num_visits);
        let init_state =
            Inkling2ShortConvState::new(num_visits, 4, kv_dim, hidden, DType::BFloat16).unwrap();

        for (v_idx, tv) in tape.visits.iter_mut().enumerate() {
            let base = (v_idx as f32) * 0.1;
            let make_pair = |c: usize, salt: f32| {
                let h_vals: Vec<f32> = (0..(3 * c))
                    .map(|i| base + salt - (i as f32) * 0.05)
                    .collect();
                let r_vals: Vec<f32> = (0..(verify_rows * (c + 8)))
                    .map(|i| base + salt + (i as f32) * 0.025)
                    .collect();
                let h = MxArray::from_float32(&h_vals, &[3, c as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                // Use strided column slices to verify strided rows_in handling!
                let big = MxArray::from_float32(&r_vals, &[verify_rows as i64, (c + 8) as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let r = big
                    .slice(&[0, 4], &[verify_rows as i64, (4 + c) as i64])
                    .unwrap();
                StreamShortConvTape {
                    hist_before: h,
                    rows: r,
                }
            };
            tv.k = Some(make_pair(kv_dim, 1.0));
            tv.v = Some(make_pair(kv_dim, 2.0));
            tv.attn_post = Some(make_pair(hidden, 3.0));
            tv.mlp_post = Some(make_pair(hidden, 4.0));
        }

        for &(path, is_contiguous) in &[
            (&[][..], false),
            (&[0, 1][..], true),
            (&[0, 1, 2, 3, 4][..], true),
            (&[0, 3][..], false),
            (&[0, 2, 5, 7][..], false),
        ] {
            let mut got_state = init_state.clone();
            tape.commit_tree(path, is_contiguous, &mut got_state)
                .unwrap();

            for (v_idx, (tv, sv)) in tape.visits.iter().zip(got_state.visits.iter()).enumerate() {
                let mut ref_v = init_state.visits[v_idx].clone();
                tv.commit_tree(path, is_contiguous, &mut ref_v).unwrap();
                for (name, got, want) in [
                    ("k", &sv.k, &ref_v.k),
                    ("v", &sv.v, &ref_v.v),
                    ("attn_post", &sv.attn_post, &ref_v.attn_post),
                    ("mlp_post", &sv.mlp_post, &ref_v.mlp_post),
                ] {
                    let g = got.astype(DType::Float32).unwrap().to_float32().unwrap();
                    let w = want.astype(DType::Float32).unwrap().to_float32().unwrap();
                    assert_eq!(
                        g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "visit={v_idx} stream={name} path={path:?}"
                    );
                }
            }
        }

        let mut err_state = init_state.clone();
        assert!(tape.commit_prefix(verify_rows + 1, &mut err_state).is_err());
        assert!(
            tape.commit_tree(&[0, verify_rows as u32], false, &mut err_state)
                .is_err()
        );
    }
}
