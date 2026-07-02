use crate::array::MxArray;
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Whether the GDN `fast::metal_kernel` kernels can run on this host. False on
/// the CUDA/Linux build, where they throw — callers must use the ops path.
/// Delegates to the shared, cached `mlx_metal_is_available()` probe.
fn metal_kernel_backend_available() -> bool {
    crate::engine::persistence::compiled_forward_backend_available()
}

/// Minimum sequence length for the chunked prefill kernel to even be *eligible*.
/// Below this the per-step recurrence always wins, so chunked is never considered.
/// (Chunked is opt-in only — see [`GdnKernel`] / [`should_use_chunked`].)
const CHUNK_THRESHOLD: i64 = 64;

/// GDN recurrence kernel selection. **Per-step is the default on EVERY GPU generation.**
///
/// History (corrected 2026-06-04): a `gen >= 17` (M5) gate once routed long prefills to the
/// chunked kernel on the unvalidated theory that M5's memory bandwidth made its `O(BT^2)`
/// tiling a net win. Measured on an M5 Max (gen 17, isolated worktree): the chunked kernel is
/// **2.8–3.5× SLOWER** end-to-end prefill TTFT than per-step (24–31× slower per isolated GDN
/// call) at `Hv=32, B=1` across 580–5384 prompt tokens — and it is ~2× slower on M3 too. The
/// chunked Metal kernel (`gated_delta_chunked.metal.inc`) is pure scalar-FMA + `simd_sum`
/// reductions with ZERO `simdgroup_matrix` / NAX matmul, so it never had a tensor-core
/// advantage. The gen gate was a stale inversion of an old M3 result that was never A/B'd on
/// M5; it is removed. Per-step is already the canonical path on M1–M4, for all `seq < 64`, and
/// all masked GDN calls — so per-step is the de-facto
/// reference everywhere.
///
/// Chunked is retained behind `MLX_GDN_KERNEL=chunked` for A/B and bring-up only. NOTE: the two
/// kernels are NOT token-identical — they differ by 1–2 bf16 ULP (two valid reduction
/// orderings), which can flip a greedy argmax and change the continuation on some long prompts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GdnKernel {
    /// Measured-best default: per-step on every arch.
    Auto,
    /// Force the per-step recurrence (also the `MLX_GDN_FORCE_PERSTEP=1` legacy toggle).
    ForcePerStep,
    /// Force the chunked prefill kernel (A/B only — changes output by 1–2 bf16 ULP).
    ForceChunked,
    /// Force the device-agnostic chunk-parallel ops path (`gated_delta_chunked_ops`).
    /// This is the default on the CUDA ops path (where the Metal kernels are absent);
    /// `MLX_GDN_KERNEL=perstep` reverts it for same-binary A/B. No effect on Metal,
    /// whose production path takes the `use_kernel=true` per-step/chunked kernels.
    ForceChunkedOps,
}

/// Read the `MLX_GDN_KERNEL` override fresh per call (`perstep` | `chunked`); also honors the
/// legacy `MLX_GDN_FORCE_PERSTEP=1`. Anything else (incl. unset) → [`GdnKernel::Auto`].
fn gdn_kernel_override() -> GdnKernel {
    parse_gdn_kernel(
        std::env::var("MLX_GDN_KERNEL").ok().as_deref(),
        std::env::var("MLX_GDN_FORCE_PERSTEP").ok().as_deref(),
    )
}

/// Pure parse of the GDN-kernel env overrides (kept env-free for race-free testing).
/// `MLX_GDN_KERNEL` takes precedence; the legacy `MLX_GDN_FORCE_PERSTEP=1/true/on` is a
/// per-step-only fallback. Unrecognized / both-unset → [`GdnKernel::Auto`].
fn parse_gdn_kernel(mlx_gdn_kernel: Option<&str>, legacy_force_perstep: Option<&str>) -> GdnKernel {
    if let Some(v) = mlx_gdn_kernel {
        match v.trim().to_ascii_lowercase().as_str() {
            "perstep" | "per_step" | "per-step" | "step" => return GdnKernel::ForcePerStep,
            "chunked_ops" | "chunkedops" | "chunked-ops" | "ops" => {
                return GdnKernel::ForceChunkedOps;
            }
            "chunked" | "chunk" => return GdnKernel::ForceChunked,
            _ => {}
        }
    }
    if matches!(
        legacy_force_perstep.map(str::trim),
        Some("1") | Some("true") | Some("on")
    ) {
        return GdnKernel::ForcePerStep;
    }
    GdnKernel::Auto
}

/// Pure routing predicate: should this GDN call take the chunked prefill kernel?
///
/// `Auto` is ALWAYS false — per-step is faster on every measured arch, so chunked only runs
/// when explicitly forced AND the call is a long (`seq >= CHUNK_THRESHOLD`), unmasked prefill
/// the chunked kernel can actually handle. `_gpu_gen` is retained for documentation and to make
/// any future arch-gating a localized one-line change; `Auto` ignores it today (M5 included).
fn should_use_chunked(seq_len: i64, mask_is_none: bool, _gpu_gen: i32, choice: GdnKernel) -> bool {
    // Chunked has no masked variant and loses on short sequences — never eligible there.
    if !mask_is_none || seq_len < CHUNK_THRESHOLD {
        return false;
    }
    match choice {
        GdnKernel::ForceChunked => true,
        // ChunkedOps is the CUDA ops-path selector, not the Metal chunked kernel this
        // predicate guards — it never selects the Metal kernel here.
        GdnKernel::Auto | GdnKernel::ForcePerStep | GdnKernel::ForceChunkedOps => false,
    }
}

/// Returns the GPU architecture generation, cached after first call.
fn gpu_architecture_gen() -> i32 {
    use std::sync::OnceLock;
    static GEN: OnceLock<i32> = OnceLock::new();
    *GEN.get_or_init(|| unsafe { sys::mlx_gpu_architecture_gen() })
}

/// Compute decay gate: g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// Uses a fused C++ implementation that builds the full expression in a single
/// FFI call, allowing MLX's graph optimizer to see the complete expression.
///
/// Shapes:
///   A_log: [Hv]
///   a: [B, T, Hv]
///   dt_bias: [Hv]
///
/// Returns: [B, T, Hv]
fn compute_g(a_log: &MxArray, a: &MxArray, dt_bias: &MxArray) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_fused_compute_g(a_log.as_raw_ptr(), a.as_raw_ptr(), dt_bias.as_raw_ptr())
    };
    MxArray::from_handle(handle, "fused_compute_g")
}

/// Log-space decay gate: `g_log = -exp(a_log) * softplus(a + dt_bias)` = `log(compute_g(...))`,
/// computed DIRECTLY rather than as `compute_g(...).log()`. Strong decay drives the exp-space
/// gate `compute_g` to underflow to 0, so `log(g)` is `-inf`; the chunked path then forms
/// `gcum_i - gcum_j = (-inf) - (-inf) = NaN` and emits garbage. The log-space form stays finite
/// (softplus is numerically stable). Mirrors the native `g_log` the fused Metal gating returns.
fn compute_g_log(a_log: &MxArray, a: &MxArray, dt_bias: &MxArray) -> Result<MxArray> {
    use crate::array::DType;
    let f32 = DType::Float32;
    let sp = Activations::softplus(&a.astype(f32)?.add(&dt_bias.astype(f32)?)?)?;
    let scale = a_log.astype(f32)?.exp()?;
    sp.mul(&scale)?.negative()
}

/// Fused gating: computes both beta and g in a single Metal kernel dispatch.
///
/// beta = sigmoid(b)
/// g = -exp(a_log) * softplus(a + dt_bias)
///
/// Returns: (beta [B, T, Hv] in input dtype, g [B, T, Hv] in f32)
fn fused_gdn_gating(
    b: &MxArray,
    a: &MxArray,
    a_log: &MxArray,
    dt_bias: &MxArray,
    num_heads: i32,
) -> Result<(MxArray, MxArray)> {
    let total_elements = b.size()? as i32;
    let mut out_beta: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_g: *mut sys::mlx_array = std::ptr::null_mut();

    let ok = unsafe {
        sys::mlx_fused_gdn_gating(
            b.as_raw_ptr(),
            a.as_raw_ptr(),
            a_log.as_raw_ptr(),
            dt_bias.as_raw_ptr(),
            num_heads,
            total_elements,
            &mut out_beta,
            &mut out_g,
        )
    };

    if !ok {
        return Err(Error::from_reason("Fused GDN gating kernel failed"));
    }

    let beta = MxArray::from_handle(out_beta, "fused_gating:beta")?;
    let g = MxArray::from_handle(out_g, "fused_gating:g")?;
    Ok((beta, g))
}

/// Chunked gated delta recurrence for prefill (BT=32 tokens per chunk).
/// Processes multiple tokens in parallel within each chunk.
fn gated_delta_chunked(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let mut out_y: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_state: *mut sys::mlx_array = std::ptr::null_mut();

    let ok = unsafe {
        sys::mlx_gated_delta_chunked(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state.as_raw_ptr(),
            &mut out_y,
            &mut out_state,
        )
    };

    if !ok {
        return Err(Error::from_reason(
            "Chunked gated delta kernel failed (check stderr for details)",
        ));
    }

    let y = MxArray::from_handle(out_y, "gated_delta_chunked:y")?;
    let new_state = MxArray::from_handle(out_state, "gated_delta_chunked:state")?;
    Ok((y, new_state))
}

/// Run the gated delta recurrence using a custom Metal kernel.
///
/// This dispatches to a fused GPU kernel that keeps recurrent state in
/// thread-local registers and uses SIMD reductions, avoiding per-timestep
/// kernel launches. ~10x faster than the ops-based sequential loop.
///
/// Shapes:
///   q: [B, T, Hk, Dk]  (already GQA-expanded to Hv heads by caller)
///   k: [B, T, Hk, Dk]  (already GQA-expanded)
///   v: [B, T, Hv, Dv]
///   g: [B, T, Hv]       - decay gate
///   beta: [B, T, Hv]    - beta (sigmoid already applied)
///   state: [B, Hv, Dv, Dk] - recurrent state
///   mask: Option<[B, T]>
///
/// Returns: (output [B, T, Hv, Dv], final_state [B, Hv, Dv, Dk])
fn gated_delta_kernel(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    mask: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let mut out_y: *mut sys::mlx_array = std::ptr::null_mut();
    let mut out_state: *mut sys::mlx_array = std::ptr::null_mut();

    let mask_ptr = match mask {
        Some(m) => m.as_raw_ptr(),
        None => std::ptr::null_mut(),
    };

    let ok = unsafe {
        sys::mlx_gated_delta_kernel(
            q.as_raw_ptr(),
            k.as_raw_ptr(),
            v.as_raw_ptr(),
            g.as_raw_ptr(),
            beta.as_raw_ptr(),
            state.as_raw_ptr(),
            mask_ptr,
            &mut out_y,
            &mut out_state,
        )
    };

    if !ok {
        return Err(Error::from_reason(
            "Metal gated delta kernel failed (check stderr for details)",
        ));
    }

    let y = MxArray::from_handle(out_y, "gated_delta_kernel:y")?;
    let new_state = MxArray::from_handle(out_state, "gated_delta_kernel:state")?;
    Ok((y, new_state))
}

/// Per-step GDN recurrence record for the eager MTP tape replay.
///
/// Captures the EXACT inputs passed to [`gated_delta_kernel`] for the whole
/// `[B, T, ...]` verify window — `q`/`k` are already GQA-expanded and
/// RMS-norm-scaled, `g` is the post-`exp` decay (`g_log.exp()`), and `beta`
/// is post-sigmoid. All handles are lazy `MxArray` clones (no eval, no copy).
///
/// On accept the replay slices each window tensor to step `t` as `[B, 1, ...]`
/// and re-runs [`gated_delta_kernel`] AT T=1 per accepted step, threading the
/// bf16 recurrent state between calls. Re-running the SAME kernel AR uses at
/// T=1 reproduces the per-token bf16 round-trip of true autoregressive decode
/// by construction — the windowed verify kernel keeps state fp32 across the
/// whole window, which is the divergence the replay corrects.
#[derive(Clone)]
pub(crate) struct GdnKernelTape {
    /// Queries `[B, T, Hv, Dk]` (GQA-expanded, RMS-norm-scaled).
    pub q: MxArray,
    /// Keys `[B, T, Hv, Dk]` (GQA-expanded, RMS-norm-scaled).
    pub k: MxArray,
    /// Values `[B, T, Hv, Dv]`.
    pub v: MxArray,
    /// Decay gate `[B, T, Hv]` (post-`exp`, i.e. `g_log.exp()`).
    pub g: MxArray,
    /// Beta `[B, T, Hv]` (post-sigmoid).
    pub beta: MxArray,
}

impl GdnKernelTape {
    /// Number of recorded window steps (`T`, = `depth + 1`).
    pub(crate) fn window_len(&self) -> Result<i64> {
        self.q.shape_at(1)
    }

    /// Replay the first `accepted_steps` recorded steps at T=1, threading the
    /// bf16 recurrent state between kernel calls. Starts from `start_state`
    /// (the pre-verify snapshot's bf16 recurrent state) and returns the
    /// AR-exact carried state after `accepted_steps` tokens.
    ///
    /// Each T=1 [`gated_delta_kernel`] call casts the recurrent state to bf16
    /// at the end (matching the AR per-token decode), so threading the bf16
    /// state across the loop reproduces autoregressive decode bit-for-bit.
    pub(crate) fn replay_recurrent_state(
        &self,
        start_state: &MxArray,
        accepted_steps: usize,
    ) -> Result<MxArray> {
        let mut state = start_state.clone();
        for t in 0..accepted_steps as i64 {
            let q_t = self.q.slice_axis(1, t, t + 1)?; // [B, 1, Hv, Dk]
            let k_t = self.k.slice_axis(1, t, t + 1)?;
            let v_t = self.v.slice_axis(1, t, t + 1)?;
            let g_t = self.g.slice_axis(1, t, t + 1)?; // [B, 1, Hv]
            let beta_t = self.beta.slice_axis(1, t, t + 1)?;
            // mask=None: the verify forward runs unmasked (same as AR decode),
            // so the replay must too.
            let (_y, new_state) =
                gated_delta_kernel(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, None)?;
            state = new_state;
        }
        Ok(state)
    }
}

/// Single timestep of the gated delta recurrence (delta rule) — ops-based fallback.
///
/// Shapes:
///   q_t: [B, 1, Hv, Dk]  (already GQA-expanded)
///   k_t: [B, 1, Hv, Dk]  (already GQA-expanded)
///   v_t: [B, 1, Hv, Dv]
///   g_t: [B, 1, Hv]
///   beta_t: [B, 1, Hv]
///   state: [B, Hv, Dv, Dk]
///   mask_t: [B, 1] or None
///   batch, num_v_heads, k_dim, v_dim: pre-extracted dimensions (avoids per-step FFI calls)
///
/// Returns: (output [B, 1, Hv, Dv], new_state [B, Hv, Dv, Dk])
fn gated_delta_step(
    q_t: &MxArray,
    k_t: &MxArray,
    v_t: &MxArray,
    g_t: &MxArray,
    beta_t: &MxArray,
    state: &MxArray,
    mask_t: Option<&MxArray>,
    batch: i64,
    num_v_heads: i64,
    k_dim: i64,
    v_dim: i64,
) -> Result<(MxArray, MxArray)> {
    // Squeeze time dimension: [B, 1, H, D] → [B, H, D]
    let q = q_t.squeeze(Some(&[1]))?; // [B, Hv, Dk]
    let k = k_t.squeeze(Some(&[1]))?; // [B, Hv, Dk]
    let v = v_t.squeeze(Some(&[1]))?; // [B, Hv, Dv]
    let g = g_t.squeeze(Some(&[1]))?; // [B, Hv]
    let beta = beta_t.squeeze(Some(&[1]))?; // [B, Hv]

    // Save old state for mask restore
    let old_state = state.clone();

    // 1. Decay existing state: state *= g[..., None, None]
    let g_4d = g.reshape(&[batch, num_v_heads, 1, 1])?;
    let state = state.mul(&g_4d)?; // [B, Hv, Dv, Dk]

    // 2. Compute kv_mem = (state * k[..., None, :]).sum(-1) → [B, Hv, Dv]
    //    k: [B, Hv, Dk] → [B, Hv, 1, Dk]
    let k_4d = k.reshape(&[batch, num_v_heads, 1, k_dim])?;
    let state_k = state.mul(&k_4d)?; // [B, Hv, Dv, Dk]
    let kv_mem = state_k.sum(Some(&[-1]), Some(false))?; // [B, Hv, Dv]

    // 3. Compute delta = (v - kv_mem) * beta[..., None] → [B, Hv, Dv]
    let v_minus_kv = v.sub(&kv_mem)?; // [B, Hv, Dv]
    let beta_3d = beta.reshape(&[batch, num_v_heads, 1])?;
    let delta = v_minus_kv.mul(&beta_3d)?; // [B, Hv, Dv]

    // 4. Update state: state += k[..., None, :] * delta[..., None]
    //    k[..., None, :]: [B, Hv, 1, Dk]
    //    delta[..., None]: [B, Hv, Dv, 1]
    let delta_4d = delta.reshape(&[batch, num_v_heads, v_dim, 1])?;
    let k_delta = k_4d.mul(&delta_4d)?; // [B, Hv, Dv, Dk]
    let new_state = state.add(&k_delta)?; // [B, Hv, Dv, Dk]

    // 5. Output: y = (new_state * q[..., None, :]).sum(-1) → [B, Hv, Dv]
    let q_4d = q.reshape(&[batch, num_v_heads, 1, k_dim])?;
    let state_q = new_state.mul(&q_4d)?; // [B, Hv, Dv, Dk]
    let y = state_q.sum(Some(&[-1]), Some(false))?; // [B, Hv, Dv]

    // Apply mask: restore old state for masked positions (not zero output)
    let new_state = if let Some(m) = mask_t {
        // m: [B, 1] → [B, 1, 1, 1] for broadcasting with [B, Hv, Dv, Dk]
        let m_4d = m.reshape(&[batch, 1, 1, 1])?;
        m_4d.where_(&new_state, &old_state)?
    } else {
        new_state
    };

    // Add time dimension back: [B, Hv, Dv] → [B, 1, Hv, Dv]
    let y_out = y.reshape(&[batch, 1, num_v_heads, v_dim])?;

    Ok((y_out, new_state))
}

/// Ops-based sequential loop fallback for the gated delta recurrence.
fn gated_delta_ops(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    mask: Option<&MxArray>,
) -> Result<(MxArray, MxArray)> {
    let seq_len = q.shape_at(1)?;

    // Extract dimensions once to avoid per-step FFI calls in gated_delta_step
    let batch = q.shape_at(0)?;
    let num_v_heads = v.shape_at(2)?;
    let k_dim = q.shape_at(3)?;
    let v_dim = v.shape_at(3)?;

    let mut current_state = state.clone();
    let mut outputs: Vec<MxArray> = Vec::with_capacity(seq_len as usize);

    for t in 0..seq_len {
        // Slice timestep t: [B, 1, ...]
        let q_t = q.slice_axis(1, t, t + 1)?;
        let k_t = k.slice_axis(1, t, t + 1)?;
        let v_t = v.slice_axis(1, t, t + 1)?;
        let g_t = g.slice_axis(1, t, t + 1)?;
        let beta_t = beta.slice_axis(1, t, t + 1)?;

        let mask_t = mask.map(|m| m.slice_axis(1, t, t + 1)).transpose()?;

        let (y_t, new_state) = gated_delta_step(
            &q_t,
            &k_t,
            &v_t,
            &g_t,
            &beta_t,
            &current_state,
            mask_t.as_ref(),
            batch,
            num_v_heads,
            k_dim,
            v_dim,
        )?;

        outputs.push(y_t);
        current_state = new_state;
    }

    // Concatenate along time dimension: [B, T, Hv, Dv]
    let output_refs: Vec<&MxArray> = outputs.iter().collect();
    let output = MxArray::concatenate_many(output_refs, Some(1))?;

    Ok((output, current_state))
}

/// Stable `(I + A)^-1` for a batched strictly-lower-triangular `A` of size `l x l`
/// (the matrix is the last two axes). `A` is nilpotent, so the inverse is the finite sum
/// `sum_m (-A)^m`; but forming it by repeated squaring overflows f32 at `l = 64` (the
/// intermediate powers reach ~1e57 before nilpotency zeros them) and loses precision
/// whenever `||A||` is large. Instead we build `M` directly by row-wise forward
/// substitution, which never forms a power of `A`:
///   `(I + A) M = I`  =>  row i:  `M[i, :] = e_i - A[i, :] @ M`  (A[i, j] = 0 for j >= i,
/// so this only uses rows `< i`, already finalized). This matches the FLA / vLLM
/// `solve_tril` reference. The `l - 1` steps are sequential but batched over all chunks,
/// so the depth is independent of sequence length.
fn invert_i_plus_strict_lower(a: &MxArray, l: i64) -> Result<MxArray> {
    use crate::array::DType;
    let f32 = DType::Float32;
    let eye = MxArray::eye(l as i32, None, None, Some(f32))?; // [L, L]
    let zeros_col = MxArray::zeros(&[l, 1], Some(f32))?;
    let ai = a.ndim()? as usize - 2; // row axis of the [.., L, L] matrix
    let mut m = eye.clone(); // M starts as I; row 0 (= e_0) is already final.
    for i in 1..l {
        let a_row = a.slice_axis(ai, i, i + 1)?; // [.., 1, L] = A[i, :]
        let am = a_row.matmul(&m)?; // [.., 1, L] = A[i, :] @ M  (uses rows < i)
        let row_i = eye.slice_axis(0, i, i + 1)?.sub(&am)?; // e_i - A[i,:]@M -> [.., 1, L]
        // Scatter row_i into row i of M: column i of the identity is the one-hot row mask.
        let row_mask = eye.slice_axis(1, i, i + 1)?.greater(&zeros_col)?; // [L, 1] bool
        m = row_mask.where_(&row_i, &m)?;
    }
    Ok(m)
}

/// Chunk-parallel port of the per-step recurrence (`gated_delta_ops`) for the CUDA
/// prefill path. Collapses the O(T) token-serial recurrence into O(T/BT) chunk-serial
/// steps of dense batched matmuls (cuBLAS / tensor cores), matching the in-tree Metal
/// chunked kernel's math (`crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc`).
///
/// Device-agnostic (portable MxArray ops) so it also runs on Metal, where the unit-test
/// parity check against `gated_delta_ops` lives. `g_log` is the LOG-space decay gate
/// (negative); `beta` has sigmoid applied; `state` is `[B, Hv, Dv, Dk]` (value-major).
/// q,k,v are already GQA-expanded to Hv. All accumulation is f32; output and final state
/// are cast back to the model dtype. Inputs are zero-padded to whole `BT`-token chunks
/// (zero — not -inf — so padded tokens contribute nothing without `0*inf=NaN`).
fn gated_delta_chunked_ops(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g_log: &MxArray,
    beta: &MxArray,
    state: &MxArray,
) -> Result<(MxArray, MxArray)> {
    use crate::array::DType;
    const BT: i64 = 64;

    let b = q.shape_at(0)?;
    let t = q.shape_at(1)?;
    let hv = q.shape_at(2)?;
    let dk = q.shape_at(3)?;
    let dv = v.shape_at(3)?;
    let c = (t + BT - 1) / BT;
    let tpad = c * BT;
    let pad_t = (tpad - t) as i32;

    let out_dtype = v.dtype()?;
    let state_dtype = state.dtype()?;
    let f32 = DType::Float32;

    // Promote to f32: state must not drift in bf16 across the chunk-carry.
    let q = q.astype(f32)?;
    let k = k.astype(f32)?;
    let v = v.astype(f32)?;
    let g_log = g_log.astype(f32)?;
    let beta = beta.astype(f32)?;
    let s0 = state.astype(f32)?; // [B, Hv, Dv, Dk]

    // Zero-pad time to a whole number of chunks (padded tokens contribute nothing).
    let (q, k, v, g_log, beta) = if pad_t > 0 {
        (
            q.pad(&[0, 0, 0, pad_t, 0, 0, 0, 0], 0.0)?,
            k.pad(&[0, 0, 0, pad_t, 0, 0, 0, 0], 0.0)?,
            v.pad(&[0, 0, 0, pad_t, 0, 0, 0, 0], 0.0)?,
            g_log.pad(&[0, 0, 0, pad_t, 0, 0], 0.0)?,
            beta.pad(&[0, 0, 0, pad_t, 0, 0], 0.0)?,
        )
    } else {
        (q, k, v, g_log, beta)
    };

    // [B, Tpad, Hv, *] -> [B, C, BT, Hv, *] -> [B, C, Hv, BT, *] (batch matmul over B,C,Hv).
    let q = q
        .reshape(&[b, c, BT, hv, dk])?
        .transpose(Some(&[0, 1, 3, 2, 4]))?;
    let k = k
        .reshape(&[b, c, BT, hv, dk])?
        .transpose(Some(&[0, 1, 3, 2, 4]))?;
    let v = v
        .reshape(&[b, c, BT, hv, dv])?
        .transpose(Some(&[0, 1, 3, 2, 4]))?;
    let g_log = g_log
        .reshape(&[b, c, BT, hv])?
        .transpose(Some(&[0, 1, 3, 2]))?;
    let beta = beta
        .reshape(&[b, c, BT, hv])?
        .transpose(Some(&[0, 1, 3, 2]))?;

    // Cumulative decay within each chunk.
    let gcum = g_log.cumsum(3)?; // inclusive prefix sum -> [B,C,Hv,BT]
    let decay_self = gcum.exp()?; // exp(gcum[i])
    // decay_mat[i,j] = exp(gcum[i]-gcum[j]); upper-tri may overflow but is always masked to 0.
    let decay_mat = gcum.expand_dims(4)?.sub(&gcum.expand_dims(3)?)?.exp()?; // [B,C,Hv,BT,BT]

    // Positional triangular masks (BT x BT).
    let idx = MxArray::arange(0.0, BT as f64, None, Some(f32))?;
    let idx_i = idx.reshape(&[BT, 1])?;
    let idx_j = idx.reshape(&[1, BT])?;
    let strict_lower = idx_j.less(&idx_i)?; // j < i
    let incl_lower = idx_j.less_equal(&idx_i)?; // j <= i
    let zeros_bb = MxArray::zeros(&[BT, BT], Some(f32))?;

    // WY system matrix A[i,j] = (j<i) * beta_i * (k_i . k_j) * decay_mat[i,j].
    let kk = k.matmul(&k.transpose(Some(&[0, 1, 2, 4, 3]))?)?;
    let beta_col = beta.expand_dims(4)?; // [B,C,Hv,BT,1]
    let a = strict_lower.where_(&kk.mul(&decay_mat)?.mul(&beta_col)?, &zeros_bb)?;

    // M = (I + A)^-1. A is strictly-lower nilpotent, but plain repeated squaring overflows
    // f32 at BT=64 (intermediate powers ~1e57); invert_i_plus_strict_lower splits into f32-safe
    // 32-blocks. See that function for the stability bound.
    let m = invert_i_plus_strict_lower(&a, BT)?;

    // S_in-independent WY factors (parallel over all chunks).
    let u = m.matmul(&v.mul(&beta_col)?)?; // M @ (beta*v) -> [B,C,Hv,BT,Dv]
    let bdk = k.mul(&beta.mul(&decay_self)?.expand_dims(4)?)?; // (beta*decay_self)*k
    let w = m.matmul(&bdk)?; // [B,C,Hv,BT,Dk]

    // Serial carry over chunks: the ONLY sequential dependency (C = T/BT steps, not T).
    let mut s_cur = s0; // [B,Hv,Dv,Dk]
    let mut delta_chunks: Vec<MxArray> = Vec::with_capacity(c as usize);
    let mut sin_chunks: Vec<MxArray> = Vec::with_capacity(c as usize);
    for ci in 0..c {
        let u_c = u.slice_axis(1, ci, ci + 1)?.squeeze(Some(&[1]))?; // [B,Hv,BT,Dv]
        let w_c = w.slice_axis(1, ci, ci + 1)?.squeeze(Some(&[1]))?; // [B,Hv,BT,Dk]
        let k_c = k.slice_axis(1, ci, ci + 1)?.squeeze(Some(&[1]))?; // [B,Hv,BT,Dk]
        let dm_c = decay_mat.slice_axis(1, ci, ci + 1)?.squeeze(Some(&[1]))?; // [B,Hv,BT,BT]
        let ds_c = decay_self.slice_axis(1, ci, ci + 1)?.squeeze(Some(&[1]))?; // [B,Hv,BT]

        // delta_c = u_c - W_c @ S_in^T.
        let sin_t = s_cur.transpose(Some(&[0, 1, 3, 2]))?; // [B,Hv,Dk,Dv]
        let delta_c = u_c.sub(&w_c.matmul(&sin_t)?)?; // [B,Hv,BT,Dv]

        // S_out = decay_total * S_in + delta_c^T @ (k_c * decay_to_end).
        let dte = dm_c.slice_axis(2, BT - 1, BT)?.squeeze(Some(&[2]))?; // last row [B,Hv,BT]
        let kd = k_c.mul(&dte.expand_dims(3)?)?; // [B,Hv,BT,Dk]
        let decay_total = ds_c.slice_axis(2, BT - 1, BT)?.reshape(&[b, hv, 1, 1])?;
        let upd = delta_c.transpose(Some(&[0, 1, 3, 2]))?.matmul(&kd)?; // [B,Hv,Dv,Dk]
        let s_out = s_cur.mul(&decay_total)?.add(&upd)?;

        sin_chunks.push(s_cur.clone());
        delta_chunks.push(delta_c);
        s_cur = s_out;
    }

    // Output (parallel over chunks once delta + S_in are known).
    let delta_refs: Vec<&MxArray> = delta_chunks.iter().collect();
    let sin_refs: Vec<&MxArray> = sin_chunks.iter().collect();
    let delta_all = MxArray::stack(delta_refs, Some(1))?; // [B,C,Hv,BT,Dv]
    let sin_all = MxArray::stack(sin_refs, Some(1))?; // [B,C,Hv,Dv,Dk]

    // inter = decay_self * (q @ S_in^T).
    let qs = q.matmul(&sin_all.transpose(Some(&[0, 1, 2, 4, 3]))?)?; // [B,C,Hv,BT,Dv]
    let inter = qs.mul(&decay_self.expand_dims(4)?)?;
    // intra = ((j<=i) * (q.k) * decay_mat) @ delta.
    let qk = q.matmul(&k.transpose(Some(&[0, 1, 2, 4, 3]))?)?; // [B,C,Hv,BT,BT]
    let aqk = incl_lower.where_(&qk.mul(&decay_mat)?, &zeros_bb)?;
    let intra = aqk.matmul(&delta_all)?; // [B,C,Hv,BT,Dv]
    let o = inter.add(&intra)?;

    // [B,C,Hv,BT,Dv] -> [B,C,BT,Hv,Dv] -> [B,Tpad,Hv,Dv] -> slice to T.
    let o = o
        .transpose(Some(&[0, 1, 3, 2, 4]))?
        .reshape(&[b, tpad, hv, dv])?;
    let o = if pad_t > 0 { o.slice_axis(1, 0, t)? } else { o };

    Ok((o.astype(out_dtype)?, s_cur.astype(state_dtype)?))
}

/// Gated delta recurrence update.
///
/// Uses a custom Metal kernel when available (GPU, Metal, Dk divisible by 32),
/// falling back to an ops-based sequential loop otherwise.
///
/// Shapes:
///   q: [B, T, Hk, Dk]   - queries
///   k: [B, T, Hk, Dk]   - keys
///   v: [B, T, Hv, Dv]   - values
///   a: [B, T, Hv]        - decay parameter
///   b: [B, T, Hv]        - beta parameter (before sigmoid)
///   A_log: [Hv]          - learnable log decay
///   dt_bias: [Hv]        - learnable bias
///   state: [B, Hv, Dv, Dk] or None - recurrent state
///   mask: [B, T] or None
///
/// Returns: (output [B, T, Hv, Dv], final_state [B, Hv, Dv, Dk])
pub fn gated_delta_update(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    a: &MxArray,
    b: &MxArray,
    a_log: &MxArray,
    dt_bias: &MxArray,
    state: Option<&MxArray>,
    mask: Option<&MxArray>,
    use_kernel: bool,
) -> Result<(MxArray, MxArray)> {
    gated_delta_update_with_tape(q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel, None)
}

/// Tape-recording variant of [`gated_delta_update`].
///
/// Identical to [`gated_delta_update`] except that, when the per-step Metal
/// kernel runs (`use_kernel`, `k_dim % 32 == 0`, not chunked), it records the
/// exact `(q, k, v, g, beta)` window tensors into `tape_sink` for the eager
/// MTP replay. The captured `q`/`k` are GQA-expanded + RMS-norm-scaled and `g`
/// is `g_log.exp()` — i.e. EXACTLY the tensors handed to the kernel, by lazy
/// `.clone()` (no eval). When `tape_sink` is `None` the behavior is
/// byte-identical to [`gated_delta_update`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_update_with_tape(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    a: &MxArray,
    b: &MxArray,
    a_log: &MxArray,
    dt_bias: &MxArray,
    state: Option<&MxArray>,
    mask: Option<&MxArray>,
    use_kernel: bool,
    mut tape_sink: Option<&mut Option<GdnKernelTape>>,
) -> Result<(MxArray, MxArray)> {
    let batch = q.shape_at(0)?;
    let num_k_heads = q.shape_at(2)?;
    let num_v_heads = v.shape_at(2)?;
    let v_dim = v.shape_at(3)?;
    let k_dim = q.shape_at(3)?;

    // The fused GDN gating + per-step/chunked recurrence are `fast::metal_kernel`
    // kernels that throw without MLX's Metal backend. On the CUDA/Linux build
    // (`mlx_metal_is_available()` is false) route straight to the
    // device-agnostic ops path instead of paying a per-layer-per-token
    // throw/catch (which would also contaminate decode-perf numbers).
    let use_kernel = use_kernel && metal_kernel_backend_available();

    // When use_kernel=false, use only ops-based paths for full differentiability (autograd).
    // compute_g builds a standard MLX expression graph via C++ (differentiable),
    // but the fused_gdn_gating Metal kernel and recurrence kernels are NOT differentiable.
    if !use_kernel {
        let beta = Activations::sigmoid(b)?;
        // compute_g returns exp(g_log) directly — use it as the decay gate without log/exp round-trip
        let g = compute_g(a_log, a, dt_bias)?;

        // GQA head expansion
        let (q, k) = if num_v_heads != num_k_heads {
            if num_k_heads == 0 {
                return Err(Error::from_reason(
                    "GatedDelta: num_k_heads is 0, cannot compute GQA repeat factor",
                ));
            }
            if num_v_heads % num_k_heads != 0 {
                return Err(Error::from_reason(format!(
                    "GatedDelta: num_v_heads ({}) must be divisible by num_k_heads ({})",
                    num_v_heads, num_k_heads
                )));
            }
            let repeat_factor = num_v_heads / num_k_heads;
            let q_expanded = q.repeat(repeat_factor as i32, 2)?;
            let k_expanded = k.repeat(repeat_factor as i32, 2)?;
            (q_expanded, k_expanded)
        } else {
            (q.clone(), k.clone())
        };

        let initial_state = match state {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
        };

        // CUDA prefill: collapse the O(T) per-step recurrence into O(T/BT) chunk-serial
        // batched matmuls (cuBLAS / tensor cores). Default on the CUDA ops path; reverts
        // to per-step under `MLX_GDN_KERNEL=perstep`. Gated to the genuine non-Metal
        // backend so the Mac autograd/training ops path stays byte-identical. Decode
        // (seq < CHUNK_THRESHOLD) and masked calls always take per-step.
        let seq_len = q.shape_at(1)?;
        let choice = gdn_kernel_override();
        if !metal_kernel_backend_available()
            && seq_len >= CHUNK_THRESHOLD
            && mask.is_none()
            && choice != GdnKernel::ForcePerStep
        {
            // Log-space gate computed directly (NOT g.log()): strong decay underflows the
            // exp-space `g` to 0, and log(0) = -inf makes the chunked decay-diff inf-inf = NaN
            // (observed as garbage on the MoE GDN layers, whose decay is stronger than dense).
            let g_log = compute_g_log(a_log, a, dt_bias)?;
            match gated_delta_chunked_ops(&q, &k, v, &g_log, &beta, &initial_state) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("[mlx-gdn] chunked_ops failed ({e}); falling back to per-step ops");
                }
            }
        }

        return gated_delta_ops(&q, &k, v, &g, &beta, &initial_state, mask);
    }

    // Compute beta = sigmoid(b) and g_log = -exp(A_log) * softplus(a + dt_bias)
    // Try fused Metal kernel first (single dispatch), fall back to separate ops.
    // g_log is the log-space gate; per-step kernel needs exp(g_log), chunked needs g_log directly.
    let (beta, g_log) = match fused_gdn_gating(b, a, a_log, dt_bias, num_v_heads as i32) {
        Ok((beta_flat, g_flat)) => {
            let seq_len_tmp = b.shape_at(1)?;
            let beta = beta_flat.reshape(&[batch, seq_len_tmp, num_v_heads])?;
            let g_log = g_flat.reshape(&[batch, seq_len_tmp, num_v_heads])?;
            (beta, g_log)
        }
        Err(_) => {
            let beta = Activations::sigmoid(b)?;
            // compute_g returns exp(g_log), so take log to get g_log
            let g = compute_g(a_log, a, dt_bias)?;
            let g_log = g.log()?;
            (beta, g_log)
        }
    };

    // GQA head expansion: repeat q,k from Hk to Hv heads
    let (q, k) = if num_v_heads != num_k_heads {
        if num_k_heads == 0 {
            return Err(Error::from_reason(
                "GatedDelta: num_k_heads is 0, cannot compute GQA repeat factor",
            ));
        }
        if num_v_heads % num_k_heads != 0 {
            return Err(Error::from_reason(format!(
                "GatedDelta: num_v_heads ({}) must be divisible by num_k_heads ({})",
                num_v_heads, num_k_heads
            )));
        }
        let repeat_factor = num_v_heads / num_k_heads;
        let q_expanded = q.repeat(repeat_factor as i32, 2)?; // [B, T, Hv, Dk]
        let k_expanded = k.repeat(repeat_factor as i32, 2)?; // [B, T, Hv, Dk]
        (q_expanded, k_expanded)
    } else {
        (q.clone(), k.clone())
    };

    // Initialize state if not provided: [B, Hv, Dv, Dk]
    // Use v's dtype to avoid f32 promotion for bf16/f16 models
    let initial_state = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
    };

    let seq_len = q.shape_at(1)?;

    // Use Metal kernel for recurrence (requires Dk divisible by 32 for SIMD register blocking)
    if k_dim % 32 == 0 {
        // GDN recurrence kernel selection. Per-step is the default on EVERY GPU generation:
        // chunked is 2.8–3.5× slower prefill on M5 and ~2× slower on M3 (see `GdnKernel`).
        // Chunked is opt-in only via `MLX_GDN_KERNEL=chunked` (A/B / bring-up), and needs g in
        // log-space directly (no exp/log roundtrip).
        //
        // Cheap eligibility first — mirrors `should_use_chunked`'s early-out (same
        // `CHUNK_THRESHOLD` / `mask` terms) so short or masked calls (every per-token decode
        // step) never pay the env + GPU-gen lookups below. The pure `should_use_chunked` still
        // owns the full contract and is unit-tested; this is just lazy argument evaluation.
        if seq_len >= CHUNK_THRESHOLD && mask.is_none() {
            let choice = gdn_kernel_override();
            if should_use_chunked(seq_len, mask.is_none(), gpu_architecture_gen(), choice) {
                match gated_delta_chunked(&q, &k, v, &g_log, &beta, &initial_state) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // An explicit `MLX_GDN_KERNEL=chunked` force must be observable when it
                        // fails — otherwise an A/B run silently measures per-step while reporting
                        // "chunked". (Auto never reaches here: it returns per-step above.)
                        if choice == GdnKernel::ForceChunked {
                            eprintln!(
                                "[mlx-gdn] MLX_GDN_KERNEL=chunked forced but the chunked kernel failed ({e}); falling back to per-step"
                            );
                        }
                        // Fall through to per-step kernel.
                    }
                }
            }
        }

        // Per-step kernel needs exponentiated decay factor
        let g = g_log.exp()?;
        if let Ok(result) = gated_delta_kernel(&q, &k, v, &g, &beta, &initial_state, mask) {
            // Record the EXACT kernel inputs (lazy clones, no eval) for the
            // eager MTP tape replay. Only the per-step kernel path is recorded —
            // verify decode always lands here (seq < CHUNK_THRESHOLD, k_dim % 32
            // == 0, mask=None), so a `None` sink on every other path is correct.
            if let Some(sink) = tape_sink.take() {
                *sink = Some(GdnKernelTape {
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    g: g.clone(),
                    beta: beta.clone(),
                });
            }
            return Ok(result);
        }
    }

    // Ops-based sequential loop fallback (also needs exp(g_log))
    let g = g_log.exp()?;
    gated_delta_ops(&q, &k, v, &g, &beta, &initial_state, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    // All GPU generations this engine targets (M1=13 … M5=17). The shipped contract is
    // arch-independent: `Auto` is per-step on every one of these — including M5 (gen 17),
    // whose old `>= 17` chunked gate was a measured-2.8–3.5×-slower stale inversion.
    const ALL_GENS: [i32; 5] = [13, 14, 15, 16, 17];

    /// Locks the SHIPPED routing intent so a future edit that re-inverts the M5 default
    /// (or makes chunked the default on any arch) trips a failing assert. See [`GdnKernel`].
    #[test]
    fn chunked_is_never_the_default_on_any_arch() {
        for gpu_gen in ALL_GENS {
            // A long, unmasked prefill — the only shape chunked is even eligible for —
            // still routes to per-step under `Auto`, on EVERY arch (M5 included).
            assert!(
                !should_use_chunked(4096, true, gpu_gen, GdnKernel::Auto),
                "Auto must route to per-step on gen {gpu_gen} (chunked is 2.8–3.5× slower on M5)",
            );
            assert!(
                !should_use_chunked(4096, true, gpu_gen, GdnKernel::ForcePerStep),
                "ForcePerStep must never select chunked (gen {gpu_gen})",
            );
            // Chunked is reachable ONLY by explicit force, and on every arch (so a future
            // default-flip can be A/B'd in both directions without a rebuild).
            assert!(
                should_use_chunked(4096, true, gpu_gen, GdnKernel::ForceChunked),
                "ForceChunked must select chunked for a long unmasked prefill (gen {gpu_gen})",
            );
        }
    }

    /// Chunked is ineligible for short or masked calls regardless of the override.
    #[test]
    fn chunked_eligibility_requires_long_unmasked_prefill() {
        for choice in [
            GdnKernel::Auto,
            GdnKernel::ForcePerStep,
            GdnKernel::ForceChunked,
        ] {
            // Below CHUNK_THRESHOLD → never chunked, even when forced.
            assert!(!should_use_chunked(CHUNK_THRESHOLD - 1, true, 17, choice));
            // Masked (decode / banded) → never chunked, even when forced (no masked variant).
            assert!(!should_use_chunked(4096, false, 17, choice));
        }
        // Exactly at the threshold, forced, unmasked → eligible.
        assert!(should_use_chunked(
            CHUNK_THRESHOLD,
            true,
            17,
            GdnKernel::ForceChunked
        ));
    }

    /// The env-override parser maps strings → [`GdnKernel`] (tested env-free, race-free).
    #[test]
    fn parse_gdn_kernel_override_semantics() {
        // Default: nothing set.
        assert_eq!(parse_gdn_kernel(None, None), GdnKernel::Auto);
        // MLX_GDN_KERNEL=chunked / perstep (case-insensitive, trimmed, aliases).
        assert_eq!(
            parse_gdn_kernel(Some("chunked"), None),
            GdnKernel::ForceChunked
        );
        assert_eq!(
            parse_gdn_kernel(Some("  CHUNK "), None),
            GdnKernel::ForceChunked
        );
        // MLX_GDN_KERNEL=chunked_ops selects the device-agnostic ops path (CUDA default).
        assert_eq!(
            parse_gdn_kernel(Some("chunked_ops"), None),
            GdnKernel::ForceChunkedOps
        );
        assert_eq!(
            parse_gdn_kernel(Some("CHUNKED-OPS"), None),
            GdnKernel::ForceChunkedOps
        );
        assert_eq!(
            parse_gdn_kernel(Some("perstep"), None),
            GdnKernel::ForcePerStep
        );
        assert_eq!(
            parse_gdn_kernel(Some("per-step"), None),
            GdnKernel::ForcePerStep
        );
        assert_eq!(
            parse_gdn_kernel(Some("Step"), None),
            GdnKernel::ForcePerStep
        );
        // MLX_GDN_KERNEL wins over the legacy toggle when it is a known value.
        assert_eq!(
            parse_gdn_kernel(Some("chunked"), Some("1")),
            GdnKernel::ForceChunked
        );
        // Unknown MLX_GDN_KERNEL falls through to the legacy toggle, then to Auto.
        assert_eq!(
            parse_gdn_kernel(Some("garbage"), Some("1")),
            GdnKernel::ForcePerStep
        );
        assert_eq!(parse_gdn_kernel(Some("garbage"), None), GdnKernel::Auto);
        // Legacy MLX_GDN_FORCE_PERSTEP truthy values only.
        assert_eq!(
            parse_gdn_kernel(None, Some("true")),
            GdnKernel::ForcePerStep
        );
        assert_eq!(parse_gdn_kernel(None, Some("on")), GdnKernel::ForcePerStep);
        assert_eq!(parse_gdn_kernel(None, Some("0")), GdnKernel::Auto);
        assert_eq!(parse_gdn_kernel(None, Some("")), GdnKernel::Auto);
    }

    use crate::array::DType;

    fn rand_bf16(shape: &[i64]) -> MxArray {
        MxArray::random_normal(shape, 0.0, 0.3, Some(DType::Float32))
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
    }

    fn max_abs_diff(a: &MxArray, b: &MxArray) -> f32 {
        let af = a.astype(DType::Float32).unwrap().to_float32().unwrap();
        let bf = b.astype(DType::Float32).unwrap().to_float32().unwrap();
        let av = af.as_ref();
        let bv = bf.as_ref();
        av.iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Diagnostic: does a per-step T=1 kernel loop (= AR decode) match
    /// recording during a windowed kernel then replaying per-step?
    #[test]
    fn tape_replay_matches_per_step_ar_loop() {
        let b = 1i64;
        let hv = 4i64;
        let dk = 32i64;
        let dv = 32i64;
        let t = 4i64;

        let q = rand_bf16(&[b, t, hv, dk]);
        let k = rand_bf16(&[b, t, hv, dk]);
        let v = rand_bf16(&[b, t, hv, dv]);
        // g in (0,1): sigmoid-ish decay.
        let g = MxArray::random_normal(&[b, t, hv], 0.0, 0.3, Some(DType::Float32)).unwrap();
        let g = Activations::sigmoid(&g).unwrap();
        let beta = MxArray::random_normal(&[b, t, hv], 0.0, 0.3, Some(DType::Float32)).unwrap();
        let beta = Activations::sigmoid(&beta).unwrap();

        let state0 = rand_bf16(&[b, hv, dv, dk]);

        // (A) Reference AR loop: per-step T=1 kernel from state0, threading bf16.
        let ar_final = {
            let mut s = state0.clone();
            for ti in 0..t {
                let q_t = q.slice_axis(1, ti, ti + 1).unwrap();
                let k_t = k.slice_axis(1, ti, ti + 1).unwrap();
                let v_t = v.slice_axis(1, ti, ti + 1).unwrap();
                let g_t = g.slice_axis(1, ti, ti + 1).unwrap();
                let beta_t = beta.slice_axis(1, ti, ti + 1).unwrap();
                let (_y, ns) =
                    gated_delta_kernel(&q_t, &k_t, &v_t, &g_t, &beta_t, &s, None).unwrap();
                s = ns;
            }
            s.eval();
            s
        };

        // (B) Windowed single call (the lossy verify-style fp32-carry path).
        let win_final = {
            let (_y, ns) = gated_delta_kernel(&q, &k, &v, &g, &beta, &state0, None).unwrap();
            ns.eval();
            ns
        };

        // (C) Replay via GdnKernelTape (records the same q,k,v,g,beta, replays
        //     per-step T=1). This is exactly what the rollback does.
        let tape = GdnKernelTape {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            g: g.clone(),
            beta: beta.clone(),
        };
        let replay_final = tape.replay_recurrent_state(&state0, t as usize).unwrap();
        let replay_final = {
            replay_final.eval();
            replay_final
        };

        let ar_vs_replay = max_abs_diff(&ar_final, &replay_final);
        let ar_vs_win = max_abs_diff(&ar_final, &win_final);
        eprintln!("TAPE_DIAG ar_vs_replay={ar_vs_replay:.6e} ar_vs_win={ar_vs_win:.6e}");

        // The replay MUST be bit-exact to the AR per-step loop.
        assert_eq!(
            ar_vs_replay, 0.0,
            "per-step tape replay must equal the AR per-step loop bit-for-bit \
             (got max_abs_diff={ar_vs_replay:.6e})"
        );
    }

    /// Parity: the chunk-parallel ops path must match the per-step recurrence (the oracle)
    /// in f32 across chunk boundaries (T = 64, 65, 127, 128, 256). A mask off-by-one,
    /// padding leak, or state-orientation bug shows up as an O(1) diff; a correct port
    /// differs only by f32 reduction order (~1e-4). Device-agnostic, so this validates the
    /// algorithm on Mac/Metal before any DGX time.
    #[test]
    fn chunked_ops_matches_per_step_ops_f32() -> Result<()> {
        use crate::array::DType;
        // Production dims (Dk=Dv=128 != BT=64 so a BT/Dk axis-swap can't hide).
        let (b, hv, dk, dv) = (1i64, 4i64, 128i64, 128i64);
        let f32 = DType::Float32;
        // q,k carry the upstream RMS-norm + Dk^-0.5 scaling, so k.k ~ O(0.1) and the
        // WY system (I+A) is well-conditioned (the per-step path is unconditionally
        // stable; the chunked Neumann inverse needs ||A|| small, as in production).
        let qk_std = 1.0 / (dk as f64).sqrt();
        let randn = |shape: &[i64], std: f64| MxArray::random_normal(shape, 0.0, std, Some(f32));
        let max_abs_diff = |a: &MxArray, x: &MxArray| -> Result<f32> {
            let d = a
                .sub(x)?
                .abs()?
                .reshape(&[-1])?
                .max(Some(&[0]), Some(false))?;
            d.eval();
            d.item_at_float32(0)
        };
        for &t in &[64i64, 65, 127, 128, 256] {
            let q = randn(&[b, t, hv, dk], qk_std)?;
            let k = randn(&[b, t, hv, dk], qk_std)?;
            let v = randn(&[b, t, hv, dv], 1.0)?;
            // Decay gate g in (0,1) via sigmoid; chunked consumes g_log = log(g).
            let g = Activations::sigmoid(&randn(&[b, t, hv], 1.0)?)?;
            let g_log = g.log()?;
            let beta = Activations::sigmoid(&randn(&[b, t, hv], 1.0)?)?;
            let state = MxArray::zeros(&[b, hv, dv, dk], Some(f32))?;

            let (o_ref, s_ref) = gated_delta_ops(&q, &k, &v, &g, &beta, &state, None)?;
            let (o_chk, s_chk) = gated_delta_chunked_ops(&q, &k, &v, &g_log, &beta, &state)?;

            let od = max_abs_diff(&o_ref, &o_chk)?;
            let sd = max_abs_diff(&s_ref, &s_chk)?;
            assert!(od < 1e-2, "T={t}: output max-abs-diff {od} exceeds tol");
            assert!(sd < 1e-2, "T={t}: state max-abs-diff {sd} exceeds tol");
        }
        Ok(())
    }

    /// Reproduces the real-model condition that overflows a naive (I+A)^-1 by repeated
    /// squaring: unit-norm but HIGHLY CORRELATED keys (k.k ~ 1, as conv1d produces) plus
    /// near-1 decay (so decay_mat does not damp A). Then ||A||_inf ~ BT, and N^32 ~ 63^32
    /// ~ 4e57 overflows f32 — the blocked inverse must stay finite and match per-step.
    /// (The random-key test above keeps k.k ~ 0.1, which never triggers this.)
    #[test]
    fn chunked_ops_stable_with_correlated_unit_norm_keys() -> Result<()> {
        use crate::array::DType;
        let (b, hv, dk, dv) = (1i64, 4i64, 128i64, 128i64);
        let f32 = DType::Float32;
        let randn = |shape: &[i64], std: f64| MxArray::random_normal(shape, 0.0, std, Some(f32));
        let max_abs_diff = |a: &MxArray, x: &MxArray| -> Result<f32> {
            let d = a
                .sub(x)?
                .abs()?
                .reshape(&[-1])?
                .max(Some(&[0]), Some(false))?;
            d.eval();
            d.item_at_float32(0)
        };
        // L2-normalize over the last axis -> unit-norm keys, matching the upstream
        // `k = Dk^-0.5 * rms_norm(k)` scaling (||k|| = 1, so |k_i . k_j| <= 1).
        let l2norm = |x: &MxArray| -> Result<MxArray> {
            let n = x.square()?.sum(Some(&[3]), Some(true))?.sqrt()?;
            x.div(&n)
        };
        for &t in &[128i64, 256] {
            // Shared base direction + small per-token noise -> rows are highly correlated,
            // so k_i . k_j ~ 1 after normalization (the regime conv1d puts the model in).
            let base = randn(&[b, 1, hv, dk], 1.0)?;
            let k = l2norm(&base.add(&randn(&[b, t, hv, dk], 0.3)?)?)?;
            let q = l2norm(&base.add(&randn(&[b, t, hv, dk], 0.3)?)?)?;
            let v = randn(&[b, t, hv, dv], 1.0)?;
            // Near-1 decay: g = sigmoid(+large) ~ 1, so g_log ~ 0 and A is undamped.
            let g = Activations::sigmoid(&randn(&[b, t, hv], 0.1)?.add_scalar(6.0)?)?;
            let g_log = g.log()?;
            let beta = Activations::sigmoid(&randn(&[b, t, hv], 1.0)?)?;
            let state = MxArray::zeros(&[b, hv, dv, dk], Some(f32))?;

            let (o_ref, s_ref) = gated_delta_ops(&q, &k, &v, &g, &beta, &state, None)?;
            let (o_chk, s_chk) = gated_delta_chunked_ops(&q, &k, &v, &g_log, &beta, &state)?;

            // Looser tol than the realistic-scale test: this is a deliberately ill-conditioned
            // stress case (the residual is f32 precision, ~0.02). The point is that the inverse
            // stays FINITE and far from garbage — repeated squaring gives ~3e5 or Inf here.
            let od = max_abs_diff(&o_ref, &o_chk)?;
            let sd = max_abs_diff(&s_ref, &s_chk)?;
            assert!(
                od.is_finite() && od < 5e-2,
                "T={t}: output max-abs-diff {od} (non-finite/large => inverse unstable)"
            );
            assert!(
                sd.is_finite() && sd < 5e-2,
                "T={t}: state max-abs-diff {sd} (non-finite/large => inverse unstable)"
            );
        }
        Ok(())
    }

    /// `compute_g_log` must equal `log(compute_g)` on normal inputs AND stay finite under
    /// strong decay, where the exp-space `compute_g` underflows to 0 and `log(0) = -inf`
    /// (the MoE GDN garbage root cause). The chunked CUDA path uses this in place of `g.log()`.
    #[test]
    fn compute_g_log_finite_under_strong_decay() -> Result<()> {
        use crate::array::DType;
        let f32 = DType::Float32;
        let (b, t, hv) = (1i64, 8i64, 4i64);
        let max_abs = |x: &MxArray| -> Result<f32> {
            let d = x.abs()?.reshape(&[-1])?.max(Some(&[0]), Some(false))?;
            d.eval();
            d.item_at_float32(0)
        };
        // Normal magnitude: no underflow, so compute_g_log ~ log(compute_g).
        let a_log = MxArray::random_normal(&[hv], 0.0, 1.0, Some(f32))?;
        let a = MxArray::random_normal(&[b, t, hv], 0.0, 1.0, Some(f32))?;
        let dt_bias = MxArray::random_normal(&[hv], 0.0, 1.0, Some(f32))?;
        let gl = compute_g_log(&a_log, &a, &dt_bias)?;
        let gl_ref = compute_g(&a_log, &a, &dt_bias)?.log()?;
        assert!(
            max_abs(&gl.sub(&gl_ref)?)? < 1e-3,
            "compute_g_log != log(compute_g) on normal inputs"
        );

        // Strong decay: exp(a_log)*softplus(a+dt_bias) ~ 20*53 ~ 1060 >> 88, so compute_g
        // underflows to 0 -> log(0) = -inf. compute_g_log stays finite (~ -1060).
        let a_big = MxArray::zeros(&[b, t, hv], Some(f32))?.add_scalar(50.0)?;
        let alog_big = MxArray::zeros(&[hv], Some(f32))?.add_scalar(3.0)?;
        let gl_strong = max_abs(&compute_g_log(&alog_big, &a_big, &dt_bias)?)?;
        assert!(
            gl_strong.is_finite(),
            "compute_g_log went non-finite under strong decay: {gl_strong}"
        );
        // Confirm the naive `g.log()` path really does blow up here, which is
        // why `compute_g_log` computes the log directly.
        let old = max_abs(&compute_g(&alog_big, &a_big, &dt_bias)?.log()?)?;
        assert!(
            !old.is_finite(),
            "expected log(compute_g) = -inf under strong decay, got {old}"
        );
        Ok(())
    }
}
