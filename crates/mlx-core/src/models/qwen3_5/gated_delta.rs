use crate::array::MxArray;
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Whether the GDN `fast::metal_kernel` kernels can run on this host. False on
/// the CUDA/Linux build, where they throw — callers must use the ops path.
/// Delegates to the shared, cached `mlx_metal_is_available()` probe.
fn metal_kernel_backend_available() -> bool {
    super::persistence_common::compiled_forward_backend_available()
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
/// M5; it is removed. Per-step is already the canonical path on M1–M4, for all `seq < 64`, all
/// masked GDN calls, and every compiled-C++ prefill path — so per-step is the de-facto
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
        GdnKernel::Auto | GdnKernel::ForcePerStep => false,
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
}
