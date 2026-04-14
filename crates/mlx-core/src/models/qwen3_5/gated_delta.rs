use crate::array::MxArray;
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::debug::log_tensor_stats;

/// Minimum sequence length to use the chunked prefill kernel.
/// On M1–M4, the chunked kernel's O(BT^2) overhead outweighs memory bandwidth savings
/// (no tensor cores to accelerate the quadratic matmuls). On M5+ (gen >= 17), the
/// Neural Accelerators make simdgroup_matrix operations ~4x faster, so chunked wins.
const CHUNK_THRESHOLD: i64 = 64;

/// Minimum GPU architecture generation for the chunked kernel.
/// M5 (gen 17) has Neural Accelerators that make chunked prefill faster than per-step.
/// On M1–M4 (gen 13–16), the per-step kernel is faster due to simpler ALU patterns.
const CHUNK_MIN_GPU_GEN: i32 = 17;

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
#[cfg(not(target_family = "wasm"))]
fn compute_g(a_log: &MxArray, a: &MxArray, dt_bias: &MxArray) -> Result<MxArray> {
    let handle = unsafe {
        sys::mlx_fused_compute_g(a_log.as_raw_ptr(), a.as_raw_ptr(), dt_bias.as_raw_ptr())
    };
    MxArray::from_handle(handle, "fused_compute_g")
}

#[cfg(target_family = "wasm")]
fn compute_g(a_log: &MxArray, a: &MxArray, dt_bias: &MxArray) -> Result<MxArray> {
    // g = exp(-exp(A_log) * softplus(a + dt_bias))
    // softplus(x) = where(x > 20, x, log(1 + exp(x)))
    let big_a = a_log.exp()?; // exp(A_log): [Hv]
    let x = a.add(dt_bias)?; // a + dt_bias: [B, T, Hv]
    let threshold = MxArray::from_float32(&[20.0f32], &[1])?; // scalar 20.0
    let sp = x.exp()?.add_scalar(1.0)?.log()?; // log(1 + exp(x))
    let sp = x.greater(&threshold)?.where_(&x, &sp)?; // where(x > 20, x, sp)
    let result = big_a.mul(&sp)?.negative()?.exp()?; // exp(-(A * sp))
    Ok(result)
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
    gated_delta_ops_inner(q, k, v, g, beta, state, mask, None)
}

fn gated_delta_ops_inner(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    g: &MxArray,
    beta: &MxArray,
    state: &MxArray,
    mask: Option<&MxArray>,
    layer_idx: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    let debug_prefix = layer_idx.map(|layer| format!("layer.{layer:02}.gdn.recurrence"));

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

        if t == 0 {
            if let Some(prefix) = debug_prefix.as_ref() {
                log_tensor_stats(&format!("{prefix}.step0.q"), &q_t);
                log_tensor_stats(&format!("{prefix}.step0.k"), &k_t);
                log_tensor_stats(&format!("{prefix}.step0.v"), &v_t);
                log_tensor_stats(&format!("{prefix}.step0.g"), &g_t);
                log_tensor_stats(&format!("{prefix}.step0.beta"), &beta_t);
                log_tensor_stats(&format!("{prefix}.step0.state_in"), &current_state);
                log_tensor_stats(&format!("{prefix}.step0.y"), &y_t);
                log_tensor_stats(&format!("{prefix}.step0.state_out"), &new_state);
            }
        }

        outputs.push(y_t);
        current_state = new_state;
    }

    // Concatenate along time dimension: [B, T, Hv, Dv]
    let output_refs: Vec<&MxArray> = outputs.iter().collect();
    let output = MxArray::concatenate_many(output_refs, Some(1))?;

    if let Some(prefix) = debug_prefix.as_ref() {
        log_tensor_stats(&format!("{prefix}.output"), &output);
        log_tensor_stats(&format!("{prefix}.final_state"), &current_state);
    }

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
    gated_delta_update_inner(q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel, None)
}

pub(crate) fn gated_delta_update_inner(
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
    layer_idx: Option<usize>,
) -> Result<(MxArray, MxArray)> {
    let batch = q.shape_at(0)?;
    let num_k_heads = q.shape_at(2)?;
    let num_v_heads = v.shape_at(2)?;
    let v_dim = v.shape_at(3)?;
    let k_dim = q.shape_at(3)?;

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
    #[cfg(not(target_family = "wasm"))]
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

    // Phase 6e: route the compute_g log-space tape through the fused
    // `mlx_gdn_compute_g_forward` FFI. The FFI builds the same 9-op
    // exp/add/log/where/negative/multiply chain that the inline Rust
    // version did, but when the Phase 6e compile flag is on (via
    // wgpuSetGdnGCompileEnabled or MLX_WGPU_COMPILE_GDN_G=1) the whole
    // element-wise chain is routed through mlx::core::compile and fuses
    // into a single Compiled kernel. Default OFF → the FFI runs its
    // eager path, which is op-for-op equivalent to what the Rust inline
    // used to build, so this is a zero-risk change on the cold path.
    #[cfg(target_family = "wasm")]
    let (beta, g_log) = {
        use std::ptr;
        let beta = Activations::sigmoid(b)?;
        let mut g_log_out: *mut sys::mlx_array = ptr::null_mut();
        let rc = unsafe {
            sys::mlx_gdn_compute_g_forward(
                a_log.as_raw_ptr(),
                a.as_raw_ptr(),
                dt_bias.as_raw_ptr(),
                &mut g_log_out,
            )
        };
        if rc != 0 || g_log_out.is_null() {
            return Err(Error::from_reason(
                "mlx_gdn_compute_g_forward returned an error",
            ));
        }
        let g_log = MxArray::from_handle(g_log_out, "gdn_compute_g_forward")?;
        (beta, g_log)
    };

    if let Some(layer) = layer_idx {
        log_tensor_stats(&format!("layer.{layer:02}.gdn.beta"), &beta);
        log_tensor_stats(&format!("layer.{layer:02}.gdn.g_log"), &g_log);
    }

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

    if let Some(layer) = layer_idx {
        log_tensor_stats(&format!("layer.{layer:02}.gdn.q_expanded"), &q);
        log_tensor_stats(&format!("layer.{layer:02}.gdn.k_expanded"), &k);
        log_tensor_stats(&format!("layer.{layer:02}.gdn.v"), v);
    }

    // Initialize state if not provided: [B, Hv, Dv, Dk]
    // Use v's dtype to avoid f32 promotion for bf16/f16 models
    let initial_state = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
    };

    if let Some(layer) = layer_idx {
        log_tensor_stats(&format!("layer.{layer:02}.gdn.state_init"), &initial_state);
        let g_exp = g_log.exp()?;
        log_tensor_stats(&format!("layer.{layer:02}.gdn.g_exp"), &g_exp);
    }

    let seq_len = q.shape_at(1)?;

    // Use Metal kernel for recurrence (requires Dk divisible by 32 for SIMD register blocking)
    #[cfg(not(target_family = "wasm"))]
    if k_dim % 32 == 0 {
        // Chunked kernel for long sequences on M5+ (Neural Accelerator-accelerated prefill).
        // On M1–M4, per-step kernel is faster (no tensor cores for O(BT^2) matmuls).
        // Chunked kernel needs g in log-space directly (no exp/log roundtrip).
        if seq_len >= CHUNK_THRESHOLD
            && mask.is_none()
            && gpu_architecture_gen() >= CHUNK_MIN_GPU_GEN
        {
            match gated_delta_chunked(&q, &k, v, &g_log, &beta, &initial_state) {
                Ok(result) => return Ok(result),
                Err(_) => {
                    // Fall through to per-step kernel
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
    gated_delta_ops_inner(&q, &k, v, &g, &beta, &initial_state, mask, layer_idx)
}
