use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;

/// Compute decay gate: g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// Shapes:
///   A_log: [Hv]
///   a: [B, T, Hv]
///   dt_bias: [Hv]
///
/// Returns: [B, T, Hv]
fn compute_g(a_log: &MxArray, a: &MxArray, dt_bias: &MxArray) -> Result<MxArray> {
    // a + dt_bias (broadcasts [Hv] to [B, T, Hv])
    let a_biased = a.add(dt_bias)?;

    // softplus(a + dt_bias) = log(1 + exp(a + dt_bias))
    let sp = Activations::softplus(&a_biased)?;

    // exp(A_log)
    let a_exp = a_log.exp()?;

    // -exp(A_log) * softplus(a + dt_bias)
    let neg_a_exp = a_exp.negative()?;
    let exponent = neg_a_exp.mul(&sp)?;

    // g = exp(exponent)
    exponent.exp()
}

/// Single timestep of the gated delta recurrence (delta rule).
///
/// Implements the correct delta rule:
///   1. Decay state: state *= g
///   2. Compute kv_mem = (state * k).sum(-1)  → retrieval from memory
///   3. Compute delta = (v - kv_mem) * beta   → error signal
///   4. Update state: state += k * delta      → write correction to memory
///   5. Output: y = (state * q).sum(-1)       → read from memory
///
/// Shapes:
///   q_t: [B, 1, Hv, Dk]  (already GQA-expanded)
///   k_t: [B, 1, Hv, Dk]  (already GQA-expanded)
///   v_t: [B, 1, Hv, Dv]
///   g_t: [B, 1, Hv]
///   beta_t: [B, 1, Hv]
///   state: [B, Hv, Dv, Dk]
///   mask_t: [B, 1] or None
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
) -> Result<(MxArray, MxArray)> {
    let batch = q_t.shape_at(0)?;
    let num_v_heads = v_t.shape_at(2)?;
    let k_dim = q_t.shape_at(3)?;
    let v_dim = v_t.shape_at(3)?;

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

/// Gated delta recurrence update (ops-based sequential loop).
///
/// This is the main entry point for linear attention recurrence.
/// Handles GQA head expansion when num_v_heads != num_k_heads.
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
) -> Result<(MxArray, MxArray)> {
    let batch = q.shape_at(0)?;
    let seq_len = q.shape_at(1)?;
    let num_k_heads = q.shape_at(2)?;
    let num_v_heads = v.shape_at(2)?;
    let v_dim = v.shape_at(3)?;
    let k_dim = q.shape_at(3)?;

    // Compute beta = sigmoid(b): [B, T, Hv]
    let beta = Activations::sigmoid(b)?;

    // Compute g = exp(-exp(A_log) * softplus(a + dt_bias)): [B, T, Hv]
    let g = compute_g(a_log, a, dt_bias)?;

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
    let mut current_state = match state {
        Some(s) => s.clone(),
        None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
    };

    // Sequential loop over timesteps
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
        )?;

        outputs.push(y_t);
        current_state = new_state;
    }

    // Concatenate along time dimension: [B, T, Hv, Dv]
    let output_refs: Vec<&MxArray> = outputs.iter().collect();
    let output = MxArray::concatenate_many(output_refs, Some(1))?;

    Ok((output, current_state))
}
