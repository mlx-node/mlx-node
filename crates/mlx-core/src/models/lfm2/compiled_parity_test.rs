//! Phase-1 component-parity gate for the lfm2 compiled C++ forward path.
//!
//! lfm2's compiled forward is not end-to-end runnable until the ShortConv
//! operator lands (Phase 2), so we validate the parity-critical novel C++ — the
//! attention pure-fn and the dense SwiGLU MLP — in ISOLATION here, against the
//! Rust-native single-layer forward. The C++ probes (`mlx_lfm2_probe_attn_seq`,
//! `mlx_lfm2_probe_dense_mlp`) register one layer's weights into the shared
//! `g_weights()` map, run the compiled `lfm2_attn_pure_fn` / `lfm2_dense_mlp`,
//! and return the output. In the `--lib` test binary nothing else touches that
//! global map, so the probes' clear→store→run→clear is race-free.

#![cfg(test)]

use crate::array::{DType, MxArray};
use crate::transformer::{KVCache, MLP};

use super::attention::Lfm2Attention;

/// Deterministic small bf16 array of `shape` from a seed (so the native and
/// probe sides receive byte-identical inputs).
fn det(shape: &[i64], seed: i64) -> MxArray {
    let n: i64 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 131 + seed * 17 + 7).rem_euclid(23)) as f32 - 11.0) * 0.03)
        .collect();
    MxArray::from_float32(&data, shape)
        .expect("from_float32")
        .astype(DType::BFloat16)
        .expect("bf16")
}

/// Deterministic RMSNorm weight (~1.0) of length `dim`.
fn det_norm(dim: i64, seed: i64) -> MxArray {
    let data: Vec<f32> = (0..dim)
        .map(|i| 1.0 + (((i + seed).rem_euclid(7)) as f32 - 3.0) * 0.04)
        .collect();
    MxArray::from_float32(&data, &[dim])
        .expect("from_float32")
        .astype(DType::BFloat16)
        .expect("bf16")
}

fn to_vec(a: &MxArray) -> Vec<f32> {
    a.astype(DType::Float32)
        .expect("f32")
        .to_float32()
        .expect("to_float32")
        .to_vec()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The compiled `lfm2_attn_pure_fn`, run as a `T`-step decode sequence, must
/// match the native `Lfm2Attention::forward` driven over the same `T` steps
/// (so multi-key softmax + RoPE offset + per-head QK RMSNorm are exercised).
#[test]
fn compiled_attn_seq_matches_native() {
    let hidden = 64i64;
    let num_heads = 4i64;
    let num_kv_heads = 2i64;
    let head_dim = 16i64; // hidden == num_heads * head_dim
    let norm_eps = 1e-5f64;
    let rope_theta = 1_000_000.0f64;
    let t = 4i64;

    let q_w = det(&[num_heads * head_dim, hidden], 1);
    let k_w = det(&[num_kv_heads * head_dim, hidden], 2);
    let v_w = det(&[num_kv_heads * head_dim, hidden], 3);
    let out_w = det(&[hidden, num_heads * head_dim], 4);
    let qn_w = det_norm(head_dim, 5);
    let kn_w = det_norm(head_dim, 6);

    // Shared input data so native (per-step [1,1,hidden]) and probe ([T,hidden])
    // get byte-identical bf16 rows.
    let x_data: Vec<f32> = (0..(t * hidden))
        .map(|i| (((i * 97 + 5).rem_euclid(19)) as f32 - 9.0) * 0.04)
        .collect();
    let x_seq = MxArray::from_float32(&x_data, &[t, hidden])
        .expect("x_seq")
        .astype(DType::BFloat16)
        .expect("bf16");

    // ----- native: T decode steps through a shared KVCache -----
    let mut attn = Lfm2Attention::new(
        hidden as i32,
        num_heads as i32,
        num_kv_heads as i32,
        head_dim as i32,
        norm_eps,
        rope_theta,
    )
    .expect("attn new");
    attn.q_proj_mut().set_weight(&q_w, "q_proj").expect("q");
    attn.k_proj_mut().set_weight(&k_w, "k_proj").expect("k");
    attn.v_proj_mut().set_weight(&v_w, "v_proj").expect("v");
    attn.out_proj_mut()
        .set_weight(&out_w, "out_proj")
        .expect("o");
    attn.set_q_layernorm_weight(&qn_w).expect("qn");
    attn.set_k_layernorm_weight(&kn_w).expect("kn");

    let mut cache = KVCache::new();
    let mut native_last: Option<MxArray> = None;
    for i in 0..t {
        let row = &x_data[(i * hidden) as usize..((i + 1) * hidden) as usize];
        let x_i = MxArray::from_float32(row, &[1, 1, hidden])
            .expect("x_i")
            .astype(DType::BFloat16)
            .expect("bf16");
        native_last = Some(
            attn.forward(&x_i, None, Some(&mut cache))
                .expect("native fwd"),
        );
    }
    let native_last = native_last.expect("ran >=1 step");

    // ----- probe: same T steps through the compiled pure-fn -----
    let out_ptr = unsafe {
        mlx_sys::mlx_lfm2_probe_attn_seq(
            x_seq.as_raw_ptr(),
            q_w.as_raw_ptr(),
            k_w.as_raw_ptr(),
            v_w.as_raw_ptr(),
            out_w.as_raw_ptr(),
            qn_w.as_raw_ptr(),
            kn_w.as_raw_ptr(),
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            rope_theta as f32,
            norm_eps as f32,
        )
    };
    assert!(!out_ptr.is_null(), "mlx_lfm2_probe_attn_seq returned null");
    let probe_out = MxArray::from_handle(out_ptr, "probe_attn").expect("probe handle");

    let nv = to_vec(&native_last); // [1,1,hidden] flattened
    let pv = to_vec(&probe_out); // [1,hidden] flattened
    let d = max_abs(&nv, &pv);
    // Same ops on both sides (matmul / fast::rms_norm / fast::rope / fast SDPA),
    // bf16 throughout — only kernel-ordering jitter, so a tight bound holds.
    assert!(
        d < 2e-2,
        "compiled attn pure-fn must match native single-layer decode: max_abs={d}"
    );
}

/// The compiled `lfm2_dense_mlp` must match the native `MLP::forward`
/// (validates `linear_proj` + `swiglu` wiring + the weight-store/transpose
/// round-trip).
#[test]
fn compiled_dense_mlp_matches_native() {
    let hidden = 64i64;
    let inter = 128i64;

    let gate_w = det(&[inter, hidden], 11);
    let up_w = det(&[inter, hidden], 12);
    let down_w = det(&[hidden, inter], 13);
    let x = det(&[3, hidden], 14); // B=3 rows

    let mut mlp = MLP::new(hidden as u32, inter as u32).expect("mlp new");
    mlp.set_gate_proj_weight(&gate_w).expect("gate");
    mlp.set_up_proj_weight(&up_w).expect("up");
    mlp.set_down_proj_weight(&down_w).expect("down");
    let native = mlp.forward(&x).expect("native mlp fwd");

    let out_ptr = unsafe {
        mlx_sys::mlx_lfm2_probe_dense_mlp(
            x.as_raw_ptr(),
            gate_w.as_raw_ptr(),
            up_w.as_raw_ptr(),
            down_w.as_raw_ptr(),
        )
    };
    assert!(!out_ptr.is_null(), "mlx_lfm2_probe_dense_mlp returned null");
    let probe = MxArray::from_handle(out_ptr, "probe_mlp").expect("probe handle");

    let d = max_abs(&to_vec(&native), &to_vec(&probe));
    assert!(
        d < 2e-2,
        "compiled dense mlp must match native MLP: max_abs={d}"
    );
}
