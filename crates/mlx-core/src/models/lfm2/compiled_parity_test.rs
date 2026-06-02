//! Phase-1 component-parity gate for the lfm2 compiled C++ forward path.
//!
//! lfm2's compiled forward is not end-to-end runnable until the full backbone
//! lands (Phase 2+), so we validate the parity-critical novel C++ — the
//! attention pure-fn, the dense SwiGLU MLP, and the ShortConv operator — in
//! ISOLATION here, against the Rust-native single-layer forward. The C++ probes
//! (`mlx_lfm2_probe_attn_seq`, `mlx_lfm2_probe_dense_mlp`,
//! `mlx_lfm2_probe_conv_seq`) register one layer's weights into the shared
//! `g_weights()` map, run the compiled pure-fn, and return the output.
//!
//! That shared map is process-global and is the SAME registry the production
//! compiled paths (qwen3.5 / qwen3.5-MoE / gemma4, and eventually lfm2) own
//! during registration + inference, serialized by
//! [`COMPILED_WEIGHTS_RWLOCK`](crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK).
//! A probe's clear→store→run→clear (which also resets the active model id) would
//! corrupt a concurrent compiled registration/inference — and qwen compiled-path
//! tests live in this very `--lib` binary. So each probe test holds that SAME
//! write lock (not a private one) for its whole transaction, making it mutually
//! exclusive with all compiled-path activity process-wide. `into_inner()`
//! recovers a poisoned lock so one failing test doesn't cascade-poison the rest.

#![cfg(test)]

use crate::array::{DType, MxArray};
use crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK;
use crate::transformer::{KVCache, MLP};

use super::attention::Lfm2Attention;
use super::short_conv::ShortConv;
use crate::models::qwen3_5::arrays_cache::ArraysCache;

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
    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
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

/// The ARRAY-OFFSET attn variant `lfm2_attn_pure_fn_arr` (fixed padded KV cache +
/// per-step static additive mask — the path the compiled decode loop uses) must
/// ALSO match the native `Lfm2Attention::forward` over the same `T`-step decode,
/// proving the fixed-cache+mask+array-offset path is numerically identical to
/// native BEFORE it is wired into `lfm2_decode_fn`.
#[test]
fn compiled_attn_arr_seq_matches_native() {
    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
    let hidden = 64i64;
    let num_heads = 4i64;
    let num_kv_heads = 2i64;
    let head_dim = 16i64;
    let norm_eps = 1e-5f64;
    let rope_theta = 1_000_000.0f64;
    let t = 4i64;

    let q_w = det(&[num_heads * head_dim, hidden], 1);
    let k_w = det(&[num_kv_heads * head_dim, hidden], 2);
    let v_w = det(&[num_kv_heads * head_dim, hidden], 3);
    let out_w = det(&[hidden, num_heads * head_dim], 4);
    let qn_w = det_norm(head_dim, 5);
    let kn_w = det_norm(head_dim, 6);

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

    // ----- probe: same T steps through the array-offset compiled variant -----
    let out_ptr = unsafe {
        mlx_sys::mlx_lfm2_probe_attn_arr_seq(
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
    assert!(
        !out_ptr.is_null(),
        "mlx_lfm2_probe_attn_arr_seq returned null"
    );
    let probe_out = MxArray::from_handle(out_ptr, "probe_attn_arr").expect("probe handle");

    let d = max_abs(&to_vec(&native_last), &to_vec(&probe_out));
    assert!(
        d < 2e-2,
        "array-offset compiled attn must match native single-layer decode: max_abs={d}"
    );
}

/// The compiled `lfm2_dense_mlp` must match the native `MLP::forward`
/// (validates `linear_proj` + `swiglu` wiring + the weight-store/transpose
/// round-trip).
#[test]
fn compiled_dense_mlp_matches_native() {
    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
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

/// Drive native `ShortConv` and the compiled `lfm2_conv_pure_fn` probe over the
/// same `T`-step decode sequence (B=1) and assert bf16 parity. Exercises the
/// split order (B,C,x), the `B*x` input gate, the depthwise conv window + the
/// conv-state carry-over, and the `C*conv_out` output gate. `conv_bias` toggles
/// the in_proj/conv/out_proj additive biases (one lfm2 config flag gates all
/// three), so both arms of the pure-fn's `if (conv_bias)` are covered.
fn run_conv_parity(conv_bias: bool) {
    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
    let hidden = 64i64;
    let l_cache = 3i64; // kernel size K; n_keep = K-1 = 2
    let t = 4i64;

    // Weight shapes & seeds:
    //   in_proj.weight  [3*hidden, hidden]   seed 21  (natural [out, in])
    //   out_proj.weight [hidden,   hidden]   seed 22  (natural [out, in])
    //   conv.weight     [hidden, l_cache, 1] seed 23  (MLX depthwise layout)
    //   in_proj.bias    [3*hidden]           seed 31
    //   conv.bias       [hidden]             seed 32
    //   out_proj.bias   [hidden]             seed 33
    let in_proj_w = det(&[3 * hidden, hidden], 21);
    let out_proj_w = det(&[hidden, hidden], 22);
    let conv_w = det(&[hidden, l_cache, 1], 23);

    let in_proj_b = det(&[3 * hidden], 31);
    let conv_b = det(&[hidden], 32);
    let out_proj_b = det(&[hidden], 33);

    // Shared input data so native (per-step [1,1,hidden]) and probe ([T,hidden])
    // get byte-identical bf16 rows.
    let x_data: Vec<f32> = (0..(t * hidden))
        .map(|i| (((i * 97 + 5).rem_euclid(19)) as f32 - 9.0) * 0.04)
        .collect();
    let x_seq = MxArray::from_float32(&x_data, &[t, hidden])
        .expect("x_seq")
        .astype(DType::BFloat16)
        .expect("bf16");

    // ----- native: T decode steps through a shared conv-state cache -----
    let mut conv = ShortConv::new(hidden as i32, l_cache as i32, conv_bias).expect("conv new");
    // in_proj / out_proj weights via the mode-aware LinearProj setter (stores a
    // plain bf16 weight on the `Standard` arm — what the probe registers and
    // `linear_proj` consumes after the store-time auto-transpose).
    conv.in_proj_mut()
        .set_weight(&in_proj_w, "in_proj")
        .expect("in_proj w");
    conv.out_proj_mut()
        .set_weight(&out_proj_w, "out_proj")
        .expect("out_proj w");
    // Depthwise conv weight is set verbatim in [H, K, 1] layout (never quantized).
    conv.set_conv_weight(&conv_w).expect("conv w");
    if conv_bias {
        conv.set_in_proj_bias(Some(&in_proj_b)).expect("in_proj b");
        conv.set_conv_bias(Some(&conv_b)).expect("conv b");
        conv.set_out_proj_bias(Some(&out_proj_b))
            .expect("out_proj b");
    }

    // One conv-state cache (slot 0), threaded across all t steps.
    let mut cache = ArraysCache::new(1);
    let mut native_last: Option<MxArray> = None;
    for i in 0..t {
        let row = &x_data[(i * hidden) as usize..((i + 1) * hidden) as usize];
        let x_i = MxArray::from_float32(row, &[1, 1, hidden])
            .expect("x_i")
            .astype(DType::BFloat16)
            .expect("bf16");
        native_last = Some(conv.forward(&x_i, Some(&mut cache)).expect("native fwd"));
    }
    let native_last = native_last.expect("ran >=1 step"); // [1,1,hidden]

    // ----- probe: same T steps through the compiled pure-fn -----
    // Null bias pointers when conv_bias=false (the probe ignores them).
    let null_ptr = std::ptr::null_mut();
    let out_ptr = unsafe {
        mlx_sys::mlx_lfm2_probe_conv_seq(
            x_seq.as_raw_ptr(),
            in_proj_w.as_raw_ptr(),
            conv_w.as_raw_ptr(),
            out_proj_w.as_raw_ptr(),
            if conv_bias {
                in_proj_b.as_raw_ptr()
            } else {
                null_ptr
            },
            if conv_bias {
                conv_b.as_raw_ptr()
            } else {
                null_ptr
            },
            if conv_bias {
                out_proj_b.as_raw_ptr()
            } else {
                null_ptr
            },
            l_cache as i32,
            if conv_bias { 1 } else { 0 },
        )
    };
    assert!(!out_ptr.is_null(), "mlx_lfm2_probe_conv_seq returned null");
    let probe_out = MxArray::from_handle(out_ptr, "probe_conv").expect("probe handle");

    let nv = to_vec(&native_last); // [1,1,hidden] flattened == hidden values
    let pv = to_vec(&probe_out); // [1,hidden] flattened == hidden values
    let d = max_abs(&nv, &pv);
    // Same ops on both sides (matmul / conv1d / elementwise), bf16 throughout —
    // only kernel-ordering jitter, so the same tight bound as the attn test holds.
    assert!(
        d < 2e-2,
        "compiled conv pure-fn must match native (conv_bias={conv_bias}): max_abs={d}"
    );
}

/// ShortConv parity WITHOUT biases (LFM2.5 production default: conv_bias=false).
#[test]
fn compiled_conv_seq_matches_native() {
    run_conv_parity(false);
}

/// ShortConv parity WITH biases (in_proj 3H + conv H + out_proj H additive
/// biases) — exercises the `conv_bias=true` code path end to end.
#[test]
fn compiled_conv_seq_matches_native_with_bias() {
    run_conv_parity(true);
}

/// 2b-1 end-to-end-SHAPED gate: the full `lfm2_decode_fn` assembly (driven via
/// the synthetic-model probe) must match a hand-assembled native `[conv, attn,
/// conv]` dense stack over the same `T`-step decode. Exercises the per-layer
/// conv/attn dispatch (from `is_attn[]`), the operator_norm→op→+res→ffn_norm→
/// mlp→+res order, the conv-state vs KV slot interleaving at uniform stride 2,
/// the final `embedding_norm`, and the tied `embed_tokens` head. The probe runs
/// `lfm2_decode_fn` EAGERLY and `mlx_lfm2_get_model_id()` is untouched, so the
/// production gate stays OFF.
fn run_decode_seq_parity(conv_bias: bool) {
    use crate::nn::{Embedding, RMSNorm};

    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());

    let hidden = 64i64;
    let num_heads = 4i64;
    let num_kv_heads = 2i64;
    let head_dim = 16i64; // hidden == num_heads * head_dim
    let inter = 128i64;
    let l_cache = 3i64;
    let vocab = 32i64;
    let t = 4i64;
    let norm_eps = 1e-5f64;
    let rope_theta = 1_000_000.0f64;
    let is_attn = [0i32, 1, 0]; // [conv, attn, conv]
    let n = is_attn.len();
    let token_ids: Vec<i32> = (0..t as i32).map(|i| (i * 7 + 1) % vocab as i32).collect();

    // ---- shared weights: the SAME arrays are fed to BOTH native and probe ----
    let embed_w = det(&[vocab, hidden], 101);
    let emb_norm_w = det_norm(hidden, 102);
    let op_norm: Vec<MxArray> = (0..n).map(|i| det_norm(hidden, 110 + i as i64)).collect();
    let ffn_norm: Vec<MxArray> = (0..n).map(|i| det_norm(hidden, 120 + i as i64)).collect();
    let gate: Vec<MxArray> = (0..n)
        .map(|i| det(&[inter, hidden], 130 + i as i64))
        .collect();
    let up: Vec<MxArray> = (0..n)
        .map(|i| det(&[inter, hidden], 140 + i as i64))
        .collect();
    let down: Vec<MxArray> = (0..n)
        .map(|i| det(&[hidden, inter], 150 + i as i64))
        .collect();

    // attn-only tensors: Some at attn layers, None (→ null ptr) at conv layers.
    let attn_t = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        (is_attn[i] == 1).then(|| det(shape, seed))
    };
    let attn_norm_t = |i: usize, seed: i64| -> Option<MxArray> {
        (is_attn[i] == 1).then(|| det_norm(head_dim, seed))
    };
    let q: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[num_heads * head_dim, hidden], 160 + i as i64))
        .collect();
    let k: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[num_kv_heads * head_dim, hidden], 170 + i as i64))
        .collect();
    let v: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[num_kv_heads * head_dim, hidden], 180 + i as i64))
        .collect();
    let o: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[hidden, num_heads * head_dim], 190 + i as i64))
        .collect();
    let qn: Vec<Option<MxArray>> = (0..n).map(|i| attn_norm_t(i, 200 + i as i64)).collect();
    let kn: Vec<Option<MxArray>> = (0..n).map(|i| attn_norm_t(i, 210 + i as i64)).collect();

    // conv-only tensors: Some at conv layers, None (→ null ptr) at attn layers.
    let conv_t = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        (is_attn[i] == 0).then(|| det(shape, seed))
    };
    let in_proj: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_t(i, &[3 * hidden, hidden], 220 + i as i64))
        .collect();
    let conv_w: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_t(i, &[hidden, l_cache, 1], 230 + i as i64))
        .collect();
    let out_proj: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_t(i, &[hidden, hidden], 240 + i as i64))
        .collect();

    // conv-bias tensors: Some at conv layers ONLY when conv_bias is on; None
    // (→ null ptr / not set) otherwise. The SAME arrays feed BOTH native + probe.
    let conv_bias_t = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        (conv_bias && is_attn[i] == 0).then(|| det(shape, seed))
    };
    let in_proj_b: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_bias_t(i, &[3 * hidden], 250 + i as i64))
        .collect();
    let conv_b: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_bias_t(i, &[hidden], 260 + i as i64))
        .collect();
    let out_proj_b: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_bias_t(i, &[hidden], 270 + i as i64))
        .collect();

    // ---- native: hand-assembled [conv, attn, conv] stack ----
    let mut embed = Embedding::new(vocab as u32, hidden as u32).expect("embed");
    embed.set_weight(&embed_w).expect("embed w");
    let mut emb_norm = RMSNorm::new(hidden as u32, Some(norm_eps)).expect("emb norm");
    emb_norm.set_weight(&emb_norm_w).expect("emb norm w");

    enum Op {
        Conv(ShortConv),
        Attn(Lfm2Attention),
    }
    let mut ops: Vec<Op> = Vec::new();
    let mut op_norms: Vec<RMSNorm> = Vec::new();
    let mut ffn_norms: Vec<RMSNorm> = Vec::new();
    let mut mlps: Vec<MLP> = Vec::new();
    for i in 0..n {
        let mut onrm = RMSNorm::new(hidden as u32, Some(norm_eps)).expect("on");
        onrm.set_weight(&op_norm[i]).expect("on w");
        op_norms.push(onrm);
        let mut fnrm = RMSNorm::new(hidden as u32, Some(norm_eps)).expect("fn");
        fnrm.set_weight(&ffn_norm[i]).expect("fn w");
        ffn_norms.push(fnrm);
        let mut mlp = MLP::new(hidden as u32, inter as u32).expect("mlp");
        mlp.set_gate_proj_weight(&gate[i]).expect("g");
        mlp.set_up_proj_weight(&up[i]).expect("u");
        mlp.set_down_proj_weight(&down[i]).expect("d");
        mlps.push(mlp);
        if is_attn[i] == 1 {
            let mut attn = Lfm2Attention::new(
                hidden as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                norm_eps,
                rope_theta,
            )
            .expect("attn");
            attn.q_proj_mut()
                .set_weight(q[i].as_ref().expect("q"), "q")
                .expect("q set");
            attn.k_proj_mut()
                .set_weight(k[i].as_ref().expect("k"), "k")
                .expect("k set");
            attn.v_proj_mut()
                .set_weight(v[i].as_ref().expect("v"), "v")
                .expect("v set");
            attn.out_proj_mut()
                .set_weight(o[i].as_ref().expect("o"), "o")
                .expect("o set");
            attn.set_q_layernorm_weight(qn[i].as_ref().expect("qn"))
                .expect("qn set");
            attn.set_k_layernorm_weight(kn[i].as_ref().expect("kn"))
                .expect("kn set");
            ops.push(Op::Attn(attn));
        } else {
            let mut conv = ShortConv::new(hidden as i32, l_cache as i32, conv_bias).expect("conv");
            conv.in_proj_mut()
                .set_weight(in_proj[i].as_ref().expect("ip"), "in_proj")
                .expect("ip set");
            conv.set_conv_weight(conv_w[i].as_ref().expect("cw"))
                .expect("cw set");
            conv.out_proj_mut()
                .set_weight(out_proj[i].as_ref().expect("op"), "out_proj")
                .expect("op set");
            if conv_bias {
                conv.set_in_proj_bias(Some(in_proj_b[i].as_ref().expect("ipb")))
                    .expect("ipb set");
                conv.set_conv_bias(Some(conv_b[i].as_ref().expect("cb")))
                    .expect("cb set");
                conv.set_out_proj_bias(Some(out_proj_b[i].as_ref().expect("opb")))
                    .expect("opb set");
            }
            ops.push(Op::Conv(conv));
        }
    }

    let mut conv_caches: Vec<Option<ArraysCache>> = (0..n)
        .map(|i| (is_attn[i] == 0).then(|| ArraysCache::new(1)))
        .collect();
    let mut kv_caches: Vec<Option<KVCache>> = (0..n)
        .map(|i| (is_attn[i] == 1).then(KVCache::new))
        .collect();

    let mut native_last: Option<MxArray> = None;
    for &tok in &token_ids {
        let ids = MxArray::from_int32(&[tok], &[1, 1]).expect("ids");
        let mut h = embed.forward(&ids).expect("embed fwd"); // [1,1,hidden]
        for i in 0..n {
            let normed = op_norms[i].forward(&h).expect("on fwd");
            let r = match &ops[i] {
                Op::Conv(c) => c
                    .forward(&normed, conv_caches[i].as_mut())
                    .expect("conv fwd"),
                Op::Attn(a) => a
                    .forward(&normed, None, kv_caches[i].as_mut())
                    .expect("attn fwd"),
            };
            h = h.add(&r).expect("res1");
            let fn_in = ffn_norms[i].forward(&h).expect("fn fwd");
            let ffn_out = mlps[i].forward(&fn_in).expect("mlp fwd");
            h = h.add(&ffn_out).expect("res2");
        }
        h = emb_norm.forward(&h).expect("emb norm fwd");
        native_last = Some(embed.as_linear(&h).expect("as_linear")); // [1,1,vocab]
    }
    let native_last = native_last.expect("ran >=1 step");

    // ---- probe: per-category pointer arrays (null for the irrelevant kind) ----
    let nullp = std::ptr::null_mut::<mlx_sys::mlx_array>();
    let optr = |slots: &[Option<MxArray>]| -> Vec<*mut mlx_sys::mlx_array> {
        slots
            .iter()
            .map(|o| o.as_ref().map_or(nullp, |a| a.as_raw_ptr()))
            .collect()
    };
    let reqr = |slots: &[MxArray]| -> Vec<*mut mlx_sys::mlx_array> {
        slots.iter().map(|a| a.as_raw_ptr()).collect()
    };
    let op_norm_p = reqr(&op_norm);
    let ffn_norm_p = reqr(&ffn_norm);
    let gate_p = reqr(&gate);
    let up_p = reqr(&up);
    let down_p = reqr(&down);
    let q_p = optr(&q);
    let k_p = optr(&k);
    let v_p = optr(&v);
    let o_p = optr(&o);
    let qn_p = optr(&qn);
    let kn_p = optr(&kn);
    let in_p = optr(&in_proj);
    let cw_p = optr(&conv_w);
    let op_p = optr(&out_proj);
    let in_b_p = optr(&in_proj_b);
    let cb_p = optr(&conv_b);
    let ob_p = optr(&out_proj_b);

    let out_ptr = unsafe {
        mlx_sys::mlx_lfm2_probe_decode_seq(
            embed_w.as_raw_ptr(),
            emb_norm_w.as_raw_ptr(),
            is_attn.as_ptr(),
            n as i32,
            hidden as i32,
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            l_cache as i32,
            rope_theta as f32,
            norm_eps as f32,
            token_ids.as_ptr(),
            t as i32,
            op_norm_p.as_ptr(),
            ffn_norm_p.as_ptr(),
            gate_p.as_ptr(),
            up_p.as_ptr(),
            down_p.as_ptr(),
            q_p.as_ptr(),
            k_p.as_ptr(),
            v_p.as_ptr(),
            o_p.as_ptr(),
            qn_p.as_ptr(),
            kn_p.as_ptr(),
            in_p.as_ptr(),
            cw_p.as_ptr(),
            op_p.as_ptr(),
            i32::from(conv_bias),
            in_b_p.as_ptr(),
            cb_p.as_ptr(),
            ob_p.as_ptr(),
        )
    };
    assert!(
        !out_ptr.is_null(),
        "mlx_lfm2_probe_decode_seq returned null"
    );
    let probe = MxArray::from_handle(out_ptr, "probe_decode").expect("probe handle");

    let d = max_abs(&to_vec(&native_last), &to_vec(&probe));
    println!("lfm2 compiled decode-seq parity (conv_bias={conv_bias}): max_abs = {d}");
    assert!(
        d < 2e-2,
        "compiled decode_fn must match native dense [conv,attn,conv] stack (conv_bias={conv_bias}): max_abs={d}"
    );
}

/// 2b-1 full-decode parity WITHOUT conv biases (LFM2.5 production default).
#[test]
fn compiled_decode_seq_matches_native() {
    run_decode_seq_parity(false);
}

/// Phase 4 Piece 1: the SAME full synthetic decode-sequence parity, but with the
/// ShortConv biases (`conv.in_proj.bias`, `conv.conv.bias`, `conv.out_proj.bias`)
/// seeded into the registry and applied on BOTH sides — the compiled
/// `lfm2_decode_fn` via `cfg.conv_bias` (threaded through the probe's `conv_bias`
/// arg + the three synthetic bias pointer arrays) and the native `ShortConv` via
/// its bias setters. Proves the threaded conv biases land under the keys
/// `get_weight` reads and the compiled decode matches native end to end.
#[test]
fn compiled_decode_seq_matches_native_with_conv_bias() {
    run_decode_seq_parity(true);
}

/// Phase-3a end-to-end-SHAPED MoE gate: the full `lfm2_decode_fn` assembly with
/// the sparse-MoE FFN branch (driven via `mlx_lfm2_probe_moe_decode_seq`) must
/// match a hand-assembled native `[conv(dense), attn(MoE), conv(MoE)]` stack over
/// the same `T`-step decode. The dense layer (idx 0 < num_dense_layers) routes
/// through the dense SwiGLU MLP; the MoE layers (idx >= num_dense_layers) route
/// through the sparse `Lfm2SparseMoeBlock` natively and `lfm2_moe_ffn` on the
/// compiled path (router softmax + selection-only expert_bias + top-k +
/// switch_mlp SwiGLU + weighted sum). The probe runs EAGERLY and
/// `mlx_lfm2_get_model_id()` is untouched, so the production gate stays OFF.
#[test]
fn compiled_moe_decode_seq_matches_native() {
    use super::config::Lfm2Config;
    use super::decoder_layer::OperatorType;
    use super::sparse_moe::Lfm2SparseMoeBlock;
    use crate::nn::{Embedding, RMSNorm};

    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());

    let hidden = 64i64;
    let num_heads = 4i64;
    let num_kv_heads = 2i64;
    let head_dim = 16i64; // hidden == num_heads * head_dim
    let inter = 128i64; // dense-layer SwiGLU hidden
    let moe_inter = 96i64; // per-expert SwiGLU hidden
    let l_cache = 3i64;
    let vocab = 32i64;
    let t = 4i64;
    let norm_eps = 1e-5f64;
    let rope_theta = 1_000_000.0f64;
    let n_exp = 4i32;
    let top_k = 2i32;
    let num_dense_layers = 1i32;
    let is_attn = [0i32, 1, 0]; // [conv, attn, conv]
    let n = is_attn.len();
    // A layer is MoE iff idx >= num_dense_layers (and num_experts > 0).
    let is_moe = |i: usize| (i as i32) >= num_dense_layers;
    let token_ids: Vec<i32> = (0..t as i32).map(|i| (i * 7 + 1) % vocab as i32).collect();

    // ---- shared weights: the SAME arrays are fed to BOTH native and probe ----
    let embed_w = det(&[vocab, hidden], 301);
    let emb_norm_w = det_norm(hidden, 302);
    let op_norm: Vec<MxArray> = (0..n).map(|i| det_norm(hidden, 310 + i as i64)).collect();
    let ffn_norm: Vec<MxArray> = (0..n).map(|i| det_norm(hidden, 320 + i as i64)).collect();

    // Dense-FFN tensors: Some at dense layers (idx < num_dense_layers), None else.
    let dense_t = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        (!is_moe(i)).then(|| det(shape, seed))
    };
    let gate: Vec<Option<MxArray>> = (0..n)
        .map(|i| dense_t(i, &[inter, hidden], 330 + i as i64))
        .collect();
    let up: Vec<Option<MxArray>> = (0..n)
        .map(|i| dense_t(i, &[inter, hidden], 340 + i as i64))
        .collect();
    let down: Vec<Option<MxArray>> = (0..n)
        .map(|i| dense_t(i, &[hidden, inter], 350 + i as i64))
        .collect();

    // MoE tensors: Some at MoE layers, None at dense layers.
    let moe_t = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        is_moe(i).then(|| det(shape, seed))
    };
    let moe_router: Vec<Option<MxArray>> = (0..n)
        .map(|i| moe_t(i, &[n_exp as i64, hidden], 360 + i as i64))
        .collect();
    let moe_bias: Vec<Option<MxArray>> = (0..n)
        .map(|i| moe_t(i, &[n_exp as i64], 370 + i as i64))
        .collect();
    let moe_gate: Vec<Option<MxArray>> = (0..n)
        .map(|i| moe_t(i, &[n_exp as i64, moe_inter, hidden], 380 + i as i64))
        .collect();
    let moe_up: Vec<Option<MxArray>> = (0..n)
        .map(|i| moe_t(i, &[n_exp as i64, moe_inter, hidden], 390 + i as i64))
        .collect();
    let moe_down: Vec<Option<MxArray>> = (0..n)
        .map(|i| moe_t(i, &[n_exp as i64, hidden, moe_inter], 400 + i as i64))
        .collect();

    // attn-only tensors: Some at attn layers, None at conv layers.
    let attn_t = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        (is_attn[i] == 1).then(|| det(shape, seed))
    };
    let attn_norm_t = |i: usize, seed: i64| -> Option<MxArray> {
        (is_attn[i] == 1).then(|| det_norm(head_dim, seed))
    };
    let q: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[num_heads * head_dim, hidden], 410 + i as i64))
        .collect();
    let k: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[num_kv_heads * head_dim, hidden], 420 + i as i64))
        .collect();
    let v: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[num_kv_heads * head_dim, hidden], 430 + i as i64))
        .collect();
    let o: Vec<Option<MxArray>> = (0..n)
        .map(|i| attn_t(i, &[hidden, num_heads * head_dim], 440 + i as i64))
        .collect();
    let qn: Vec<Option<MxArray>> = (0..n).map(|i| attn_norm_t(i, 450 + i as i64)).collect();
    let kn: Vec<Option<MxArray>> = (0..n).map(|i| attn_norm_t(i, 460 + i as i64)).collect();

    // conv-only tensors: Some at conv layers, None at attn layers.
    let conv_tn = |i: usize, shape: &[i64], seed: i64| -> Option<MxArray> {
        (is_attn[i] == 0).then(|| det(shape, seed))
    };
    let in_proj: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_tn(i, &[3 * hidden, hidden], 470 + i as i64))
        .collect();
    let conv_w: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_tn(i, &[hidden, l_cache, 1], 480 + i as i64))
        .collect();
    let out_proj: Vec<Option<MxArray>> = (0..n)
        .map(|i| conv_tn(i, &[hidden, hidden], 490 + i as i64))
        .collect();

    // ---- native: hand-assembled [conv(dense), attn(MoE), conv(MoE)] stack ----
    let mut embed = Embedding::new(vocab as u32, hidden as u32).expect("embed");
    embed.set_weight(&embed_w).expect("embed w");
    let mut emb_norm = RMSNorm::new(hidden as u32, Some(norm_eps)).expect("emb norm");
    emb_norm.set_weight(&emb_norm_w).expect("emb norm w");

    // Minimal Lfm2Config for Lfm2SparseMoeBlock::new (only the MoE fields matter).
    let moe_cfg = Lfm2Config {
        vocab_size: vocab as i32,
        hidden_size: hidden as i32,
        num_hidden_layers: n as i32,
        num_attention_heads: num_heads as i32,
        num_key_value_heads: num_kv_heads as i32,
        max_position_embeddings: 128,
        norm_eps,
        conv_bias: false,
        conv_l_cache: l_cache as i32,
        block_dim: hidden as i32,
        block_ff_dim: inter as i32,
        block_multiple_of: 256,
        block_ffn_dim_multiplier: 1.0,
        block_auto_adjust_ff_dim: false,
        rope_theta,
        layer_types: vec!["conv".into(), "full_attention".into(), "conv".into()],
        tie_embedding: true,
        eos_token_id: 7,
        bos_token_id: 1,
        pad_token_id: 0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        use_block_paged_cache: None,
        intermediate_size: Some(inter as i32),
        moe_intermediate_size: Some(moe_inter as i32),
        num_experts: Some(n_exp),
        num_experts_per_tok: Some(top_k),
        num_dense_layers: Some(num_dense_layers),
        norm_topk_prob: Some(true),
        use_expert_bias: Some(true),
    };

    // FFN holder: dense SwiGLU MLP (dense layers) or sparse MoE block (MoE layers).
    enum Ffn {
        Dense(MLP),
        Moe(Lfm2SparseMoeBlock),
    }
    // Build per-layer native modules directly (not full Lfm2DecoderLayer, which
    // bundles its own norms): conv/attn operator + the two RMSNorms + FFN.
    let mut ops: Vec<OperatorType> = Vec::new();
    let mut op_norms: Vec<RMSNorm> = Vec::new();
    let mut ffn_norms: Vec<RMSNorm> = Vec::new();
    let mut ffns: Vec<Ffn> = Vec::new();
    for i in 0..n {
        let mut onrm = RMSNorm::new(hidden as u32, Some(norm_eps)).expect("on");
        onrm.set_weight(&op_norm[i]).expect("on w");
        op_norms.push(onrm);
        let mut fnrm = RMSNorm::new(hidden as u32, Some(norm_eps)).expect("fn");
        fnrm.set_weight(&ffn_norm[i]).expect("fn w");
        ffn_norms.push(fnrm);

        if is_moe(i) {
            let mut moe = Lfm2SparseMoeBlock::new(&moe_cfg).expect("moe new");
            // `set_gate_weight` is the ROUTER gate (a Result); the three
            // `set_switch_mlp_*` setters return `()` (no Result) — do not chain
            // `.expect()` on them.
            moe.set_gate_weight(moe_router[i].as_ref().expect("router"))
                .expect("router set");
            moe.set_expert_bias(moe_bias[i].as_ref().expect("bias"))
                .expect("bias set");
            moe.set_switch_mlp_gate_proj_weight(moe_gate[i].as_ref().expect("g"));
            moe.set_switch_mlp_up_proj_weight(moe_up[i].as_ref().expect("u"));
            moe.set_switch_mlp_down_proj_weight(moe_down[i].as_ref().expect("d"));
            ffns.push(Ffn::Moe(moe));
        } else {
            let mut mlp = MLP::new(hidden as u32, inter as u32).expect("mlp");
            mlp.set_gate_proj_weight(gate[i].as_ref().expect("g"))
                .expect("g");
            mlp.set_up_proj_weight(up[i].as_ref().expect("u"))
                .expect("u");
            mlp.set_down_proj_weight(down[i].as_ref().expect("d"))
                .expect("d");
            ffns.push(Ffn::Dense(mlp));
        }

        if is_attn[i] == 1 {
            let mut attn = Lfm2Attention::new(
                hidden as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                norm_eps,
                rope_theta,
            )
            .expect("attn");
            attn.q_proj_mut()
                .set_weight(q[i].as_ref().expect("q"), "q_proj")
                .expect("set q");
            attn.k_proj_mut()
                .set_weight(k[i].as_ref().expect("k"), "k_proj")
                .expect("set k");
            attn.v_proj_mut()
                .set_weight(v[i].as_ref().expect("v"), "v_proj")
                .expect("set v");
            attn.out_proj_mut()
                .set_weight(o[i].as_ref().expect("o"), "out_proj")
                .expect("set o");
            attn.set_q_layernorm_weight(qn[i].as_ref().expect("qn"))
                .expect("set qn");
            attn.set_k_layernorm_weight(kn[i].as_ref().expect("kn"))
                .expect("set kn");
            ops.push(OperatorType::Attention(attn));
        } else {
            let mut conv = ShortConv::new(hidden as i32, l_cache as i32, false).expect("conv");
            conv.in_proj_mut()
                .set_weight(in_proj[i].as_ref().expect("in"), "in_proj")
                .expect("set in");
            conv.set_conv_weight(conv_w[i].as_ref().expect("cw"))
                .expect("set conv");
            conv.out_proj_mut()
                .set_weight(out_proj[i].as_ref().expect("out"), "out_proj")
                .expect("set out");
            ops.push(OperatorType::Conv(conv));
        }
    }

    let mut conv_caches: Vec<Option<ArraysCache>> = (0..n)
        .map(|i| (is_attn[i] == 0).then(|| ArraysCache::new(1)))
        .collect();
    let mut kv_caches: Vec<Option<KVCache>> = (0..n)
        .map(|i| (is_attn[i] == 1).then(KVCache::new))
        .collect();

    let mut native_last: Option<MxArray> = None;
    for &tok in &token_ids {
        let ids = MxArray::from_int32(&[tok], &[1, 1]).expect("ids");
        let mut h = embed.forward(&ids).expect("embed fwd"); // [1,1,hidden]
        for i in 0..n {
            let normed = op_norms[i].forward(&h).expect("on fwd");
            let r = match &ops[i] {
                OperatorType::Conv(c) => c
                    .forward(&normed, conv_caches[i].as_mut())
                    .expect("conv fwd"),
                OperatorType::Attention(a) => a
                    .forward(&normed, None, kv_caches[i].as_mut())
                    .expect("attn fwd"),
            };
            h = h.add(&r).expect("res1");
            let fn_in = ffn_norms[i].forward(&h).expect("fn fwd");
            let ffn_out = match &ffns[i] {
                Ffn::Dense(m) => m.forward(&fn_in).expect("dense mlp fwd"),
                Ffn::Moe(m) => m.forward(&fn_in).expect("moe fwd"),
            };
            h = h.add(&ffn_out).expect("res2");
        }
        h = emb_norm.forward(&h).expect("emb norm fwd");
        native_last = Some(embed.as_linear(&h).expect("as_linear")); // [1,1,vocab]
    }
    let native_last = native_last.expect("ran >=1 step");

    // ---- probe: per-category pointer arrays (null for the irrelevant kind) ----
    let nullp = std::ptr::null_mut::<mlx_sys::mlx_array>();
    let optr = |slots: &[Option<MxArray>]| -> Vec<*mut mlx_sys::mlx_array> {
        slots
            .iter()
            .map(|o| o.as_ref().map_or(nullp, |a| a.as_raw_ptr()))
            .collect()
    };
    let reqr = |slots: &[MxArray]| -> Vec<*mut mlx_sys::mlx_array> {
        slots.iter().map(|a| a.as_raw_ptr()).collect()
    };
    let op_norm_p = reqr(&op_norm);
    let ffn_norm_p = reqr(&ffn_norm);
    let gate_p = optr(&gate);
    let up_p = optr(&up);
    let down_p = optr(&down);
    let q_p = optr(&q);
    let k_p = optr(&k);
    let v_p = optr(&v);
    let o_p = optr(&o);
    let qn_p = optr(&qn);
    let kn_p = optr(&kn);
    let in_p = optr(&in_proj);
    let cw_p = optr(&conv_w);
    let op_p = optr(&out_proj);
    let mr_p = optr(&moe_router);
    let mb_p = optr(&moe_bias);
    let mg_p = optr(&moe_gate);
    let mu_p = optr(&moe_up);
    let md_p = optr(&moe_down);

    let out_ptr = unsafe {
        mlx_sys::mlx_lfm2_probe_moe_decode_seq(
            embed_w.as_raw_ptr(),
            emb_norm_w.as_raw_ptr(),
            is_attn.as_ptr(),
            n as i32,
            hidden as i32,
            num_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            l_cache as i32,
            rope_theta as f32,
            norm_eps as f32,
            n_exp,
            top_k,
            num_dense_layers,
            1, // norm_topk_prob
            1, // use_expert_bias
            0, // use_sigmoid (lfm2 uses softmax)
            token_ids.as_ptr(),
            t as i32,
            op_norm_p.as_ptr(),
            ffn_norm_p.as_ptr(),
            gate_p.as_ptr(),
            up_p.as_ptr(),
            down_p.as_ptr(),
            q_p.as_ptr(),
            k_p.as_ptr(),
            v_p.as_ptr(),
            o_p.as_ptr(),
            qn_p.as_ptr(),
            kn_p.as_ptr(),
            in_p.as_ptr(),
            cw_p.as_ptr(),
            op_p.as_ptr(),
            mr_p.as_ptr(),
            mb_p.as_ptr(),
            mg_p.as_ptr(),
            mu_p.as_ptr(),
            md_p.as_ptr(),
        )
    };
    assert!(
        !out_ptr.is_null(),
        "mlx_lfm2_probe_moe_decode_seq returned null"
    );
    let probe = MxArray::from_handle(out_ptr, "probe_moe_decode").expect("probe handle");

    let d = max_abs(&to_vec(&native_last), &to_vec(&probe));
    assert!(
        d < 2e-2,
        "compiled MoE decode_fn must match native [conv(dense),attn(MoE),conv(MoE)] stack: max_abs={d}"
    );
}

/// DECISIVE H1/H2 experiment: COMPILED-vs-EAGER synthetic MoE.
///
/// Drives the process-global `compiled_lfm2_decode()` (NOT eager
/// `lfm2_decode_fn`) with a FIXED 3-layer synthetic MoE stack and compares the
/// last-step logits against the EAGER run of the SAME fn on the SAME weights.
/// The ONLY variable is `mlx::core::compile`. A nonzero diff isolates a compile
/// effect on the MoE FFN path.
///
/// (1) WELL-SEPARATED router (`expert_bias` gaps of 4.0 dominate the <1.0
///     softmax spread -> top-k selection is bias-ranked, input-independent, NO
///     near-ties). compiled == eager here => `compile()` selects and computes
///     the MoE correctly (rules out a structural compile bug on this stack).
/// (2) NEAR-TIE router (`expert_bias` gaps of 1e-4, E=32/k=4 fan-out matching
///     the real 8B model -> selection decided by softmax(routing) near-ties,
///     FP-fusion sensitive). Diagnostic: a nonzero diff here positively
///     confirms the near-tie selection-flip mechanism (H2).
///
/// Runs in its OWN test so its fixed synthetic topology bakes into the compiled
/// static cleanly. `WS_TOL`, `NT_MIN_DIVERGENCE`, `ASSERT_NT_GT_WS` env vars
/// allow bisecting the magnitudes from the command line.
#[test]
fn compiled_moe_vs_eager_well_separated_is_clean() {
    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());

    let seed = 0x10F2_3C0Du64;

    // (1) Well-separated: compiled MUST match eager.
    let mut ws_maxabs = f32::NAN;
    let rc = unsafe { mlx_sys::mlx_lfm2_probe_moe_compiled_vs_eager(seed, 1, &mut ws_maxabs) };
    assert_eq!(rc, 0, "well-separated probe failed (rc={rc})");
    eprintln!("[H1/H2] well-separated compiled-vs-eager max_abs = {ws_maxabs:e}");
    let ws_tol: f32 = std::env::var("WS_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2e-2);
    assert!(
        ws_maxabs < ws_tol,
        "well-separated compiled MoE diverged from eager: max_abs={ws_maxabs} \
         (>= {ws_tol} would indicate a real compile bug / H1)"
    );

    // (2) Near-tie: diagnostic. Records the diff to characterize the
    // selection-flip mechanism. Must at least run cleanly.
    let mut nt_maxabs = f32::NAN;
    let rc2 = unsafe { mlx_sys::mlx_lfm2_probe_moe_compiled_vs_eager(seed, 0, &mut nt_maxabs) };
    assert_eq!(rc2, 0, "near-tie probe failed (rc={rc2})");
    eprintln!("[H1/H2] near-tie compiled-vs-eager max_abs = {nt_maxabs:e}");

    // Optional positive confirmation of the selection-flip mechanism.
    if let Some(min_div) = std::env::var("NT_MIN_DIVERGENCE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        assert!(
            nt_maxabs >= min_div,
            "near-tie divergence {nt_maxabs} < required {min_div}"
        );
    }
    if std::env::var("ASSERT_NT_GT_WS").is_ok() {
        assert!(
            nt_maxabs > ws_maxabs,
            "near-tie {nt_maxabs} not greater than well-separated {ws_maxabs}"
        );
    }
}

/// A->B MODEL-SWAP regression test — directly locks in the compile-epoch
/// invalidation fix for the stale-compiled-closure (frozen weight-constant)
/// hazard.
///
/// The compiled `lfm2_decode_fn` graph captures its weight CONSTANTS at trace
/// time. Before the epoch fix, the closure was an init-once `static auto`, so a
/// SECOND lfm2 model that re-took the compiled path with the same decode-graph
/// shapes would silently replay the FIRST model's frozen weights. The C++ probe
/// builds two distinct synthetic MoE models A and B (same fixed topology,
/// different seeded weights) in one process, runs A compiled (caching the
/// graph), then re-registers B and bumps the compile epoch
/// (`mlx_lfm2_invalidate_compiled`, mirroring `register_weights_with_cpp`), then
/// runs B both compiled and eager.
///
/// Assertions:
///   * `b_comp_vs_b_eager < 2e-2` — B's compiled output matches B's EAGER output
///     (the closure recompiled against B's live constants). WITHOUT the epoch
///     bump this would equal `b_comp_vs_a_comp` (A's frozen graph replayed) and
///     blow past tolerance — that is the regression this test guards.
///   * `b_comp_vs_a_comp` is LARGE (well above tolerance) — proves A and B are
///     genuinely different models, so a stale-graph reuse is distinguishable
///     from two coincidentally-equal models (without this the first assertion
///     would be vacuous).
///   * `a_comp_vs_a_eager < 2e-2` — compile is faithful for A too (rules out a
///     pre-existing compile bug masking the result).
///
/// Holds `COMPILED_WEIGHTS_RWLOCK` (write, poison-recovered) per the probe
/// contract (the probe is DESTRUCTIVE on the shared `g_weights()` map).
#[test]
fn compiled_moe_ab_model_swap_recompiles() {
    let _guard = COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());

    // Distinct seeds => distinct A and B weights.
    let seed_a = 0x1111_2222_3333_4444u64;
    let seed_b = 0xAAAA_BBBB_CCCC_DDDDu64;
    // A THIRD distinct seed for the no-bump warm-up pre-seed below. It MUST differ
    // from `seed_a`: the warm probe FORCES the cached stale closure to be
    // `warm_seed`'s model (see the warm-probe contract below), so it must freeze
    // DIFFERENT constants than MODEL A. If `warm_seed == seed_a`, the stale closure
    // would replay constants byte-identical to MODEL A's and (wrongly) still
    // produce A's correct logits, making the MODEL-A epoch-bump non-load-bearing
    // and the F3 gold-standard vacuous. With a different seed, removing the MODEL-A
    // bump makes A's compiled run replay `warm_seed`'s weights and
    // `a_comp_vs_a_eager` blows past PARITY_TOL — which is the regression the
    // MODEL-A bump must defeat.
    let warm_seed = 0x7777_8888_9999_AAAAu64;

    // F3 soundness: PRE-SEED a compiled closure at the current epoch BEFORE the
    // measured probe so the A-side stale-closure hazard manifests
    // DETERMINISTICALLY. The dedicated `warm_compiled_no_bump` probe registers its
    // own synthetic `warm_seed` weights, then — crucially — performs a SAME-EPOCH
    // RESET of the cached closure (drops any closure a prior same-epoch probe left
    // WITHOUT advancing the epoch) and runs one COMPILED decode. The reset forces
    // that decode to RE-TRACE against `warm_seed`'s constants, so the closure it
    // leaves cached at the current epoch is DETERMINISTICALLY `warm_seed`'s model —
    // regardless of what closure any earlier same-epoch probe had cached (e.g. the
    // well-separated compiled-vs-eager probe). It then clears the weights and, by
    // design, does NOT bump the compile epoch. So the measured A->B probe below
    // re-enters with `warm_seed`'s stale closure cached at this epoch: WITHOUT the
    // MODEL-A `build_model` epoch bump (the F3 production-style fix), MODEL A's
    // compiled run reuses that stale closure, replays `warm_seed`'s frozen
    // constants, and `a_comp_vs_a_eager` blows past PARITY_TOL — i.e. removing the
    // MODEL-A bump makes THIS test fail. WITH the bump, MODEL A is epoch-fresh and
    // all three deltas hold. The same-epoch reset is what makes the pre-seed
    // load-bearing AND order-independent: the stale closure is `warm_seed`'s model
    // whether or not another probe ran first in this process.
    let warm_rc = unsafe { mlx_sys::mlx_lfm2_probe_warm_compiled_no_bump(warm_seed) };
    assert_eq!(
        warm_rc, 0,
        "no-bump warm-up pre-seed probe failed (rc={warm_rc})"
    );

    let mut b_comp_vs_b_eager = f32::NAN;
    let mut b_comp_vs_a_comp = f32::NAN;
    let mut a_comp_vs_a_eager = f32::NAN;
    let rc = unsafe {
        mlx_sys::mlx_lfm2_probe_moe_ab_swap(
            seed_a,
            seed_b,
            &mut b_comp_vs_b_eager,
            &mut b_comp_vs_a_comp,
            &mut a_comp_vs_a_eager,
        )
    };
    assert_eq!(rc, 0, "A->B swap probe failed (rc={rc})");
    eprintln!(
        "[A->B swap] b_comp_vs_b_eager={b_comp_vs_b_eager:e} \
         b_comp_vs_a_comp={b_comp_vs_a_comp:e} a_comp_vs_a_eager={a_comp_vs_a_eager:e}"
    );

    // The models must actually differ, or the regression assertion below is
    // vacuous: reusing A's stale graph would have to produce a measurably wrong
    // answer for B. Require the A/B gap to be well clear of the parity tol.
    assert!(
        b_comp_vs_a_comp > 1e-1,
        "A and B are too close to distinguish a stale-graph reuse \
         (b_comp_vs_a_comp={b_comp_vs_a_comp}); the test cannot prove the fix"
    );

    // Compiled-vs-eager parity bound — the SAME 2e-2 the sibling lfm2 parity
    // gates use; we do NOT weaken it. With the per-epoch-`fun_id` retrace in
    // place (the C++ `detail::compile` path) the freshly-traced B graph captures
    // B's LIVE constants, so empirically BOTH `a_comp_vs_a_eager` AND
    // `b_comp_vs_b_eager` come out at EXACTLY 0.0 (eager and compiled run the
    // identical op DAG on the identical constants). A real stale-closure
    // regression replays A's frozen graph for B, pushing `b_comp_vs_b_eager` up
    // to ~`b_comp_vs_a_comp` (~0.67 here) — ~34x over this tolerance — so the
    // regression is caught decisively.
    const PARITY_TOL: f32 = 2e-2;

    // A's compile is faithful (no pre-existing compile bug).
    assert!(
        a_comp_vs_a_eager < PARITY_TOL,
        "model A compiled diverged from A eager: max_abs={a_comp_vs_a_eager} \
         (a pre-existing compile bug would invalidate this regression test)"
    );

    // THE FIX: B's compiled output matches B's EAGER output, i.e. the closure
    // RE-TRACED against B's live constants. Without the epoch bump (or with the
    // public `compile`, which replays the address-keyed cached trace) this would
    // instead equal b_comp_vs_a_comp (A's frozen graph replayed) — far above
    // PARITY_TOL — so this assertion is what fails on a regression.
    assert!(
        b_comp_vs_b_eager < PARITY_TOL,
        "STALE COMPILED CLOSURE: model B compiled output diverged from B eager \
         (max_abs={b_comp_vs_b_eager}); the compiled graph froze model A's weight \
         constants and was reused for B (b_comp_vs_a_comp={b_comp_vs_a_comp}). \
         The compile-epoch invalidation (mlx_lfm2_invalidate_compiled) did not take."
    );
}
