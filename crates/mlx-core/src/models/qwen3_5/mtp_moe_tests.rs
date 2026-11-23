//! Unit tests for the Qwen3.5 MoE MTP module.
//!
//! Tests that allocate MLX arrays require Metal. We skip when the
//! tiny config fails to construct — same pattern as the dense MTP
//! tests in `crates/mlx-core/src/models/qwen3_5/mtp.rs`.

use super::*;
use crate::array::DType;
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;

fn new_moe_mtp(cfg: &Qwen3_5MoeConfig) -> Result<Qwen3_5MTPModule> {
    let (base, moe) = cfg.split();
    Qwen3_5MTPModule::new_with_moe(&base, Some(&moe))
}

fn moe_mtp_mlp_variant(cfg: &Qwen3_5MoeConfig) -> crate::models::mtp_drafter::DrafterBodyVariant {
    let (base, moe) = cfg.split();
    Qwen3_5MTPModule::mtp_mlp_variant(&base, Some(&moe))
}

fn tiny_mtp_cfg() -> Qwen3_5MoeConfig {
    // hidden_size and head_dim chosen to keep the test cheap while
    // staying compatible with Qwen3.5 attention constraints (head_dim
    // divisible by 2 for RoPE). full_attention_interval=4 makes layer
    // 3 a full-attention layer; n_mtp_layers=1. num_experts=4 keeps
    // SparseMoeBlock construction cheap.
    Qwen3_5MoeConfig {
        rope_scaling: None,
        vocab_size: 1024,
        hidden_size: 64,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 1024,
        pad_token_id: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        shared_expert_intermediate_size: Some(64),
        moe_intermediate_size: Some(64),
        norm_topk_prob: true,
        mlp_only_layers: None,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 1,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
    }
}

fn build_mtp_or_skip(test_name: &str) -> Option<(Qwen3_5MTPModule, Qwen3_5MoeConfig)> {
    let cfg = tiny_mtp_cfg();
    match new_moe_mtp(&cfg) {
        Ok(m) => Some((m, cfg)),
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping {test_name} (MLX/Metal unavailable): {msg}");
                None
            } else {
                panic!("unexpected Qwen3_5MTPModule::new failure in {test_name}: {msg}");
            }
        }
    }
}

#[test]
fn ctor_constructs_one_layer_mtp() {
    let Some((mtp, cfg)) = build_mtp_or_skip("ctor_constructs_one_layer_mtp") else {
        return;
    };
    assert_eq!(mtp.num_layers(), 1);
    // Layer must be full-attention (enforced by Qwen3_5MTPModule::new).
    assert!(
        matches!(mtp.layers[0].attn, AttentionType::Full(_)),
        "MTP DecoderLayer must be full-attention; got Linear"
    );
    // With decoder_sparse_step=1 and no mlp_only_layers, fa_idx=3 is
    // an MoE layer; the MTP layer should be MoE-flavored to mirror
    // what the main decoder would build at the same layer_idx.
    let fa_idx = (cfg.full_attention_interval - 1) as usize;
    assert!(cfg.is_moe_layer(fa_idx));
    assert!(
        matches!(mtp.layers[0].mlp, MLPType::MoE(_)),
        "MTP DecoderLayer must mirror the main model's MLP flavor at fa_idx; expected MoE"
    );
}

#[test]
fn ctor_rejects_zero_mtp_layers() {
    let mut cfg = tiny_mtp_cfg();
    cfg.n_mtp_layers = 0;
    match new_moe_mtp(&cfg) {
        Ok(_) => panic!("n_mtp_layers=0 must fail"),
        Err(err) => {
            let msg = err.reason.to_string();
            assert!(
                msg.contains("n_mtp_layers"),
                "error must mention n_mtp_layers; got: {msg}"
            );
        }
    }
}

#[test]
fn ctor_rejects_all_linear_config() {
    // full_attention_interval<=0 means every layer is GDN; the MTP
    // ctor must refuse to silently produce linear-attention MTP
    // layers.
    let mut cfg = tiny_mtp_cfg();
    cfg.full_attention_interval = 0;
    match new_moe_mtp(&cfg) {
        Ok(_) => panic!("all-linear config must be rejected by the MTP ctor"),
        Err(err) => {
            let msg = err.reason.to_string();
            assert!(
                msg.contains("GDN") || msg.contains("linear"),
                "error must mention GDN/linear rejection; got: {msg}"
            );
        }
    }
}

#[test]
fn apply_weights_rejects_explicit_or_stored_plain_fp8_before_dense_fallback() {
    let label = "apply_weights_rejects_explicit_or_stored_plain_fp8_before_dense_fallback";
    let default_plq = PerLayerQuant {
        bits: 4,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };
    let gate_plq = PerLayerQuant {
        bits: 8,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };

    let Some((mut mtp, _)) = build_mtp_or_skip(label) else {
        return;
    };
    let mut stored_fp8 = HashMap::from([
        (
            "mtp.fc.weight".to_string(),
            MxArray::from_uint8(&[0; 8], &[2, 4]).unwrap(),
        ),
        (
            "mtp.fc.scales".to_string(),
            MxArray::from_float32(&[1.0, 1.0], &[2, 1]).unwrap(),
        ),
    ]);
    let err = mtp
        .apply_weights(
            &mut stored_fp8,
            default_plq,
            Some(gate_plq),
            &HashMap::new(),
            DType::BFloat16,
        )
        .expect_err("raw Uint8 MoE MTP storage must not reach Linear::set_weight");
    assert!(err.reason.contains("Uint8 MTP storage"), "{}", err.reason);
    assert!(err.reason.contains("dense fallback"), "{}", err.reason);

    // Uint8 weight + Uint8 scales is structurally MXFP8, but the MoE
    // resolver must still agree on Mxfp8 for this exact projection. A
    // missing override falls back to affine here and therefore rejects.
    let Some((mut mtp, _)) = build_mtp_or_skip(label) else {
        return;
    };
    let mut stale_mxfp8 = HashMap::from([
        (
            "mtp.fc.weight".to_string(),
            MxArray::from_uint8(&[0; 8], &[2, 4]).unwrap(),
        ),
        (
            "mtp.fc.scales".to_string(),
            MxArray::from_uint8(&[127, 127], &[2, 1]).unwrap(),
        ),
    ]);
    let err = mtp
        .apply_weights(
            &mut stale_mxfp8,
            default_plq,
            Some(gate_plq),
            &HashMap::new(),
            DType::BFloat16,
        )
        .expect_err("MXFP8 MoE MTP storage with stale affine metadata must reject");
    assert!(err.reason.contains("MXFP8 MTP storage"), "{}", err.reason);
    assert!(err.reason.contains("resolves to Affine"), "{}", err.reason);

    let Some((mut mtp, _)) = build_mtp_or_skip(label) else {
        return;
    };
    let mut dense = HashMap::from([(
        "mtp.fc.weight".to_string(),
        MxArray::from_float32(&[0.0; 8], &[2, 4])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap(),
    )]);
    let explicit = HashMap::from([(
        "mtp.fc".to_string(),
        PerLayerQuant {
            bits: 8,
            group_size: crate::quant::fp8_weight::FP8_E4M3_GROUP_SIZE,
            mode: PerLayerMode::Fp8E4m3,
            input_amax: None,
            layout: Default::default(),
        },
    )]);
    let err = mtp
        .apply_weights(
            &mut dense,
            default_plq,
            Some(gate_plq),
            &explicit,
            DType::BFloat16,
        )
        .expect_err("explicit fp8_e4m3 MoE MTP metadata must reject");
    assert!(err.reason.contains("explicit fp8_e4m3"), "{}", err.reason);
}

#[test]
fn fresh_caches_match_num_layers() {
    let cfg = tiny_mtp_cfg();
    let caches = Qwen3_5MTPModule::fresh_caches(&cfg.to_dense_config());
    assert_eq!(caches.len(), cfg.n_mtp_layers as usize);
    for c in &caches {
        assert!(
            matches!(c, Qwen3_5LayerCache::FullAttention(_)),
            "MTP fresh_caches must be FullAttention slots"
        );
    }
}

/// Contract guard: `get_parameters()` must emit a SUPERSET of every key
/// the MoE loader requires for an MTP head. The MoE post-sanitize gate
/// (`qwen3_5_moe/persistence.rs`) calls `missing_required_mtp_keys(..,
/// DrafterBodyVariant::Moe, ..)`; we assert against that same shared
/// source of truth so save can never drift below what load needs. With
/// the tiny config (decoder_sparse_step=1) the MTP layer is MoE-flavored,
/// so this also locks the `switch_mlp.*` + `mlp.gate` MoE MLP keys.
#[test]
fn get_parameters_is_superset_of_required_keys() {
    use crate::models::mtp_drafter::{DrafterBodyVariant, missing_required_mtp_keys};

    let Some((mtp, cfg)) = build_mtp_or_skip("get_parameters_is_superset_of_required_keys") else {
        return;
    };

    let params = mtp.get_parameters();
    let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, cfg.n_mtp_layers);
    assert!(
        missing.is_empty(),
        "get_parameters() must emit every loader-required MoE MTP key; missing: {missing:?}"
    );

    // Sanity: a freshly-constructed (random-init) module is bf16, so the
    // dense-only save path is permitted to serialize it.
    assert!(
        !mtp.has_quantized_weights(),
        "a bf16-constructed MoE MTP module must not report quantized weights"
    );
}

/// Affine-quantize a floating-point weight via `mlx_quantize`, returning
/// `(packed_weight, scales, biases)`. Works for 2D Linear weights and 3D
/// SwitchLinear expert stacks alike (MLX groups along the last axis). Used
/// only by the quantized-apply test below. Returns `None` if MLX/Metal is
/// unavailable so the caller can skip.
fn quantize_affine_or_skip(
    weight: &MxArray,
    group_size: i32,
    bits: i32,
) -> Option<(MxArray, MxArray, MxArray)> {
    let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_quantize(
            weight.as_raw_ptr(),
            group_size,
            bits,
            c"affine".as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    if !ok || out_q.is_null() || out_s.is_null() || out_b.is_null() {
        return None;
    }
    Some((
        MxArray::from_handle(out_q, "q").ok()?,
        MxArray::from_handle(out_s, "s").ok()?,
        MxArray::from_handle(out_b, "b").ok()?,
    ))
}

/// Fix 3 (Task 36) load-side proof: a quantized MoE-flavored MTP weight set
/// (packed `Uint32` `.weight` + `.scales`/`.biases`) loaded via
/// `apply_weights` with the AUGMENTED `per_layer_quant` (mirroring what
/// `load_with_thread` now injects from `mtplx_mtp_quantization`) installs
/// quantized backends for the router gate, switch_mlp experts, shared
/// expert + gate, and attention — proving produce (Fix 2) + reload (Fix 3)
/// agree on the uniform 4-bit/gs32 affine PLQ.
#[test]
fn quantized_moe_mtp_apply_weights_installs_quantized_sublayers() {
    use crate::models::mtp_drafter::MTP_MOE_LAYER_LINEAR_SUFFIXES;
    use crate::models::quant_dispatch::{PerLayerMode, default_per_layer_quant};
    use crate::models::qwen3_5::persistence::augment_mtplx_mtp_quantization_with_suffixes;
    use serde_json::json;

    let label = "quantized_moe_mtp_apply_weights_installs_quantized_sublayers";

    // Source bf16 params from a freshly constructed MoE-flavored MTP module.
    let Some((bf16_mtp, cfg)) = build_mtp_or_skip(label) else {
        return;
    };
    // tiny_mtp_cfg (decoder_sparse_step=1) → MoE-flavored MTP layer.
    assert!(matches!(bf16_mtp.layers[0].mlp, MLPType::MoE(_)));
    let bf16_params = bf16_mtp.get_parameters();

    // The exact set of quantizable linear prefixes for a MoE MTP layer.
    let quantizable_prefixes: Vec<String> = (0..cfg.n_mtp_layers as usize)
        .flat_map(|l| {
            MTP_MOE_LAYER_LINEAR_SUFFIXES
                .iter()
                .map(move |s| format!("mtp.layers.{l}.{s}"))
        })
        .collect();
    let quantizable: std::collections::HashSet<&str> =
        quantizable_prefixes.iter().map(String::as_str).collect();

    // Build the quantized param set: replace each quantizable `<prefix>.weight`
    // with packed weight + scales + biases; copy every other tensor verbatim.
    let mut q_params: HashMap<String, MxArray> = HashMap::new();
    for (name, arr) in &bf16_params {
        let Some(prefix) = name.strip_suffix(".weight") else {
            q_params.insert(name.clone(), arr.clone());
            continue;
        };
        if !quantizable.contains(prefix) {
            q_params.insert(name.clone(), arr.clone());
            continue;
        }
        let Some((packed, scales, biases)) = quantize_affine_or_skip(arr, 32, 4) else {
            eprintln!("skipping {label} (MLX/Metal unavailable during quantize)");
            return;
        };
        q_params.insert(format!("{prefix}.weight"), packed);
        q_params.insert(format!("{prefix}.scales"), scales);
        q_params.insert(format!("{prefix}.biases"), biases);
    }

    // Augment the per-layer-quant table exactly as `load_with_thread` does
    // for a MoE-flavored MTP layer.
    let raw = json!({
        "mtplx_mtp_quantization": {
            "prequantized": true,
            "policy": "all",
            "bits": 4,
            "group_size": 32,
            "mode": "affine"
        }
    });
    let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
    augment_mtplx_mtp_quantization_with_suffixes(
        &raw,
        cfg.n_mtp_layers,
        &MTP_MOE_LAYER_LINEAR_SUFFIXES,
        &mut per_layer_quant,
    )
    .expect("valid MTP quantization metadata");

    // Fresh module to load the quantized weights into.
    let Some((mut mtp, _cfg2)) = build_mtp_or_skip(label) else {
        return;
    };
    let default_plq = default_per_layer_quant(4, 32, PerLayerMode::Affine);
    // 8-bit affine gate default (canonical recipe), to PROVE the augmented
    // 4-bit override wins over it for the router gate (Option A).
    let default_gate_plq = PerLayerQuant {
        bits: 8,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };
    if let Err(err) = mtp.apply_weights(
        &mut q_params,
        default_plq,
        Some(default_gate_plq),
        &per_layer_quant,
        DType::BFloat16,
    ) {
        let msg = err.reason.to_string();
        if msg.contains("Metal") || msg.contains("device") {
            eprintln!("skipping {label} (MLX/Metal unavailable during apply): {msg}");
            return;
        }
        panic!("unexpected apply_weights failure in {label}: {msg}");
    }

    // Top-level aggregate: SOMETHING quantized.
    assert!(
        mtp.has_quantized_weights(),
        "MoE MTP head must report quantized weights after quantized apply"
    );

    // Per-layer granular assertions.
    for layer in &mtp.layers {
        match &layer.attn {
            AttentionType::Full(a) => assert!(
                a.is_quantized(),
                "MTP attention must be quantized (q/k/v/o_proj packed)"
            ),
            AttentionType::Linear(_) => panic!("MTP layer must be full-attention"),
        }
        match &layer.mlp {
            MLPType::MoE(moe) => {
                assert!(
                    moe.is_quantized(),
                    "MoE block must report quantized (gate/switch_mlp/shared_expert/gate)"
                );
                assert!(
                    moe.get_switch_mlp().is_quantized(),
                    "switch_mlp experts must be quantized"
                );
            }
            MLPType::Dense(_) => panic!("tiny_mtp_cfg MTP layer must be MoE-flavored"),
        }
    }
}

/// Regression for the Codex [high] dense-flavored-MoE-MTP load gate.
///
/// A MoE backbone whose MTP layer resolves to a DENSE-flavored layer
/// (here: `fa_idx ∈ mlp_only_layers`) emits dense
/// `mtp.layers.{i}.mlp.{gate,up,down}_proj.weight`, NOT the MoE
/// `switch_mlp.* + mlp.gate` schema. The completeness gate MUST derive
/// its expected variant from `Qwen3_5MTPModule::mtp_mlp_variant`
/// (`is_moe_layer(fa_idx)`); a hardcoded `Moe` variant would reject this
/// complete checkpoint and silently disable MTP. We:
///   1. confirm the chosen config really is dense-flavored
///      (`is_linear_layer(fa_idx) == false` so construction succeeds, and
///      `is_moe_layer(fa_idx) == false` so the MLP is dense);
///   2. assert `get_parameters()` is COMPLETE under the derived (Dense)
///      variant (save/load agree); and
///   3. assert a hardcoded `Moe` variant flags it incomplete — proving
///      the flavor-derivation is load-bearing, not a no-op.
#[test]
fn dense_flavored_moe_mtp_load_gate_uses_derived_variant() {
    use crate::models::mtp_drafter::{DrafterBodyVariant, missing_required_mtp_keys};

    // Start from the tiny MoE config (fa_idx = 3, full-attention) but force
    // layer 3 dense via mlp_only_layers — the canonical decoder_sparse_step=1
    // tests can't reach a dense-flavored MTP layer.
    let mut cfg = tiny_mtp_cfg();
    let fa_idx = (cfg.full_attention_interval - 1).max(0) as usize;
    cfg.mlp_only_layers = Some(vec![fa_idx as i32]);

    // The MTP layer must still be full-attention (construction precondition)
    // AND dense-flavored (the gap this test guards).
    assert!(
        !cfg.is_linear_layer(fa_idx),
        "fa_idx must be a full-attention layer so the MTP ctor succeeds"
    );
    assert!(
        !cfg.is_moe_layer(fa_idx),
        "mlp_only_layers override must make fa_idx dense-flavored"
    );
    assert_eq!(
        moe_mtp_mlp_variant(&cfg),
        DrafterBodyVariant::Dense,
        "derived MTP MLP variant must be Dense for a dense-flavored MoE-MTP layer"
    );

    let mtp = match new_moe_mtp(&cfg) {
        Ok(m) => m,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!(
                    "skipping dense_flavored_moe_mtp_load_gate_uses_derived_variant \
                     (MLX/Metal unavailable): {msg}"
                );
                return;
            }
            panic!("unexpected Qwen3_5MTPModule::new failure: {msg}");
        }
    };

    // The constructed layer must really be dense-flavored.
    assert!(
        matches!(mtp.layers[0].mlp, MLPType::Dense(_)),
        "MTP DecoderLayer must be dense-flavored when is_moe_layer(fa_idx) is false"
    );

    let params = mtp.get_parameters();

    // (1) save/load agree under the DERIVED variant (the fix).
    let derived = moe_mtp_mlp_variant(&cfg);
    let missing_derived = missing_required_mtp_keys(&params, derived, cfg.n_mtp_layers);
    assert!(
        missing_derived.is_empty(),
        "dense-flavored get_parameters() must be complete under the derived Dense variant; \
         missing: {missing_derived:?}"
    );

    // (2) CONTRAST: a hardcoded `Moe` gate WRONGLY rejects this
    // complete dense-flavored checkpoint (demands switch_mlp.* + mlp.gate
    // that a dense MLP never emits). This is what makes the derivation
    // load-bearing.
    let missing_hardcoded_moe =
        missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, cfg.n_mtp_layers);
    assert!(
        !missing_hardcoded_moe.is_empty(),
        "hardcoded-Moe gate MUST flag a dense-flavored MoE-MTP checkpoint incomplete \
         (proves the flavor-derivation is load-bearing)"
    );
    assert!(
        missing_hardcoded_moe
            .iter()
            .any(|k| k == "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"),
        "hardcoded-Moe gate must miss switch_mlp.*; got: {missing_hardcoded_moe:?}"
    );
    assert!(
        missing_hardcoded_moe
            .iter()
            .any(|k| k == "mtp.layers.0.mlp.gate.weight"),
        "hardcoded-Moe gate must miss the router mlp.gate; got: {missing_hardcoded_moe:?}"
    );
}

#[test]
fn forward_shape_matches_input() {
    // Random init (no weights loaded); only checks the forward
    // signature and output shape. Skips if MLX/Metal init fails.
    let Some((mut mtp, cfg)) = build_mtp_or_skip("forward_shape_matches_input") else {
        return;
    };

    let hidden = cfg.hidden_size as i64;
    let shape = [1i64, 1, hidden];

    let prev_hidden = match MxArray::random_normal(&shape, 0.0, 1.0, Some(DType::BFloat16)) {
        Ok(a) => a,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping forward_shape_matches_input (Metal unavailable): {msg}");
                return;
            }
            panic!("unexpected random_normal failure: {msg}");
        }
    };
    let prev_emb = MxArray::random_normal(&shape, 0.0, 1.0, Some(DType::BFloat16))
        .expect("prev_emb random_normal");

    let mut caches = Qwen3_5MTPModule::fresh_caches(&cfg.to_dense_config());
    let out = match mtp.forward(&prev_hidden, &prev_emb, Some(&mut caches)) {
        Ok(o) => o,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping forward_shape_matches_input (Metal unavailable): {msg}");
                return;
            }
            panic!("unexpected forward failure: {msg}");
        }
    };

    let out_shape = out.shape().expect("output shape");
    assert_eq!(out_shape.as_ref(), &[1i64, 1, hidden]);
}

/// Spec-compliance guard for the gate-PLQ routing fix.
///
/// `apply_weights` resolves per-projection PLQs through
/// `effective_plq_for(prefix, per_layer_quant, default_plq,
/// Some(default_gate_plq))`. For canonical recipes (`mixed_2_6`,
/// `mixed_3_4`, `qwen3_5`) the global default is 4-bit affine but
/// router gates are 8-bit affine, and MTP keys are skipped in the
/// per-layer override table — so the gate prefix MUST fall back to
/// `default_gate_plq`, not `default_plq`. A direct
/// `per_layer_quant.get(prefix).unwrap_or(default_plq)` simplification
/// would silently return 4-bit affine and load `mtp.layers.0.mlp.gate`
/// with the wrong bits/group_size.
///
/// We exercise the same indirection the closure inside
/// `apply_weights` uses, with the exact prefix the loader builds for
/// the first MTP layer's router gate.
#[test]
fn apply_weights_routes_gate_to_default_gate_plq() {
    let default_plq = PerLayerQuant {
        bits: 4,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };
    let default_gate_plq = PerLayerQuant {
        bits: 8,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };
    // Empty override table — MTP keys are never recorded here, so
    // `effective_plq_for` must take the gate-default fallback.
    let per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();

    let got_gate = effective_plq_for(
        "mtp.layers.0.mlp.gate",
        &per_layer_quant,
        default_plq,
        Some(default_gate_plq),
    );
    assert_eq!(
        got_gate, default_gate_plq,
        "mtp router gate must route to default_gate_plq (8-bit affine), not default_plq \
         (4-bit affine); regression of the W2 simplification bug"
    );
    assert_ne!(
        got_gate, default_plq,
        "must not fall back to default_plq for the gate prefix"
    );

    let got_shared_gate = effective_plq_for(
        "mtp.layers.0.mlp.shared_expert_gate",
        &per_layer_quant,
        default_plq,
        Some(default_gate_plq),
    );
    assert_eq!(
        got_shared_gate, default_gate_plq,
        "mtp shared_expert_gate must also route to default_gate_plq"
    );

    // Plain non-gate projection must still use default_plq.
    let got_qproj = effective_plq_for(
        "mtp.layers.0.self_attn.q_proj",
        &per_layer_quant,
        default_plq,
        Some(default_gate_plq),
    );
    assert_eq!(
        got_qproj, default_plq,
        "non-gate projections must use default_plq"
    );
}

/// Run `apply_weights` for an fc-install fixture. Every absent per-layer
/// tensor is skipped by its `if let Some(w)` guard, so an fc-only param
/// set never errors. Returns `false` (and prints a skip note) when
/// MLX/Metal is unavailable so the caller bails cleanly.
fn apply_fc_or_skip(
    mtp: &mut Qwen3_5MTPModule,
    params: &mut HashMap<String, MxArray>,
    default_plq: PerLayerQuant,
    default_gate_plq: Option<PerLayerQuant>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    label: &str,
) -> bool {
    match mtp.apply_weights(
        params,
        default_plq,
        default_gate_plq,
        per_layer_quant,
        DType::BFloat16,
    ) {
        Ok(()) => true,
        Err(err) => {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                false
            } else {
                panic!("unexpected apply_weights failure in {label}: {msg}");
            }
        }
    }
}

/// The MoE MTP `fc` projection must install through the mode-aware
/// `LinearProj` dispatch: a quantized fc honors its per-layer mode as
/// `LinearProj::Quantized`, and a bf16 fc (no `.scales`) stays
/// `LinearProj::Standard` — the identical dense matmul.
///
/// Regression guard for the old `self.fc.load_quantized(...)` install,
/// which hardcoded affine dequant: a non-affine (mxfp8/nvfp4) quantized
/// fc could only load as affine (crash / wrong dequant). The fabricated
/// packed tensors are never run through `forward`, so their exact shapes
/// do not matter — only the install path (mode dispatch) is asserted.
#[test]
fn mtp_fc_installs_mode_aware_linearproj() {
    use crate::models::quantized_linear::{
        DEFAULT_QUANT_MODE, LinearProj, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE,
    };
    let label = "mtp_fc_installs_mode_aware_linearproj";

    // fc: Linear(hidden*2 -> hidden); weight is [out=hidden, in=hidden*2].
    let hidden = tiny_mtp_cfg().hidden_size as i64;
    let out = hidden;
    let inp = hidden * 2;

    let u32_arr = |shape: &[i64]| {
        let n: i64 = shape.iter().product();
        MxArray::from_float32(&vec![0.0f32; n as usize], shape)
            .expect("from_float32")
            .astype(DType::Uint32)
            .expect("uint32")
    };
    let u8_arr = |shape: &[i64]| {
        let n: i64 = shape.iter().product();
        MxArray::from_float32(&vec![1.0f32; n as usize], shape)
            .expect("from_float32")
            .astype(DType::Uint8)
            .expect("uint8")
    };
    let bf16_arr = |shape: &[i64], v: f32| {
        let n: i64 = shape.iter().product();
        MxArray::from_float32(&vec![v; n as usize], shape)
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16")
    };

    // Closure fallback only; each quantized case names its mtp.fc mode
    // explicitly in `per_layer_quant`. The gate default is irrelevant to
    // the fc prefix (fc is not a router gate).
    let default_plq = PerLayerQuant {
        bits: 4,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };
    let default_gate_plq = PerLayerQuant {
        bits: 8,
        group_size: 64,
        mode: PerLayerMode::Affine,
        input_amax: None,
        layout: Default::default(),
    };

    // (a) affine quantized fc → Quantized (mode "affine").
    {
        let Some((mut mtp, _cfg)) = build_mtp_or_skip(label) else {
            return;
        };
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("mtp.fc.weight".into(), u32_arr(&[out, inp / 8]));
        params.insert("mtp.fc.scales".into(), bf16_arr(&[out, inp / 32], 1.0));
        params.insert("mtp.fc.biases".into(), bf16_arr(&[out, inp / 32], 0.0));
        let mut plq: HashMap<String, PerLayerQuant> = HashMap::new();
        plq.insert(
            "mtp.fc".into(),
            PerLayerQuant {
                bits: 4,
                group_size: 32,
                mode: PerLayerMode::Affine,
                input_amax: None,
                layout: Default::default(),
            },
        );
        if !apply_fc_or_skip(
            &mut mtp,
            &mut params,
            default_plq,
            Some(default_gate_plq),
            &plq,
            label,
        ) {
            return;
        }
        assert!(
            matches!(mtp.fc, LinearProj::Quantized(_)),
            "affine mtp.fc must install as LinearProj::Quantized"
        );
        if let LinearProj::Quantized(ref ql) = mtp.fc {
            assert_eq!(
                ql.mode(),
                DEFAULT_QUANT_MODE,
                "affine fc must keep affine mode"
            );
        }
    }

    // (b) mxfp8 quantized fc → Quantized (mode "mxfp8"). The case the old
    //     affine-only `load_quantized` could not represent.
    {
        let Some((mut mtp, _cfg)) = build_mtp_or_skip(label) else {
            return;
        };
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("mtp.fc.weight".into(), u8_arr(&[out, inp]));
        params.insert("mtp.fc.scales".into(), u8_arr(&[out, inp / 32]));
        let mut plq: HashMap<String, PerLayerQuant> = HashMap::new();
        plq.insert(
            "mtp.fc".into(),
            PerLayerQuant {
                bits: MXFP8_BITS,
                group_size: MXFP8_GROUP_SIZE,
                mode: PerLayerMode::Mxfp8,
                input_amax: None,
                layout: Default::default(),
            },
        );
        if !apply_fc_or_skip(
            &mut mtp,
            &mut params,
            default_plq,
            Some(default_gate_plq),
            &plq,
            label,
        ) {
            return;
        }
        assert!(
            matches!(mtp.fc, LinearProj::Quantized(_)),
            "mxfp8 mtp.fc must install as LinearProj::Quantized"
        );
        if let LinearProj::Quantized(ref ql) = mtp.fc {
            assert_eq!(ql.mode(), MXFP8_MODE, "mxfp8 fc must keep mxfp8 mode");
        }
    }

    // (c) bf16 fc (no `.scales`) → Standard.
    {
        let Some((mut mtp, _cfg)) = build_mtp_or_skip(label) else {
            return;
        };
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("mtp.fc.weight".into(), bf16_arr(&[out, inp], 0.01));
        if !apply_fc_or_skip(
            &mut mtp,
            &mut params,
            default_plq,
            Some(default_gate_plq),
            &HashMap::new(),
            label,
        ) {
            return;
        }
        assert!(
            matches!(mtp.fc, LinearProj::Standard(_)),
            "bf16 mtp.fc must install as LinearProj::Standard"
        );
    }
}
