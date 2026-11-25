#[test]
fn draft_execution_policy_is_scoped_and_honors_explicit_overrides() {
    use super::DraftExecutionPolicy;
    let dense = DraftExecutionPolicy::from_values(None, None);
    let moe = dense.with_moe_profile(true);
    assert!(!dense.compile_draft() && !dense.tensor_attention());
    assert!(moe.compile_draft() && moe.tensor_attention());
    let cuda = dense.with_dense_cuda_profile(true);
    assert!(cuda.compile_draft() && !cuda.tensor_attention());
    assert!(!cuda.with_dense_cuda_profile(false).compile_draft());
    assert!(!moe.with_moe_profile(false).compile_draft());
    for (value, enabled) in [("0", false), ("1", true), ("true", false), ("", false)] {
        for supported in [false, true] {
            let policy = DraftExecutionPolicy::from_values(Some(value), Some(value))
                .with_moe_profile(supported);
            assert_eq!(policy.compile_draft(), enabled);
            assert_eq!(policy.tensor_attention(), enabled);
        }
    }
    let mixed = DraftExecutionPolicy::from_values(Some("0"), None).with_moe_profile(true);
    assert!(!mixed.compile_draft() && mixed.tensor_attention());
    let mixed = DraftExecutionPolicy::from_values(None, Some("0")).with_moe_profile(true);
    assert!(mixed.compile_draft() && !mixed.tensor_attention());
    // A second load's explicit settings do not change either earlier snapshot.
    let _other = DraftExecutionPolicy::from_values(Some("0"), Some("0"));
    assert!(moe.compile_draft() && moe.tensor_attention());
    assert!(!dense.compile_draft() && !dense.tensor_attention());
}

#[test]
fn draft_execution_defaults_require_the_measured_moe_structure() {
    let target = [2048, 40, 256, 8, 512];
    let config = DFlash2Config {
        block_size: 7,
        mask_token_id: 248077,
        target_layers: vec![1, 6, 11, 16, 22, 27, 32, 37],
        target_num_layers: 40,
        hidden_size: 2048,
        intermediate_size: 6144,
        num_hidden_layers: 6,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 248320,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 262144,
        sliding_window: 2048,
        conv_group_size: 16,
        conv_kernel_size: 2,
        selector_rank: 256,
        selector_top_k: 16,
        rope_theta: 10000000.0,
        final_logit_softcapping: None,
        selector_softcap_mode: None,
        valid_vocab_size: None,
        capture_final_norm: false,
    };
    assert!(config.supports_moe_execution_defaults(target));
    for index in 0..target.len() {
        let mut other = target;
        other[index] -= 1;
        assert!(!config.supports_moe_execution_defaults(other));
    }
    for index in 0..14 {
        let mut other = config.clone();
        match index {
            0 => other.hidden_size -= 1,
            1 => other.intermediate_size -= 1,
            2 => other.target_num_layers -= 1,
            3 => {
                other.target_layers.pop();
            }
            4 => other.num_hidden_layers -= 1,
            5 => other.num_attention_heads -= 1,
            6 => other.num_key_value_heads -= 1,
            7 => other.head_dim -= 1,
            8 => other.sliding_window -= 1,
            9 => other.block_size -= 1,
            10 => other.conv_group_size -= 1,
            11 => other.conv_kernel_size -= 1,
            12 => other.selector_rank -= 1,
            13 => other.selector_top_k -= 1,
            _ => unreachable!(),
        }
        assert!(
            !other.supports_moe_execution_defaults(target),
            "field {index}"
        );
    }
}
