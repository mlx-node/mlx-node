use super::*;

#[test]
fn dflash_override_preserves_generation_defaults_and_request_precedence() {
    let defaults = crate::engine::ModelGenerationDefaults {
        temperature: Some(0.35),
        top_k: Some(17),
        top_p: Some(0.82),
        min_p: Some(0.04),
        repetition_penalty: Some(1.08),
        ..crate::engine::ModelGenerationDefaults::default()
    };
    let ordinary = resolve_qwen35_chat_params(&ChatConfig::default(), &defaults, None);
    let sampling = ordinary.sampling_config.expect("sampling config");
    assert_eq!(sampling.temperature, Some(0.35));
    assert_eq!(sampling.top_k, Some(17));
    assert_eq!(ordinary.repetition_penalty, 1.08);
    assert_eq!(ordinary.mtp_depth, 1);

    let params = resolve_qwen35_chat_params(&ChatConfig::default(), &defaults, Some(8));
    let sampling = params.sampling_config.expect("sampling config");
    assert_eq!(sampling.temperature, Some(0.35));
    assert_eq!(sampling.top_k, Some(17));
    assert_eq!(sampling.top_p, Some(0.82));
    assert_eq!(sampling.min_p, Some(0.04));
    assert_eq!(params.repetition_penalty, 1.08);
    assert_eq!(params.mtp_depth, 8);

    let explicit = resolve_qwen35_chat_params(
        &ChatConfig {
            temperature: Some(0.0),
            top_k: Some(3),
            repetition_penalty: Some(1.0),
            mtp_depth: Some(2),
            ..ChatConfig::default()
        },
        &defaults,
        Some(8),
    );
    let sampling = explicit.sampling_config.expect("sampling config");
    assert_eq!(sampling.temperature, Some(0.0));
    assert_eq!(sampling.top_k, Some(3));
    assert_eq!(explicit.repetition_penalty, 1.0);
    assert_eq!(explicit.mtp_depth, 8);

    // A native-MTP request still owns its chain depth; a block draft always
    // uses its checkpoint width, including when callers supplied the former
    // depth-three experiment or a value above the checkpoint width.
    for depth in [0, 3, 99] {
        let config = ChatConfig {
            mtp_depth: Some(depth),
            ..ChatConfig::default()
        };
        let draft = resolve_qwen35_chat_params(&config, &defaults, Some(7));
        assert_eq!(draft.mtp_depth, 7);
    }
    let native = resolve_qwen35_chat_params(
        &ChatConfig {
            mtp_depth: Some(3),
            ..ChatConfig::default()
        },
        &defaults,
        None,
    );
    assert_eq!(native.mtp_depth, 3);
}

#[test]
fn dflash_disables_adaptive_fallback_without_changing_native_mtp() {
    let defaults = crate::engine::ModelGenerationDefaults::default();
    for adaptive in [None, Some(false), Some(true)] {
        let config = ChatConfig {
            mtp_adaptive_depth: adaptive,
            ..ChatConfig::default()
        };
        let draft = resolve_qwen35_chat_params(&config, &defaults, Some(8));
        assert!(!draft.mtp_adaptive_depth);
        assert_eq!(draft.mtp_depth, 8);
        let native = resolve_qwen35_chat_params(&config, &defaults, None);
        assert_eq!(native.mtp_adaptive_depth, adaptive.unwrap_or(false));
    }
}
