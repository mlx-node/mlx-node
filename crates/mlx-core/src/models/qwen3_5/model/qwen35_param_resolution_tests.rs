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
    assert_eq!(explicit.mtp_depth, 2);
}
