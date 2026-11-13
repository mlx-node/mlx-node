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

/// A load with no `persistPagedCache` takes `MLX_PERSIST_PAGED_CACHE`, the
/// env every host entry writes from `--no-persist-cache` > env > its default
/// (on): on keeps a persistent tier, off a temporary one removed at close.
/// Mutation PS2 (`disk_tier` reads the option only) → red.
#[test]
fn load_takes_the_shared_persist_switch() {
    let mode = |options: &Qwen35LoadOptions, env: Option<&str>| {
        let tier = options.disk_tier_with_env("Qwen3.5", env).unwrap();
        (tier.persistent, tier.cache_disk)
    };
    let auto = crate::disk_tier::CacheDisk::Auto;
    let unset = Qwen35LoadOptions::default();
    assert_eq!(mode(&unset, Some("1")), (true, auto));
    assert_eq!(mode(&unset, Some("0")), (false, auto));
    assert_eq!(mode(&unset, None), (false, auto));
    let off = Qwen35LoadOptions {
        persist_paged_cache: Some(false),
        ..Default::default()
    };
    assert_eq!(mode(&off, Some("1")), (false, auto));
    let on = Qwen35LoadOptions {
        persist_paged_cache: Some(true),
        ..Default::default()
    };
    assert_eq!(mode(&on, Some("0")), (true, auto));
    let no_tier = Qwen35LoadOptions {
        cache_disk: Some("0".to_string()),
        ..Default::default()
    };
    assert_eq!(
        mode(&no_tier, Some("1")),
        (false, crate::disk_tier::CacheDisk::Off)
    );
    let dir = Qwen35LoadOptions {
        cache_dir: Some("/c".to_string()),
        ..Default::default()
    };
    assert_eq!(
        dir.disk_tier_with_env("Qwen3.5", Some("0"))
            .unwrap_err()
            .reason,
        "Qwen3.5 load: --cache-dir needs the persistent cache --no-persist-cache turns off"
    );
    assert!(
        dir.disk_tier_with_env("Qwen3.5", Some("1"))
            .unwrap()
            .persistent
    );
}
