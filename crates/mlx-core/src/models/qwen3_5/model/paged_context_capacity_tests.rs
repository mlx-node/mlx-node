use super::*;
use crate::engine::types::ChatConfig;

fn make_params(max_new_tokens: i32) -> engine::ChatParams {
    extract_chat_params(&ChatConfig {
        cache_salt: None,
        cache_owner_id: None,
        cache_root_owner_id: None,
        max_new_tokens: Some(max_new_tokens),
        ..ChatConfig::default()
    })
}

#[test]
fn rejects_a_prompt_larger_than_the_physical_pool() {
    let mut params = make_params(32);
    let err = constrain_paged_context_params("test", 65, 64, &mut params).unwrap_err();
    assert!(
        err.reason
            .starts_with("context_length_exceeded: rendered prompt has 65 tokens")
    );
}

#[test]
fn clamps_output_with_last_sampled_token_accounting() {
    let mut params = make_params(100);
    constrain_paged_context_params("test", 48, 64, &mut params).unwrap();
    assert_eq!(params.max_new_tokens, 17);

    let mut full_prompt = make_params(100);
    constrain_paged_context_params("test", 64, 64, &mut full_prompt).unwrap();
    assert_eq!(full_prompt.max_new_tokens, 1);
}

#[test]
fn preserves_an_already_safe_output_budget() {
    let mut params = make_params(10);
    constrain_paged_context_params("test", 48, 64, &mut params).unwrap();
    assert_eq!(params.max_new_tokens, 10);
}

#[test]
fn scheduler_usable_window_is_stricter_than_trained_min_pool_when_recurrent_is_nonzero() {
    use crate::engine::hybrid_scheduler::{pool_tokens_after_recurrent, scheduled_turn_context};
    let trained = 1_048_576;
    let pool = 1_048_576;
    let rec = 48 * 1024 * 1024;
    let usable = pool_tokens_after_recurrent(pool, 16, 1024, rec);
    let old = trained.min(pool);
    let effective = scheduled_turn_context(trained, usable, None);
    assert!(rec > 0);
    assert!(usable < pool);
    assert!(effective < old);
    assert_eq!(effective, usable);
    assert_eq!(
        scheduled_turn_context(trained, usable, Some(32_768)),
        32_768
    );
}
