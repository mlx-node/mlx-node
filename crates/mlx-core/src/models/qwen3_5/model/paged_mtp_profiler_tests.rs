use super::*;

#[test]
fn paged_mtp_profiler_is_gated_away_from_ar() {
    assert!(begin_paged_mtp_profiler(false, 37).is_none());
    assert!(begin_paged_mtp_profiler(true, 37).is_some());
}

#[test]
fn paged_mtp_profiler_starts_with_fresh_suffix_metadata() {
    let mut profiler = crate::decode_profiler::DecodeProfiler::new("paged_mtp_test", "qwen3_5");
    profiler.enable_for_test();

    let mut profiler = configure_paged_mtp_profiler(profiler, 37);
    let (prompt_tokens, prefill_active, memory_before, prefill_ms) =
        profiler.turn_start_state_for_test();
    assert_eq!(prompt_tokens, 37);
    assert!(prefill_active);
    assert!(memory_before);
    assert_eq!(prefill_ms, 0.0);

    profiler.end_prefill();
    let (_, prefill_active, _, prefill_ms) = profiler.turn_start_state_for_test();
    assert!(!prefill_active);
    assert!(prefill_ms.is_finite());
}
