//! Diagnostics override a model-local plan; no selected value is portable.
use crate::engine::decode_tuning::DecodePlan;
use std::sync::OnceLock;

pub(super) fn enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("MLX_MUSE_DECODE_TUNING").as_deref() != Ok("0"))
}

pub(super) fn override_plan(mut plan: DecodePlan) -> DecodePlan {
    static EARLY: OnceLock<Option<usize>> = OnceLock::new();
    static STRIPES: OnceLock<Option<u32>> = OnceLock::new();
    if let Some(depth) = *EARLY.get_or_init(|| {
        std::env::var("MLX_MUSE_DECODE_EARLY_EVAL_LAYERS")
            .ok()?
            .parse()
            .ok()
    }) {
        plan.early_layers = depth;
    }
    if let Some(stripes) = *STRIPES.get_or_init(|| {
        let value: u32 = std::env::var("MLX_MUSE_GROUPED_STRIPES")
            .ok()?
            .parse()
            .ok()?;
        (value == 0 || ((4..=1024).contains(&value) && value.is_power_of_two())).then_some(value)
    }) {
        plan.grouped_stripes = Some(stripes);
    }
    plan
}

pub(super) fn current_plan() -> DecodePlan {
    crate::engine::decode_tuning::current_plan().unwrap_or_default()
}
