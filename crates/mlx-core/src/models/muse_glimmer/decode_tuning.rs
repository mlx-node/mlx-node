//! Diagnostics override a model-local plan; no selected value is portable.
use crate::engine::decode_tuning::DecodePlan;
use std::sync::OnceLock;

/// Apply the process-local diagnostic overrides to a single-row decode plan.
/// A forced partition count (`MLX_MUSE_GROUPED_STRIPES`) is bounded by
/// `max_stripes`, the kernel's live cap (kernel limit, work tiles, temporary
/// storage, buffer length), exactly like the device rule it replaces.
pub(super) fn override_plan(mut plan: DecodePlan, max_stripes: u32) -> DecodePlan {
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
        plan.grouped_stripes = Some(capped_stripes(stripes, max_stripes));
    }
    plan
}

/// A requested partition count under the live cap: never above
/// `max_stripes` (a power of two, so the result stays one), and generic V2
/// (0) when the cap admits no grouped partition.
fn capped_stripes(stripes: u32, max_stripes: u32) -> u32 {
    if max_stripes < 4 {
        0
    } else {
        stripes.min(max_stripes)
    }
}

pub(super) fn current_plan() -> DecodePlan {
    crate::engine::decode_tuning::current_plan().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_stripes_respect_the_live_cap() {
        // (forced, cap) -> launched
        for (forced, cap, expected) in [
            (1_024, 32, 32),
            (1_024, 1_024, 1_024),
            (16, 1_024, 16),
            (4, 4, 4),
            (64, 0, 0),
            (64, 2, 0),
            (0, 1_024, 0),
        ] {
            assert_eq!(
                capped_stripes(forced, cap),
                expected,
                "forced={forced} cap={cap}"
            );
        }
    }
}
