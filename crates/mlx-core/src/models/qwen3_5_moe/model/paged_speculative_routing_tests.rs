//! Routing gate for the MoE paged native-MTP lane.
//!
//! Composes the plan this family actually publishes
//! (`qwen35_moe_speculative_plan`) with the engine's resolver, so the gate
//! fails if either half of the flip is missing: the published
//! `supports_paged_attention` flag, or the paged core the generic driver
//! dispatches into.
//!
//! The core half is not asserted here — it lives with the driver
//! (`engine::paged_turn`'s
//! `a_speculative_plan_runs_the_family_core_and_the_shared_epilogue` and
//! `publishing_the_flag_without_a_core_fails_the_turn_closed`). What THIS
//! module pins is that MoE reaches that dispatch at all, and that the flat
//! lane still exists for adapter-less checkpoints.

use super::{qwen35_moe_media_plan, qwen35_moe_speculative_plan};
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, PagedAttentionPlan, SpeculativeKind, TurnPath,
    TurnPlan, TurnRequest,
};

/// The execution plan a fully loaded vision-capable MoE checkpoint
/// publishes, assembled from the same two helpers `execution_plan` uses.
fn execution(paged: bool, has_mtp: bool) -> ExecutionPlan {
    ExecutionPlan {
        media: qwen35_moe_media_plan(true, true, paged),
        paged_attention: paged.then_some(PagedAttentionPlan {
            supports_delta: true,
        }),
        speculative: has_mtp.then(qwen35_moe_speculative_plan),
    }
}

fn text_mtp_request(is_delta: bool) -> TurnRequest {
    TurnRequest {
        is_delta,
        input_media: MediaCapabilities::NONE,
        context_media: MediaCapabilities::NONE,
        speculative_requested: true,
        streaming: false,
    }
}

/// A paged text turn that asked for MTP resolves to NATIVE MTP on the
/// PAGED handler, never a silent autoregressive downgrade.
///
/// Catches, verified by mutation: setting `supports_paged_attention: false`
/// back in `qwen35_moe_speculative_plan` — the decoder then resolves to
/// `Autoregressive` and the turn decodes with the MTP head loaded but
/// never engaged.
#[test]
fn a_paged_text_turn_that_requested_mtp_plans_native_mtp_not_a_downgrade() {
    for is_delta in [false, true] {
        let plan = TurnPlan::resolve(execution(true, true), text_mtp_request(is_delta));
        assert!(plan.use_paged_attention, "is_delta={is_delta}");
        assert_eq!(
            plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
            "a paged MoE text turn must keep native MTP (is_delta={is_delta})"
        );
        assert_eq!(
            plan.path(),
            TurnPath::Paged,
            "paged speculation runs on the paged handler (is_delta={is_delta})"
        );
    }
}

/// The FLAT lane survives the flip: an adapter-less MoE checkpoint (sym8,
/// or `use_block_paged_cache: false`) still plans native MTP and still
/// routes to the flat speculative handler.
///
/// Catches, verified by mutation: deleting the flat MTP arm's plan by
/// gating `speculative` on the adapter — the flat checkpoint then plans
/// `Autoregressive` and loses MTP entirely.
#[test]
fn an_adapter_less_checkpoint_still_plans_flat_native_mtp() {
    let plan = TurnPlan::resolve(execution(false, true), text_mtp_request(false));
    assert!(!plan.use_paged_attention);
    assert_eq!(
        plan.decoder,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    );
    assert_eq!(plan.path(), TurnPath::Speculative);
}

/// An image-bearing turn keeps the autoregressive decoder on both lanes:
/// there is no MoE hidden-emitting prefill to seed a drafter from image
/// features.
#[test]
fn an_image_turn_never_plans_speculative_decode() {
    let request = TurnRequest {
        is_delta: false,
        input_media: MediaCapabilities::IMAGES,
        context_media: MediaCapabilities::NONE,
        speculative_requested: true,
        streaming: false,
    };
    for paged in [false, true] {
        let plan = TurnPlan::resolve(execution(paged, true), request);
        assert_eq!(plan.decoder, DecoderPlan::Autoregressive, "paged={paged}");
        assert_eq!(plan.path(), TurnPath::Multimodal, "paged={paged}");
    }
}
