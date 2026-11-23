//! Routing gate for the MoE paged native-MTP lane.

use super::qwen35_dense_media_plan;
use crate::engine::plan::{
    DecoderPlan, ExecutionPlan, MediaCapabilities, PagedAttentionPlan, SpeculativeKind,
    SpeculativePlan, TurnPath, TurnPlan, TurnRequest,
};

fn qwen35_moe_speculative_plan() -> SpeculativePlan {
    SpeculativePlan {
        kind: SpeculativeKind::NativeMtp,
        supported_input_media: MediaCapabilities::NONE,
        supported_context_media: MediaCapabilities::NONE,
        supports_paged_attention: true,
        supports_streaming: true,
    }
}

fn execution(paged: bool, has_mtp: bool) -> ExecutionPlan {
    ExecutionPlan {
        media: qwen35_dense_media_plan(true, true, paged),
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
