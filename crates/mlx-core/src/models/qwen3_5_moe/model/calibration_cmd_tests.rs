//! Reachability/construction tests for the MoE activation-amax calibration
//! command. A full model-load calibration is exercised end-to-end by the
//! dense/MoE e2e; here we lock in that the MoE command variant exists, is
//! constructible, and carries the same `{texts, calib_seq, reply}` shape the
//! NAPI dispatch (`calibrate_activation_amax_raw`) sends — so a `qwen3_5_moe`
//! checkpoint has a command to route to (the finding-1 gap was that no MoE
//! command existed at all).

use super::*;

/// The MoE `CalibratePrefillRaw` variant is constructible and matchable with
/// exactly the fields the NAPI driver sends. Proves finding-1's MoE command
/// is reachable without needing a 19GB model load.
#[test]
fn moe_calibrate_prefill_raw_cmd_constructs_and_carries_fields() {
    let (tx, _rx) = tokio::sync::oneshot::channel::<napi::Result<u32>>();
    let cmd = Qwen35MoeCmd::CalibratePrefillRaw {
        texts: vec!["hello world".to_string(), "second row".to_string()],
        calib_seq: 128,
        reply: tx,
    };
    match cmd {
        Qwen35MoeCmd::CalibratePrefillRaw {
            texts, calib_seq, ..
        } => {
            assert_eq!(texts.len(), 2);
            assert_eq!(calib_seq, 128);
        }
        _ => panic!("expected CalibratePrefillRaw variant"),
    }
}
