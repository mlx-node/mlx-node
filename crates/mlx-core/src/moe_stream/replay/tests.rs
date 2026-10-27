use super::*;

const TRACE: &str = r#"{"kind":"qwen4_exp_expert_route_trace","schema_version":1,"mode":"ar","prompt_tokens":"2","layer_count":2,"expert_count":8,"top_k":2}
{"kind":"routes","sequence":0,"pass":0,"entry":0,"history_len":2,"source":"prefill_device_commit","layer":0,"rows":2,"top_k":2,"expert_ids":[1,2,2,3]}
{"kind":"routes","sequence":1,"pass":0,"entry":1,"history_len":2,"source":"prefill_device_commit","layer":1,"rows":2,"top_k":2,"expert_ids":[4,5,5,6]}
{"kind":"routes","sequence":2,"pass":1,"entry":0,"history_len":3,"source":"decode_device_commit","layer":0,"rows":1,"top_k":2,"expert_ids":[1,7]}
{"kind":"routes","sequence":3,"pass":1,"entry":1,"history_len":3,"source":"decode_device_commit","layer":1,"rows":1,"top_k":2,"expert_ids":[4,5]}
{"kind":"routes","sequence":4,"pass":2,"entry":0,"history_len":4,"source":"decode_device_commit","layer":0,"rows":1,"top_k":2,"expert_ids":[7,1]}
{"kind":"routes","sequence":5,"pass":2,"entry":1,"history_len":4,"source":"decode_device_commit","layer":1,"rows":1,"top_k":2,"expert_ids":[5,6]}
"#;

#[test]
fn reads_schema_1() {
    let t = read_trace(TRACE.as_bytes()).unwrap();
    assert_eq!(
        (t.layers, t.experts, t.top_k, t.prompt_tokens),
        (2, 8, 2, 2)
    );
    assert_eq!(t.steps.len(), 6);
    assert_eq!(
        t.steps[0],
        Step {
            layer: 0,
            rows: 2,
            ids: vec![1, 2, 2, 3]
        }
    );
    assert!(read_trace(TRACE.replace("\"top_k\":2}", "\"top_k\":3}").as_bytes()).is_err());
    assert!(read_trace(TRACE.replace("[4,5]", "[4,9]").as_bytes()).is_err());
    assert!(read_trace(&b"{\"kind\":\"other\"}\n"[..]).is_err());
}

/// Hand-counted: prefill stages 3 + 3 experts; with cache slots decode step
/// 1 misses all 4 (victims, staging tags unused), step 2 hits 1, 7, 5 and
/// misses 6.
#[test]
fn replay_counts_by_hand() {
    let t = read_trace(TRACE.as_bytes()).unwrap();
    let none = replay_lru(&t, 0);
    assert_eq!((none.uses, none.hits, none.decode_uses), (14, 0, 8));
    // 0 slots: decode stages too; only expert 7 is not already staged for its layer
    assert_eq!(none.reads, 6 + 1);
    let big = replay_lru(&t, 16);
    assert_eq!((big.hits, big.decode_hits, big.reads), (3, 3, 6 + 4 + 1));
    assert!((big.decode_h() - 3.0 / 8.0).abs() < 1e-12);
    // 1 slot: each decode step keeps one victim; the next step's wanted set
    // never evicts its own hit
    let one = replay_lru(&t, 1);
    assert!(one.hits <= big.hits && one.reads >= big.reads, "{one:?}");
}
