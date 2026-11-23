//! End-to-end `Qwen35Inner` + `ClairicEngine` integration unit tests for
//! [`mlx_clairic::orchestrator::extract_json`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use mlx_clairic::embedded::RenderedPdfPageData;
use mlx_clairic::errors::{CANCELLED_MESSAGE, INVALID_REQUEST_PREFIX};
use mlx_clairic::grammar::GrammarEngine;
use mlx_clairic::profile::ClairicProfile;
use serde_json::{Map, Value, json};

use crate::array::{DType, MxArray};
use crate::models::clairic::ar_engine::extract_json_on_inner;
use crate::models::clairic::head::{self, ClairicHead, ClairicHeadConfig};
use crate::models::paddleocr_vl::processing::ImageProcessorConfig;
use crate::models::qwen3_5::attention::RopeMode;
use crate::models::qwen3_5::config::{Qwen3_5Config, Qwen3_5MultimodalTokens};
use crate::models::qwen3_5::decoder_layer::AttentionType;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::tokenizer::Qwen3Tokenizer;
use crate::vision::projector::SpatialProjector;
use crate::vision::qwen::encoder::{QwenVisionConfig, QwenVisionEncoder};
use crate::vision::qwen::processing::QwenImageProcessor;

#[cfg(test)]
mod tests {
    use super::*;

    fn clef_tokenizer_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../__test__/fixtures/clef/tokenizer.json")
    }

    fn det_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = ((x >> 40) as f32) / ((1u32 << 24) as f32);
            out.push((u * 2.0 - 1.0) * scale);
        }
        out
    }

    fn det_arr(seed: u64, shape: &[i64], scale: f32) -> MxArray {
        let n: usize = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
        MxArray::from_float32(&det_vec(seed, n, scale), shape).unwrap()
    }

    fn synthetic_clairic_head(hidden_size: i64, bins: i64) -> ClairicHead {
        let cfg = ClairicHeadConfig {
            hidden_size,
            bins,
            pointer_width: 16,
            readout: "lexical".to_string(),
            route_width: 32,
            route_layers: 1,
            route_heads: 4,
            slot_rank: 8,
            estimate_lexical: true,
            span_trim: true,
            loop_feedback: false,
            answer_slots: true,
            answer_slot_routing: true,
            number_groups: true,
            number_pointer: true,
        };
        let mut weights = HashMap::new();
        let h = cfg.hidden_size;
        let w = cfg.route_width;
        let p = cfg.pointer_width;
        let r = cfg.slot_rank;
        let mut seed = 202u64;

        fn add(weights: &mut HashMap<String, MxArray>, seed: &mut u64, name: &str, shape: &[i64]) {
            let numel: usize = shape.iter().map(|&d| d as usize).product::<usize>().max(1);
            let data = if name.ends_with(".weight") && shape.len() == 1 {
                vec![1.0f32; numel]
            } else if name == "lexical_scale" {
                vec![2.0, 2.0, 1.5]
            } else {
                det_vec(*seed, numel, 0.2)
            };
            *seed += 1;
            weights.insert(
                name.to_string(),
                MxArray::from_float32(&data, shape).unwrap(),
            );
        }
        fn norm(weights: &mut HashMap<String, MxArray>, seed: &mut u64, prefix: &str, width: i64) {
            add(weights, seed, &format!("{prefix}.weight"), &[width]);
            add(weights, seed, &format!("{prefix}.bias"), &[width]);
        }
        fn linear(
            weights: &mut HashMap<String, MxArray>,
            seed: &mut u64,
            prefix: &str,
            inp: i64,
            out: i64,
            bias: bool,
        ) {
            add(weights, seed, &format!("{prefix}.weight"), &[out, inp]);
            if bias {
                add(weights, seed, &format!("{prefix}.bias"), &[out]);
            }
        }
        fn attn(weights: &mut HashMap<String, MxArray>, seed: &mut u64, prefix: &str, width: i64) {
            add(
                weights,
                seed,
                &format!("{prefix}.in_proj_weight"),
                &[3 * width, width],
            );
            add(
                weights,
                seed,
                &format!("{prefix}.in_proj_bias"),
                &[3 * width],
            );
            linear(
                weights,
                seed,
                &format!("{prefix}.out_proj"),
                width,
                width,
                true,
            );
        }

        add(&mut weights, &mut seed, "slot_route_scale", &[8]);
        norm(&mut weights, &mut seed, "norm", h);
        norm(&mut weights, &mut seed, "token_norm", h);
        linear(&mut weights, &mut seed, "null", h, 1, true);
        linear(&mut weights, &mut seed, "span_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "span_key", h, 2 * p, true);
        add(&mut weights, &mut seed, "length_bias", &[head::MAX_SPAN]);
        linear(&mut weights, &mut seed, "tag_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "tag_key", h, p, true);
        add(&mut weights, &mut seed, "tag_bias", &[2]);
        linear(
            &mut weights,
            &mut seed,
            "trim",
            h,
            2 * head::TRIM_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "route_in", h, w, false);
        add(
            &mut weights,
            &mut seed,
            "kind_embedding.weight",
            &[head::KIND_EMBEDDING_ROWS, w],
        );
        linear(&mut weights, &mut seed, "memory_in", h, w, false);
        for n in ["norm1", "norm2", "norm3"] {
            norm(&mut weights, &mut seed, &format!("route_layers.0.{n}"), w);
        }
        attn(&mut weights, &mut seed, "route_layers.0.self_attn", w);
        attn(&mut weights, &mut seed, "route_layers.0.multihead_attn", w);
        linear(
            &mut weights,
            &mut seed,
            "route_layers.0.linear1",
            w,
            4 * w,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "route_layers.0.linear2",
            4 * w,
            w,
            true,
        );
        norm(&mut weights, &mut seed, "route_norm", w);
        linear(&mut weights, &mut seed, "route_out", w, h, true);
        add(&mut weights, &mut seed, "route_gate", &[]);
        add(&mut weights, &mut seed, "digit_rows", &[10, h]);
        add(&mut weights, &mut seed, "month_rows", &[12, h]);
        add(&mut weights, &mut seed, "sign_rows", &[2, h]);
        add(&mut weights, &mut seed, "lexical_scale", &[3]);
        linear(
            &mut weights,
            &mut seed,
            "slot_down",
            h,
            (head::DIGIT_SLOTS + 2) * r,
            false,
        );
        add(
            &mut weights,
            &mut seed,
            "slot_up",
            &[head::DIGIT_SLOTS, r, 10],
        );
        add(&mut weights, &mut seed, "month_up", &[r, 12]);
        add(&mut weights, &mut seed, "sign_up", &[r, 2]);
        linear(
            &mut weights,
            &mut seed,
            "slot_residual",
            w,
            head::DIGIT_SLOTS * 10 + 12 + 2,
            true,
        );
        add(&mut weights, &mut seed, "residual_gate", &[]);
        linear(
            &mut weights,
            &mut seed,
            "exponent",
            h,
            head::EXPONENT_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "item_end", h, 1, true);
        linear(&mut weights, &mut seed, "bins", h, cfg.bins, true);
        linear(&mut weights, &mut seed, "offset", h, cfg.bins, true);
        linear(&mut weights, &mut seed, "overflow", h, 2, true);
        linear(&mut weights, &mut seed, "estimate_gate", h, 1, true);
        linear(
            &mut weights,
            &mut seed,
            "estimate_spread",
            h,
            1 + head::LEXICAL_DIGITS,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "number_group_down",
            h,
            head::GROUP_RANK,
            true,
        );
        linear(
            &mut weights,
            &mut seed,
            "number_group_up",
            head::GROUP_RANK,
            head::NUMBER_GROUPS * head::GROUP_CLASSES,
            true,
        );
        linear(&mut weights, &mut seed, "copy_query", h, 2 * p, true);
        linear(&mut weights, &mut seed, "copy_gate", h, 1, true);

        ClairicHead::from_weights(cfg, weights, hidden_size, false).expect("build synthetic head")
    }

    fn synthetic_vision(text_hidden: i32) -> (QwenImageProcessor, QwenVisionEncoder) {
        let proc_cfg = ImageProcessorConfig {
            min_pixels: 32 * 32,
            max_pixels: 64 * 64,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
            do_rescale: true,
            do_normalize: true,
        };
        let processor = QwenImageProcessor::new(Some(proc_cfg));

        let vdim = 32i32;
        let merge = 2i32;
        let vcfg = QwenVisionConfig {
            hidden_size: vdim,
            intermediate_size: 64,
            num_heads: 2,
            num_layers: 0,
            patch_size: 16,
            spatial_merge_size: merge,
            image_size: 64,
            out_hidden_size: text_hidden,
        };
        let mut encoder = QwenVisionEncoder::new(vcfg).expect("QwenVisionEncoder::new");
        let patch_w = det_arr(901, &[vdim as i64, 16, 16, 3], 0.05);
        encoder.set_patch_embed(&patch_w, None).unwrap();
        let merged_in = (vdim * merge * merge) as i64;
        let merger = SpatialProjector::new(
            merge as u32,
            &MxArray::ones(&[vdim as i64], Some(DType::Float32)).unwrap(),
            &MxArray::zeros(&[vdim as i64], Some(DType::Float32)).unwrap(),
            &det_arr(902, &[64, merged_in], 0.05),
            &MxArray::zeros(&[64], Some(DType::Float32)).unwrap(),
            &det_arr(903, &[text_hidden as i64, 64], 0.05),
            &MxArray::zeros(&[text_hidden as i64], Some(DType::Float32)).unwrap(),
        )
        .expect("SpatialProjector::new");
        encoder.set_merger(merger);
        (processor, encoder)
    }

    fn synthetic_clairic_inner(with_vision: bool) -> Qwen35Inner {
        let tok_path = clef_tokenizer_path();
        let tokenizer = Arc::new(
            Qwen3Tokenizer::load_from_file_sync(tok_path.to_str().unwrap())
                .expect("load clef fixture tokenizer"),
        );
        let vocab_size = tokenizer.vocab_size() as i32;
        let eos_id = tokenizer
            .token_to_id("<|im_end|>".to_string())
            .filter(|&id| (id as i32) < vocab_size)
            .unwrap_or_else(|| {
                let candidate = tokenizer.get_eos_token_id();
                if (candidate as i32) < vocab_size {
                    candidate
                } else {
                    (vocab_size - 1).max(0) as u32
                }
            });
        let pad_id = {
            let candidate = tokenizer.get_pad_token_id();
            if (candidate as i32) < vocab_size {
                candidate
            } else {
                0
            }
        };
        let vision_end_id: i32 = 7.min(vocab_size - 1);
        let image_token_id: i32 = 8.min(vocab_size - 1);

        let config = Qwen3_5Config {
            vocab_size,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 2048,
            pad_token_id: pad_id as i32,
            eos_token_id: eos_id as i32,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            n_mtp_layers: 0,
            qwen35_gguf_gdn_layout: None,
            kv_format: Some("bf16".to_string()),
            rope_scaling: None,
        };

        let mut inner = Qwen35Inner::new(config.clone()).expect("Qwen35Inner::new");
        let h = config.hidden_size as i64;
        let v = config.vocab_size as i64;
        inner
            .embedding
            .set_weight(&det_arr(11, &[v, h], 0.15))
            .unwrap();
        inner
            .final_norm
            .set_weight(&MxArray::ones(&[h], Some(DType::Float32)).unwrap())
            .unwrap();

        let key_dim = config.linear_key_dim() as i64;
        let val_dim = config.linear_value_dim() as i64;
        let conv_dim = config.linear_conv_dim() as i64;
        let hv = config.linear_num_value_heads as i64;
        let nh = config.num_heads as i64;
        let nkv = config.num_kv_heads as i64;
        let hd = config.head_dim as i64;
        let inter = config.intermediate_size as i64;

        for (i, layer) in inner.layers.iter_mut().enumerate() {
            let s = (i as u64 + 1) * 31;
            match &mut layer.attn {
                AttentionType::Linear(gdn) => {
                    gdn.set_in_proj_qkvz_weight(&det_arr(
                        s + 1,
                        &[2 * key_dim + 2 * val_dim, h],
                        0.1,
                    ))
                    .unwrap();
                    gdn.set_in_proj_ba_weight(&det_arr(s + 2, &[2 * hv, h], 0.1))
                        .unwrap();
                    gdn.set_conv1d_weight(&det_arr(s + 3, &[conv_dim, 4, 1], 0.15), DType::Float32)
                        .unwrap();
                    gdn.set_out_proj_weight(&det_arr(s + 4, &[h, val_dim], 0.1))
                        .unwrap();
                }
                AttentionType::Full(attn) => {
                    attn.set_q_proj_weight(&det_arr(s + 5, &[2 * nh * hd, h], 0.1))
                        .unwrap();
                    attn.set_k_proj_weight(&det_arr(s + 6, &[nkv * hd, h], 0.1))
                        .unwrap();
                    attn.set_v_proj_weight(&det_arr(s + 7, &[nkv * hd, h], 0.1))
                        .unwrap();
                    attn.set_o_proj_weight(&det_arr(s + 8, &[h, nh * hd], 0.1))
                        .unwrap();
                }
            }
            if let crate::models::quantized_linear::MLPVariant::Standard(mlp) = &mut layer.mlp {
                mlp.set_gate_proj_weight(&det_arr(s + 9, &[inter, h], 0.08))
                    .unwrap();
                mlp.set_up_proj_weight(&det_arr(s + 10, &[inter, h], 0.08))
                    .unwrap();
                mlp.set_down_proj_weight(&det_arr(s + 11, &[h, inter], 0.08))
                    .unwrap();
            }
        }

        inner
            .init_mrope_layers(
                vec![2, 1, 1],
                config.rope_theta,
                config.max_position_embeddings,
            )
            .unwrap();
        inner.set_rope_mode_layers(RopeMode::Temporal);
        inner.clairic_multimodal = Qwen3_5MultimodalTokens {
            image_token_id,
            vision_end_token_id: vision_end_id,
            mrope_section: vec![2, 1, 1],
        };
        inner.spatial_merge_size = Some(2);

        if with_vision {
            let (proc, enc) = synthetic_vision(config.hidden_size);
            inner.image_processor = Some(Arc::new(proc));
            inner.vision_encoder = Some(Arc::new(enc));
        }

        let grammar_engine = GrammarEngine::new(
            tokenizer.inner_arc(),
            vocab_size as usize,
            eos_id,
            vec![eos_id],
        )
        .expect("build GrammarEngine");
        let head = synthetic_clairic_head(h, 256);
        inner.clairic_profile = Some(ClairicProfile::Flash);
        inner.clairic_head = Some(head);
        inner.clairic_grammar = Some(Arc::new(grammar_engine));
        inner.set_tokenizer(tokenizer);
        inner
    }

    #[test]
    fn end_to_end_dual_path_text_extraction_with_quantiles() {
        let mut inner = synthetic_clairic_inner(false);
        let req = json!({
            "model": "clairic-flash",
            "state": "Acme Corp Invoice #1042 dated 2026-03-15. Subtotal $100.00, Tax $8.50. Paid: true. Status: open. Tags: alpha, beta.",
            "reference_date": "2026-03-20",
            "options": {
                "include_quantiles": true,
                "return_state_text": true
            },
            "fields": {
                "vendor": { "type": "text", "instructions": "Merchant name." },
                "subtotal": { "type": "number", "decimals": 2, "instructions": "Subtotal amount." },
                "paid": { "type": "boolean", "instructions": "Whether the invoice is paid." },
                "status": { "type": "enum", "enum": ["open", "closed"], "instructions": "Invoice status." },
                "tags": { "type": "list", "items": "text", "max_items": 2, "instructions": "Listed tags." },
                "grand_total": { "type": "compute", "decimals": 2, "instructions": "Subtotal plus tax." },
                "score": { "type": "estimate", "min": 0, "max": 100, "scale": "linear", "instructions": "Reliability score." },
                "revenue_fc": { "type": "forecast", "min": 10, "max": 1000, "scale": "log", "horizon": 2, "interval": "month", "instructions": "Monthly revenue." },
                "derived_due": { "type": "date", "instructions": "Due date 30 days after invoice date." }
            }
        });

        let raw_out = extract_json_on_inner(&mut inner, &req.to_string(), Vec::new(), None)
            .expect("extract_json_on_inner succeeded");
        let resp: Value = serde_json::from_str(&raw_out).expect("valid response JSON");

        assert_eq!(resp["model"], "clairic-flash");
        assert!(resp["truncated"].is_boolean());
        assert_eq!(
            resp["state_text"].as_str().unwrap(),
            req["state"].as_str().unwrap()
        );

        let usage = resp["usage"].as_object().unwrap();
        assert!(usage["state_tokens"].as_u64().unwrap() > 0);
        assert_eq!(usage["image_tokens"].as_u64().unwrap(), 0);
        assert!(usage["input_tokens"].as_u64().unwrap() > 0);
        assert!(usage["passes"].as_u64().unwrap() >= 1);

        let answers = resp["answers"].as_object().unwrap();
        assert_eq!(answers.len(), 9);
        for (field, expected_type) in [
            ("vendor", "text"),
            ("subtotal", "number"),
            ("paid", "boolean"),
            ("status", "enum"),
            ("tags", "list"),
            ("grand_total", "compute"),
            ("score", "estimate"),
            ("revenue_fc", "forecast"),
            ("derived_due", "date"),
        ] {
            let ans = answers[field].as_object().unwrap();
            assert_eq!(ans["type"], expected_type, "type mismatch on {field}");
            if matches!(field, "score" | "revenue_fc") {
                assert!(ans.contains_key("median"), "missing median on {field}");
                assert!(ans.contains_key("p10"), "missing p10 on {field}");
                assert!(ans.contains_key("p90"), "missing p90 on {field}");
            } else {
                assert!(
                    ans.contains_key("confidence") && ans.contains_key("p_null"),
                    "missing confidence/p_null on {field}: {ans:?}"
                );
            }
        }
        assert!(answers["score"].get("quantiles").is_some());
        assert_eq!(answers["revenue_fc"]["steps"].as_array().unwrap().len(), 2);
        assert!(answers["revenue_fc"]["steps"][0].get("quantiles").is_some());
    }

    #[test]
    fn end_to_end_rendered_pdf_page_vision_extraction() {
        let mut inner = synthetic_clairic_inner(true);
        let page = RenderedPdfPageData {
            rgb: vec![240u8; 32 * 32 * 3],
            width: 32,
            height: 32,
            document_index: 0,
            page_index: 0,
        };
        let req = json!({
            "state": "Page 1 caption: Invoice total $42.00.",
            "fields": {
                "is_invoice": { "type": "boolean", "instructions": "Is this an invoice?" },
                "risk": { "type": "estimate", "min": 0, "max": 1, "scale": "linear", "instructions": "Risk score." }
            }
        });

        let raw_out = extract_json_on_inner(&mut inner, &req.to_string(), vec![page], None)
            .expect("PDF vision extract_json_on_inner succeeded");
        let resp: Value = serde_json::from_str(&raw_out).expect("valid response JSON");

        assert_eq!(resp["model"], "clairic-flash");
        assert!(
            resp["usage"]["image_tokens"].as_u64().unwrap() > 0,
            "expected positive image_tokens for rendered PDF page"
        );
        let answers = resp["answers"].as_object().unwrap();
        assert_eq!(answers["is_invoice"]["type"], "boolean");
        assert_eq!(answers["risk"]["type"], "estimate");
        // Quantiles omitted by default (`include_quantiles: false`)
        assert!(answers["risk"].get("quantiles").is_none());
    }

    #[test]
    fn rejects_invalid_requests_with_prefix() {
        let mut inner = synthetic_clairic_inner(false);

        // 1. Unknown top-level key
        let bad_key = json!({
            "state": "hello",
            "fields": { "a": { "type": "text" } },
            "bogus": 123
        });
        let err =
            extract_json_on_inner(&mut inner, &bad_key.to_string(), Vec::new(), None).unwrap_err();
        assert!(
            err.reason.starts_with(INVALID_REQUEST_PREFIX),
            "got: {}",
            err.reason
        );

        // 2. Bad field type
        let bad_type = json!({
            "state": "hello",
            "fields": { "a": { "type": "imaginary_type" } }
        });
        let err =
            extract_json_on_inner(&mut inner, &bad_type.to_string(), Vec::new(), None).unwrap_err();
        assert!(
            err.reason.starts_with(INVALID_REQUEST_PREFIX),
            "got: {}",
            err.reason
        );

        // 3. Missing fields
        let missing_fields = json!({ "state": "hello" });
        let err = extract_json_on_inner(&mut inner, &missing_fields.to_string(), Vec::new(), None)
            .unwrap_err();
        assert!(
            err.reason.starts_with(INVALID_REQUEST_PREFIX),
            "got: {}",
            err.reason
        );

        // 4. >4 images
        let too_many_images = json!({
            "state": "hello",
            "images": ["a", "b", "c", "d", "e"],
            "fields": { "a": { "type": "text" } }
        });
        let err = extract_json_on_inner(&mut inner, &too_many_images.to_string(), Vec::new(), None)
            .unwrap_err();
        assert!(
            err.reason.starts_with(INVALID_REQUEST_PREFIX),
            "got: {}",
            err.reason
        );

        // 5. Empty state with no media
        let empty_state = json!({
            "state": "   ",
            "fields": { "a": { "type": "text" } }
        });
        let err = extract_json_on_inner(&mut inner, &empty_state.to_string(), Vec::new(), None)
            .unwrap_err();
        assert!(
            err.reason.starts_with(INVALID_REQUEST_PREFIX),
            "got: {}",
            err.reason
        );
    }

    #[test]
    fn cancellation_flag_aborts_cleanly() {
        let mut inner = synthetic_clairic_inner(false);
        let flag = Arc::new(AtomicBool::new(true));
        flag.store(true, Ordering::Release);

        let req = json!({
            "state": "hello world",
            "fields": { "a": { "type": "boolean" } }
        });
        let err = extract_json_on_inner(&mut inner, &req.to_string(), Vec::new(), Some(&flag))
            .unwrap_err();
        assert_eq!(err.reason, CANCELLED_MESSAGE);
    }

    #[test]
    fn accepts_renamed_checkpoint_model_ids_and_rejects_blank_or_non_string() {
        let mut inner = synthetic_clairic_inner(false);

        // Renamed local checkpoint ID (e.g., advertised by `/v1/models`) must be accepted.
        let renamed = json!({
            "model": "clairic-flash-4bit",
            "state": "Paid: true.",
            "fields": {
                "paid": { "type": "boolean", "instructions": "Whether paid." }
            }
        });
        let raw_out = extract_json_on_inner(&mut inner, &renamed.to_string(), Vec::new(), None)
            .expect("renamed checkpoint model ID must be accepted");
        let resp: Value = serde_json::from_str(&raw_out).expect("valid JSON");
        assert_eq!(resp["model"], "clairic-flash");

        // Blank string model ID must be rejected with invalid_request.
        for bad_model in [json!(""), json!("   "), json!(123), json!(true), json!([])] {
            let req = json!({
                "model": bad_model,
                "state": "Paid: true.",
                "fields": { "paid": { "type": "boolean" } }
            });
            let err =
                extract_json_on_inner(&mut inner, &req.to_string(), Vec::new(), None).unwrap_err();
            assert!(
                err.reason.starts_with(INVALID_REQUEST_PREFIX),
                "expected invalid request for model={bad_model}, got: {}",
                err.reason
            );
        }
    }

    #[test]
    fn rejects_excessive_list_expansion_without_oom() {
        let mut inner = synthetic_clairic_inner(false);

        // 1. Huge explicit `max_items` values must fail cleanly without allocating.
        for huge in [1_000_000_000u64, u32::MAX as u64] {
            let req = json!({
                "state": "a, b, c",
                "fields": {
                    "items": { "type": "list", "items": "text", "max_items": huge }
                }
            });
            let err =
                extract_json_on_inner(&mut inner, &req.to_string(), Vec::new(), None).unwrap_err();
            assert!(
                err.reason.starts_with(INVALID_REQUEST_PREFIX),
                "expected invalid request for max_items={huge}, got: {}",
                err.reason
            );
        }

        // 2. Multi-list expansion exceeding MAX_TOTAL_EXPANDED_TASKS (5 lists * 501 = 2505 > 2048).
        let mut multi_fields = Map::new();
        for i in 0..5 {
            multi_fields.insert(
                format!("list_{i}"),
                json!({ "type": "list", "items": "text", "max_items": 500 }),
            );
        }
        let multi_req = json!({
            "state": "alpha beta",
            "fields": Value::Object(multi_fields),
        });
        let err = extract_json_on_inner(&mut inner, &multi_req.to_string(), Vec::new(), None)
            .unwrap_err();
        assert!(
            err.reason.starts_with(INVALID_REQUEST_PREFIX),
            "expected invalid request for >MAX_TOTAL_EXPANDED_TASKS, got: {}",
            err.reason
        );
    }
}
