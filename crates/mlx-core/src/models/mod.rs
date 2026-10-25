/**
 * Models Module
 *
 * Contains all model implementations.
 */
pub(crate) mod attention_core;
pub(crate) mod chat_napi;
pub mod clef;
pub(crate) mod forward;
pub mod gemma4;
pub mod harrier;
pub mod int8_gemm;
pub mod jina_reranker;
pub mod k2_horizon;
pub mod lfm2;
pub mod mtp_drafter;
pub mod muse_glimmer;
pub mod nemotron_h;
pub mod paddleocr_vl;
pub mod paged_config;
pub mod pp_doc_ori;
pub mod pp_doc_unwarp;
pub mod pp_doclayout_v3;
pub mod pp_text_det;
pub mod pp_text_rec;
pub mod privacy_filter;
pub mod qianfan_ocr;
pub mod quant_dispatch;
pub mod quantized_linear;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_moe;
pub mod qwen3_asr;
pub mod qwen3_tts;
pub mod qwen4_exp;
pub(crate) mod rerank_checkpoint;
pub(crate) mod rerank_runtime;
pub mod xlm_roberta_reranker;

#[cfg(test)]
mod free_released_wiring_tests {
    use crate::array::memory::TEST_FREE_RELEASED_SITES;
    use crate::engine::checkpoint_source::CheckpointSource;
    use crate::utils::gguf::NativeGgufFamily;
    use crate::utils::gguf_test_fixtures::write_import_fixture;

    fn guard_enabled_in(file: &str) -> bool {
        TEST_FREE_RELEASED_SITES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|site| site.ends_with(file))
    }

    #[tokio::test]
    async fn every_gguf_loader_enables_free_released_sources() {
        let work = std::env::temp_dir().join(format!("mlx-free-released-{}", uuid::Uuid::new_v4()));
        let cases = [
            (
                "qwen35-dense-tied",
                NativeGgufFamily::Qwen35,
                "models/qwen3_5/persistence.rs",
            ),
            (
                "qwen35moe-3d-experts",
                NativeGgufFamily::Qwen35Moe,
                "models/qwen3_5_moe/persistence.rs",
            ),
            (
                "gemma4-mmproj",
                NativeGgufFamily::Gemma4,
                "models/gemma4/persistence.rs",
            ),
            (
                "muse-draft",
                NativeGgufFamily::MuseGlimmer,
                "models/muse_glimmer/persistence.rs",
            ),
        ];
        let mut missing = Vec::new();
        for (fixture, family, site) in cases {
            let fx = write_import_fixture(fixture, &work.join(fixture));
            if family == NativeGgufFamily::Qwen35 {
                let config = work.join(fixture).join("config.json");
                let mut raw: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
                raw["intermediate_size"] = 512.into();
                std::fs::write(&config, raw.to_string()).unwrap();
            }
            let source = CheckpointSource::resolve_in(&fx.main, family, Some(&work.join("cache")))
                .await
                .unwrap();
            assert!(
                matches!(source, CheckpointSource::Gguf(_)),
                "{fixture} resolved to a plain directory"
            );
            let outcome = match family {
                NativeGgufFamily::Qwen35 => {
                    super::qwen3_5::persistence::load_with_source(&source, None, None)
                        .await
                        .map(drop)
                }
                NativeGgufFamily::Qwen35Moe => {
                    super::qwen3_5_moe::persistence::load_with_source(&source, None, None)
                        .await
                        .map(drop)
                }
                NativeGgufFamily::Gemma4 => {
                    super::gemma4::Gemma4Model::load_with_source(&source, None)
                        .await
                        .map(drop)
                }
                NativeGgufFamily::MuseGlimmer => {
                    super::muse_glimmer::persistence::load_with_source(&source)
                        .await
                        .map(drop)
                }
                NativeGgufFamily::Clef => unreachable!(),
            };
            if outcome.is_err() {
                source.release_lease();
            }
            if !guard_enabled_in(site) {
                missing.push(format!("{site} (load: {outcome:?})"));
            }
        }
        std::fs::remove_dir_all(&work).ok();
        assert!(
            missing.is_empty(),
            "GGUF loads that never enabled FreeReleasedSources: {missing:?}"
        );
    }
}
