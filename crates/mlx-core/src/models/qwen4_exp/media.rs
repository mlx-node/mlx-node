//! Lazily loaded Qwen3-VL vision tower, shared with Qwen3.5, with a bounded
//! image admission policy and Qwen4's interleaved three-axis RoPE.
use super::{Inner, config::Config, weights::Store};
use crate::array::MxArray;
use crate::engine::backend::{PagedBackend, TurnOutput, WholeTurnArgs};
use crate::vision::qwen::prompt::{
    IMAGE_TOKEN_ID, compute_image_token_counts_per_image, get_rope_index, inject_image_placeholders,
};
use crate::vision::qwen::{
    encoder::{QwenVisionConfig, QwenVisionEncoder},
    processing::QwenImageProcessor,
};
use napi::{Error, Result};
use std::collections::HashMap;

pub(super) const IMAGE_LIMITS: crate::vision::qwen::prompt::ImageLimits =
    crate::vision::qwen::prompt::ImageLimits {
        max_images: 4,
        max_pixels: 16_777_216,
        max_encoded_bytes: 32 << 20,
    };

pub(super) fn image_processor() -> QwenImageProcessor {
    QwenImageProcessor::new(Some(
        crate::models::paddleocr_vl::processing::ImageProcessorConfig {
            min_pixels: 65536,
            max_pixels: 262144,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            do_rescale: true,
            do_normalize: true,
        },
    ))
}

pub struct Vision {
    config: Option<QwenVisionConfig>,
    encoder: Option<QwenVisionEncoder>,
}
pub struct Prepared {
    pub embeddings: MxArray,
    pub positions: Vec<[i64; 3]>,
    pub delta: i64,
    pub digests: Vec<crate::engine::cache::ImageCacheDigest>,
}

/// The vision tower is bounded as a whole, but its merger matrix is larger
/// than one staging read. Assemble it without relaxing the store's read cap.
pub(super) fn load_vision_tensor(store: &mut Store, name: &str) -> Result<MxArray> {
    let d = store.descriptor(name)?;
    let rows = d.rows()?;
    let width = d.width()?;
    if !name.starts_with("visual.") || rows as u128 * width as u128 * 4 > (2u128 << 30) {
        return Err(Error::from_reason(
            "Qwen4 vision tensor exceeds its materialization budget",
        ));
    }
    let shape: Vec<i64> = d.shape.iter().map(|&x| x as i64).collect();
    let chunk = (super::weights::MAX_READ_BYTES as usize / 4 / width).max(1);
    let mut values = Vec::new();
    for first in (0..rows).step_by(chunk) {
        values.push(
            store
                .read(name, first, (rows - first).min(chunk))?
                .dense()?,
        );
    }
    let value = if values.len() == 1 {
        values.remove(0)
    } else {
        MxArray::concatenate_many(values.iter().collect(), Some(0))?
    }
    .reshape(&shape)?;
    MxArray::eval_arrays_with_context(&[&value], "qwen4::media::bounded_tensor")?;
    Ok(value)
}

impl Vision {
    pub fn available(&self) -> bool {
        self.config.is_some()
    }
    pub fn metadata(raw: &serde_json::Value, store: &Store, target: &Config) -> Result<Self> {
        if raw.get("vision_config").is_none() {
            return Ok(Self {
                config: None,
                encoder: None,
            });
        }
        let v = &raw["vision_config"];
        let cfg = crate::vision::qwen::weights::parse_vision_config(raw);
        if cfg.hidden_size <= 0
            || cfg.hidden_size > 2048
            || cfg.intermediate_size <= 0
            || cfg.intermediate_size > 8192
            || cfg.num_heads <= 0
            || cfg.num_heads > 64
            || cfg.num_layers <= 0
            || cfg.num_layers > 64
            || cfg.hidden_size % cfg.num_heads != 0
            || (cfg.hidden_size / cfg.num_heads) % 4 != 0
            || cfg.out_hidden_size as usize != target.hidden_size
            || cfg.patch_size != 16
            || cfg.spatial_merge_size != 2
            || v["temporal_patch_size"].as_i64() != Some(2)
            || v["deepstack_visual_indexes"]
                .as_array()
                .is_some_and(|a| !a.is_empty())
        {
            return Err(Error::from_reason("Unsupported Qwen4 vision geometry"));
        }
        let tensors: Vec<_> = store
            .tensors
            .iter()
            .filter(|(key, _)| key.starts_with("visual."))
            .collect();
        let bytes: u128 = tensors
            .iter()
            .map(|(_, t)| t.shape.iter().map(|&d| d as u128).product::<u128>() * 4)
            .sum();
        if tensors.is_empty()
            || bytes > (2u128 << 30)
            || !store.tensors.contains_key("visual.patch_embed.proj.weight")
        {
            return Err(Error::from_reason(
                "Qwen4 vision tensors missing or exceed the 2 GiB F32-equivalent budget",
            ));
        }
        if store.tensors["visual.patch_embed.proj.weight"].shape
            != [cfg.hidden_size as usize, 3, 2, 16, 16]
        {
            return Err(Error::from_reason(
                "Qwen4 vision patch tensor shape mismatch",
            ));
        }
        Ok(Self {
            config: Some(cfg),
            encoder: None,
        })
    }
    fn load(&mut self, store: &mut Store) -> Result<()> {
        if self.encoder.is_some() {
            return Ok(());
        }
        let cfg = self.config.clone().ok_or_else(|| Error::from_reason("Qwen4 GGUF omits vision weights; set auxiliaryModelPath to the matching original HF checkpoint"))?;
        let mut params = HashMap::new();
        let mut names: Vec<_> = store
            .tensors
            .keys()
            .filter(|k| k.starts_with("visual."))
            .cloned()
            .collect();
        names.sort();
        for key in names {
            let mut value = load_vision_tensor(store, &key)?;
            if key == "visual.patch_embed.proj.weight" && value.ndim()? == 5 {
                value = value.transpose(Some(&[0, 2, 3, 4, 1]))?;
            }
            MxArray::eval_arrays_with_context(&[&value], "qwen4::media::value")?;
            params.insert(key.trim_start_matches("visual.").to_string(), value);
        }
        let mut encoder = QwenVisionEncoder::new(cfg.clone())?;
        crate::vision::qwen::weights::load_vision_weights(&mut encoder, &params, &cfg)?;
        self.encoder = Some(encoder);
        Ok(())
    }
}
impl Inner {
    pub fn run_media(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        if args.media.images.is_empty() || !args.media.audio.is_empty() {
            return Err(Error::from_reason(
                "Qwen4 accepts 1-4 images per rendered conversation; audio is not part of this checkpoint",
            ));
        }
        self.decoder.check_cancelled()?;
        IMAGE_LIMITS.validate(args.media.images)?;
        let processor = image_processor();
        let refs: Vec<_> = args.media.images.iter().map(Vec::as_slice).collect();
        let counts = processor.plan_merged_token_counts(&refs, 2)?;
        let tokens = inject_image_placeholders(args.tokens, &counts)?;
        if tokens.len() > self.decoder.config.effective_context_limit() {
            return Err(Error::from_reason(
                "Qwen4 expanded image prompt exceeds context budget",
            ));
        }
        self.vision.load(&mut self.decoder.weights)?;
        let encoder = self
            .vision
            .encoder
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen4 vision encoder was not initialized"))?;
        let mut grids = Vec::new();
        let mut features = Vec::new();
        for bytes in refs {
            self.decoder.check_cancelled()?;
            let image = processor.process_many(&[bytes])?;
            let grid = image.grid_thw();
            let actual = compute_image_token_counts_per_image(&grid, 2)?;
            let feature = encoder.forward(&image.pixel_values().expand_dims(0)?, &grid)?;
            MxArray::eval_arrays_with_context(&[&feature], "qwen4::media::feature")?;
            if feature.shape_at(0)? as usize != actual[0] {
                return Err(Error::from_reason("Qwen4 vision feature count mismatch"));
            }
            features.push(feature);
            grids.push(grid);
            crate::array::memory::clear_cache();
        }
        let text_dtype = self.decoder.embed_token(tokens[0])?.dtype()?;
        let features =
            MxArray::concatenate_many(features.iter().collect(), Some(0))?.astype(text_dtype)?;
        let grid = MxArray::concatenate_many(grids.iter().collect(), Some(0))?;
        let ids = MxArray::from_uint32(&tokens, &[1, tokens.len() as i64])?;
        let (positions, delta) = get_rope_index(&ids, Some(&grid), 2, IMAGE_TOKEN_ID)?;
        let positions = positions.to_int32()?;
        let axes = (0..tokens.len())
            .map(|i| {
                [
                    positions[i] as i64,
                    positions[tokens.len() + i] as i64,
                    positions[2 * tokens.len() + i] as i64,
                ]
            })
            .collect();
        let mut embeddings = Vec::with_capacity(tokens.len());
        let mut feature_row = 0;
        for &token in &tokens {
            let embedding = if token == IMAGE_TOKEN_ID as u32 {
                let row = features
                    .slice_axis(0, feature_row, feature_row + 1)?
                    .expand_dims(0)?;
                feature_row += 1;
                row
            } else {
                self.decoder.embed_token(token)?
            };
            embeddings.push(embedding);
        }
        let embeddings = MxArray::concatenate_many(embeddings.iter().collect(), Some(1))?;
        MxArray::eval_arrays_with_context(&[&embeddings], "qwen4::media::embeddings")?;
        self.abort_paged_turn();
        self.media_prefill = Some(Prepared {
            embeddings,
            positions: axes,
            delta,
            digests: crate::engine::cache::compute_image_cache_keys(args.media.images).1,
        });
        let mut expanded = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: args.tokenizer,
            eos_id: args.eos_id,
            config: args.config,
            params: args.params,
            thinking: args.thinking,
            plan: args.plan,
            sink: args.sink,
            cancelled: args.cancelled,
            media: args.media,
        };
        let result = crate::engine::paged_turn::run_paged_turn(self, &mut expanded);
        self.media_prefill = None;
        result
    }
}
