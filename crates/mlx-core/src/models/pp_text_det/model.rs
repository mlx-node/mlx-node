//! Text Detection Full Model
//!
//! Top-level model combining HGNetV2 backbone, LKPAN neck, and DBHead.
//! Provides the NAPI `TextDetModel` class with `load()` and `detect()` methods.

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::models::pp_doclayout_v3::backbone::HGNetV2Backbone;

use super::config::TextDetConfig;
use super::db_head::PFHeadLocal;
use super::lkpan::LKPAN;
use super::postprocessing::{TextBox, postprocess_db};
use super::processing::TextDetImageProcessor;

/// PP-OCRv5 Text Detection model (DBNet with PPHGNetV2 backbone).
///
/// Detects text lines in document images and returns bounding boxes.
#[napi(js_name = "TextDetModel")]
pub struct TextDetModel {
    config: TextDetConfig,
    backbone: HGNetV2Backbone,
    neck: LKPAN,
    head: PFHeadLocal,
    image_processor: TextDetImageProcessor,
}

#[napi]
impl TextDetModel {
    /// Load a TextDetModel from a directory containing model.safetensors.
    ///
    /// # Arguments
    /// * `model_path` - Path to model directory
    #[napi(factory)]
    pub fn load(model_path: String) -> Result<Self> {
        super::persistence::load_model(&model_path)
    }

    /// Detect text lines in an image.
    ///
    /// # Arguments
    /// * `image_data` - Encoded image bytes (PNG/JPEG)
    /// * `threshold` - Optional detection threshold (default from config, typically 0.3)
    ///
    /// # Returns
    /// * Vec of TextBox with bounding boxes and confidence scores
    #[napi]
    pub fn detect(&self, image_data: Buffer, threshold: Option<f64>) -> Result<Vec<TextBox>> {
        let det_threshold = threshold.unwrap_or(self.config.det_threshold);

        // 1. Preprocess
        let (pixel_values, orig_h, orig_w, resized_h, resized_w) =
            self.image_processor.process(&image_data)?;

        // 2. Run model
        let prob_map = self.forward(&pixel_values)?;

        // 3. Postprocess
        let map_shape = prob_map.shape()?;
        let map_h = map_shape[1] as usize;
        let map_w = map_shape[2] as usize;

        prob_map.eval();
        let prob_data = prob_map.to_float32()?;
        let prob_vec: Vec<f32> = prob_data.to_vec();

        Ok(postprocess_db(
            &prob_vec,
            map_h,
            map_w,
            orig_h,
            orig_w,
            resized_h,
            resized_w,
            det_threshold,
            self.config.box_threshold,
            self.config.unclip_ratio,
            self.config.max_candidates,
            self.config.min_size,
        ))
    }

    /// Detect text lines from raw RGB pixel data.
    ///
    /// # Arguments
    /// * `rgb_data` - Raw RGB pixel data
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `threshold` - Optional detection threshold (default from config)
    ///
    /// # Returns
    /// * Vec of TextBox with bounding boxes and confidence scores
    #[napi]
    pub fn detect_crop(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
        threshold: Option<f64>,
    ) -> Result<Vec<TextBox>> {
        let det_threshold = threshold.unwrap_or(self.config.det_threshold);

        let (pixel_values, resized_h, resized_w) =
            self.image_processor.process_crop(rgb_data, width, height)?;

        let prob_map = self.forward(&pixel_values)?;

        let map_shape = prob_map.shape()?;
        let map_h = map_shape[1] as usize;
        let map_w = map_shape[2] as usize;

        prob_map.eval();
        let prob_data = prob_map.to_float32()?;
        let prob_vec: Vec<f32> = prob_data.to_vec();

        Ok(postprocess_db(
            &prob_vec,
            map_h,
            map_w,
            height,
            width,
            resized_h,
            resized_w,
            det_threshold,
            self.config.box_threshold,
            self.config.unclip_ratio,
            self.config.max_candidates,
            self.config.min_size,
        ))
    }

    /// Run the full model forward pass.
    fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        // Backbone: extract multi-scale features
        let features = self.backbone.forward(pixel_values)?;

        // Neck: LKPAN fuses features via FPN + PAN and returns a single tensor
        // [B, H, W, 256] at the finest feature level resolution
        let fused = self.neck.forward(&features)?;

        // Head: produce probability map
        self.head.forward(&fused)
    }
}

/// Create an initialized TextDetModel (called from persistence).
pub(super) fn create_model(
    config: TextDetConfig,
    backbone: HGNetV2Backbone,
    neck: LKPAN,
    head: PFHeadLocal,
) -> TextDetModel {
    TextDetModel {
        config,
        backbone,
        neck,
        head,
        image_processor: TextDetImageProcessor::new(),
    }
}
