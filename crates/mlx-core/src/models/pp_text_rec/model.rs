//! Text Recognition Full Model
//!
//! Top-level model combining HGNetV2 backbone, SVTR neck, and CTC head.
//! Provides the NAPI `TextRecModel` class with `load()` and `recognize()` methods.

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::models::pp_doclayout_v3::backbone::HGNetV2Backbone;

use super::config::TextRecConfig;
use super::ctc_head::{CTCHead, ctc_greedy_decode};
use super::dictionary::CharDictionary;
use super::processing::TextRecImageProcessor;
use super::svtr_neck::SVTRNeck;

/// Result of text recognition.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct RecResult {
    /// Recognized text
    pub text: String,
    /// Confidence score (mean character probability)
    pub score: f64,
}

/// PP-OCRv5 Text Recognition model (PPHGNetV2 + SVTR + CTC).
///
/// Recognizes text from cropped text line images.
#[napi(js_name = "TextRecModel")]
pub struct TextRecModel {
    backbone: HGNetV2Backbone,
    neck: SVTRNeck,
    head: CTCHead,
    dictionary: CharDictionary,
    image_processor: TextRecImageProcessor,
}

#[napi]
impl TextRecModel {
    /// Load a TextRecModel from a directory containing model.safetensors.
    ///
    /// # Arguments
    /// * `model_path` - Path to model directory
    /// * `dict_path` - Path to character dictionary text file
    #[napi(factory)]
    pub fn load(model_path: String, dict_path: String) -> Result<Self> {
        super::persistence::load_model(&model_path, &dict_path)
    }

    /// Recognize text from encoded image bytes.
    ///
    /// # Arguments
    /// * `image_data` - Encoded image bytes (PNG/JPEG)
    ///
    /// # Returns
    /// * RecResult with recognized text and confidence score
    #[napi]
    pub fn recognize(&self, image_data: Buffer) -> Result<RecResult> {
        let pixel_values = self.image_processor.process(&image_data)?;
        let logits = self.forward(&pixel_values)?;
        let decoded = ctc_greedy_decode(&logits)?;

        if decoded.is_empty() {
            return Ok(RecResult {
                text: String::new(),
                score: 0.0,
            });
        }

        let (indices, scores) = &decoded[0];
        let (text, score) = self.dictionary.decode_with_score(indices, scores);

        Ok(RecResult {
            text,
            score: score as f64,
        })
    }

    /// Recognize text from multiple encoded images.
    ///
    /// # Arguments
    /// * `images` - Vec of encoded image bytes (PNG/JPEG)
    ///
    /// # Returns
    /// * Vec of RecResult with recognized text and confidence scores
    #[napi]
    pub fn recognize_batch(&self, images: Vec<Buffer>) -> Result<Vec<RecResult>> {
        let mut results = Vec::with_capacity(images.len());

        // Process images one at a time through the model
        // (batching requires all images to have same dimensions after processing)
        for image_data in &images {
            let pixel_values = self.image_processor.process(image_data)?;
            let logits = self.forward(&pixel_values)?;
            let decoded = ctc_greedy_decode(&logits)?;

            if decoded.is_empty() {
                results.push(RecResult {
                    text: String::new(),
                    score: 0.0,
                });
                continue;
            }

            let (indices, scores) = &decoded[0];
            let (text, score) = self.dictionary.decode_with_score(indices, scores);

            results.push(RecResult {
                text,
                score: score as f64,
            });
        }

        Ok(results)
    }

    /// Recognize text from raw RGB crop data.
    ///
    /// # Arguments
    /// * `rgb_data` - Raw RGB pixel data of a cropped text line
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// * RecResult with recognized text and confidence score
    #[napi]
    pub fn recognize_crop(&self, rgb_data: &[u8], width: u32, height: u32) -> Result<RecResult> {
        let pixel_values = self.image_processor.process_crop(rgb_data, width, height)?;
        let logits = self.forward(&pixel_values)?;
        let decoded = ctc_greedy_decode(&logits)?;

        if decoded.is_empty() {
            return Ok(RecResult {
                text: String::new(),
                score: 0.0,
            });
        }

        let (indices, scores) = &decoded[0];
        let (text, score) = self.dictionary.decode_with_score(indices, scores);

        Ok(RecResult {
            text,
            score: score as f64,
        })
    }

    /// Run the full model forward pass.
    fn forward(&self, pixel_values: &MxArray) -> Result<MxArray> {
        // Backbone: extract features (only last stage for text rec)
        let features = self.backbone.forward(pixel_values)?;
        let last_feature = features
            .last()
            .ok_or_else(|| Error::from_reason("Backbone produced no features"))?;

        // Apply avg_pool2d with kernel [3, 2] to match PaddleOCR's text_rec mode.
        // PaddleOCR backbone.forward() does: x = F.avg_pool2d(x, [3, 2])
        // when text_rec=True and not training.
        // This collapses H from 3 to 1 and halves W.
        // Implementation: reshape [B, H, W, C] -> [B, H/3, 3, W/2, 2, C] -> mean(axes=[2,4])
        let pooled = avg_pool2d_k3x2(last_feature)?;

        // Neck: SVTR with EncoderWithSVTR architecture
        // Input is [B, 1, W/2, C] after pooling. Returns [B, 1, W/2, dims].
        let spatial = self.neck.forward(&pooled)?;

        // Im2Seq: assert H==1, squeeze H, result is [B, W, dims]
        // PaddleOCR's Im2Seq does: assert H==1; x = x.squeeze(axis=2); x = x.transpose([0,2,1])
        // In NHWC, H is axis 1. We squeeze axis 1 to get [B, W, dims].
        let shape = spatial.shape()?;
        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];
        let dims = shape[3];

        let sequence = if h == 1 {
            spatial.reshape(&[batch, w, dims])?
        } else {
            // Fallback: pool height dimension if somehow H > 1
            spatial.mean(Some(&[1]), Some(false))?
        };

        // Head: CTC projection
        self.head.forward(&sequence)
    }
}

/// Average pooling with kernel [3, 2] and stride [3, 2] (stride defaults to kernel size).
///
/// Matches PaddleOCR's `F.avg_pool2d(x, [3, 2])` in the backbone's text_rec forward.
/// Input: [B, H, W, C] (NHWC)
/// Output: [B, floor(H/3), floor(W/2), C] (NHWC)
///
/// Implemented as reshape + mean since MLX doesn't have a native avg_pool2d FFI:
///   [B, H', W', C] -> [B, H'/3, 3, W'/2, 2, C] -> mean(axes=[2, 4]) -> [B, H'/3, W'/2, C]
/// where H' and W' are truncated to be divisible by 3 and 2 respectively,
/// matching PyTorch/PaddlePaddle's truncation behavior for non-divisible inputs.
fn avg_pool2d_k3x2(input: &MxArray) -> Result<MxArray> {
    let shape = input.shape()?;
    if shape.len() != 4 {
        return Err(Error::from_reason(format!(
            "avg_pool2d_k3x2: expected 4D input [B, H, W, C], got {}D",
            shape.len()
        )));
    }
    let batch = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    let h_out = h / 3;
    let w_out = w / 2;

    if h_out == 0 || w_out == 0 {
        return Err(Error::from_reason(format!(
            "avg_pool2d_k3x2: input H={} or W={} too small for kernel [3, 2]",
            h, w
        )));
    }

    // Truncate to exact multiples (matching PyTorch's floor behavior)
    let h_trunc = h_out * 3;
    let w_trunc = w_out * 2;
    let truncated = if h_trunc != h || w_trunc != w {
        input.slice(&[0, 0, 0, 0], &[batch, h_trunc, w_trunc, c])?
    } else {
        input.clone()
    };

    // Reshape: [B, H', W', C] -> [B, H'/3, 3, W'/2, 2, C]
    let reshaped = truncated.reshape(&[batch, h_out, 3, w_out, 2, c])?;

    // Mean over the kernel dimensions (axes 2 and 4) to compute the average
    let pooled = reshaped.mean(Some(&[2, 4]), Some(false))?;

    Ok(pooled)
}

/// Create an initialized TextRecModel (called from persistence).
pub(super) fn create_model(
    config: &TextRecConfig,
    backbone: HGNetV2Backbone,
    neck: SVTRNeck,
    head: CTCHead,
    dictionary: CharDictionary,
) -> TextRecModel {
    let image_processor =
        TextRecImageProcessor::new(config.input_height as u32, config.input_max_width as u32);
    TextRecModel {
        backbone,
        neck,
        head,
        dictionary,
        image_processor,
    }
}
