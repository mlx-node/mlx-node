//! Horizontal-stretched vertical collage supplement for visual PO reference extraction.
//!
//! Exact port of upstream `clairic_runtime/po_image_supplement.py`.

use std::collections::{HashMap, HashSet};

use crate::errors::Result;
use image::RgbImage;
use serde_json::{Map, Value};

use crate::{
    embedded::{ValidatedImage, encode_rgb_png_to_validated},
    errors::invalid_request,
};

use super::{
    DEFAULT_MAX_PIXELS, focus, is_truthy, paste_rgb, resize_bicubic, state_only_spec, white_canvas,
};

/// Supplement version identifier matching `po_image_supplement.VERSION`.
pub const VERSION: &str = "27b-po-all-content-horizontal2-supplement-v2";
/// Maximum pixel budget for the rendered collage matching `po_image_supplement.MAX_PIXELS`.
pub const MAX_PIXELS: usize = 1_600_000;
/// Identifier spelling wire version matching `identifier_spelling.VERSION`.
pub const IDENTIFIER_SPELLING_VERSION: &str = "identifier-codepoint-spelling-v1";

const HORIZONTAL_SCALE: f64 = 2.0;
const VERTICAL_GAP_PX: u32 = 32;

/// Focus each input image, stretch its width `2.0x` bicubic, stack vertically with 32px
/// solid white gaps, and downscale bicubic if the collage exceeds [`MAX_PIXELS`].
///
/// Exact geometric port of `po_image_supplement.py::supplement`.
pub fn supplement_image(images: &[&RgbImage]) -> Result<RgbImage> {
    if images.is_empty() {
        return Err(invalid_request(
            "PO supplement requires at least one source image",
        ));
    }

    let mut views: Vec<RgbImage> = Vec::with_capacity(images.len());
    for image in images {
        let focused = focus::focus_document_or_clone(image, DEFAULT_MAX_PIXELS, 2.0);
        let stretched_w = ((f64::from(focused.width()) * HORIZONTAL_SCALE).round() as u32).max(1);
        let stretched = resize_bicubic(&focused, stretched_w, focused.height().max(1));
        views.push(stretched);
    }

    let board_w = views.iter().map(RgbImage::width).max().unwrap_or(1);
    let total_h: u32 = views
        .iter()
        .map(RgbImage::height)
        .fold(0u32, u32::saturating_add)
        .saturating_add(VERTICAL_GAP_PX.saturating_mul((views.len() - 1) as u32));

    let mut board = white_canvas(board_w, total_h.max(1));
    let mut y = 0u32;
    for view in &views {
        paste_rgb(&mut board, view, 0, y);
        y = y
            .saturating_add(view.height())
            .saturating_add(VERTICAL_GAP_PX);
    }

    let area = f64::from(board.width()) * f64::from(board.height());
    let scale = 1.0_f64.min(((MAX_PIXELS as f64) / area).sqrt());
    if scale < 1.0 {
        let new_w = ((f64::from(board.width()) * scale) as u32).max(1);
        let new_h = ((f64::from(board.height()) * scale) as u32).max(1);
        board = resize_bicubic(&board, new_w, new_h);
    }

    Ok(board)
}

/// Build a PNG-encoded [`ValidatedImage`] collage from `images`, matching
/// `po_image_supplement.py::supplement`.
pub fn supplement(images: &[&RgbImage]) -> Result<ValidatedImage> {
    let board = supplement_image(images)?;
    encode_rgb_png_to_validated(&board)
}

/// Build the exact supplement list for a request's original images:
/// - When `originals.len() == 2`, creates one individual supplement per image.
/// - Otherwise creates a single combined vertical collage of all images.
///
/// Exact port of `po_image_supplement.py:141-145`.
pub fn build_supplements(originals: &[RgbImage]) -> Result<Vec<ValidatedImage>> {
    if originals.len() == 2 {
        let mut out = Vec::with_capacity(2);
        for img in originals {
            out.push(supplement(&[img])?);
        }
        Ok(out)
    } else {
        let refs: Vec<&RgbImage> = originals.iter().collect();
        Ok(vec![supplement(&refs)?])
    }
}

/// Return the field names eligible for the PO image supplement.
///
/// Exact port of `po_image_supplement.py::selected_fields`.
pub fn selected_fields(
    fields: &Map<String, Value>,
    image_count: usize,
    has_documents: bool,
    active_media_tasks: &HashSet<String>,
    spelling_wire_by_field: &HashMap<String, String>,
) -> Vec<String> {
    if !matches!(image_count, 1..=3) || has_documents {
        return Vec::new();
    }
    let mut selected = Vec::new();
    for (name, spec) in fields {
        if !active_media_tasks.contains(name) {
            continue;
        }
        let wire_matches = spelling_wire_by_field
            .get(name)
            .is_some_and(|w| w == IDENTIFIER_SPELLING_VERSION);
        if wire_matches && !is_truthy(spec.get("knowledge")) && !state_only_spec(spec) {
            selected.push(name.clone());
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use serde_json::json;

    #[test]
    fn stretches_horizontally_and_stacks_with_white_gaps() {
        let img1 = RgbImage::from_pixel(100, 50, Rgb([20, 20, 20]));
        let img2 = RgbImage::from_pixel(120, 40, Rgb([40, 40, 40]));

        // Dense images won't be cropped by focus_document, so stretched sizes are:
        // img1 -> 200x50, img2 -> 240x40; collage = 240 x (50 + 32 + 40 = 122)
        let board = supplement_image(&[&img1, &img2]).unwrap();
        assert_eq!(board.dimensions(), (240, 122));
        // Gap row y = 60 must be solid white
        assert_eq!(board.get_pixel(10, 60).0, [255, 255, 255]);

        // Validated PNG wrapper roundtrips
        let validated = supplement(&[&img1]).unwrap();
        assert_eq!(validated.content_type, "image/png");
        let decoded = validated.to_rgb_image().unwrap();
        assert_eq!(decoded.dimensions(), (200, 50));
    }

    #[test]
    fn downscales_collage_when_exceeding_max_pixels() {
        let big = RgbImage::from_pixel(1200, 1000, Rgb([50, 50, 50]));
        // Stretched = 2400 x 1000 = 2_400_000 > 1_600_000 -> downscaled <= 1_600_000
        let board = supplement_image(&[&big]).unwrap();
        let pixels = (board.width() as usize) * (board.height() as usize);
        assert!(pixels <= MAX_PIXELS, "pixels={pixels}");
    }

    #[test]
    fn selected_fields_enforces_count_and_wire_contract() {
        let schema: Map<String, Value> = serde_json::from_value(json!({
            "po_number": { "type": "text", "instructions": "Read purchase order number." },
            "state_po": { "type": "text", "instructions": "From state text only, read PO number." }
        }))
        .unwrap();
        let active: HashSet<String> = schema.keys().cloned().collect();
        let mut wires = HashMap::new();
        wires.insert(
            "po_number".to_string(),
            IDENTIFIER_SPELLING_VERSION.to_string(),
        );
        wires.insert(
            "state_po".to_string(),
            IDENTIFIER_SPELLING_VERSION.to_string(),
        );

        assert_eq!(
            selected_fields(&schema, 1, false, &active, &wires),
            vec!["po_number".to_string()]
        );
        assert!(selected_fields(&schema, 4, false, &active, &wires).is_empty());
        assert!(selected_fields(&schema, 1, true, &active, &wires).is_empty());
    }
}
