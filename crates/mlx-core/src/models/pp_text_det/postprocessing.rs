//! DB PostProcessor
//!
//! Converts the probability map output by DBHead into text bounding boxes.
//!
//! Algorithm:
//! 1. Binarize the probability map (threshold > det_threshold)
//! 2. Connected component labeling (pure Rust, no OpenCV)
//! 3. Extract bounding rectangle per component
//! 4. Score each box (mean probability inside box)
//! 5. Filter by box_threshold
//! 6. Expand boxes by unclip_ratio
//! 7. Map coordinates back to original image size

use napi_derive::napi;

/// A detected text bounding box.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TextBox {
    /// Bounding box in original image coordinates [x1, y1, x2, y2]
    pub bbox: Vec<f64>,
    /// Detection confidence score (mean probability inside box)
    pub score: f64,
}

/// Post-process the probability map into text boxes.
///
/// # Arguments
/// * `prob_map` - Probability map values (flattened, H*W)
/// * `map_h` - Probability map height
/// * `map_w` - Probability map width
/// * `orig_h` - Original image height
/// * `orig_w` - Original image width
/// * `_resized_h` - Resized image height (unused; map_h is used for scaling)
/// * `_resized_w` - Resized image width (unused; map_w is used for scaling)
/// * `det_threshold` - Binarization threshold
/// * `box_threshold` - Minimum mean score inside box
/// * `unclip_ratio` - Box expansion ratio
/// * `max_candidates` - Maximum number of candidates
/// * `min_size` - Minimum box side length
pub fn postprocess_db(
    prob_map: &[f32],
    map_h: usize,
    map_w: usize,
    orig_h: u32,
    orig_w: u32,
    _resized_h: u32,
    _resized_w: u32,
    det_threshold: f64,
    box_threshold: f64,
    unclip_ratio: f64,
    max_candidates: usize,
    min_size: f64,
) -> Vec<TextBox> {
    // 1. Binarize
    let mut binary: Vec<bool> = Vec::with_capacity(map_h * map_w);
    for &p in prob_map.iter().take(map_h * map_w) {
        binary.push(p as f64 > det_threshold);
    }

    // 2. Connected component labeling
    let labels = connected_components(&binary, map_h, map_w);
    let num_labels = *labels.iter().max().unwrap_or(&0);

    if num_labels == 0 {
        return Vec::new();
    }

    // 3. Extract bounding rectangles and scores per component
    //
    // PaddleOCR's boxes_from_bitmap() order:
    //   get_mini_boxes -> first min_size check -> score check -> unclip
    //   -> get_mini_boxes again -> second min_size check -> scale coordinates
    let mut boxes: Vec<TextBox> = Vec::new();

    for label_id in 1..=num_labels {
        // Find bounding rect
        let mut min_x = map_w;
        let mut min_y = map_h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut score_sum = 0.0f64;
        let mut count = 0usize;

        for y in 0..map_h {
            for x in 0..map_w {
                let idx = y * map_w + x;
                if labels[idx] == label_id {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    score_sum += prob_map[idx] as f64;
                    count += 1;
                }
            }
        }

        if count == 0 {
            continue;
        }

        let box_w = (max_x - min_x + 1) as f64;
        let box_h = (max_y - min_y + 1) as f64;

        // First min_size check (PaddleOCR: sside < min_size -> continue)
        // This comes BEFORE the score check, matching PaddleOCR's order.
        if box_w.min(box_h) < min_size {
            continue;
        }

        // Score check (PaddleOCR: box_thresh > score -> continue)
        let mean_score = score_sum / count as f64;
        if mean_score < box_threshold {
            continue;
        }

        // Expand box by unclip_ratio using the Clipper library offset formula.
        //
        // PaddleOCR's DBPostProcessor.unclip() computes:
        //   distance = poly.area * unclip_ratio / poly.length
        // where poly.length is the full perimeter (Shapely convention).
        //
        // For an axis-aligned rectangle of width w and height h:
        //   area = w * h
        //   perimeter = 2 * (w + h)
        //   distance = (w * h * unclip_ratio) / (2 * (w + h))
        //
        // PaddleOCR then uses pyclipper.PyclipperOffset with JT_ROUND to expand
        // each polygon edge outward by `distance` pixels (Minkowski sum). For a
        // rectangle, this is equivalent to shifting each edge outward by `distance`:
        //   x1 -= distance, y1 -= distance, x2 += distance, y2 += distance
        //
        // Reference: PaddleOCR/ppocr/postprocess/db_postprocess.py, DBPostProcess.unclip()
        let perimeter = 2.0 * (box_w + box_h);
        let area = box_w * box_h;
        let distance = area * unclip_ratio / perimeter;

        let x1 = (min_x as f64 - distance).max(0.0);
        let y1 = (min_y as f64 - distance).max(0.0);
        let x2 = (max_x as f64 + distance).min(map_w as f64 - 1.0);
        let y2 = (max_y as f64 + distance).min(map_h as f64 - 1.0);

        // Second min_size check after unclip (PaddleOCR: sside < min_size + 2)
        let exp_w = x2 - x1;
        let exp_h = y2 - y1;
        if exp_w.min(exp_h) < min_size + 2.0 {
            continue;
        }

        // Map back to original image coordinates with rounding.
        // PaddleOCR uses np.round() when scaling coordinates:
        //   box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        //   box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        let scale_x = orig_w as f64 / map_w as f64;
        let scale_y = orig_h as f64 / map_h as f64;

        let ox1 = (x1 * scale_x).round().max(0.0).min(orig_w as f64);
        let oy1 = (y1 * scale_y).round().max(0.0).min(orig_h as f64);
        let ox2 = (x2 * scale_x).round().max(0.0).min(orig_w as f64);
        let oy2 = (y2 * scale_y).round().max(0.0).min(orig_h as f64);

        boxes.push(TextBox {
            bbox: vec![ox1, oy1, ox2, oy2],
            score: mean_score,
        });

        if boxes.len() >= max_candidates {
            break;
        }
    }

    // Sort spatially: top-to-bottom (y1 ascending), then left-to-right (x1 ascending).
    // This matches PaddleOCR's contour-order output which is roughly spatial,
    // and is more useful for OCR pipelines than score-based sorting.
    boxes.sort_by(|a, b| {
        let y_cmp = a.bbox[1]
            .partial_cmp(&b.bbox[1])
            .unwrap_or(std::cmp::Ordering::Equal);
        if y_cmp != std::cmp::Ordering::Equal {
            y_cmp
        } else {
            a.bbox[0]
                .partial_cmp(&b.bbox[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    boxes
}

/// Connected component labeling using union-find.
///
/// 8-connected components on a binary image (matches cv2.findContours behavior).
/// Checks all 8 neighbors: up, left, up-left diagonal, and up-right diagonal.
/// Returns label array (0 = background, 1..N = component IDs).
fn connected_components(binary: &[bool], h: usize, w: usize) -> Vec<usize> {
    let n = h * w;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return;
        }
        if rank[rx] < rank[ry] {
            parent[rx] = ry;
        } else if rank[rx] > rank[ry] {
            parent[ry] = rx;
        } else {
            parent[ry] = rx;
            rank[rx] += 1;
        }
    }

    // First pass: union adjacent foreground pixels (8-connectivity)
    // For each pixel (y, x), check previously visited neighbors:
    //   (y-1, x-1) above-left, (y-1, x) above, (y-1, x+1) above-right, (y, x-1) left
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !binary[idx] {
                continue;
            }
            // Check left neighbor
            if x > 0 && binary[idx - 1] {
                union(&mut parent, &mut rank, idx, idx - 1);
            }
            // Check above neighbor
            if y > 0 && binary[(y - 1) * w + x] {
                union(&mut parent, &mut rank, idx, (y - 1) * w + x);
            }
            // Check above-left diagonal
            if y > 0 && x > 0 && binary[(y - 1) * w + (x - 1)] {
                union(&mut parent, &mut rank, idx, (y - 1) * w + (x - 1));
            }
            // Check above-right diagonal
            if y > 0 && x + 1 < w && binary[(y - 1) * w + (x + 1)] {
                union(&mut parent, &mut rank, idx, (y - 1) * w + (x + 1));
            }
        }
    }

    // Assign contiguous labels
    let mut label_map: Vec<usize> = vec![0; n];
    let mut next_label = 1usize;
    let mut root_to_label = std::collections::HashMap::new();

    for i in 0..n {
        if !binary[i] {
            continue;
        }
        let root = find(&mut parent, i);
        let label = *root_to_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        label_map[i] = label;
    }

    label_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connected_components_simple() {
        // 4x4 binary image with two components (no diagonal connection between them)
        #[rustfmt::skip]
        let binary = vec![
            true,  true,  false, false,
            true,  false, false, false,
            false, false, true,  true,
            false, false, true,  true,
        ];
        let labels = connected_components(&binary, 4, 4);
        // First component (top-left L-shape) should have same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[4]);
        // Second component (bottom-right square) should have same label
        assert_eq!(labels[10], labels[11]);
        assert_eq!(labels[10], labels[14]);
        assert_eq!(labels[10], labels[15]);
        // Two components should have different labels
        assert_ne!(labels[0], labels[10]);
        // Background should be 0
        assert_eq!(labels[2], 0);
    }

    #[test]
    fn test_connected_components_8_connectivity() {
        // 3x3 binary image where pixels are only connected diagonally.
        // With 4-connectivity these would be separate; with 8-connectivity
        // they should be the same component.
        #[rustfmt::skip]
        let binary = vec![
            true,  false, false,
            false, true,  false,
            false, false, true,
        ];
        let labels = connected_components(&binary, 3, 3);
        // All three diagonal pixels should be in the same component (8-connected)
        assert_ne!(labels[0], 0);
        assert_eq!(labels[0], labels[4]); // (0,0) and (1,1)
        assert_eq!(labels[0], labels[8]); // (0,0) and (2,2)
    }

    #[test]
    fn test_postprocess_empty() {
        let prob_map = vec![0.0f32; 100];
        let boxes = postprocess_db(
            &prob_map, 10, 10, 100, 100, 10, 10, 0.3, 0.6, 1.5, 1000, 3.0,
        );
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_postprocess_single_box() {
        let mut prob_map = vec![0.0f32; 400]; // 20x20
        // Place a high-probability region
        for y in 5..15 {
            for x in 5..15 {
                prob_map[y * 20 + x] = 0.9;
            }
        }
        let boxes = postprocess_db(
            &prob_map, 20, 20, 200, 200, 20, 20, 0.3, 0.6, 1.5, 1000, 3.0,
        );
        assert!(!boxes.is_empty());
        assert!(boxes[0].score > 0.8);
    }
}
