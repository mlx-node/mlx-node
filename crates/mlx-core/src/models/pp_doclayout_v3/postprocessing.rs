//! PP-DocLayoutV3 Post-Processing
//!
//! Post-processing for detection outputs: score thresholding, box decoding,
//! and reading order extraction.

use crate::array::MxArray;
use crate::nn::activations::Activations;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

/// A single detected layout element.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LayoutElement {
    /// Detection confidence score
    pub score: f64,
    /// Class label ID (0-24)
    pub label: u32,
    /// Human-readable label name (e.g., "title", "text", "table")
    pub label_name: String,
    /// Bounding box in original image coordinates [x1, y1, x2, y2]
    pub bbox: Vec<f64>,
    /// Reading order index (0 = first element to read)
    pub order: u32,
}

/// Post-process model outputs into layout elements.
///
/// # Arguments
/// * `logits` - Class logits [batch, num_queries, num_labels]
/// * `pred_boxes` - Predicted boxes in (cx, cy, w, h) format [batch, num_queries, 4]
/// * `order_logits` - Reading order logits [batch, num_queries, num_queries]
/// * `orig_h` - Original image height
/// * `orig_w` - Original image width
/// * `threshold` - Score threshold for detection (default 0.5)
/// * `id2label` - Mapping from class ID to label name
///
/// # Returns
/// Vec of LayoutElements sorted by reading order
pub fn postprocess_detection(
    logits: &MxArray,
    pred_boxes: &MxArray,
    order_logits: &MxArray,
    orig_h: u32,
    orig_w: u32,
    threshold: f64,
    id2label: &HashMap<String, String>,
) -> Result<Vec<LayoutElement>> {
    // Apply sigmoid to logits to get scores
    let scores = Activations::sigmoid(logits)?;
    scores.eval();

    // Get shapes
    let scores_shape = scores.shape()?;
    let scores_shape: Vec<i64> = scores_shape.as_ref().to_vec();
    let num_queries = scores_shape[1] as usize;
    let num_labels = scores_shape[2] as usize;

    // Read scores data
    let scores_data = scores.to_float32()?;
    let scores_vec: Vec<f32> = scores_data.to_vec();

    // Read box data
    pred_boxes.eval();
    let boxes_data = pred_boxes.to_float32()?;
    let boxes_vec: Vec<f32> = boxes_data.to_vec();

    // Top-k flattened selection (matches HuggingFace reference implementation):
    // Instead of per-query argmax (one detection per query), we flatten all
    // (query, class) scores and select the top num_queries entries globally.
    // This allows multiple detections per query with different class labels.

    // Build vec of (score, flattened_index) for all query-label pairs
    let total = num_queries * num_labels;
    let mut scored_indices: Vec<(f32, usize)> = scores_vec[..total]
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();

    // Sort by score descending
    scored_indices
        .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top num_queries entries, decode flattened index, apply threshold
    let mut detections: Vec<(usize, u32, f64, [f64; 4])> = Vec::new();

    let oh = orig_h as f64;
    let ow = orig_w as f64;

    for &(score, flat_idx) in scored_indices.iter().take(num_queries) {
        if (score as f64) < threshold {
            continue;
        }

        let query_idx = flat_idx / num_labels;
        let label_idx = (flat_idx % num_labels) as u32;

        // Convert box from (cx, cy, w, h) normalized to (x1, y1, x2, y2) in original image coords
        let cx = boxes_vec[query_idx * 4] as f64;
        let cy = boxes_vec[query_idx * 4 + 1] as f64;
        let bw = boxes_vec[query_idx * 4 + 2] as f64;
        let bh = boxes_vec[query_idx * 4 + 3] as f64;

        let x1 = ((cx - bw / 2.0) * ow).max(0.0);
        let y1 = ((cy - bh / 2.0) * oh).max(0.0);
        let x2 = ((cx + bw / 2.0) * ow).min(ow);
        let y2 = ((cy + bh / 2.0) * oh).min(oh);

        detections.push((query_idx, label_idx, score as f64, [x1, y1, x2, y2]));
    }

    if detections.is_empty() {
        return Ok(Vec::new());
    }

    // Compute reading order from order_logits
    // order_logits: [1, num_queries, num_queries]
    // Computes global pairwise votes over all queries, ranks them ascending,
    // then gathers ranks for selected detections only.
    let order_scores = compute_reading_order(order_logits, &detections, num_queries)?;

    // Build (order_score, detection_index) pairs and sort by order_score ascending.
    // This matches the HuggingFace reference which sorts filtered detections by order_seq.
    let mut ordered_indices: Vec<(u32, usize)> = order_scores
        .iter()
        .enumerate()
        .map(|(i, &o)| (o, i))
        .collect();
    ordered_indices.sort_by_key(|&(o, _)| o);

    // Build LayoutElements in reading order with contiguous 0-based order values
    let mut elements: Vec<LayoutElement> = Vec::with_capacity(detections.len());
    for (new_order, &(_, det_idx)) in ordered_indices.iter().enumerate() {
        let (_, label, score, bbox) = &detections[det_idx];
        let label_name = id2label
            .get(&label.to_string())
            .cloned()
            .unwrap_or_else(|| format!("class_{}", label));

        elements.push(LayoutElement {
            score: *score,
            label: *label,
            label_name,
            bbox: bbox.to_vec(),
            order: new_order as u32,
        });
    }

    // Fixup: promote doc_title elements near the top of the page to the front.
    // The reading order model sometimes misranks page-spanning titles in
    // multi-column layouts.
    fixup_doc_title_order(&mut elements, oh);

    Ok(elements)
}

/// Promote doc_title elements near the top of the page to the front of reading order.
///
/// The pairwise reading order model can misrank page-spanning titles in
/// multi-column newspaper layouts. This heuristic detects doc_title elements
/// whose top edge is in the upper 15% of the image and moves them to order 0,
/// shifting other elements down.
fn fixup_doc_title_order(elements: &mut [LayoutElement], image_height: f64) {
    if elements.is_empty() {
        return;
    }

    let top_threshold = image_height * 0.15;

    // Find doc_title elements near the top, sorted by y1 (topmost first)
    let mut top_titles: Vec<usize> = elements
        .iter()
        .enumerate()
        .filter(|(_, e)| e.label_name == "doc_title" && e.bbox[1] < top_threshold)
        .map(|(i, _)| i)
        .collect();

    if top_titles.is_empty() {
        return;
    }

    // Sort by y position (topmost first)
    top_titles.sort_by(|&a, &b| {
        elements[a].bbox[1]
            .partial_cmp(&elements[b].bbox[1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Check if they're already at the front
    let already_first = top_titles
        .iter()
        .enumerate()
        .all(|(rank, &idx)| elements[idx].order == rank as u32);
    if already_first {
        return;
    }

    // Collect the title indices as a set for quick lookup
    let title_set: std::collections::HashSet<usize> = top_titles.iter().copied().collect();
    let num_promoted = top_titles.len() as u32;

    // Assign order 0, 1, ... to the promoted titles
    for (rank, &idx) in top_titles.iter().enumerate() {
        elements[idx].order = rank as u32;
    }

    // Shift all other elements' order by num_promoted
    for (i, el) in elements.iter_mut().enumerate() {
        if !title_set.contains(&i) {
            el.order += num_promoted;
        }
    }

    // Re-sort the slice by the new order and renormalize to contiguous indices
    elements.sort_by_key(|e| e.order);
    for (i, el) in elements.iter_mut().enumerate() {
        el.order = i as u32;
    }
}

/// Compute reading order from pairwise order logits.
///
/// Matches the HuggingFace reference implementation:
///   1. Compute votes GLOBALLY over ALL queries (not just detected ones)
///   2. Argsort ASCENDING to get global ranking (lowest vote = rank 0 = read first)
///   3. Assign ranks via inverse permutation
///   4. Gather ranks for detected queries only
///
/// # Arguments
/// * `order_logits` - [1, num_queries, num_queries] pairwise ordering logits
/// * `detections` - Vec of (query_idx, label, score, bbox) for detected elements
/// * `num_queries` - Total number of queries
///
/// # Returns
/// Vec of reading order ranks (0-indexed) aligned with detections
fn compute_reading_order(
    order_logits: &MxArray,
    detections: &[(usize, u32, f64, [f64; 4])],
    num_queries: usize,
) -> Result<Vec<u32>> {
    let n = detections.len();
    if n <= 1 {
        return Ok(vec![0; n]);
    }

    // Apply sigmoid to order logits
    let order_sigmoid = Activations::sigmoid(order_logits)?;
    order_sigmoid.eval();
    let order_data = order_sigmoid.to_float32()?;
    let order_vec: Vec<f32> = order_data.to_vec();

    // Step 1: Compute votes GLOBALLY for ALL queries
    // For each query i: vote[i] = sum_j>i(score[i,j]) + sum_j<i(1 - score[j,i])
    // This matches: triu(diagonal=1).sum(dim=1) + (1 - scores.T).tril(diagonal=-1).sum(dim=1)
    let mut votes: Vec<f64> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let mut vote = 0.0f64;
        for j in 0..num_queries {
            if j > i {
                // Upper triangle: score[i, j]
                vote += order_vec[i * num_queries + j] as f64;
            } else if j < i {
                // Lower triangle: 1 - score[j, i]
                vote += 1.0 - order_vec[j * num_queries + i] as f64;
            }
        }
        votes.push(vote);
    }

    // Step 2: Argsort ASCENDING (lowest vote → rank 0 → read first)
    let mut sorted_indices: Vec<usize> = (0..num_queries).collect();
    sorted_indices.sort_by(|&a, &b| {
        votes[a]
            .partial_cmp(&votes[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 3: Assign ranks (inverse permutation)
    let mut order_seq = vec![0u32; num_queries];
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        order_seq[idx] = rank as u32;
    }

    // Step 4: Gather ranks for detected queries only
    let ranks: Vec<u32> = detections
        .iter()
        .map(|(q, _, _, _)| order_seq[*q])
        .collect();

    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_element_struct() {
        let elem = LayoutElement {
            score: 0.95,
            label: 0,
            label_name: "title".to_string(),
            bbox: vec![10.0, 20.0, 300.0, 50.0],
            order: 0,
        };
        assert_eq!(elem.label, 0);
        assert_eq!(elem.label_name, "title");
        assert_eq!(elem.bbox.len(), 4);
    }
}
