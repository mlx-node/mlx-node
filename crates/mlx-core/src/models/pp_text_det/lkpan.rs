//! LKPAN Neck (Large Kernel PAN) with optional IntraCL
//!
//! Feature Pyramid Network with Path Aggregation using large-kernel convolutions.
//! Corresponds to PaddleOCR's LKPAN class (mode="large") from
//! `ppocr/modeling/necks/db_fpn.py`.
//!
//! Architecture:
//! 1. ins_conv: 1x1 Conv2D per level projecting backbone channels -> out_channels (256)
//! 2. FPN top-down: upsample + add (fuse coarse -> fine)
//! 3. inp_conv: 9x9 Conv2D per level reducing out_channels -> out_channels/4 (64)
//! 4. PAN bottom-up: pan_head_conv (3x3, stride=2) downsamples + add
//! 5. pan_lat_conv: 9x9 Conv2D per level (final lateral convolution)
//! 6. Optional IntraCL blocks applied after pan_lat_conv
//! 7. Upsample all levels to finest resolution -> concatenate (4 x 64 = 256)
//!
//! All convolutions are plain Conv2D with NO batch normalization, NO activation,
//! and NO bias -- matching PaddleOCR's LKPAN in "large" mode.
//!
//! ## IntraCLBlock
//!
//! Intra-Class Level attention module. Applies directional convolution attention
//! using cascaded multi-scale (7x7, 5x5, 3x3) convolutions with vertical and
//! horizontal stripe convolution branches, followed by channel expansion and residual.
//!
//! Reference: PaddleOCR's `ppocr/modeling/necks/intracl.py`

use crate::array::MxArray;
use crate::nn::activations::Activations;
use napi::bindgen_prelude::*;

use crate::models::pp_doclayout_v3::backbone::{FrozenBatchNorm2d, NativeConv2d};

// ============================================================================
// IntraCLBlock
// ============================================================================

/// IntraCLBlock: Intra-class level attention block.
///
/// Applies directional convolution attention using cascaded multi-scale convolutions.
///
/// Architecture:
///   1. conv1x1_reduce: channels -> channels // reduce_factor
///   2. Cascaded 7->5->3 with square + vertical + horizontal branches at each level:
///      x_7 = c_7x7(x_reduced) + v_7x1(x_reduced) + q_1x7(x_reduced)
///      x_5 = c_5x5(x_7) + v_5x1(x_7) + q_1x5(x_7)
///      x_3 = c_3x3(x_5) + v_3x1(x_5) + q_1x3(x_5)
///   3. conv1x1_return: channels // reduce_factor -> channels
///   4. BatchNorm + ReLU
///   5. Residual: output = input + relation
pub struct IntraCLBlock {
    /// Channel reduction: Conv2d(channels -> channels//rf, k=1)
    conv1x1_reduce: NativeConv2d,
    /// Channel expansion: Conv2d(channels//rf -> channels, k=1)
    conv1x1_return: NativeConv2d,

    // 7x7 level
    c_layer_7x7: NativeConv2d,
    v_layer_7x1: NativeConv2d,
    q_layer_1x7: NativeConv2d,

    // 5x5 level
    c_layer_5x5: NativeConv2d,
    v_layer_5x1: NativeConv2d,
    q_layer_1x5: NativeConv2d,

    // 3x3 level
    c_layer_3x3: NativeConv2d,
    v_layer_3x1: NativeConv2d,
    q_layer_1x3: NativeConv2d,

    /// BatchNorm2D(channels)
    bn: FrozenBatchNorm2d,
}

impl IntraCLBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        conv1x1_reduce: NativeConv2d,
        conv1x1_return: NativeConv2d,
        c_layer_7x7: NativeConv2d,
        v_layer_7x1: NativeConv2d,
        q_layer_1x7: NativeConv2d,
        c_layer_5x5: NativeConv2d,
        v_layer_5x1: NativeConv2d,
        q_layer_1x5: NativeConv2d,
        c_layer_3x3: NativeConv2d,
        v_layer_3x1: NativeConv2d,
        q_layer_1x3: NativeConv2d,
        bn: FrozenBatchNorm2d,
    ) -> Self {
        Self {
            conv1x1_reduce,
            conv1x1_return,
            c_layer_7x7,
            v_layer_7x1,
            q_layer_1x7,
            c_layer_5x5,
            v_layer_5x1,
            q_layer_1x5,
            c_layer_3x3,
            v_layer_3x1,
            q_layer_1x3,
            bn,
        }
    }

    /// Forward pass: apply IntraCL attention.
    ///
    /// Input: [B, H, W, C] (NHWC)
    /// Output: [B, H, W, C] (NHWC) — same shape with residual connection
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        // 1. Channel reduction
        let x_new = self.conv1x1_reduce.forward(input)?;

        // 2. Cascaded 7 -> 5 -> 3
        let x_7_c = self.c_layer_7x7.forward(&x_new)?;
        let x_7_v = self.v_layer_7x1.forward(&x_new)?;
        let x_7_q = self.q_layer_1x7.forward(&x_new)?;
        let x_7 = x_7_c.add(&x_7_v)?.add(&x_7_q)?;

        let x_5_c = self.c_layer_5x5.forward(&x_7)?;
        let x_5_v = self.v_layer_5x1.forward(&x_7)?;
        let x_5_q = self.q_layer_1x5.forward(&x_7)?;
        let x_5 = x_5_c.add(&x_5_v)?.add(&x_5_q)?;

        let x_3_c = self.c_layer_3x3.forward(&x_5)?;
        let x_3_v = self.v_layer_3x1.forward(&x_5)?;
        let x_3_q = self.q_layer_1x3.forward(&x_5)?;
        let x_3 = x_3_c.add(&x_3_v)?.add(&x_3_q)?;

        // 3. Channel expansion
        let x_relation = self.conv1x1_return.forward(&x_3)?;

        // 4. BN + ReLU
        let x_relation = self.bn.forward(&x_relation)?;
        let x_relation = Activations::relu(&x_relation)?;

        // 5. Residual
        input.add(&x_relation)
    }
}

// ============================================================================
// LKPAN
// ============================================================================

/// Large Kernel PAN neck for text detection.
///
/// Takes multi-scale backbone features and produces a single fused feature map
/// by combining FPN top-down + PAN bottom-up paths with 9x9 convolutions.
///
/// Returns a single tensor [B, H, W, out_channels] where out_channels = 256.
pub struct LKPAN {
    /// 1x1 Conv2D projections (one per level) -- project to out_channels
    ins_convs: Vec<NativeConv2d>,
    /// 9x9 Conv2D reducing out_channels -> out_channels/4 (one per level)
    inp_convs: Vec<NativeConv2d>,
    /// 3x3 stride-2 Conv2D for PAN downsampling (n_levels - 1)
    pan_head_convs: Vec<NativeConv2d>,
    /// 9x9 Conv2D for PAN lateral output (one per level)
    pan_lat_convs: Vec<NativeConv2d>,
    /// Optional IntraCL blocks (one per level), applied after pan_lat_conv
    intracl_blocks: Option<Vec<IntraCLBlock>>,
}

impl LKPAN {
    pub fn new(
        ins_convs: Vec<NativeConv2d>,
        inp_convs: Vec<NativeConv2d>,
        pan_head_convs: Vec<NativeConv2d>,
        pan_lat_convs: Vec<NativeConv2d>,
    ) -> Self {
        Self {
            ins_convs,
            inp_convs,
            pan_head_convs,
            pan_lat_convs,
            intracl_blocks: None,
        }
    }

    /// Create LKPAN with IntraCL blocks.
    pub fn with_intracl(
        ins_convs: Vec<NativeConv2d>,
        inp_convs: Vec<NativeConv2d>,
        pan_head_convs: Vec<NativeConv2d>,
        pan_lat_convs: Vec<NativeConv2d>,
        intracl_blocks: Vec<IntraCLBlock>,
    ) -> Self {
        Self {
            ins_convs,
            inp_convs,
            pan_head_convs,
            pan_lat_convs,
            intracl_blocks: Some(intracl_blocks),
        }
    }

    /// Forward pass matching PaddleOCR's LKPAN exactly.
    ///
    /// # Arguments
    /// * `features` - Vec of backbone feature maps [B, H_i, W_i, C_i] (NHWC),
    ///   ordered from finest (stage1) to coarsest (stage4)
    ///
    /// # Returns
    /// * Single fused feature map [B, H_finest, W_finest, out_channels]
    pub fn forward(&self, features: &[MxArray]) -> Result<MxArray> {
        let n = features.len();
        if n != 4 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("LKPAN expects exactly 4 feature levels, got {}", n),
            ));
        }

        // 1. ins_conv: project all levels to out_channels (256)
        let mut ins: Vec<MxArray> = Vec::with_capacity(n);
        for (i, feat) in features.iter().enumerate() {
            ins.push(self.ins_convs[i].forward(feat)?);
        }

        // 2. FPN top-down: from coarsest to finest
        // Indices: 0=finest (c2), 1=c3, 2=c4, 3=coarsest (c5)
        let out4 = ins[2].add(&upsample_nearest_to_match(&ins[3], &ins[2])?)?;
        let out3 = ins[1].add(&upsample_nearest_to_match(&out4, &ins[1])?)?;
        let out2 = ins[0].add(&upsample_nearest_to_match(&out3, &ins[0])?)?;

        // 3. inp_conv: reduce 256 -> 64 on each level
        // Note: coarsest level (in5) uses raw projection (no top-down fusion)
        let f5 = self.inp_convs[3].forward(&ins[3])?;
        let f4 = self.inp_convs[2].forward(&out4)?;
        let f3 = self.inp_convs[1].forward(&out3)?;
        let f2 = self.inp_convs[0].forward(&out2)?;

        // 4. PAN bottom-up: stride-2 downsample + add
        let pan3 = f3.add(&self.pan_head_convs[0].forward(&f2)?)?;
        let pan4 = f4.add(&self.pan_head_convs[1].forward(&pan3)?)?;
        let pan5 = f5.add(&self.pan_head_convs[2].forward(&pan4)?)?;

        // 5. pan_lat_conv: 9x9 lateral on each level
        let mut p2 = self.pan_lat_convs[0].forward(&f2)?;
        let mut p3 = self.pan_lat_convs[1].forward(&pan3)?;
        let mut p4 = self.pan_lat_convs[2].forward(&pan4)?;
        let mut p5 = self.pan_lat_convs[3].forward(&pan5)?;

        // 5.5. Optional IntraCL blocks
        // PaddleOCR ordering: incl4(p5), incl3(p4), incl2(p3), incl1(p2)
        if let Some(ref blocks) = self.intracl_blocks {
            p5 = blocks[3].forward(&p5)?;
            p4 = blocks[2].forward(&p4)?;
            p3 = blocks[1].forward(&p3)?;
            p2 = blocks[0].forward(&p2)?;
        }

        // 6. Upsample all to finest resolution and concatenate
        let p5_up = upsample_nearest_to_match(&p5, &p2)?;
        let p4_up = upsample_nearest_to_match(&p4, &p2)?;
        let p3_up = upsample_nearest_to_match(&p3, &p2)?;

        // Concat along channels: [p5, p4, p3, p2] = 4 x 64 = 256
        MxArray::concatenate_many(vec![&p5_up, &p4_up, &p3_up, &p2], Some(3))
    }
}

/// Upsample `src` using nearest neighbor to match the spatial dimensions of `target`.
fn upsample_nearest_to_match(src: &MxArray, target: &MxArray) -> Result<MxArray> {
    let target_shape = target.shape()?;
    let target_h = target_shape[1];
    let target_w = target_shape[2];

    let mut x = src.clone();
    loop {
        let shape = x.shape()?;
        let h = shape[1];
        let w = shape[2];
        if h >= target_h && w >= target_w {
            break;
        }
        // 2x nearest upsample
        let batch = shape[0];
        let c = shape[3];
        let expanded = x.reshape(&[batch, h, 1, w, 1, c])?;
        let upsampled = expanded.broadcast_to(&[batch, h, 2, w, 2, c])?;
        x = upsampled.reshape(&[batch, h * 2, w * 2, c])?;
    }

    // Slice to exact target size if we overshot
    let shape = x.shape()?;
    if shape[1] != target_h || shape[2] != target_w {
        x = x.slice(&[0, 0, 0, 0], &[shape[0], target_h, target_w, shape[3]])?;
    }

    Ok(x)
}
