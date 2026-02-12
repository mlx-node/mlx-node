//! SVTR Neck (EncoderWithSVTR)
//!
//! Matches PaddleOCR's EncoderWithSVTR from `ppocr/modeling/necks/rnn.py`.
//! Unlike the simpler pool→transformer approach, this keeps 2D spatial structure
//! throughout and uses conv bottlenecks for dimensionality reduction.
//!
//! Architecture (from PaddleOCR reference, kernel_size=[1,3] from YAML config):
//!   1. conv1: Conv2d(C → C/8, k=[1,3], pad=[0,1])      — channel reduction
//!   2. conv2: Conv2d(C/8 → hidden, k=1, pad=0)          — project to hidden_dims
//!   3. Flatten: [B, H, W, hidden] → [B, H*W, hidden]
//!   4. N transformer encoder blocks (pre-norm, Swish activation, mlp_ratio=2.0)
//!   5. Reshape: [B, H*W, hidden] → [B, H, W, hidden]
//!   6. conv3: Conv2d(hidden → C, k=1, pad=0)            — project back
//!   7. Concatenate conv3 output with shortcut (from input) → [B, H, W, 2C]
//!   8. conv4: Conv2d(2C → C/8, k=[1,3], pad=[0,1])     — reduce concatenated
//!   9. conv1x1: Conv2d(C/8 → dims, k=1, pad=0)         — final projection
//!      Output: [B, H, W, dims]
//!
//! The model.rs caller then pools height to get [B, W, dims] for CTC.

use crate::array::MxArray;
use crate::models::pp_doclayout_v3::backbone::{FrozenBatchNorm2d, NativeConv2d};
use crate::nn::activations::Activations;
use crate::nn::{LayerNorm, Linear};
use napi::bindgen_prelude::*;

// ============================================================================
// ConvBNLayer helper (conv + optional BN + optional activation)
// ============================================================================

/// Conv2d + BatchNorm layer used in SVTR neck bottleneck convolutions.
pub struct SVTRConvBN {
    conv: NativeConv2d,
    bn: FrozenBatchNorm2d,
    activation: String,
}

impl SVTRConvBN {
    pub fn new(conv: NativeConv2d, bn: FrozenBatchNorm2d, activation: &str) -> Self {
        Self {
            conv,
            bn,
            activation: activation.to_string(),
        }
    }

    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let x = self.conv.forward(input)?;
        let x = self.bn.forward(&x)?;
        match self.activation.as_str() {
            "swish" | "silu" => Activations::silu(&x),
            "relu" => Activations::relu(&x),
            "none" => Ok(x),
            _ => Activations::silu(&x),
        }
    }
}

// ============================================================================
// Pre-norm Transformer Encoder Layer
// ============================================================================

/// A single SVTR transformer encoder layer with **pre-norm**.
///
/// PaddleOCR's EncoderWithSVTR uses `Block(prenorm=False)` which applies
/// LayerNorm BEFORE the sublayer (attention/FFN), then adds the residual.
/// This is the standard pre-norm pattern.
///
/// Uses a FUSED QKV projection (single Linear projecting to 3*dim),
/// matching PaddleOCR's `Attention` class.
///
/// FFN uses Swish (SiLU) activation and mlp_ratio=2.0 by default.
pub struct SVTREncoderLayer {
    /// Fused Q, K, V projection: Linear(dim, dim * 3)
    qkv: Linear,
    /// Output projection: Linear(dim, dim)
    proj: Linear,
    num_heads: i32,
    head_dim: i32,
    /// LayerNorm before self-attention (pre-norm)
    norm1: LayerNorm,
    /// FFN
    fc1: Linear,
    fc2: Linear,
    /// LayerNorm before FFN (pre-norm)
    norm2: LayerNorm,
}

impl SVTREncoderLayer {
    pub fn new(
        qkv: Linear,
        proj: Linear,
        num_heads: i32,
        embed_dim: i32,
        norm1: LayerNorm,
        fc1: Linear,
        fc2: Linear,
        norm2: LayerNorm,
    ) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            norm1,
            fc1,
            fc2,
            norm2,
        }
    }

    /// Forward pass: pre-norm transformer encoder layer.
    ///
    /// PaddleOCR's Block with prenorm=False does:
    ///   x = x + mixer(norm1(x))
    ///   x = x + mlp(norm2(x))
    /// This is the standard pre-norm pattern (norm before sublayer).
    ///
    /// Uses fused QKV projection matching PaddleOCR's Attention class:
    ///   qkv = self.qkv(x).reshape(B, seq, 3, num_heads, head_dim).transpose(2,0,3,1,4)
    ///   q, k, v = qkv[0] * scale, qkv[1], qkv[2]
    ///
    /// Input/Output: [B, seq_len, hidden_dim]
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let shape = input.shape()?;
        let batch = shape[0];
        let seq_len = shape[1];
        let embed_dim = shape[2];

        // Pre-norm: LayerNorm BEFORE attention
        let normed = self.norm1.forward(input)?;

        // Fused QKV projection: [B, seq, dim] → [B, seq, 3*dim]
        let qkv = self.qkv.forward(&normed)?;

        let nh = self.num_heads as i64;
        let hd = self.head_dim as i64;

        // Reshape: [B, seq, 3*dim] → [B, seq, 3, nh, hd] → [3, B, nh, seq, hd]
        let qkv = qkv
            .reshape(&[batch, seq_len, 3, nh, hd])?
            .transpose(Some(&[2, 0, 3, 1, 4]))?;

        // Split into Q, K, V along the first dimension (size 3)
        // qkv shape: [3, B, nh, seq, hd]
        let q = qkv.slice_axis(0, 0, 1)?.squeeze(Some(&[0]))?; // [B, nh, seq, hd]
        let k = qkv.slice_axis(0, 1, 2)?.squeeze(Some(&[0]))?; // [B, nh, seq, hd]
        let v = qkv.slice_axis(0, 2, 3)?.squeeze(Some(&[0]))?; // [B, nh, seq, hd]

        // Scale Q
        let scaling = (self.head_dim as f64).powf(-0.5);
        let q = q.mul_scalar(scaling)?;

        // Attention: Q @ K^T → softmax → @ V
        let k_t = k.transpose(Some(&[0, 1, 3, 2]))?;
        let attn = q.matmul(&k_t)?;
        let attn = Activations::softmax(&attn, Some(-1))?;
        let attn_out = attn.matmul(&v)?;

        // Reshape back: [B, nh, seq, hd] → [B, seq, embed_dim]
        let attn_out = attn_out
            .transpose(Some(&[0, 2, 1, 3]))?
            .reshape(&[batch, seq_len, embed_dim])?;
        let attn_out = self.proj.forward(&attn_out)?;

        // Residual add to UNNORMALIZED input
        let x = input.add(&attn_out)?;

        // Pre-norm: LayerNorm BEFORE FFN
        let normed = self.norm2.forward(&x)?;
        let ff = self.fc1.forward(&normed)?;
        let ff = Activations::silu(&ff)?;
        let ff = self.fc2.forward(&ff)?;

        // Residual add
        x.add(&ff)
    }
}

// ============================================================================
// SVTRNeck (EncoderWithSVTR)
// ============================================================================

/// SVTR Neck matching PaddleOCR's EncoderWithSVTR architecture.
///
/// Keeps 2D spatial structure throughout, uses conv bottlenecks for
/// dimensionality reduction, and concatenates the shortcut with
/// transformer output.
pub struct SVTRNeck {
    /// Conv2d(C → C/8, k=[1,3], pad=[0,1]) — channel reduction
    conv1: SVTRConvBN,
    /// Conv2d(C/8 → hidden, k=1, pad=0) — project to hidden
    conv2: SVTRConvBN,
    /// Transformer encoder layers (pre-norm, Swish FFN)
    layers: Vec<SVTREncoderLayer>,
    /// Final LayerNorm applied after all transformer blocks (eps=1e-6)
    final_norm: LayerNorm,
    /// Conv2d(hidden → C, k=1, pad=0) — project back
    conv3: SVTRConvBN,
    /// Conv2d(2C → C/8, k=[1,3], pad=[0,1]) — reduce concatenated
    conv4: SVTRConvBN,
    /// Conv2d(C/8 → dims, k=1, pad=0) — final projection to output dims
    conv1x1: SVTRConvBN,
}

impl SVTRNeck {
    pub fn new(
        conv1: SVTRConvBN,
        conv2: SVTRConvBN,
        layers: Vec<SVTREncoderLayer>,
        final_norm: LayerNorm,
        conv3: SVTRConvBN,
        conv4: SVTRConvBN,
        conv1x1: SVTRConvBN,
    ) -> Self {
        Self {
            conv1,
            conv2,
            layers,
            final_norm,
            conv3,
            conv4,
            conv1x1,
        }
    }

    /// Forward pass.
    ///
    /// Input: [B, H, W, C] from backbone (NHWC)
    /// Output: [B, H, W, dims] spatial features
    ///
    /// The caller (model.rs) pools height to get [B, W, dims] for CTC.
    pub fn forward(&self, input: &MxArray) -> Result<MxArray> {
        let shape = input.shape()?;
        let batch = shape[0];
        let h = shape[1];
        let w = shape[2];

        // Save shortcut for later concatenation
        let shortcut = input.clone();

        // Step 1-2: Conv bottleneck to reduce to hidden dims
        let x = self.conv1.forward(input)?;
        let x = self.conv2.forward(&x)?;

        // Step 3: Flatten spatial dims for transformer
        let hidden_shape = x.shape()?;
        let hidden_dim = hidden_shape[3];
        let x = x.reshape(&[batch, h * w, hidden_dim])?;

        // Step 4: Transformer encoder layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Step 4.5: Final LayerNorm after all transformer blocks
        let x = self.final_norm.forward(&x)?;

        // Step 5: Reshape back to 2D spatial
        let x = x.reshape(&[batch, h, w, hidden_dim])?;

        // Step 6: Project back to original channel count
        let x = self.conv3.forward(&x)?;

        // Step 7: Concatenate shortcut (first) with conv3 output (second) along channels
        // PaddleOCR: z = paddle.concat((h, z), axis=1) — h=shortcut, z=conv3 output
        // In NHWC, axis=3 is the channel dimension
        let x = MxArray::concatenate(&shortcut, &x, 3)?;

        // Step 8-9: Reduce and project to output dims
        let x = self.conv4.forward(&x)?;
        self.conv1x1.forward(&x)
    }
}
