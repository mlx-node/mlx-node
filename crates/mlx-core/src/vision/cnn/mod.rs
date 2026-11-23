//! Shared CNN Building Blocks, HGNetV2 Backbone, Persistence, and Preprocessing
//!
//! Used by the Paddle document analysis models (`pp_doclayout_v3`, `pp_doc_ori`,
//! `pp_doc_unwarp`, `pp_text_det`, `pp_text_rec`) and shared tensor loaders.

pub mod hgnetv2;
pub mod layers;
pub mod persistence;
pub mod processing;

pub use hgnetv2::{
    HGBlock, HGBlockLayer, HGNetV2Backbone, HGNetV2Config, HGNetV2Embeddings, HGNetV2Encoder,
    HGNetV2Stage,
};
pub use layers::{
    ConvBNAct, ConvBNActLight, FrozenBatchNorm2d, LearnableAffineBlock, MaxPool2d, NativeConv2d,
    NativeConvTranspose2d, apply_activation,
};
pub(crate) use persistence::{
    get_tensor, load_conv_bn_act, load_conv_bn_act_with_options, load_conv_transpose2d,
    load_conv2d, load_frozen_bn, load_hgnetv2_backbone, load_layer_norm, load_linear,
};
pub use processing::rgb_to_nhwc_f32;
