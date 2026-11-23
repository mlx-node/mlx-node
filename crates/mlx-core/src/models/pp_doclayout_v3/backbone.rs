//! Re-exports of shared CNN / HGNetV2 backbone types from `crate::vision::cnn`.

pub use crate::vision::cnn::{
    ConvBNAct, ConvBNActLight, FrozenBatchNorm2d, HGBlock, HGBlockLayer, HGNetV2Backbone,
    HGNetV2Embeddings, HGNetV2Encoder, HGNetV2Stage, LearnableAffineBlock, MaxPool2d, NativeConv2d,
    NativeConvTranspose2d,
};
