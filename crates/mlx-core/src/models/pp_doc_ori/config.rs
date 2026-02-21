//! PP-LCNet_x1_0 Architecture Configuration
//!
//! PP-LCNet uses depthwise separable convolutions organized into block groups.
//! With scale=1.0, channels are used directly without scaling.
//!
//! NET_CONFIG from PaddleOCR's det_pp_lcnet.py:
//! Each entry: [kernel_size, in_channels, out_channels, stride, use_se]

/// Block configuration: (kernel_size, in_channels, out_channels, stride, use_se)
pub type BlockConfig = (i32, i32, i32, i32, bool);

/// PP-LCNet architecture configuration for scale=1.0
pub struct PPLCNetConfig {
    /// Number of output classes (4 for orientation: 0/90/180/270)
    pub num_classes: i32,
    /// Initial conv output channels (before blocks)
    pub stem_channels: i32,
    /// Block group configurations
    pub blocks: Vec<(&'static str, Vec<BlockConfig>)>,
}

impl Default for PPLCNetConfig {
    fn default() -> Self {
        Self {
            num_classes: 4,
            stem_channels: 16,
            blocks: vec![
                // blocks2: k, in_c, out_c, stride, use_se
                ("blocks2", vec![(3, 16, 32, 1, false)]),
                // blocks3
                (
                    "blocks3",
                    vec![(3, 32, 64, 2, false), (3, 64, 64, 1, false)],
                ),
                // blocks4
                (
                    "blocks4",
                    vec![(3, 64, 128, 2, false), (3, 128, 128, 1, false)],
                ),
                // blocks5
                (
                    "blocks5",
                    vec![
                        (3, 128, 256, 2, false),
                        (5, 256, 256, 1, false),
                        (5, 256, 256, 1, false),
                        (5, 256, 256, 1, false),
                        (5, 256, 256, 1, false),
                        (5, 256, 256, 1, false),
                    ],
                ),
                // blocks6 - SE modules enabled
                (
                    "blocks6",
                    vec![(5, 256, 512, 2, true), (5, 512, 512, 1, true)],
                ),
            ],
        }
    }
}

impl PPLCNetConfig {
    /// Get the final output channels (last block's out_channels)
    pub fn head_in_channels(&self) -> i32 {
        self.blocks
            .last()
            .and_then(|(_, blocks)| blocks.last())
            .map(|b| b.2)
            .unwrap_or(512)
    }
}
