//! UVDoc Architecture Configuration

/// UVDoc configuration
pub struct UVDocConfig {
    /// Base filter count (default: 32)
    pub num_filter: i32,
    /// Kernel size for residual blocks (default: 5)
    pub kernel_size: i32,
    /// Input image size (width, height) per cv2.resize convention
    pub img_size: (u32, u32),
    /// Filter multipliers for each level: [1, 2, 4, 8, 16]
    pub map_num: [i32; 5],
    /// Block counts for each ResnetStraight layer (3 layers)
    pub block_nums: [usize; 3],
    /// Stride for each ResnetStraight layer (3 layers)
    pub stride: [i32; 3],
}

impl Default for UVDocConfig {
    fn default() -> Self {
        Self {
            num_filter: 32,
            kernel_size: 5,
            img_size: (488, 712),
            map_num: [1, 2, 4, 8, 16],
            block_nums: [3, 4, 6],
            stride: [1, 2, 2],
        }
    }
}
