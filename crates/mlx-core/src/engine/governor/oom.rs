//! A GPU command that exhausted Metal memory (Splash `metal_out_of_memory`).
//!
//! MLX keeps a command buffer's error in its completion handler and throws
//! it at the next encoder or synchronize, so the error reaches whatever ran
//! next, not the command that failed.

/// MLX's texts for an allocation or a command that ran out of Metal memory:
/// `allocator.cpp`'s two `malloc` refusals and `device.cpp`'s command-buffer
/// failure with `MTLCommandBufferErrorOutOfMemory`'s description.
pub(crate) fn is_gpu_out_of_memory(text: &str) -> bool {
    text.contains("[malloc] Unable to allocate")
        || text.contains("[metal::malloc] Resource limit")
        || (text.contains("[METAL] Command buffer execution failed")
            && (text.contains("Insufficient Memory") || text.contains("OutOfMemory")))
}

/// The error every request the reset fails ends with. "out of memory" is
/// absent: the TS layer reads that phrase as a load refusal.
pub(crate) fn metal_out_of_memory(detail: &str) -> String {
    format!(
        "metal_out_of_memory: a GPU command exhausted Metal memory ({detail}); the request was \
         not at fault; retry"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutation O2 (any command-buffer failure matches) → red (the invalid
    /// resource failure is read as memory).
    #[test]
    fn gpu_oom_text_is_recognized() {
        for text in [
            "[malloc] Unable to allocate 1073741824 bytes.",
            "[metal::malloc] Resource limit (499000) exceeded.",
            "[METAL] Command buffer execution failed: Insufficient Memory \
             (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).",
        ] {
            assert!(is_gpu_out_of_memory(text), "{text}");
        }
        assert!(!is_gpu_out_of_memory(
            "[METAL] Command buffer execution failed: Invalid Resource \
             (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)."
        ));
        assert!(metal_out_of_memory("x").starts_with("metal_out_of_memory: "));
        assert!(!metal_out_of_memory("x").contains("out of memory"));
    }
}
