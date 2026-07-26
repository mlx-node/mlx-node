#![cfg(target_os = "macos")]

//! Small, ownership-aware adapter over the generated `objc2-metal` bindings.
//!
//! This intentionally exposes only the compute and buffer operations used by
//! mlx-node. Keeping the compatibility layer here lets the paged-attention
//! control flow stay unchanged while avoiding the deprecated `metal-rs`,
//! `objc`, and `block` crates.

use std::ffi::c_void;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

use foreign_types::ForeignTypeRef;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFunction,
    MTLLibrary,
};

pub use objc2_metal::MTLResourceOptions;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Metal grid or threadgroup dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl MTLSize {
    #[inline]
    pub const fn new(width: u64, height: u64, depth: u64) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    #[inline]
    fn into_objc(self) -> objc2_metal::MTLSize {
        objc2_metal::MTLSize {
            width: self.width as usize,
            height: self.height as usize,
            depth: self.depth as usize,
        }
    }
}

type ObjcBuffer = ProtocolObject<dyn MTLBuffer>;

mod private {
    pub trait Sealed {}
}

/// A scalar whose initialized object representation can be copied into a Metal buffer.
///
/// This trait is sealed so safe buffer creation cannot be extended to types
/// with padding or otherwise uninitialized bytes.
pub trait MetalBufferElement: private::Sealed {}

macro_rules! impl_metal_buffer_element {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl private::Sealed for $ty {}
            impl MetalBufferElement for $ty {}
        )+
    };
}

impl_metal_buffer_element!(u8, u16, u32, i32, i64, f32);

/// A borrowed `MTLBuffer` reference.
///
/// The transparent wrapper preserves the raw-pointer bridge used when MLX
/// lends its buffer to the Rust kernel dispatcher.
#[repr(transparent)]
pub struct BufferRef(ObjcBuffer);

unsafe impl ForeignTypeRef for BufferRef {
    type CType = AnyObject;
}

// SAFETY: Metal resources are designed for cross-thread submission and the
// wrapper exposes no unsynchronized CPU mutation beyond Metal's own API.
unsafe impl Send for BufferRef {}
unsafe impl Sync for BufferRef {}

impl BufferRef {
    #[inline]
    fn inner(&self) -> &ObjcBuffer {
        &self.0
    }

    #[inline]
    pub fn contents(&self) -> *mut c_void {
        self.0.contents().as_ptr()
    }

    #[inline]
    pub fn length(&self) -> u64 {
        self.0.length() as u64
    }
}

/// An owned retain on an `MTLBuffer`.
#[derive(Clone)]
pub struct Buffer(Retained<ObjcBuffer>);

// SAFETY: same as `BufferRef`; cloning only changes the Objective-C retain
// count, which is thread-safe.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    #[inline]
    pub fn as_ptr(&self) -> *mut AnyObject {
        Retained::as_ptr(&self.0).cast_mut().cast::<AnyObject>()
    }

    /// Retain a live, borrowed `MTLBuffer*` and return an owned wrapper.
    ///
    /// # Safety
    /// `ptr` must be a non-null, live `MTLBuffer*` for the duration of the
    /// retain operation.
    #[inline]
    pub unsafe fn retain_from_ptr(ptr: *mut c_void) -> Self {
        let ptr = ptr.cast::<ObjcBuffer>();
        // SAFETY: upheld by this method's caller. Taking an explicit +1 here
        // lets ordinary RAII release it after the synchronous dispatch.
        Self(unsafe { Retained::retain(ptr) }.expect("MTLBuffer pointer must not be null"))
    }
}

impl Deref for Buffer {
    type Target = BufferRef;

    fn deref(&self) -> &Self::Target {
        // SAFETY: both wrappers are transparent over the same Objective-C
        // object pointer and `Buffer` keeps the object retained.
        unsafe { &*(self.0.as_ref() as *const ObjcBuffer).cast::<BufferRef>() }
    }
}

/// Re-export the pointer traits used by existing MLX/Metal bridge code.
pub mod foreign_types {
    pub use foreign_types::ForeignTypeRef;
}

#[derive(Clone)]
pub struct Device(Retained<ProtocolObject<dyn MTLDevice>>);

impl Device {
    #[inline]
    pub fn system_default() -> Option<Self> {
        objc2_metal::MTLCreateSystemDefaultDevice().map(Self)
    }

    pub fn new_library_with_file(&self, path: &Path) -> Result<Library, String> {
        let path = path
            .to_str()
            .ok_or_else(|| format!("Metal library path is not UTF-8: {}", path.display()))?;
        let path = NSString::from_str(path);
        #[allow(deprecated)]
        self.0
            .newLibraryWithFile_error(&path)
            .map(Library)
            .map_err(error_string)
    }

    #[inline]
    pub fn new_command_queue(&self) -> CommandQueue {
        CommandQueue(
            self.0
                .newCommandQueue()
                .expect("Metal device returned no command queue"),
        )
    }

    #[inline]
    pub fn new_buffer(&self, length: u64, options: MTLResourceOptions) -> Buffer {
        Buffer(
            self.0
                .newBufferWithLength_options(length as usize, options)
                .expect("Metal device failed to allocate buffer"),
        )
    }

    /// Allocate a Metal buffer initialized from a slice of plain scalar values.
    #[inline]
    pub fn new_buffer_with_slice<T: MetalBufferElement>(
        &self,
        values: &[T],
        options: MTLResourceOptions,
    ) -> Buffer {
        // SAFETY: the sealed element set has no padding or uninitialized bytes,
        // and the slice remains readable for its complete inferred byte length
        // until Metal's synchronous copy returns.
        unsafe {
            self.new_buffer_with_data(
                values.as_ptr().cast(),
                std::mem::size_of_val(values) as u64,
                options,
            )
        }
    }

    /// Allocate a Metal buffer initialized from one plain scalar value.
    #[inline]
    pub fn new_buffer_with_value<T: MetalBufferElement>(
        &self,
        value: &T,
        options: MTLResourceOptions,
    ) -> Buffer {
        self.new_buffer_with_slice(std::slice::from_ref(value), options)
    }

    /// Allocate a Metal buffer and copy `length` bytes from `bytes`.
    ///
    /// # Safety
    ///
    /// `bytes` must be non-null and point to at least `length` readable bytes.
    /// The region must remain valid until this method returns; Metal copies it
    /// synchronously and does not retain the source pointer.
    #[inline]
    pub unsafe fn new_buffer_with_data(
        &self,
        bytes: *const c_void,
        length: u64,
        options: MTLResourceOptions,
    ) -> Buffer {
        let bytes = NonNull::new(bytes.cast_mut()).expect("buffer source pointer must not be null");
        let length = usize::try_from(length).expect("Metal buffer length does not fit usize");
        // SAFETY: upheld by this method's caller; Metal copies the source
        // region before returning.
        Buffer(
            unsafe {
                self.0
                    .newBufferWithBytes_length_options(bytes, length, options)
            }
            .expect("Metal device failed to allocate initialized buffer"),
        )
    }

    #[inline]
    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipelineState, String> {
        self.0
            .newComputePipelineStateWithFunction_error(&function.0)
            .map(ComputePipelineState)
            .map_err(error_string)
    }

    #[inline]
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.0.recommendedMaxWorkingSetSize()
    }

    #[inline]
    pub fn current_allocated_size(&self) -> u64 {
        self.0.currentAllocatedSize() as u64
    }
}

fn error_string(error: Retained<NSError>) -> String {
    error.localizedDescription().to_string()
}

#[derive(Clone)]
pub struct Library(Retained<ProtocolObject<dyn MTLLibrary>>);

impl Library {
    pub fn get_function(&self, name: &str, _constants: Option<()>) -> Result<Function, String> {
        let name = NSString::from_str(name);
        self.0
            .newFunctionWithName(&name)
            .map(Function)
            .ok_or_else(|| format!("Metal function `{name}` was not found"))
    }
}

pub struct Function(Retained<ProtocolObject<dyn MTLFunction>>);

#[derive(Clone)]
pub struct ComputePipelineState(Retained<ProtocolObject<dyn MTLComputePipelineState>>);

impl ComputePipelineState {
    #[inline]
    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.0.maxTotalThreadsPerThreadgroup() as u64
    }
}

#[derive(Clone)]
pub struct CommandQueue(Retained<ProtocolObject<dyn MTLCommandQueue>>);

impl CommandQueue {
    #[inline]
    pub fn new_command_buffer(&self) -> CommandBuffer {
        CommandBuffer(
            self.0
                .commandBuffer()
                .expect("Metal command queue returned no command buffer"),
        )
    }
}

pub struct CommandBuffer(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

impl CommandBuffer {
    #[inline]
    pub fn new_compute_command_encoder(&self) -> ComputeCommandEncoder {
        ComputeCommandEncoder(
            self.0
                .computeCommandEncoder()
                .expect("Metal command buffer returned no compute encoder"),
        )
    }

    #[inline]
    pub fn new_blit_command_encoder(&self) -> BlitCommandEncoder {
        BlitCommandEncoder(
            self.0
                .blitCommandEncoder()
                .expect("Metal command buffer returned no blit encoder"),
        )
    }

    #[inline]
    pub fn commit(&self) {
        self.0.commit();
    }

    #[inline]
    pub fn wait_until_completed(&self) {
        self.0.waitUntilCompleted();
    }

    /// Lifecycle stage this buffer is in. After `commit` + `wait_until_completed`
    /// Metal leaves it at either [`CommandBufferStatus::Completed`] or
    /// [`CommandBufferStatus::Error`]; anything else means the buffer never ran
    /// the work the caller submitted.
    #[inline]
    pub fn status(&self) -> CommandBufferStatus {
        CommandBufferStatus::from_raw(self.0.status())
    }

    /// The `NSError` Metal attached when execution aborted, if any. Metal
    /// leaves this nil for a buffer that completed, and may also leave it nil
    /// for an aborted one, so absence of an error is not evidence of success —
    /// read [`Self::status`] for that.
    #[inline]
    pub fn error(&self) -> Option<String> {
        self.0.error().map(error_string)
    }
}

/// Owned mirror of `MTLCommandBufferStatus`.
///
/// `Unknown` carries the raw value so a future Metal release that adds a stage
/// is reported rather than silently mapped onto an existing one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
    Unknown(u64),
}

impl CommandBufferStatus {
    #[inline]
    fn from_raw(status: MTLCommandBufferStatus) -> Self {
        match status {
            MTLCommandBufferStatus::NotEnqueued => Self::NotEnqueued,
            MTLCommandBufferStatus::Enqueued => Self::Enqueued,
            MTLCommandBufferStatus::Committed => Self::Committed,
            MTLCommandBufferStatus::Scheduled => Self::Scheduled,
            MTLCommandBufferStatus::Completed => Self::Completed,
            MTLCommandBufferStatus::Error => Self::Error,
            other => Self::Unknown(other.0 as u64),
        }
    }
}

/// Decide whether a submitted command buffer actually ran its work.
///
/// Only [`CommandBufferStatus::Completed`] counts as success. Every other
/// stage — including an `Error` for which Metal attached no `NSError` — is a
/// failure, because the GPU work either aborted or never reached the device.
/// Callers that skip this check cannot tell an aborted blit from a successful
/// one: the destination simply keeps whatever bytes it already held.
///
/// `context` names the submitting operation so the message identifies which
/// transfer failed.
pub fn command_buffer_outcome(
    status: CommandBufferStatus,
    error: Option<String>,
    context: &str,
) -> Result<(), String> {
    if status == CommandBufferStatus::Completed {
        return Ok(());
    }
    Err(format!(
        "{context}: Metal command buffer did not complete (status {status:?}): {}",
        error.as_deref().unwrap_or("no NSError reported")
    ))
}

pub struct ComputeCommandEncoder(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>);

impl ComputeCommandEncoder {
    #[inline]
    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipelineState) {
        self.0.setComputePipelineState(&pipeline.0);
    }

    #[inline]
    pub fn set_buffer(&self, index: u64, buffer: Option<&BufferRef>, offset: u64) {
        // SAFETY: command encoding is synchronous; the caller keeps each
        // buffer alive through command-buffer completion and supplies valid
        // offsets/indices for its kernel signature.
        unsafe {
            self.0.setBuffer_offset_atIndex(
                buffer.map(BufferRef::inner),
                offset as usize,
                index as usize,
            );
        }
    }

    #[inline]
    pub fn set_threadgroup_memory_length(&self, index: u64, length: u64) {
        // SAFETY: the kernel defines the matching dynamic threadgroup slot.
        unsafe {
            self.0
                .setThreadgroupMemoryLength_atIndex(length as usize, index as usize);
        }
    }

    #[inline]
    pub fn dispatch_thread_groups(&self, groups: MTLSize, threads: MTLSize) {
        self.0
            .dispatchThreadgroups_threadsPerThreadgroup(groups.into_objc(), threads.into_objc());
    }

    #[inline]
    pub fn end_encoding(&self) {
        self.0.endEncoding();
    }
}

pub struct BlitCommandEncoder(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>);

impl BlitCommandEncoder {
    #[inline]
    pub fn copy_from_buffer(
        &self,
        source: &BufferRef,
        source_offset: u64,
        destination: &BufferRef,
        destination_offset: u64,
        size: u64,
    ) {
        // SAFETY: callers validate buffer sizes and keep both resources alive
        // until the enclosing command buffer completes.
        unsafe {
            self.0
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source.inner(),
                    source_offset as usize,
                    destination.inner(),
                    destination_offset as usize,
                    size as usize,
                );
        }
    }

    #[inline]
    pub fn end_encoding(&self) {
        self.0.endEncoding();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn retained_resources_are_send_sync() {
        assert_send_sync::<Buffer>();
        assert_send_sync::<Device>();
        assert_send_sync::<Library>();
        assert_send_sync::<ComputePipelineState>();
        assert_send_sync::<CommandQueue>();
    }

    #[test]
    fn initialized_buffer_helpers_copy_exact_values() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping initialized buffer copy test: no Metal device");
            return;
        };

        let values = [0x1020_3040u32, 0x5060_7080, 0x90a0_b0c0];
        let slice_buffer =
            device.new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared);
        assert_eq!(slice_buffer.length(), std::mem::size_of_val(&values) as u64);
        // SAFETY: StorageModeShared exposes CPU-readable contents, and the
        // constructor allocated exactly `values.len()` initialized `u32`s.
        let copied = unsafe {
            std::slice::from_raw_parts(slice_buffer.contents().cast::<u32>(), values.len())
        };
        assert_eq!(copied, values);

        let value = -1234i32;
        let value_buffer =
            device.new_buffer_with_value(&value, MTLResourceOptions::StorageModeShared);
        assert_eq!(value_buffer.length(), std::mem::size_of::<i32>() as u64);
        // SAFETY: same StorageModeShared and initialized-length guarantees as
        // above, for one `i32`.
        assert_eq!(unsafe { *value_buffer.contents().cast::<i32>() }, value);
    }

    /// The success predicate must be "status is Completed", not "no NSError"
    /// and not "status is Error". Metal can abort a buffer without attaching
    /// an `NSError`, and a buffer stuck at `Scheduled` or at an
    /// unrecognized future stage never ran the submitted work either. Both of
    /// those would be read as success by the looser predicates.
    #[test]
    fn only_a_completed_command_buffer_is_success() {
        assert!(command_buffer_outcome(CommandBufferStatus::Completed, None, "ctx").is_ok());
        assert!(
            command_buffer_outcome(
                CommandBufferStatus::Error,
                Some("Ignored (for causing prior/excessive GPU errors)".to_string()),
                "ctx",
            )
            .is_err()
        );
        // An aborted buffer with a nil `error` is the case a `error.is_some()`
        // predicate would wave through.
        assert!(command_buffer_outcome(CommandBufferStatus::Error, None, "ctx").is_err());
        for status in [
            CommandBufferStatus::NotEnqueued,
            CommandBufferStatus::Enqueued,
            CommandBufferStatus::Committed,
            CommandBufferStatus::Scheduled,
            CommandBufferStatus::Unknown(9),
        ] {
            assert!(
                command_buffer_outcome(status, None, "ctx").is_err(),
                "{status:?} must not be reported as success"
            );
        }
    }

    /// A failure has to say which submission failed and what Metal reported,
    /// otherwise an operator sees an unattributed error from a process that
    /// submits command buffers from a dozen places.
    #[test]
    fn failure_message_names_the_context_and_the_metal_error() {
        let message = command_buffer_outcome(
            CommandBufferStatus::Error,
            Some("device removed".to_string()),
            "LayerKVPool::write_block_all_layers",
        )
        .expect_err("an errored buffer must not be Ok");
        assert!(
            message.contains("LayerKVPool::write_block_all_layers"),
            "message must name the caller: {message}"
        );
        assert!(
            message.contains("device removed"),
            "message must carry the NSError text: {message}"
        );

        let message = command_buffer_outcome(CommandBufferStatus::Scheduled, None, "ctx")
            .expect_err("a non-completed buffer must not be Ok");
        assert!(
            message.contains("Scheduled"),
            "message must name the observed status: {message}"
        );
        assert!(
            message.contains("no NSError"),
            "message must say the error was absent rather than omit it: {message}"
        );
    }

    /// `status()` on a healthy submission must be `Completed`, so the check
    /// added at every commit/wait site cannot reject working GPU transfers.
    #[test]
    fn healthy_submission_reports_completed() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping healthy submission status test: no Metal device");
            return;
        };
        let queue = device.new_command_queue();
        let source =
            device.new_buffer_with_slice(&[1u32, 2, 3, 4], MTLResourceOptions::StorageModeShared);
        let destination = device.new_buffer(16, MTLResourceOptions::StorageModeShared);

        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&source, 0, &destination, 0, 16);
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        assert!(
            command_buffer_outcome(command_buffer.status(), command_buffer.error(), "healthy")
                .is_ok()
        );
    }
}
