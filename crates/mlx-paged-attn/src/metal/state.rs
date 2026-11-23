//! Metal state management - device, residency library, pipeline states.
//!
//! Paged-attention and KV-write kernels live in `mlx-sys`'s `paged_attn.metallib`.
//! `MetalState` keeps the shared Metal device, command queue, and the tiny
//! embedded `residency_kick` metallib used by [`crate::residency`].

use metal::{ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// Embedded metallib bytes (compiled at build time)
#[cfg(mlx_node_metal_enabled)]
const METALLIB_BYTES: &[u8] = include_bytes!(env!("PAGED_ATTN_METALLIB"));

/// Global Metal state singleton
static METAL_STATE: OnceLock<Result<MetalState, String>> = OnceLock::new();

/// Cached Metal resources for paged attention
pub struct MetalState {
    /// Metal device
    pub device: Device,
    /// Compiled Metal library
    pub library: Library,
    /// Reusable command queue (avoids per-dispatch allocation)
    pub command_queue: metal::CommandQueue,
    /// Pipeline states keyed by kernel name (uses RwLock for interior mutability)
    pipelines: RwLock<HashMap<String, ComputePipelineState>>,
}

impl MetalState {
    /// Get the global Metal state singleton
    pub fn get() -> Result<&'static MetalState, String> {
        METAL_STATE
            .get_or_init(Self::init)
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// Initialize Metal state
    fn init() -> Result<MetalState, String> {
        #[cfg(not(mlx_node_metal_enabled))]
        return Err("No Metal device found: Metal backend disabled at build time".to_string());

        #[cfg(mlx_node_metal_enabled)]
        {
            let device = Device::system_default().ok_or("No Metal device found")?;

            let library = device
                .new_library_with_data(METALLIB_BYTES)
                .map_err(|e| format!("Failed to load metallib: {}", e))?;

            let command_queue = device
                .try_new_command_queue()
                .ok_or_else(|| "Metal device returned no command queue".to_string())?;

            Ok(MetalState {
                device,
                library,
                command_queue,
                pipelines: RwLock::new(HashMap::new()),
            })
        }
    }

    /// Get or create a compute pipeline for a kernel
    ///
    /// FORKED: Kernels are now template-specialized, so we just load by name
    /// without needing to set function constants. The kernel name encodes
    /// all the options (e.g. the _alibi suffix).
    ///
    /// Pipelines are cached for performance - subsequent calls with the same
    /// kernel name return the cached pipeline.
    pub fn get_pipeline(&self, kernel_name: &str) -> Result<ComputePipelineState, String> {
        // Check cache first (read lock)
        {
            let cache = self
                .pipelines
                .read()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            if let Some(pipeline) = cache.get(kernel_name) {
                return Ok(pipeline.clone());
            }
        }

        // Cache miss - create new pipeline (write lock)
        let mut cache = self
            .pipelines
            .write()
            .map_err(|e| format!("Lock poisoned: {}", e))?;

        // Double-check after acquiring write lock (another thread may have inserted)
        if let Some(pipeline) = cache.get(kernel_name) {
            return Ok(pipeline.clone());
        }

        // FORKED: No function constants needed - kernels are template-specialized
        let function = self
            .library
            .get_function(kernel_name, None)
            .map_err(|e| format!("Kernel '{}' not found: {}", kernel_name, e))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Failed to create pipeline for '{}': {}", kernel_name, e))?;

        // Insert into cache
        cache.insert(kernel_name.to_string(), pipeline.clone());

        Ok(pipeline)
    }
}

/// Data types supported by Metal kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDtype {
    Float32,
    Float16,
    BFloat16,
    /// Splash `kv::Format::Int8`: symmetric int8 with one f32 scale per
    /// token, head and side (`splash_store_kv_row`).
    Int8,
}

impl MetalDtype {
    /// Get the Metal kernel type string
    pub fn type_string(&self) -> &'static str {
        match self {
            MetalDtype::Float32 => "float",
            MetalDtype::Float16 => "half",
            MetalDtype::BFloat16 => "bfloat16_t",
            MetalDtype::Int8 => "char",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            MetalDtype::Float32 => 4,
            MetalDtype::Float16 | MetalDtype::BFloat16 => 2,
            MetalDtype::Int8 => 1,
        }
    }

    /// Dtype of the rows written into and read out of a cache of this
    /// dtype. Splash io is always bf16 for an int8 cache.
    pub fn io_dtype(&self) -> MetalDtype {
        match self {
            MetalDtype::Int8 => MetalDtype::BFloat16,
            other => *other,
        }
    }

    /// Scale bytes per token, head and side (Splash `kv::Layout`
    /// `bytesPerLayerPage`: an f32 per row for int8, none otherwise).
    pub fn scale_bytes(&self) -> usize {
        match self {
            MetalDtype::Int8 => 4,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_state_init() {
        match MetalState::get() {
            Ok(_) => {}
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_metal_state_init: {e}");
            }
            Err(e) => panic!("unexpected MetalState::get failure: {e}"),
        }
    }

    const TEST_CHILD: &str = "MLX_PAGED_ATTN_TEST_CHILD";

    /// This process is the child [`run_child_with_tmpdir`] started for `name`.
    fn is_child(name: &str) -> bool {
        std::env::var(TEST_CHILD).is_ok_and(|child| child == name)
    }

    /// Runs test `name` alone in a child test process whose `TMPDIR` is
    /// `tmpdir`. `MetalState` is a process-wide singleton, so only a fresh
    /// process loads the library again.
    fn run_child_with_tmpdir(name: &str, tmpdir: &std::path::Path) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--test-threads=1"])
            .env(TEST_CHILD, name)
            .env("TMPDIR", tmpdir)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed"),
            "{name} in its own process:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mlx-paged-attn-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Loads the library and builds one pipeline; `false` on a host with no
    /// Metal device.
    fn load_and_build(test: &str) -> bool {
        let state = match MetalState::get() {
            Ok(state) => state,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping {test}: {e}");
                return false;
            }
            Err(e) => panic!("{test}: MetalState::get: {e}"),
        };
        if let Err(e) = state.get_pipeline("residency_kick") {
            panic!("{test}: residency_kick: {e}");
        }
        true
    }

    /// The library loads with a temp dir that does not exist: init needs no
    /// file.
    #[test]
    fn metal_state_needs_no_temp_dir() {
        const NAME: &str = "metal::state::tests::metal_state_needs_no_temp_dir";
        if is_child(NAME) {
            let tmpdir = std::env::temp_dir();
            assert!(!tmpdir.exists(), "{} exists", tmpdir.display());
            load_and_build(NAME);
            return;
        }
        let parent = scratch_dir("no-tmp");
        run_child_with_tmpdir(NAME, &parent.join("missing"));
        let _ = std::fs::remove_dir_all(&parent);
    }

    /// Init writes nothing into the temp dir.
    #[test]
    fn metal_state_writes_no_file() {
        const NAME: &str = "metal::state::tests::metal_state_writes_no_file";
        if is_child(NAME) {
            let tmpdir = std::env::temp_dir();
            if load_and_build(NAME) {
                let left: Vec<_> = std::fs::read_dir(&tmpdir)
                    .unwrap()
                    .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
                    .collect();
                assert!(left.is_empty(), "init wrote into the temp dir: {left:?}");
            }
            return;
        }
        let dir = scratch_dir("empty-tmp");
        run_child_with_tmpdir(NAME, &dir);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
