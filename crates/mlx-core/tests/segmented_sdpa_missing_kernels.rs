//! A `paged_attn.metallib` without the prebuilt segmented SDPA kernels is a
//! packaging error, not "segmented SDPA unsupported here": the capability
//! probe returns -1 instead of 0 and the forward entry point returns null
//! instead of falling back to concatenated SDPA.
//!
//! This binary holds one test because the paged library is loaded once per
//! process: it points `MLX_PAGED_ATTN_METALLIB` at a metallib without the
//! bridge kernels (built here with `xcrun`) before anything loads it.
//!
//!   cargo test -p mlx-core --test segmented_sdpa_missing_kernels

#![cfg(target_os = "macos")]

use std::path::Path;
use std::process::Command;

use mlx_core::array::MxArray;

fn xcrun(args: &[&str], dir: &Path) {
    let status = Command::new("xcrun")
        .args(args)
        .current_dir(dir)
        .status()
        .expect("xcrun");
    assert!(status.success(), "xcrun {args:?} failed");
}

/// A valid metallib holding one unrelated kernel.
fn metallib_without_bridge_kernels(dir: &Path) -> std::path::PathBuf {
    std::fs::write(
        dir.join("other.metal"),
        "kernel void other(device float* x [[buffer(0)]]) { x[0] = 1.0f; }\n",
    )
    .expect("write other.metal");
    xcrun(
        &[
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "other.metal",
            "-o",
            "other.air",
        ],
        dir,
    );
    xcrun(
        &[
            "-sdk",
            "macosx",
            "metallib",
            "other.air",
            "-o",
            "other.metallib",
        ],
        dir,
    );
    dir.join("other.metallib")
}

fn bf16(shape: &[i64]) -> MxArray {
    let len = shape.iter().product::<i64>() as usize;
    MxArray::from_bfloat16(&vec![0x3f80u16; len], shape).expect("bf16 array")
}

fn metal_required() -> bool {
    std::env::var("MLX_TEST_REQUIRE_METAL").is_ok_and(|value| value.trim() == "1")
}

#[test]
fn missing_segmented_kernels_are_a_hard_error() {
    // Not `mlx_metal_is_available()`: a Metal build returns true without a
    // device, and the probe below reads a failed device init as 0.
    // SAFETY: nullary probe that constructs the device and catches internally.
    if unsafe { mlx_sys::mlx_gpu_architecture_gen() } <= 0 {
        assert!(
            !metal_required(),
            "MLX_TEST_REQUIRE_METAL=1 but no Metal device"
        );
        eprintln!("skipping: no Metal device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("segmented-missing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let metallib = metallib_without_bridge_kernels(&dir);
    // SAFETY: the only test in this binary; nothing has read the variable or
    // loaded the paged library yet.
    unsafe { std::env::set_var("MLX_PAGED_ATTN_METALLIB", &metallib) };

    // SAFETY: plain FFI call; catches internally.
    let max_q = unsafe { mlx_sys::mlx_segmented_sdpa_max_query_length(6) };
    assert_eq!(
        max_q, -1,
        "a missing segmented kernel must read as a packaging error (-1), not unsupported (0)"
    );

    let q = bf16(&[1, 6, 2, 256]);
    let pk = bf16(&[1, 1, 4, 256]);
    let pv = bf16(&[1, 1, 4, 256]);
    let nk = bf16(&[1, 1, 2, 256]);
    let nv = bf16(&[1, 1, 2, 256]);
    for causal in [false, true] {
        // SAFETY: valid array handles that outlive the call.
        let out = unsafe {
            mlx_sys::mlx_segmented_sdpa_forward(
                q.as_raw_ptr(),
                pk.as_raw_ptr(),
                pv.as_raw_ptr(),
                nk.as_raw_ptr(),
                nv.as_raw_ptr(),
                0.0625,
                causal,
            )
        };
        assert!(
            out.is_null(),
            "causal={causal}: segmented SDPA fell back to concatenated K/V with its kernels missing"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
