use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=metal/");
    println!("cargo:rerun-if-env-changed=MLX_DISABLE_METAL");
    // The metallib's min-OS flag reads this too — same declaration mlx-sys
    // makes, so a changed target recompiles the standalone library as well.
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rustc-check-cfg=cfg(mlx_node_metal_enabled)");

    // A build script runs for the host, so compile-time `cfg(target_os)` would
    // misroute cross-target builds. Match mlx-sys's effective backend decision
    // using Cargo's target metadata and the shared CPU-only escape hatch.
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is not set");
    let build_metal = target_os == "macos" && env::var_os("MLX_DISABLE_METAL").is_none();
    if !build_metal {
        println!("cargo:rustc-env=PAGED_ATTN_METALLIB=");
        return;
    }

    println!("cargo:rustc-cfg=mlx_node_metal_enabled");
    compile_metal_shaders();
}

/// Deployment-target floor for the metallib when `MACOSX_DEPLOYMENT_TARGET`
/// is unset — kept equal to `crates/mlx-sys/build.rs`'s constant.
///
/// Without a floor the metal compiler targets the BUILD HOST: a build on
/// macOS 27 emits `air64_v29` ("language version 4.1") shaders that a macOS
/// 26 host cannot load — the paged-attn library is loaded lazily (first KV
/// pool construction), so the failure surfaces as an HTTP 500 from the
/// server long after the app itself started. Measured: default
/// `xcrun metal` = `air64_v29-apple-macosx27.0.0`,
/// `-mmacosx-version-min=26.0` = `air64_v28-apple-macosx26.0.0`.
const MACOS_DEPLOYMENT_TARGET_FLOOR: &str = "26.0";

/// The build host's macOS version as `(major, minor)`, via `sw_vers`.
fn host_macos_version() -> Option<(u64, u64)> {
    // The absolute path survives build environments with a stripped PATH —
    // a PATH lookup that fails here would silently drop the floor back to
    // the toolchain default (the air64_v29 problem above).
    let output = Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut parts = text.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|part| part.parse().ok()).unwrap_or(0);
    Some((major, minor))
}

/// The floor applied when `MACOSX_DEPLOYMENT_TARGET` is unset: the project
/// floor, never ABOVE the build host's own version — pinning 26.0 on macOS
/// 14 or 15 (where a local source build is documented to work) would emit a
/// metallib the host cannot load and an older SDK can reject. The floor only
/// matters on hosts newer than it, so the unset fallback caps at the host
/// version — the same rule `crates/mlx-sys/build.rs` applies.
fn default_macos_deployment_target() -> Option<String> {
    let (major, minor) = host_macos_version()?;
    if (major, minor) > (26, 0) {
        Some(MACOS_DEPLOYMENT_TARGET_FLOOR.to_string())
    } else {
        Some(format!("{major}.{minor}"))
    }
}

fn macos_deployment_target() -> Option<String> {
    env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(default_macos_deployment_target)
}

fn compile_metal_shaders() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let metal_dir = PathBuf::from(&manifest_dir).join("metal");
    let deployment_target = macos_deployment_target();

    // Metal source files to compile
    let metal_files = [
        "attention/paged_attention.metal",
        "cache/reshape_and_cache.metal",
        "cache/copy_blocks.metal",
    ];

    let mut air_files = Vec::new();

    // Compile each Metal file to AIR (Apple Intermediate Representation)
    for file in &metal_files {
        let src_path = metal_dir.join(file);
        let air_name = file.replace('/', "_").replace(".metal", ".air");
        let air_path = PathBuf::from(&out_dir).join(&air_name);

        println!("cargo:rerun-if-changed={}", src_path.display());

        // Compile Metal to AIR. The min-OS flag pins the AIR language
        // version to the deployment floor — see MACOS_DEPLOYMENT_TARGET_FLOOR.
        let mut compile_cmd = Command::new("xcrun");
        compile_cmd.args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            src_path.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
            // Include path for header files
            "-I",
            metal_dir.to_str().unwrap(),
            // Optimization flags
            "-O3",
            "-ffast-math",
        ]);
        if let Some(target) = &deployment_target {
            compile_cmd.arg(format!("-mmacosx-version-min={target}"));
        }
        let status = compile_cmd.status().expect("Failed to execute xcrun metal");

        if !status.success() {
            panic!(
                "Metal compilation failed for {}: exit code {:?}",
                file,
                status.code()
            );
        }

        air_files.push(air_path);
    }

    // Link all AIR files into a single metallib
    let metallib_path = PathBuf::from(&out_dir).join("paged_attn.metallib");

    let mut link_cmd = Command::new("xcrun");
    link_cmd.args(["-sdk", "macosx", "metallib"]);

    for air in &air_files {
        link_cmd.arg(air.to_str().unwrap());
    }

    link_cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let status = link_cmd.status().expect("Failed to execute xcrun metallib");

    if !status.success() {
        panic!(
            "Metal library linking failed: exit code {:?}",
            status.code()
        );
    }

    // Export the metallib path for runtime loading
    println!(
        "cargo:rustc-env=PAGED_ATTN_METALLIB={}",
        metallib_path.display()
    );

    // Also copy the metallib to a known location for distribution
    let dist_metallib = PathBuf::from(&manifest_dir).join("paged_attn.metallib");
    std::fs::copy(&metallib_path, &dist_metallib).ok();
}
