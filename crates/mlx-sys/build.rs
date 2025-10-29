use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn metal_toolchain_available() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-v"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn add_link_search(path: &Path) {
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/mlx.cpp");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mlx_dir = manifest_dir.join("mlx");

    if !mlx_dir.join("CMakeLists.txt").exists() {
        panic!("expected mlx/CMakeLists.txt relative to crate");
    }

    let metal_disabled = env::var_os("MLX_DISABLE_METAL").is_some();
    if !metal_disabled && !metal_toolchain_available() {
        panic!(
            "Metal toolchain not found. Install it with `xcodebuild -downloadComponent MetalToolchain` or set MLX_DISABLE_METAL=1 to force a CPU-only build."
        );
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is not set");

    let mut cfg = cmake::Config::new(&mlx_dir);
    cfg.define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("MLX_BUILD_METAL", if metal_disabled { "OFF" } else { "ON" })
        .define(
            "CMAKE_OSX_ARCHITECTURES",
            if target_arch == "aarch64" {
                "arm64"
            } else {
                "x86_64"
            },
        );

    if target_os == "macos" {
        let sdk_path = Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .expect("Failed to get SDK path")
            .stdout
            .to_vec();
        let sdk_path = String::from_utf8(sdk_path).expect("Failed to convert SDK path to string");
        let sdk_path = sdk_path.trim();
        cfg.define("CMAKE_C_COMPILER", "clang")
            .define("CMAKE_CXX_COMPILER", "clang++")
            .cflag(format!("-isysroot {sdk_path}"))
            .cxxflag(format!("-isysroot {sdk_path}"));
    }

    let dst = cfg.build();

    let lib_candidates = [
        dst.join("lib"),
        dst.join("build").join("lib"),
        dst.join("build").join("Release"),
        dst.join("build").join("mlx"),
        dst.join("build").join("mlx").join("lib"),
    ];
    let mut found = false;
    for candidate in lib_candidates.iter() {
        if candidate.exists() {
            add_link_search(candidate);
            found = true;
        }
    }
    if !found {
        panic!(
            "unable to locate MLX build artifacts under {}; expected lib directories to exist",
            dst.display()
        );
    }

    println!("cargo:rustc-link-lib=static=mlx");

    if !metal_disabled {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=c++");

    let include_source = mlx_dir.join("mlx");
    let include_generated = dst.join("include");

    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&include_source)
        .include(&mlx_dir);

    if target_os == "macos" {
        bridge.compiler("clang++");
    }

    if include_generated.exists() {
        bridge.include(&include_generated);
    }
    bridge
        .file(manifest_dir.join("src/mlx.cpp"))
        .compile("mlx_ffi");

    println!("cargo:rustc-link-lib=static=mlx_ffi");
}
