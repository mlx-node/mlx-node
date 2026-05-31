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

/// Compile the paged-attention `.metal` sources into
/// `<out_dir>/paged_attn.metallib`. The kernels live in
/// `crates/mlx-paged-attn/metal/`. mlx-sys's own
/// `mlx_paged_dispatch.cpp` resolves this metallib at runtime by
/// looking next to the loaded binary (the .node addon copies it
/// alongside `mlx.metallib` during the package-build step).
///
/// Mirror of `crates/mlx-paged-attn/build.rs`'s metal-shader compile:
/// same `xcrun -sdk macosx metal -O3 -ffast-math` invocation, same
/// link step.
fn compile_paged_attn_metallib(manifest_dir: &Path, out_dir: &Path) -> PathBuf {
    let metal_src_dir = manifest_dir
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent")
        .join("mlx-paged-attn")
        .join("metal");
    if !metal_src_dir.exists() {
        panic!(
            "expected paged-attn metal sources at {}",
            metal_src_dir.display()
        );
    }

    println!("cargo:rerun-if-changed={}", metal_src_dir.display());
    let walk = walk_metal_dir(&metal_src_dir);
    for path in &walk {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let metal_files = [
        "attention/paged_attention.metal",
        "cache/reshape_and_cache.metal",
        "cache/copy_blocks.metal",
    ];

    let mut air_files = Vec::new();
    for file in &metal_files {
        let src_path = metal_src_dir.join(file);
        let air_name = file.replace('/', "_").replace(".metal", ".air");
        let air_path = out_dir.join(&air_name);

        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                src_path.to_str().unwrap(),
                "-o",
                air_path.to_str().unwrap(),
                "-I",
                metal_src_dir.to_str().unwrap(),
                "-O3",
                "-ffast-math",
            ])
            .status()
            .expect("Failed to execute xcrun metal");
        if !status.success() {
            panic!(
                "Metal compilation failed for {}: exit code {:?}",
                file,
                status.code()
            );
        }
        air_files.push(air_path);
    }

    let metallib_path = out_dir.join("paged_attn.metallib");
    let mut link_cmd = Command::new("xcrun");
    link_cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        link_cmd.arg(air.to_str().unwrap());
    }
    link_cmd.args(["-o", metallib_path.to_str().unwrap()]);
    let status = link_cmd.status().expect("Failed to execute xcrun metallib");
    if !status.success() {
        panic!(
            "Paged-attn metallib linking failed: exit code {:?}",
            status.code()
        );
    }

    metallib_path
}

/// Walk ancestors of `start` looking for a directory whose final name
/// equals `name`. Returns the matching ancestor's path, or `None`.
fn find_ancestor_with_name(start: &Path, name: &str) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        if ancestor
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .as_deref()
            == Some(name)
        {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn walk_metal_dir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                out.extend(walk_metal_dir(&p));
            } else if let Some(ext) = p.extension()
                && ext == "metal"
            {
                out.push(p);
            }
        }
    }
    out
}

fn add_link_search(path: &Path) {
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn resolve_build_tool(env_key: &str, candidates: &[&str]) -> String {
    if let Ok(value) = env::var(env_key)
        && !value.is_empty()
    {
        return value;
    }

    let path_dirs = env::var_os("PATH")
        .map(|path| env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default();

    for candidate in candidates {
        let candidate_path = Path::new(candidate);
        if candidate_path.is_absolute() && candidate_path.exists() {
            return candidate.to_string();
        }
        for dir in &path_dirs {
            let path = dir.join(candidate);
            if path.exists() {
                return path.to_string_lossy().to_string();
            }
        }
    }

    candidates
        .first()
        .expect("resolve_build_tool requires at least one candidate")
        .to_string()
}

fn xcrun_find(tool: &str) -> Option<String> {
    let output = Command::new("xcrun").args(["--find", tool]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (!path.is_empty()).then(|| path.to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=mlx");
    // Watch all C++ source files, headers, and Metal kernel includes
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
    if let Ok(entries) = std::fs::read_dir(&src_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension()
                && (ext == "cpp" || ext == "h")
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    let metal_dir = src_dir.join("metal");
    if let Ok(entries) = std::fs::read_dir(&metal_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension()
                && ext == "inc"
            {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

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

    // Compile the paged-attention `.metallib` BEFORE we run the cmake
    // build for MLX. Both products land in `OUT_DIR`. Skipped if Metal
    // is disabled (no Metal toolchain) — the C++ side guards the
    // dispatch with `target_os = "macos"` and the runtime
    // `paged_attn_metallib_path` lookup will throw if the metallib is
    // not findable.
    let out_dir_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let paged_metallib_path = if !metal_disabled {
        Some(compile_paged_attn_metallib(&manifest_dir, &out_dir_path))
    } else {
        None
    };

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
        let default_c_compiler = xcrun_find("clang").unwrap_or_else(|| "clang".to_string());
        let default_cxx_compiler = xcrun_find("clang++").unwrap_or_else(|| "clang++".to_string());
        let default_ar = xcrun_find("ar").unwrap_or_else(|| "/usr/bin/ar".to_string());
        let default_ranlib = xcrun_find("ranlib").unwrap_or_else(|| "/usr/bin/ranlib".to_string());
        let c_compiler = resolve_build_tool(
            "CC",
            &[default_c_compiler.as_str(), "/usr/bin/clang", "clang"],
        );
        let cxx_compiler = resolve_build_tool(
            "CXX",
            &[default_cxx_compiler.as_str(), "/usr/bin/clang++", "clang++"],
        );
        let ar = resolve_build_tool("AR", &[default_ar.as_str(), "/usr/bin/ar", "ar"]);
        let ranlib = resolve_build_tool(
            "RANLIB",
            &[default_ranlib.as_str(), "/usr/bin/ranlib", "ranlib"],
        );
        let sdk_path = Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .expect("Failed to get SDK path")
            .stdout
            .to_vec();
        let sdk_path = String::from_utf8(sdk_path).expect("Failed to convert SDK path to string");
        let sdk_path = sdk_path.trim();
        cfg.define("CMAKE_C_COMPILER", c_compiler)
            .define("CMAKE_CXX_COMPILER", cxx_compiler)
            .define("CMAKE_AR", &ar)
            .define("CMAKE_RANLIB", &ranlib)
            .define("CMAKE_C_COMPILER_AR", &ar)
            .define("CMAKE_CXX_COMPILER_AR", &ar)
            .define("CMAKE_C_COMPILER_RANLIB", &ranlib)
            .define("CMAKE_CXX_COMPILER_RANLIB", &ranlib)
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

    // Co-locate `paged_attn.metallib` with `mlx.metallib`. Both must
    // ship next to the loaded binary at runtime (see
    // `packages/core/build.ts::copyMetallib` which copies the cmake
    // output's `lib/mlx.metallib` into the addon directory).
    //
    // Also copy to common locations next to test binaries:
    //   - `target/<profile>/`        (cargo test --release / debug)
    //   - `target/<profile>/deps/`   (where Rust integration tests run)
    //   - `target/<arch>/<profile>/{,deps/}` (cross-target)
    // so `cargo test` works without manual env var setup. The runtime
    // `dladdr`-based lookup in mlx_paged_dispatch.cpp finds the addon
    // binary's parent directory and looks for `paged_attn.metallib`
    // there.
    if let Some(paged_metallib) = paged_metallib_path.as_ref() {
        for candidate in lib_candidates.iter() {
            if candidate.exists() {
                let dst_path = candidate.join("paged_attn.metallib");
                if let Err(e) = std::fs::copy(paged_metallib, &dst_path) {
                    panic!(
                        "Failed to copy paged_attn.metallib to {}: {e}",
                        dst_path.display()
                    );
                }
            }
        }

        // Copy to test/binary-output directories: cargo passes
        // OUT_DIR but test binaries live at target/<profile>/deps/.
        // Walk up to find the target dir.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        // OUT_DIR shape: target/<arch>/<profile>/build/mlx-sys-<hash>/out
        // Want:           target/<arch>/<profile>/{,deps/}
        // and:            target/<profile>/{,deps/} (default-target build)
        if let Some(profile_dir) = find_ancestor_with_name(&out_path, "build")
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        {
            let mut sinks = vec![profile_dir.clone(), profile_dir.join("deps")];
            // Also try walking one level above to support per-target
            // dirs (target/<arch>/<profile> path layout).
            if let Some(parent) = profile_dir.parent()
                && parent
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .as_deref()
                    != Some("target")
            {
                sinks.push(parent.join("deps"));
            }
            for sink in sinks {
                if sink.exists() {
                    let dst = sink.join("paged_attn.metallib");
                    let _ = std::fs::copy(paged_metallib, &dst);
                }
            }
        }
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
        .define("MLX_STATIC", None)
        .include(&include_source)
        .include(&mlx_dir);

    if target_os == "macos" {
        bridge.compiler("clang++");
    }

    if include_generated.exists() {
        bridge.include(&include_generated);
        // metal-cpp installs to `<install>/include/metal_cpp/Metal/Metal.hpp`.
        // mlx_paged_dispatch.cpp (Phase 2) needs it because the public
        // `mlx::core::metal::Device` API exposes `MTL::*` types from
        // `<Metal/Metal.hpp>`. The CMake build links MLX against
        // metal_cpp transitively but the cc-rs C++ bridge must be told
        // explicitly.
        let metal_cpp_include = include_generated.join("metal_cpp");
        if metal_cpp_include.exists() {
            bridge.include(&metal_cpp_include);
        }
    }
    // Add src/ as include path for metal/*.metal.inc includes
    bridge.include(&src_dir);

    // Compile all .cpp files in src/ (split from original monolithic mlx.cpp)
    for entry in std::fs::read_dir(&src_dir).expect("Failed to read src directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "cpp") {
            bridge.file(&path);
        }
    }
    bridge.compile("mlx_ffi");

    println!("cargo:rustc-link-lib=static=mlx_ffi");
}
