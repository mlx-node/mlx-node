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

    // Read the target OS/arch up front: they decide whether we build the
    // Metal backend at all. On macOS we build Metal (unless explicitly
    // disabled); on Linux we build the CUDA backend and there is no Metal
    // toolchain to look for.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is not set");
    let is_macos = target_os == "macos";

    let metal_disabled = env::var_os("MLX_DISABLE_METAL").is_some();
    // Metal is built only on macOS and only when not explicitly disabled.
    // On Linux this is always false → no xcrun/metallib, CUDA backend instead.
    let build_metal = is_macos && !metal_disabled;

    if build_metal && !metal_toolchain_available() {
        panic!(
            "Metal toolchain not found. Install it with `xcodebuild -downloadComponent MetalToolchain` or set MLX_DISABLE_METAL=1 to force a CPU-only build."
        );
    }

    // Compile the paged-attention `.metallib` BEFORE we run the cmake
    // build for MLX. Both products land in `OUT_DIR`. Skipped when Metal
    // is not built (Metal disabled, or non-macOS host) — the C++ side guards
    // the dispatch with `target_os = "macos"` and the runtime
    // `paged_attn_metallib_path` lookup will throw if the metallib is
    // not findable.
    let out_dir_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let paged_metallib_path = if build_metal {
        Some(compile_paged_attn_metallib(&manifest_dir, &out_dir_path))
    } else {
        None
    };

    let mut cfg = cmake::Config::new(&mlx_dir);
    cfg.define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("MLX_BUILD_METAL", if build_metal { "ON" } else { "OFF" });

    // `CMAKE_OSX_ARCHITECTURES` is an Apple-only knob; setting it on Linux
    // confuses the GCC/CUDA toolchain. Only emit it on macOS.
    if is_macos {
        cfg.define(
            "CMAKE_OSX_ARCHITECTURES",
            if target_arch == "aarch64" {
                "arm64"
            } else {
                "x86_64"
            },
        );
    }

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
    } else if target_os == "linux" {
        // Linux/CUDA build. The MLX submodule's CMake auto-detects the GPU
        // architecture, but on a GPU-less / headless configure host the
        // detection query is empty and FATALs. Pass the arch explicitly for
        // determinism (`121a` is what MLX auto-detected on the GB10 host;
        // override via MLX_CUDA_ARCHITECTURES). Release build type so the
        // benchmark numbers are not skewed by an unoptimized default.
        let cuda_archs = env::var("MLX_CUDA_ARCHITECTURES").unwrap_or_else(|_| "121a".into());
        cfg.define("MLX_BUILD_CUDA", "ON")
            .define("MLX_BUILD_METAL", "OFF")
            .define("MLX_BUILD_CPU", "ON")
            .define("MLX_CUDA_ARCHITECTURES", &cuda_archs)
            .define("CMAKE_BUILD_TYPE", "Release");
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

    if is_macos {
        if build_metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=QuartzCore");
        }
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "linux" {
        // CUDA runtime + math libs are PRIVATE deps of the static libmlx —
        // they are not re-exported through the .a, so we must re-declare them
        // for the final link of the .node addon. Search both the CUDA lib dir
        // and the stubs dir (libcuda.so lives only in stubs at build time;
        // the real driver is found at runtime via LD_LIBRARY_PATH).
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".into());
        println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        println!("cargo:rustc-link-search=native={cuda_path}/lib64/stubs");
        for l in ["cudart", "cublas", "cublasLt", "cufft", "nvrtc", "cuda"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        // cuDNN: MLX 9.x uses the split cuDNN. The umbrella `cudnn` shim
        // re-exports the sub-libraries on most installs; if the link reports
        // unresolved cuDNN symbols, the real sub-lib names (cudnn_graph,
        // cudnn_ops, cudnn_engines_*) are added below after verifying against
        // the cmake link line.
        println!("cargo:rustc-link-lib=dylib=cudnn");
        // MLX keeps its CPU backend on even for a CUDA build (some ops have no
        // CUDA path), so its CBLAS/LAPACK symbols (e.g. `cblas_dgemm`) are
        // PRIVATE deps of static libmlx that must be re-declared for the final
        // link. `libblas.so` provides the CBLAS interface; `liblapack.so` the
        // LAPACK ops. (MLX's CMake found these at /usr/lib/aarch64-linux-gnu.)
        for l in ["lapack", "blas"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        for l in ["stdc++", "dl", "pthread"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
    }

    let include_source = mlx_dir.join("mlx");
    let include_generated = dst.join("include");

    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .warnings(false)
        .define("MLX_STATIC", None)
        .include(&include_source)
        .include(&mlx_dir);

    if is_macos {
        // macOS keeps C++17 (its clang accepts MLX's defaulted operator==
        // under C++17). Unchanged from the original build to guarantee no
        // codegen drift on the macOS path.
        bridge.std("c++17");
        bridge.compiler("clang++");
    } else {
        // MLX itself is built with `CMAKE_CXX_STANDARD 20`; its public headers
        // use C++20-only constructs (e.g. defaulted `operator==` in device.h /
        // stream.h). GCC enforces the standard strictly, so the bridge must
        // match C++20 to consume those headers and to resolve the same
        // `slice_update` overloads MLX compiled against.
        bridge.std("c++20");
    }

    if include_generated.exists() {
        bridge.include(&include_generated);
        // metal-cpp installs to `<install>/include/metal_cpp/Metal/Metal.hpp`.
        // mlx_paged_dispatch.cpp needs it because the public
        // `mlx::core::metal::Device` API exposes `MTL::*` types from
        // `<Metal/Metal.hpp>`. The CMake build links MLX against
        // metal_cpp transitively but the cc-rs C++ bridge must be told
        // explicitly. Only present / needed on macOS — the CUDA build has
        // no metal_cpp and excludes the TUs that consume it.
        if is_macos {
            let metal_cpp_include = include_generated.join("metal_cpp");
            if metal_cpp_include.exists() {
                bridge.include(&metal_cpp_include);
            }
        }
    }
    // Add src/ as include path for metal/*.metal.inc includes
    bridge.include(&src_dir);

    // Translation units that depend on Metal *by header* (raw `MTL::` types
    // or `#include mlx/backend/metal/device.h`). These do not compile on a
    // non-Metal host. They are excluded from the Linux build; the symbols
    // their `eval_gpu` consumers need (`paged_kv_write` / `paged_attention` /
    // `paged_attention_varlen`) are provided as runtime-throwing stubs in
    // `mlx_paged_stubs_linux.cpp`, and the Rust callers fall back to the
    // eager path via the `mlx_metal_is_available()` gates.
    const METAL_ONLY_TUS: &[&str] = &[
        "mlx_paged_dispatch.cpp", // raw MTL:: types
        "mlx_paged_ops.cpp",      // #include mlx/backend/metal/device.h
        "mlx_paged_profile.cpp",  // __APPLE__-guarded body + Metal include
    ];

    // Compile all .cpp files in src/ (split from original monolithic mlx.cpp)
    for entry in std::fs::read_dir(&src_dir).expect("Failed to read src directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "cpp") {
            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if !is_macos && METAL_ONLY_TUS.contains(&file_name.as_str()) {
                continue;
            }
            bridge.file(&path);
        }
    }
    bridge.compile("mlx_ffi");

    println!("cargo:rustc-link-lib=static=mlx_ffi");
}
