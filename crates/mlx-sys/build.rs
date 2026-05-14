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

/// Compile paged-attention Metal kernels into a colocated metallib for native
/// builds. WASM/WebGPU builds do not use this path.
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
    for path in walk_metal_dir(&metal_src_dir) {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let metal_files = [
        "attention/paged_attention.metal",
        "cache/reshape_and_cache.metal",
        "cache/copy_blocks.metal",
    ];

    let mut air_files = Vec::new();
    for file in metal_files {
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
            let path = entry.path();
            if path.is_dir() {
                out.extend(walk_metal_dir(&path));
            } else if path.extension().is_some_and(|ext| ext == "metal") {
                out.push(path);
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

fn is_wasm_target(target: &str) -> bool {
    target.contains("wasm32") && target.contains("wasi")
}

fn build_wasi(_manifest_dir: &Path, mlx_dir: &Path, src_dir: &Path) {
    let wasi_sdk = env::var("WASI_SDK_PATH").unwrap_or_else(|_| "/opt/wasi-sdk".to_string());
    let wasi_sdk = PathBuf::from(wasi_sdk);

    let sysroot = wasi_sdk.join("share/wasi-sysroot");
    if !sysroot.exists() {
        panic!(
            "WASI-SDK sysroot not found at {}. Set WASI_SDK_PATH.",
            sysroot.display()
        );
    }

    // Build MLX with cmake directly (bypass cmake crate's flag conflicts).
    // The cmake crate adds --target=wasm32-wasi which conflicts with
    // --target=wasm32-wasip1-threads from the toolchain file.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("mlx-build");
    std::fs::create_dir_all(&build_dir).unwrap();

    let clang = wasi_sdk.join("bin/clang");
    let clangxx = wasi_sdk.join("bin/clang++");
    let ar = wasi_sdk.join("bin/ar");
    let ranlib = wasi_sdk.join("bin/ranlib");
    let wasm_target = "--target=wasm32-wasip1-threads";
    let sysroot_flag = format!("--sysroot={}", sysroot.display());
    let common_flags = format!(
        "{wasm_target} {sysroot_flag} -pthread -fPIC -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_SIGNAL"
    );
    let c_flags = common_flags.clone();
    // -fno-inline: prevent ALL inlining so that inline virtual methods
    // (DEFINE_NAME, UnaryPrimitive forwarders, etc.) are emitted as
    // standalone functions. Without this, wasm-ld GC discards inlined
    // virtual methods, leaving vtable entries pointing to garbage.
    // -femit-all-decls: force emission of ALL inline virtual methods.
    // Without this, Clang's Itanium ABI marks inline virtuals as
    // 'available_externally' in the key function's TU, causing vtable
    // entries to reference undefined symbols that wasm-ld resolves incorrectly.
    // MLX_WGPU_LOG_KERNELS: emit one stderr line per unique matmul kernel
    // variant the first time it is compiled. Used to verify that the packed
    // bf16 kernel variant (entry_name containing "_bf16p") is actually
    // dispatching for production Qwen3.5 shapes. Zero steady-state overhead
    // because the log lives inside the shader-module cache-miss lambda.
    let cxx_flags = format!("{common_flags} -fwasm-exceptions -DMLX_WGPU_LOG_KERNELS=1");

    let status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg(mlx_dir)
        .args(["-G", "Unix Makefiles"])
        .arg(format!("-DCMAKE_C_COMPILER={}", clang.display()))
        .arg(format!("-DCMAKE_CXX_COMPILER={}", clangxx.display()))
        .arg(format!("-DCMAKE_AR={}", ar.display()))
        .arg(format!("-DCMAKE_RANLIB={}", ranlib.display()))
        .arg(format!("-DCMAKE_SYSROOT={}", sysroot.display()))
        .arg("-DCMAKE_SYSTEM_NAME=WASI")
        .arg("-DCMAKE_SYSTEM_PROCESSOR=wasm32")
        .arg("-DCMAKE_CROSSCOMPILING=ON")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG")
        .arg(format!("-DCMAKE_C_FLAGS={c_flags}"))
        .arg(format!("-DCMAKE_CXX_FLAGS={cxx_flags}"))
        .arg(format!(
            "-DCMAKE_EXE_LINKER_FLAGS={wasm_target} {sysroot_flag} -pthread -Wl,--allow-undefined"
        ))
        .arg("-DCMAKE_C_COMPILER_WORKS=ON")
        .arg("-DCMAKE_CXX_COMPILER_WORKS=ON")
        .arg("-DMLX_BUILD_TESTS=OFF")
        .arg("-DMLX_BUILD_EXAMPLES=OFF")
        .arg("-DMLX_BUILD_BENCHMARKS=OFF")
        .arg("-DMLX_BUILD_PYTHON_BINDINGS=OFF")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DMLX_BUILD_METAL=OFF")
        .arg("-DMLX_BUILD_CUDA=OFF")
        .arg("-DMLX_BUILD_CPU=ON")
        .arg("-DMLX_BUILD_WEBGPU=ON")
        .arg("-DWEBGPU_BACKEND=WASI_IMPORT")
        .arg("-DMLX_NO_BLAS=ON")
        .arg("-DMLX_BUILD_GGUF=OFF")
        .arg("-DMLX_BUILD_SAFETENSORS=ON")
        .arg("-DWASI_SDK=ON")
        .arg("-DMLX_USE_CCACHE=OFF")
        .status()
        .expect("Failed to run cmake configure");
    if !status.success() {
        panic!("cmake configure failed");
    }

    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let status = Command::new("cmake")
        .current_dir(&build_dir)
        .args(["--build", ".", "--", &format!("-j{nproc}")])
        .status()
        .expect("Failed to run cmake build");
    if !status.success() {
        panic!("cmake build failed");
    }

    // Find the built library
    let lib_candidates = [
        build_dir.join("lib"),
        build_dir.join("mlx"),
        build_dir.join("Release"),
    ];
    for candidate in &lib_candidates {
        if candidate.exists() {
            add_link_search(candidate);
        }
    }
    // Also search the build dir itself
    add_link_search(&build_dir);

    // whole-archive: force wasm-ld to include ALL object files from libmlx.a.
    // Without this, virtual method implementations (referenced only via vtable
    // data relocations in the WASM data section) get discarded during archive
    // scanning, causing "function signature mismatch" at runtime.
    println!("cargo:rustc-link-lib=static:+whole-archive=mlx");

    // No framework linking for WASI

    let include_source = mlx_dir.join("mlx");
    let include_generated = build_dir.join("include");

    let clangxx_threads = wasi_sdk.join("bin/wasm32-wasip1-threads-clang++");

    let mut bridge = cc::Build::new();
    bridge
        .compiler(&clangxx_threads)
        .cpp(true)
        .std("c++20")
        .warnings(false)
        .define("MLX_STATIC", None)
        .define("MLX_USE_WEBGPU", None)
        .define("WEBGPU_BACKEND_WASI_IMPORT", None)
        // Do NOT define MLX_USE_METAL
        .include(&include_source)
        .include(mlx_dir)
        .flag("-pthread")
        .flag("-fexceptions")
        .flag("-fwasm-exceptions")
        .flag("-D_WASI_EMULATED_MMAN")
        .flag("-D_WASI_EMULATED_SIGNAL")
        // Match CMake's visibility flags to prevent vtable corruption.
        // Without these, inline virtual overrides in UnaryPrimitive get
        // different visibility between libmlx.a and libmlx_ffi.a, causing
        // wasm-ld to pick inconsistent vtable thunks → call_indirect mismatch.
        .flag("-fvisibility=hidden")
        .flag("-fvisibility-inlines-hidden");

    if include_generated.exists() {
        bridge.include(&include_generated);
    }
    bridge.include(src_dir);

    // Compile all .cpp files in src/
    for entry in std::fs::read_dir(src_dir).expect("Failed to read src directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "cpp") {
            bridge.file(&path);
        }
    }
    bridge.compile("mlx_ffi");

    println!("cargo:rustc-link-lib=static=mlx_ffi");

    // Link C++ standard library and ABI for WASI (provides operator new/delete,
    // exception handling, and RTTI — these are left as imports without this).
    let sysroot_lib = wasi_sdk.join("share/wasi-sysroot/lib/wasm32-wasip1-threads");
    println!("cargo:rustc-link-search=native={}", sysroot_lib.display());
    println!("cargo:rustc-link-lib=static=c++");
    println!("cargo:rustc-link-lib=static=c++abi");
}

fn build_native(mlx_dir: &Path, src_dir: &Path) {
    let metal_disabled = env::var_os("MLX_DISABLE_METAL").is_some();
    if !metal_disabled && !metal_toolchain_available() {
        panic!(
            "Metal toolchain not found. Install it with `xcodebuild -downloadComponent MetalToolchain` or set MLX_DISABLE_METAL=1 to force a CPU-only build."
        );
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let paged_metallib_path = if !metal_disabled {
        Some(compile_paged_attn_metallib(&manifest_dir, &out_dir_path))
    } else {
        None
    };

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is not set");

    let mut cfg = cmake::Config::new(mlx_dir);
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

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        if let Some(profile_dir) = find_ancestor_with_name(&out_path, "build")
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        {
            let mut sinks = vec![profile_dir.clone(), profile_dir.join("deps")];
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
                    let _ = std::fs::copy(paged_metallib, sink.join("paged_attn.metallib"));
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
        .include(mlx_dir);

    if target_os == "macos" {
        bridge.compiler("clang++");
    }

    if include_generated.exists() {
        bridge.include(&include_generated);
        let metal_cpp_include = include_generated.join("metal_cpp");
        if metal_cpp_include.exists() {
            bridge.include(&metal_cpp_include);
        }
    }
    // Add src/ as include path for metal/*.metal.inc includes
    bridge.include(src_dir);

    // Compile all .cpp files in src/
    for entry in std::fs::read_dir(src_dir).expect("Failed to read src directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "cpp") {
            bridge.file(&path);
        }
    }
    bridge.compile("mlx_ffi");

    println!("cargo:rustc-link-lib=static=mlx_ffi");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=mlx");

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

    let target = env::var("TARGET").unwrap_or_default();

    if is_wasm_target(&target) {
        build_wasi(&manifest_dir, &mlx_dir, &src_dir);
    } else {
        build_native(&mlx_dir, &src_dir);
    }
}
