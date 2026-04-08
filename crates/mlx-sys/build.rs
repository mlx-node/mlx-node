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
    let cxx_flags = format!("{common_flags} -fwasm-exceptions");

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
        .arg(format!("-DCMAKE_EXE_LINKER_FLAGS={wasm_target} {sysroot_flag} -pthread -Wl,--allow-undefined"))
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
        .flag("-fvisibility-inlines-hidden")
        ;

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
    let sysroot_lib = wasi_sdk
        .join("share/wasi-sysroot/lib/wasm32-wasip1-threads");
    println!("cargo:rustc-link-search=native={}", sysroot_lib.display());
    println!("cargo:rustc-link-lib=static=c++");
    println!("cargo:rustc-link-lib=static=c++abi");
    // Increase WASM stack size to 8MB (default is 64KB-1MB, too small for deep
    // model call stacks: chat_sync → forward → 24 layers → matmul → ...)
    println!("cargo:rustc-link-arg=-Wl,-z,stack-size=8388608");
}

fn build_native(manifest_dir: &Path, mlx_dir: &Path, src_dir: &Path) {
    let metal_disabled = env::var_os("MLX_DISABLE_METAL").is_some();
    if !metal_disabled && !metal_toolchain_available() {
        panic!(
            "Metal toolchain not found. Install it with `xcodebuild -downloadComponent MetalToolchain` or set MLX_DISABLE_METAL=1 to force a CPU-only build."
        );
    }

    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS is not set");

    let mut cfg = cmake::Config::new(mlx_dir);
    cfg.define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define(
            "MLX_BUILD_METAL",
            if metal_disabled { "OFF" } else { "ON" },
        )
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
        let sdk_path =
            String::from_utf8(sdk_path).expect("Failed to convert SDK path to string");
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
        .define("MLX_STATIC", None)
        .include(&include_source)
        .include(mlx_dir);

    if target_os == "macos" {
        bridge.compiler("clang++");
    }

    if include_generated.exists() {
        bridge.include(&include_generated);
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
        build_native(&manifest_dir, &mlx_dir, &src_dir);
    }
}
