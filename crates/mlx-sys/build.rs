use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Port the reference's Metal residency and custom-kernel cache without changing
/// the MLX gitlink. Derived host files live in OUT_DIR; the narrow replacements fail
/// loudly if a future MLX update changes their integration points.
#[deny(clippy::unwrap_used, clippy::expect_used)]
fn metal_residency_overlay(manifest: &Path, mlx: &Path, out_dir: &Path) -> io::Result<PathBuf> {
    let write_changed = |path: PathBuf, bytes: &[u8]| -> io::Result<()> {
        match std::fs::read(&path) {
            Ok(existing) if existing == bytes => return Ok(()),
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(build_file_error("read overlay", &path, error)),
        }
        std::fs::write(&path, bytes)
            .map_err(|error| build_file_error("write overlay", &path, error))
    };
    let root = out_dir.join("metal-residency");
    let output = root.join("mlx/backend/metal");
    std::fs::create_dir_all(&output)
        .map_err(|error| build_file_error("create overlay directory", &output, error))?;
    let source = mlx.join("mlx/backend/metal");
    let port = manifest.join("metal-residency");
    let replace = |text: &mut String, from: &str, to: &str| -> io::Result<()> {
        if text.matches(from).count() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("MLX residency integration drift: {from}"),
            ));
        }
        *text = text.replacen(from, to, 1);
        Ok(())
    };
    for name in ["resident.h", "resident.cpp", "overlay.cmake"] {
        println!("cargo:rerun-if-changed={}", port.join(name).display());
    }
    for name in ["resident.h", "resident.cpp"] {
        let path = port.join(name);
        let bytes = std::fs::read(&path)
            .map_err(|error| build_file_error("read residency source", &path, error))?;
        write_changed(output.join(name), &bytes)?;
    }
    for name in ["device.h", "device.cpp"] {
        println!("cargo:rerun-if-changed={}", source.join(name).display());
        let mut text = read_build_source(&source.join(name))?;
        if name == "device.h" {
            replace(
                &mut text,
                "  Device& device_;",
                "  Device& device_;\n  ResidencySet& residency_set_;\n  uint64_t sets_attached_{0};",
            )?;
        } else {
            replace(
                &mut text,
                "    : device_(d) {",
                "    : device_(d), residency_set_(residency_set) {",
            )?;
            replace(
                &mut text,
                "  if (residency_set.mtl_residency_set()) {\n    queue_->addResidencySet(residency_set.mtl_residency_set());\n  }",
                "  residency_set_.attach_new_sets(queue_.get(), sets_attached_);",
            )?;
            replace(
                &mut text,
                "void CommandEncoder::commit(std::function<void()> completion) {",
                "void CommandEncoder::commit(std::function<void()> completion) {\n  // Metal fixes residency at commit, including sets created after this queue.\n  residency_set_.attach_new_sets(queue_.get(), sets_attached_);",
            )?;
        }
        write_changed(output.join(name), text.as_bytes())?;
    }
    // The reference keys custom libraries by name, source and compile options.
    // Compute that immutable key with the primitive, so cached graphs do not
    // rescan the complete Metal source on every dispatch. Every CMake/bridge
    // translation unit must see this same generated class layout.
    let header = mlx.join("mlx/fast_primitives.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let mut text = read_build_source(&header)?;
    replace(
        &mut text,
        "#include <optional>",
        "#include <cstdlib>\n#include <functional>\n#include <optional>",
    )?;
    replace(
        &mut text,
        "        compile_options_(compile_options) {}",
        "        compile_options_(compile_options),\n        library_name_(hash_cache_enabled() ? name_ + \"_mlx_node_\" +\n            std::to_string(std::hash<std::string>{}(source_)) + \"_\" +\n            std::to_string(compile_options_) : std::string{}) {}",
    )?;
    replace(
        &mut text,
        "  CompileOptions::Data compile_options_;",
        "  CompileOptions::Data compile_options_;\n  std::string library_name_;\n  static bool hash_cache_enabled() {\n    static const bool enabled = [] {\n      const char* value = std::getenv(\"MLX_METAL_HASH_KERNEL_CACHE\");\n      return value && std::string(value) == \"1\";\n    }();\n    return enabled;\n  }",
    )?;
    write_changed(root.join("mlx/fast_primitives.h"), text.as_bytes())?;
    let kernel = source.join("custom_kernel.cpp");
    println!("cargo:rerun-if-changed={}", kernel.display());
    let mut text = read_build_source(&kernel)?;
    replace(
        &mut text,
        "  {\n    // Clear kernels from the device library cache if needed",
        "  // Process-start experiment; an empty key retains the original cache.\n  const bool hashed = !library_name_.empty();\n  if (!hashed) {\n    // Clear kernels from the device library cache if needed",
    )?;
    replace(
        &mut text,
        "      name_, compile_options_, [this] { return metal::utils() + source_; });",
        "      hashed ? library_name_ : name_, compile_options_,\n      [this] { return metal::utils() + source_; });",
    )?;
    write_changed(output.join("custom_kernel.cpp"), text.as_bytes())?;
    Ok(root)
}

fn build_file_error(action: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("Failed to {action} {}: {error}", path.display()),
    )
}

fn read_build_source(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path)
        .map_err(|error| build_file_error("read build source", path, error))
}

fn metal_toolchain_available() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-v"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Explicit deployment-target floor for the macOS build products. When unset,
/// every toolchain stamps its own default and the artifact floor silently
/// tracks the build machine: the metal compiler uses the SDK's default min-OS
/// and MLX's CMake defaults `CMAKE_OSX_DEPLOYMENT_TARGET` to the build host's
/// macOS version. Setting `MACOSX_DEPLOYMENT_TARGET` (already honored by
/// rustc and cc for the Rust side) makes the floor deliberate for the CMake
/// and metallib products too.
fn macos_deployment_target() -> Option<String> {
    env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .filter(|v| !v.is_empty())
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

        let mut compile_cmd = Command::new("xcrun");
        compile_cmd.args([
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
        ]);
        // Pin the metallib's min-OS stamp when a floor is requested, matching
        // what MLX's kernel CMake does for mlx.metallib. The metal driver
        // reads MACOSX_DEPLOYMENT_TARGET from the environment too, but the
        // explicit flag keeps the floor visible in the command line.
        if let Some(target) = macos_deployment_target() {
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

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=mlx");
    // The deployment-target floor participates in both the CMake configure
    // and the paged-attn metallib compile; changing it must re-run this
    // script or the caches keep the old floor.
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    // This changes the CMake backend set, bridge translation-unit set, and
    // exported C ABI. Re-run even when Cargo otherwise considers the inputs
    // unchanged so toggling CPU-only mode cannot reuse Metal artifacts.
    println!("cargo:rerun-if-env-changed=MLX_DISABLE_METAL");
    // Cargo recursively watches directories, including added/removed bridge
    // files and shader includes nested under metal/common or a model family.
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
    println!("cargo:rerun-if-changed={}", src_dir.display());

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
    let residency_overlay = build_metal
        .then(|| metal_residency_overlay(&manifest_dir, &mlx_dir, &out_dir_path))
        .transpose()?;
    if let Some(overlay) = &residency_overlay {
        cfg.define("MLX_NODE_RESIDENCY_OVERLAY", overlay);
        cfg.define(
            "CMAKE_PROJECT_INCLUDE",
            manifest_dir.join("metal-residency/overlay.cmake"),
        );
    } else {
        // Clear a cached include if this build directory switches to CPU-only.
        cfg.define("CMAKE_PROJECT_INCLUDE", "");
    }
    cfg.define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("MLX_BUILD_METAL", if build_metal { "ON" } else { "OFF" });

    // `CMAKE_OSX_ARCHITECTURES` is an Apple-only knob; setting it on Linux
    // confuses the GCC/CUDA toolchain. Only emit it on macOS.
    if is_macos {
        // napi-rs invokes Cargo with an explicit Darwin target even when that
        // target is the native Apple-silicon host. Some CMake launches then
        // retain an x86_64 host processor while honoring the arm64 compiler
        // flags, and upstream MLX rejects the configure before it can inspect
        // CMAKE_OSX_ARCHITECTURES. Pin both views of the target architecture.
        cfg.define("CMAKE_SYSTEM_NAME", "Darwin");
        cfg.define(
            "CMAKE_SYSTEM_PROCESSOR",
            if target_arch == "aarch64" {
                "arm64"
            } else {
                "x86_64"
            },
        );
        cfg.define(
            "CMAKE_OSX_ARCHITECTURES",
            if target_arch == "aarch64" {
                "arm64"
            } else {
                "x86_64"
            },
        );
        // Forward an explicit deployment-target floor as a -D define: a
        // define overrides a stale CMAKE_OSX_DEPLOYMENT_TARGET already
        // recorded in CMakeCache.txt (e.g. a CI-restored cargo cache),
        // which the environment variable alone cannot. When unset, MLX's
        // CMakeLists defaults the floor to the build host's macOS version.
        if let Some(deployment_target) = macos_deployment_target() {
            cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);
        }
        // Upstream MLX only builds the NAX (gen-17 tensor-core) kernels when
        // the deployment floor is >= 26.2 and otherwise compiles the dispatch
        // out via MLX_METAL_NO_NAX. The vendored fork branch
        // (mlx-node/mlx#nax-macos-26-0-floor) adds MLX_METAL_FORCE_NAX to
        // decouple kernel presence from the floor, so one published artifact
        // can keep a macOS 26.0 floor AND carry the NAX kernels. The NAX
        // kernels themselves still compile at -mmacosx-version-min=26.2 —
        // they need the 26.2 tensor-ops ABI (lower targets select MPP's
        // pre-26.2 compatibility intrinsics, which miscompute) — while the
        // metallib links at the floor, so it loads on all of macOS 26.
        // Runtime dispatch (`is_nax_available`: gpu gen >= 17 && macOS >=
        // 26.2) keeps pre-26.2 machines from ever instantiating the
        // 26.2-targeted functions. The option is inert when the floor is
        // already >= 26.2 and when the SDK cannot build NAX (SDK < 26.2 or
        // MSL < 4.0).
        cfg.define("MLX_METAL_FORCE_NAX", "ON");
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
        // Rust links with -nodefaultlibs. Clang 21's availability checks use
        // __isPlatformVersionAtLeast from compiler-rt, which the C++ driver
        // normally supplies automatically. Carry that runtime explicitly so
        // native tests and addons targeting older macOS versions both link.
        if let Ok(runtime) = Command::new(&cxx_compiler)
            .arg("-print-file-name=libclang_rt.osx.a")
            .output()
            && runtime.status.success()
        {
            let path = PathBuf::from(String::from_utf8_lossy(&runtime.stdout).trim());
            if path.is_absolute()
                && path.is_file()
                && let Some(parent) = path.parent()
            {
                println!("cargo:rustc-link-search=native={}", parent.display());
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
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

        // `-Werror=switch` cannot be reached with `.cxxflag(...)`: the `cmake`
        // crate composes `CMAKE_CXX_FLAGS` as our flags followed by the ones
        // `cc` produces, and `cc` ends that list with `-w`, which clang honors
        // over every `-W`/`-Werror=` flag no matter where it sits on the
        // command line. The include below runs inside MLX's configure step,
        // where it can rewrite the composed string instead of appending to it.
        // Clang-only, hence macOS-only: `-Wno-everything` is not a GCC flag.
        let switch_diagnostic = manifest_dir
            .join("cmake")
            .join("switch-exhaustiveness.cmake");
        println!("cargo:rerun-if-changed={}", switch_diagnostic.display());
        cfg.define(
            "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
            switch_diagnostic
                .to_str()
                .expect("switch-exhaustiveness.cmake path is not valid UTF-8"),
        );
    } else if target_os == "linux" {
        // Linux/CUDA build. The MLX submodule's CMake auto-detects the GPU
        // architecture, but on a GPU-less / headless configure host the
        // detection query is empty and FATALs. Pass the arch explicitly for
        // determinism (`121a` is what MLX auto-detected on the GB10 host;
        // override via MLX_CUDA_ARCHITECTURES). Release build type so the
        // benchmark numbers are not skewed by an unoptimized default.
        //
        // `switch-exhaustiveness.cmake` is deliberately not applied here: it
        // rewrites the composed flags with `-Wno-everything`, which only
        // clang understands, and nvcc drives host compilation through GCC.
        // A `switch` over QuantizationMode that misses an enumerator is
        // therefore a build error on macOS only. CUDA sources are compiled
        // nowhere else in CI, so any such switch has to be walked by hand.
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
    if let Some(overlay) = &residency_overlay {
        // Device contains ResidencySet by value: all bridge code must see the
        // same class layout as libmlx, before the original vendor headers.
        bridge.include(overlay);
    }
    bridge
        .cpp(true)
        .warnings(false)
        .define("MLX_STATIC", None)
        .include(&include_source)
        .include(&mlx_dir);

    if build_metal || target_os == "linux" {
        bridge.define("MLX_NODE_GPU_ENABLED", None);
    }

    // `__APPLE__` alone does not mean this build contains MLX's Metal backend:
    // `MLX_DISABLE_METAL=1` is a supported CPU-only macOS configuration.  Keep
    // bridge translation units from including/calling Metal-only APIs unless
    // the CMake build above actually enabled them.
    if build_metal {
        bridge.define("MLX_NODE_METAL_ENABLED", None);
        // Custom quantized kernels reuse the vendored K-quant arithmetic even
        // in the normal precompiled-metallib build, where MLX does not export
        // its optional JIT preambles. Generate private copies from source so
        // the helper kernels cannot drift from the linked quantization code.
        let preambles = out_dir_path.join("quantized-preambles");
        let script = mlx_dir.join("mlx/backend/metal/make_compiled_preamble.sh");
        for (source_name, name) in [
            ("steel/gemm/gemm", "gemm"),
            ("quantized_utils", "quantized_utils"),
            ("kquant", "kquant"),
            ("steel/gemm/nax", "nax"),
            ("kquant_nax", "kquant_nax"),
        ] {
            let status = Command::new("bash")
                .arg(&script)
                .arg(&preambles)
                .arg("clang")
                .arg(&mlx_dir)
                .arg(source_name)
                .status()
                .map_err(|error| build_file_error("run preamble generator", &script, error))?;
            if !status.success() {
                return Err(io::Error::other(format!(
                    "Quantized {name} Metal preamble failed: {status}"
                )));
            }
            let path = preambles.join(format!("{name}.cpp"));
            let source = read_build_source(&path)?.replace(
                "namespace mlx::core::metal",
                "namespace mlx::core::quantized_preamble",
            );
            std::fs::write(&path, source).map_err(|error| {
                build_file_error("write private quantized preamble", &path, error)
            })?;
            bridge.file(path);
            println!(
                "cargo:rerun-if-changed={}",
                mlx_dir
                    .join(format!("mlx/backend/metal/kernels/{source_name}.h"))
                    .display()
            );
        }
    }

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
    // Add src/ as include path for metal/{common,<family>}/*.metal.inc includes
    bridge.include(&src_dir);

    // Translation units that depend on Metal *by header* (raw `MTL::` types
    // or `#include mlx/backend/metal/device.h`). These do not compile on a
    // non-Metal host. They are excluded from every non-Metal build (including
    // `MLX_DISABLE_METAL=1` on macOS); the symbols
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
            if !build_metal && METAL_ONLY_TUS.contains(&file_name.as_str()) {
                continue;
            }
            bridge.file(&path);
        }
    }
    bridge.compile("mlx_ffi");

    println!("cargo:rustc-link-lib=static=mlx_ffi");
    Ok(())
}
