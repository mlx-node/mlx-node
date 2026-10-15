use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// macOS deployment-target floor of the Metal build products.
///
/// MLX builds its NAX (gen-17 tensor-core) kernels only at a deployment target
/// of 26.2 or newer (`mlx/backend/metal/kernels/CMakeLists.txt`); below that it
/// defines MLX_METAL_NO_NAX and silently ships without them, so a Metal build
/// below the floor fails (`assert_nax_buildable`). The toolchains' own default
/// is the BUILD HOST, which is never used: a build on macOS 27 emits
/// `air64_v29` shaders ("language version 4.1") that a macOS 26 host refuses
/// to load at runtime while the rest of the app launches fine (measured:
/// default `xcrun metal` = `air64_v29-apple-macosx27.0.0`,
/// `-mmacosx-version-min=26.2` = `air64_v28-apple-macosx26.2.0`).
const MACOS_DEPLOYMENT_TARGET_FLOOR: &str = "26.2";

/// The build host's macOS version as `(major, minor)`, via `sw_vers`.
fn host_macos_version() -> Option<(u64, u64)> {
    // The absolute path survives build environments with a stripped PATH.
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

/// The deployment target of the macOS build products: `MACOSX_DEPLOYMENT_TARGET`
/// when set (rustc and cc honor it for the Rust side too), else the floor —
/// capped at an older build host's own version, so a CPU-only build
/// (`MLX_DISABLE_METAL=1`) there still emits binaries the host can run. A Metal
/// build on such a host then fails in `assert_nax_buildable`.
fn macos_deployment_target() -> String {
    if let Some(target) = env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .filter(|v| !v.is_empty())
    {
        return target;
    }
    match host_macos_version().map(|(major, minor)| format!("{major}.{minor}")) {
        Some(host) if !version_at_least(&host, MACOS_DEPLOYMENT_TARGET_FLOOR) => host,
        _ => MACOS_DEPLOYMENT_TARGET_FLOOR.to_string(),
    }
}

/// Fails the build unless MLX's own NAX condition holds
/// (`mlx/backend/metal/kernels/CMakeLists.txt`): deployment target >= 26.2,
/// macOS SDK >= 26.2, and Metal language >= 4.0 at that target. When it does
/// not, MLX defines MLX_METAL_NO_NAX and builds a working library without the
/// NAX kernels, which nothing downstream would notice.
fn assert_nax_buildable(deployment_target: &str) {
    if !version_at_least(deployment_target, MACOS_DEPLOYMENT_TARGET_FLOOR) {
        let source = if env::var("MACOSX_DEPLOYMENT_TARGET").is_ok_and(|v| !v.is_empty()) {
            "MACOSX_DEPLOYMENT_TARGET"
        } else {
            "the build host's macOS version (MACOSX_DEPLOYMENT_TARGET is unset)"
        };
        panic!(
            "macOS deployment target {deployment_target} (from {source}) is below mlx-node's \
             floor {MACOS_DEPLOYMENT_TARGET_FLOOR}. MLX builds its NAX kernels only at \
             {MACOS_DEPLOYMENT_TARGET_FLOOR} or newer and would otherwise build without them \
             (MLX_METAL_NO_NAX). Set MACOSX_DEPLOYMENT_TARGET={MACOS_DEPLOYMENT_TARGET_FLOOR} \
             (or newer), or MLX_DISABLE_METAL=1 for a CPU-only build."
        );
    }
    let sdk = command_stdout(Command::new("xcrun").args(["-sdk", "macosx", "--show-sdk-version"]));
    if !sdk
        .as_deref()
        .is_some_and(|v| version_at_least(v, MACOS_DEPLOYMENT_TARGET_FLOOR))
    {
        panic!(
            "macOS SDK {} is below {MACOS_DEPLOYMENT_TARGET_FLOOR}: MLX would build without its \
             NAX kernels (MLX_METAL_NO_NAX). Select an Xcode with the macOS \
             {MACOS_DEPLOYMENT_TARGET_FLOOR} SDK or newer (`xcode-select`, `SDKROOT`).",
            sdk.as_deref().unwrap_or("<unknown>")
        );
    }
    let metal_version = (|| {
        let mut child = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-E", "-x", "metal", "-P", "-"])
            .arg(format!("-mmacosx-version-min={deployment_target}"))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        {
            use std::io::Write;
            child.stdin.take()?.write_all(b"__METAL_VERSION__\n").ok()?;
        }
        let output = child.wait_with_output().ok()?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .and_then(|line| line.trim().parse::<u32>().ok())
    })();
    if !metal_version.is_some_and(|version| version >= 400) {
        panic!(
            "Metal language version {metal_version:?} at -mmacosx-version-min={deployment_target} \
             is below 4.0: MLX would build without its NAX kernels (MLX_METAL_NO_NAX)."
        );
    }
}

/// One `.metal` → `.air` compile. `args` are every flag except the input,
/// output and dependency-file paths.
struct AirJob {
    src: PathBuf,
    air: PathBuf,
    args: Vec<String>,
}

/// FNV-1a: stable across Rust releases, unlike `DefaultHasher`, so a stamp
/// written by one toolchain still reads correctly under the next.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn command_stdout(cmd: &mut Command) -> Option<String> {
    let output = cmd.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Paths listed in a Make-format dependency file (`-MD -MF`), target excluded.
fn depfile_paths(text: &str) -> Vec<PathBuf> {
    let joined = text.replace("\\\n", " ");
    let body = joined.split_once(": ").map_or("", |(_, deps)| deps);
    let mut paths = Vec::new();
    let mut current = String::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&' ') {
            current.push(' ');
            chars.next();
        } else if c.is_whitespace() {
            if !current.is_empty() {
                paths.push(PathBuf::from(std::mem::take(&mut current)));
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        paths.push(PathBuf::from(current));
    }
    paths
}

/// The stamp a job's `.air` was built from: toolchain, flags, and the content
/// hash of the source and every header it included.
fn air_stamp(job: &AirJob, toolchain: &str, deps: &[PathBuf]) -> Option<String> {
    let mut stamp = format!("toolchain {toolchain}\nargs {}\n", job.args.join(" "));
    for dep in deps {
        let bytes = std::fs::read(dep).ok()?;
        stamp.push_str(&format!("{:016x} {}\n", fnv1a(&bytes), dep.display()));
    }
    Some(stamp)
}

fn air_is_current(job: &AirJob, toolchain: &str) -> bool {
    let stamp_path = job.air.with_extension("air.stamp");
    let (Ok(stamp), true) = (std::fs::read_to_string(&stamp_path), job.air.exists()) else {
        return false;
    };
    let deps: Vec<PathBuf> = stamp
        .lines()
        .skip(2)
        .filter_map(|line| line.split_once(' ').map(|(_, path)| PathBuf::from(path)))
        .collect();
    air_stamp(job, toolchain, &deps).as_deref() == Some(stamp.as_str())
}

fn compile_air(job: &AirJob, toolchain: &str) {
    if air_is_current(job, toolchain) {
        return;
    }
    let depfile = job.air.with_extension("air.d");
    let stamp_path = job.air.with_extension("air.stamp");
    let _ = std::fs::remove_file(&stamp_path);
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal"])
        .args(&job.args)
        .arg("-c")
        .arg(&job.src)
        .arg("-o")
        .arg(&job.air)
        .arg("-MD")
        .arg("-MF")
        .arg(&depfile)
        .status()
        .expect("Failed to execute xcrun metal");
    if !status.success() {
        panic!(
            "Metal compilation failed for {} ({}): exit code {:?}",
            job.src.display(),
            job.args.join(" "),
            status.code()
        );
    }
    // A missing stamp only costs a recompile next time.
    if let Some(stamp) = std::fs::read_to_string(&depfile)
        .ok()
        .and_then(|text| air_stamp(job, toolchain, &depfile_paths(&text)))
    {
        let _ = std::fs::write(&stamp_path, stamp);
    }
}

/// Dotted-version compare (`26.2` vs `26.0.1`), missing parts read as 0.
fn version_at_least(version: &str, floor: &str) -> bool {
    let parse = |v: &str| -> Vec<u64> { v.split('.').map(|p| p.parse().unwrap_or(0)).collect() };
    let (a, b) = (parse(version), parse(floor));
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        if x != y {
            return x > y;
        }
    }
    true
}

/// Build `<out_dir>/paged_attn.metallib`: the paged-attention kernels
/// (`crates/mlx-paged-attn/metal/`) and the prebuilt bridge kernels: K-quant
/// (`src/metal/kquant/`), segmented SDPA (`src/metal/segmented_sdpa/`) and
/// mixed affine `qmv_wide` (`src/metal/affine_mixed/`).
/// `mlx_paged_dispatch.cpp` resolves it at runtime next to the loaded binary
/// (the package build copies it beside `mlx.metallib`).
///
/// The bridge `.air` files use MLX's own kernel flags (`-fno-fast-math`, no
/// `-O`, no `-std`), which keep their bits equal to MLX's prebuilt kernels and
/// to the JIT builds they replace; the paged-attention flags (`-O3
/// -ffast-math`) would change them.
fn compile_paged_attn_metallib(
    manifest_dir: &Path,
    mlx_dir: &Path,
    out_dir: &Path,
    deployment_target: &str,
) -> PathBuf {
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

    let min_os = format!("-mmacosx-version-min={deployment_target}");

    let mut jobs = Vec::new();
    for file in [
        "attention/paged_attention.metal",
        "cache/reshape_and_cache.metal",
        "cache/copy_blocks.metal",
    ] {
        let args = vec![
            "-I".to_string(),
            metal_src_dir.display().to_string(),
            "-O3".to_string(),
            "-ffast-math".to_string(),
            min_os.clone(),
        ];
        jobs.push(AirJob {
            src: metal_src_dir.join(file),
            air: out_dir.join(file.replace('/', "_").replace(".metal", ".air")),
            args,
        });
    }

    let bridge_dir = manifest_dir.join("src").join("metal");
    let kquant_jobs = |name: &str| -> Vec<AirJob> {
        (0..3)
            .map(|dtype| {
                let args = vec![
                    "-x".to_string(),
                    "metal".to_string(),
                    "-fno-fast-math".to_string(),
                    format!("-DKQUANT_DTYPE={dtype}"),
                    "-I".to_string(),
                    mlx_dir.display().to_string(),
                    min_os.clone(),
                ];
                AirJob {
                    src: bridge_dir.join("kquant").join(format!("{name}.metal")),
                    air: out_dir.join(format!("{name}_{dtype}.air")),
                    args,
                }
            })
            .collect()
    };
    jobs.extend(kquant_jobs("kquant"));
    jobs.extend(kquant_jobs("kquant_nax"));
    // bfloat16-only M = 8 tensor-op kernels: one compile, no KQUANT_DTYPE.
    jobs.push(AirJob {
        src: bridge_dir.join("kquant").join("kquant_m8_nax.metal"),
        air: out_dir.join("kquant_m8_nax.air"),
        args: vec![
            "-x".to_string(),
            "metal".to_string(),
            "-fno-fast-math".to_string(),
            "-I".to_string(),
            mlx_dir.display().to_string(),
            min_os.clone(),
        ],
    });
    // Self-contained sources (no MLX headers), with the same flags. The
    // tensor-op verify kernel includes MetalPerformancePrimitives from the
    // SDK; `assert_nax_buildable` has already checked the deployment target.
    for file in [
        "segmented_sdpa/sdpa_segmented.metal",
        "segmented_sdpa/sdpa_segmented_nax.metal",
        "affine_mixed/affine_qmv_wide_mixed.metal",
    ] {
        let args = vec![
            "-x".to_string(),
            "metal".to_string(),
            "-fno-fast-math".to_string(),
            min_os.clone(),
        ];
        jobs.push(AirJob {
            src: bridge_dir.join(file),
            air: out_dir.join(file.replace('/', "_").replace(".metal", ".air")),
            args,
        });
    }

    let toolchain = [
        command_stdout(Command::new("xcrun").args(["-sdk", "macosx", "metal", "--version"])),
        command_stdout(Command::new("xcrun").args(["-sdk", "macosx", "--show-sdk-path"])),
        command_stdout(Command::new("xcrun").args(["-sdk", "macosx", "--show-sdk-version"])),
    ]
    .map(|part| part.unwrap_or_default())
    .join(" | ")
    .replace('\n', " ");
    std::thread::scope(|scope| {
        for job in &jobs {
            let toolchain = &toolchain;
            scope.spawn(move || compile_air(job, toolchain));
        }
    });

    let metallib_path = out_dir.join("paged_attn.metallib");
    let mut link_cmd = Command::new("xcrun");
    link_cmd.args(["-sdk", "macosx", "metallib"]);
    for job in &jobs {
        link_cmd.arg(&job.air);
    }
    link_cmd.arg("-o").arg(&metallib_path);
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

/// `CXX`, else the Xcode toolchain's clang++, else whatever `clang++` is on PATH.
fn macos_cxx_compiler() -> String {
    let default = xcrun_find("clang++").unwrap_or_else(|| "clang++".to_string());
    resolve_build_tool("CXX", &[default.as_str(), "/usr/bin/clang++", "clang++"])
}

/// Rust links with -nodefaultlibs. Clang 21's availability checks use
/// __isPlatformVersionAtLeast from compiler-rt, which the C++ driver
/// normally supplies automatically. Carry that runtime explicitly so
/// native tests and addons targeting older macOS versions both link.
/// Needed whether libmlx.a was built here or downloaded: the symbol is
/// referenced from MLX's own objects.
fn link_clang_runtime(cxx_compiler: &str) {
    if let Ok(runtime) = Command::new(cxx_compiler)
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
}

// =============================================================================
// Prebuilt MLX
// =============================================================================
//
// The upstream-MLX cmake build (static lib + headers + mlx.metallib) takes
// ~10 minutes per cargo profile and changes only when the submodule pin
// moves. `mlx-prebuilt.json` pins a GitHub Release holding that build output
// for the current pin; when the checked-out submodule matches the pin, the
// artifact is downloaded once per machine (into a cache shared across
// profiles) instead of rebuilt. Only MLX itself is prebuilt: the bridge
// (`libmlx_ffi.a`), its preambles and `paged_attn.metallib` change often,
// build in seconds, and stay source-built. docs/mlx-fork.md "Prebuilt MLX".

const PREBUILT_PIN_FILE: &str = "mlx-prebuilt.json";
const PREBUILT_MARKER: &str = ".release-tag";

/// Where the MLX static library, headers and `mlx.metallib` come from.
enum MlxSource {
    /// `cmake::Config::build()` of the submodule. `export`: also write the
    /// prebuilt artifact set there (`MLX_PREBUILT_EXPORT`).
    Source {
        reason: String,
        export: Option<PathBuf>,
    },
    /// An unpacked artifact: `lib/libmlx.a`, `lib/mlx.metallib`, `include/`.
    Prebuilt { dir: PathBuf, label: String },
}

/// The part of `mlx-prebuilt.json` this build needs. Parsed by string
/// search (the file is flat and its values are plain hex/paths) so the
/// build script does not pull in a JSON crate.
struct PrebuiltPin {
    repo: String,
    mlx_commit: String,
    release_tag: String,
    asset_file: Option<String>,
    asset_sha256: Option<String>,
}

/// The string value of the first `"key": "value"` at or after `from`.
fn json_string_after(text: &str, key: &str, from: usize) -> Option<String> {
    let needle = format!("\"{key}\"");
    let key_at = from + text.get(from..)?.find(&needle)?;
    let rest = text[key_at + needle.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn parse_prebuilt_pin(text: &str, triple: &str) -> Option<PrebuiltPin> {
    let repo = json_string_after(text, "repo", 0)?;
    let mlx_commit = json_string_after(text, "mlx_commit", 0)?;
    let release_tag = json_string_after(text, "release_tag", 0)?;
    let mut asset_file = None;
    let mut asset_sha256 = None;
    if let Some(assets_at) = text.find("\"assets\"")
        && let Some(triple_at) = text[assets_at..]
            .find(&format!("\"{triple}\""))
            .map(|at| assets_at + at)
    {
        // The asset object ends at its closing brace; keys past it belong
        // to the next triple.
        let object_end = text[triple_at..]
            .find('}')
            .map_or(text.len(), |at| triple_at + at);
        let object = &text[..object_end];
        asset_file = json_string_after(object, "file", triple_at);
        asset_sha256 = json_string_after(object, "sha256", triple_at);
    }
    Some(PrebuiltPin {
        repo,
        mlx_commit,
        release_tag,
        asset_file,
        asset_sha256,
    })
}

fn env_non_empty(key: &str) -> Option<String> {
    env::var(key).ok().filter(|v| !v.is_empty())
}

/// Root of the download/unpack cache. Shared across cargo profiles on
/// purpose: one download per machine. `packages/core/metallib-select.ts`
/// (`resolvePrebuiltMetallibPath`) mirrors this rule and must stay in step.
fn prebuilt_cache_root(manifest_dir: &Path) -> PathBuf {
    if let Some(dir) = env_non_empty("MLX_PREBUILT_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    for key in ["CARGO_TARGET_DIR", "CARGO_BUILD_TARGET_DIR"] {
        if let Some(dir) = env_non_empty(key) {
            return PathBuf::from(dir).join("mlx-prebuilt");
        }
    }
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("CARGO_MANIFEST_DIR is <repo>/crates/mlx-sys")
        .join("target")
        .join("mlx-prebuilt")
}

/// `mlx-prebuilt-<mlx sha12>-<build key8>`. The release is keyed by every
/// input that shapes the artifact: the submodule commit and the cmake build
/// key (`prebuilt_build_key`). Changing a cmake define or a file under
/// `cmake/` without moving the submodule therefore names a NEW release, and
/// a pin that still names the old one falls back to the source build instead
/// of linking an artifact built from other inputs.
fn release_tag_for(commit: &str, build_key: &str) -> String {
    format!(
        "mlx-prebuilt-{}-{}",
        &commit[..commit.len().min(12)],
        &build_key[..build_key.len().min(8)]
    )
}

/// The cmake defines that are the same on every machine for a given target
/// (no compiler paths, no SDK root, no OUT_DIR). `build_mlx_from_source`
/// starts from this list, and `prebuilt_build_key` hashes it, so the two
/// cannot drift apart.
fn portable_cmake_defines(
    build_metal: bool,
    target_os: &str,
    target_arch: &str,
    deployment_target: Option<&str>,
) -> Vec<(String, String)> {
    let mut defines: Vec<(String, String)> = Vec::new();
    let mut define = |key: &str, value: &str| defines.push((key.to_string(), value.to_string()));
    // CMakeCache.txt from an older mlx-sys may still name the deleted
    // metal-residency/overlay.cmake; configure fails until it is cleared.
    define("CMAKE_PROJECT_INCLUDE", "");
    define("MLX_BUILD_TESTS", "OFF");
    define("MLX_BUILD_EXAMPLES", "OFF");
    define("MLX_BUILD_BENCHMARKS", "OFF");
    define("MLX_BUILD_PYTHON_BINDINGS", "OFF");
    define("BUILD_SHARED_LIBS", "OFF");
    define("MLX_BUILD_METAL", if build_metal { "ON" } else { "OFF" });
    // `CMAKE_OSX_ARCHITECTURES` is an Apple-only knob; setting it on Linux
    // confuses the GCC/CUDA toolchain. Only emit it on macOS.
    if target_os == "macos" {
        // napi-rs invokes Cargo with an explicit Darwin target even when that
        // target is the native Apple-silicon host. Some CMake launches then
        // retain an x86_64 host processor while honoring the arm64 compiler
        // flags, and upstream MLX rejects the configure before it can inspect
        // CMAKE_OSX_ARCHITECTURES. Pin both views of the target architecture.
        let arch = if target_arch == "aarch64" {
            "arm64"
        } else {
            "x86_64"
        };
        define("CMAKE_SYSTEM_NAME", "Darwin");
        define("CMAKE_SYSTEM_PROCESSOR", arch);
        define("CMAKE_OSX_ARCHITECTURES", arch);
        // A -D define overrides a stale CMAKE_OSX_DEPLOYMENT_TARGET already
        // recorded in CMakeCache.txt (e.g. a CI-restored cargo cache), which
        // the environment variable alone cannot.
        if let Some(deployment_target) = deployment_target {
            define("CMAKE_OSX_DEPLOYMENT_TARGET", deployment_target);
        }
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
        define("MLX_BUILD_CUDA", "ON");
        define("MLX_BUILD_METAL", "OFF");
        define("MLX_BUILD_CPU", "ON");
        define("MLX_CUDA_ARCHITECTURES", &cuda_archs);
        define("CMAKE_BUILD_TYPE", "Release");
    }
    defines
}

/// Revision of the artifact-shaping logic that is NOT captured by the
/// portable defines or `cmake/`: the macOS toolchain selection and
/// `-isysroot` flags in `build_mlx_from_source`, and the export layout in
/// `export_prebuilt`. Those use host-specific paths, so they cannot be
/// hashed into the key directly. BUMP THIS when changing any of them, or
/// the workflow computes the existing immutable tag and keeps the old asset.
const PREBUILT_BUILD_KEY_REVISION: u32 = 1;

/// sha256 over the portable cmake defines, every file under
/// `crates/mlx-sys/cmake/` (they run inside MLX's configure) and
/// `PREBUILT_BUILD_KEY_REVISION`. Together with the submodule commit this is
/// everything that decides the artifact's bytes, up to the toolchain.
fn prebuilt_build_key(manifest_dir: &Path, defines: &[(String, String)]) -> io::Result<String> {
    let mut canonical = format!("revision={PREBUILT_BUILD_KEY_REVISION}\n");
    for (key, value) in defines {
        canonical.push_str(&format!("-D{key}={value}\n"));
    }
    let cmake_dir = manifest_dir.join("cmake");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&cmake_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect();
    files.sort();
    for file in files {
        let name = file.file_name().unwrap().to_string_lossy().into_owned();
        canonical.push_str(&format!("cmake/{name}={}\n", sha256_file(&file)?));
    }
    let tmp = env::temp_dir().join(format!("mlx-prebuilt-key-{}.txt", std::process::id()));
    std::fs::write(&tmp, canonical)?;
    let key = sha256_file(&tmp);
    let _ = std::fs::remove_file(&tmp);
    key
}

/// `(HEAD sha, dirty)` of the submodule checkout, or the git error.
fn submodule_state(mlx_dir: &Path) -> Result<(String, bool), String> {
    let head = command_stdout(
        Command::new("git")
            .args(["-C"])
            .arg(mlx_dir)
            .args(["rev-parse", "HEAD"]),
    )
    .ok_or_else(|| "git rev-parse HEAD failed (not a git checkout?)".to_string())?;
    let status = command_stdout(Command::new("git").args(["-C"]).arg(mlx_dir).args([
        "status",
        "--porcelain",
        "--untracked-files=no",
    ]))
    .ok_or_else(|| "git status failed".to_string())?;
    Ok((head, !status.is_empty()))
}

fn sha256_file(path: &Path) -> io::Result<String> {
    // The host tools avoid a hashing build-dependency: `shasum` ships with
    // macOS, `sha256sum` with coreutils on Linux.
    let output = if cfg!(target_os = "macos") {
        Command::new("shasum")
            .args(["-a", "256"])
            .arg(path)
            .output()
    } else {
        Command::new("sha256sum").arg(path).output()
    }?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "sha256 of {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| io::Error::other("empty sha256 output"))
}

fn marker_content(release_tag: &str, sha256: &str) -> String {
    format!("{release_tag}\nsha256:{sha256}\n")
}

/// The files every consumer path reads; a cache dir missing one of them is
/// treated as absent rather than discovered at link time.
fn prebuilt_layout_complete(dir: &Path) -> bool {
    dir.join("lib").join("libmlx.a").is_file()
        && dir.join("lib").join("mlx.metallib").is_file()
        && dir.join("include").join("mlx").is_dir()
}

/// A complete unpack of the pinned artifact: marker (tag + digest) matches
/// and the layout is there. The marker is written last, so a partial
/// unpack never passes.
fn prebuilt_unpack_is_current(dir: &Path, release_tag: &str, sha256: &str) -> bool {
    std::fs::read_to_string(dir.join(PREBUILT_MARKER))
        .is_ok_and(|text| text.trim() == marker_content(release_tag, sha256).trim())
        && prebuilt_layout_complete(dir)
}

fn curl_download(url: &str, dest: &Path, token: Option<&str>) -> Result<(), String> {
    let mut cmd = Command::new("curl");
    // `-L` follows the redirect to objects.githubusercontent.com; curl drops
    // the Authorization header across hosts by default, which is wanted.
    cmd.args(["-fsSL", "--retry", "3", "-o"]).arg(dest);
    if let Some(token) = token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {token}"));
    }
    let output = cmd
        .arg(url)
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "curl exit {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// Download, verify and unpack the pinned artifact into `unpack_dir`.
/// `Err` = the download itself failed (the caller falls back to source);
/// a digest mismatch panics instead: the pin or the asset is wrong and a
/// silent rebuild would hide that.
fn fetch_prebuilt(
    pin: &PrebuiltPin,
    asset_file: &str,
    sha256: &str,
    unpack_dir: &Path,
) -> Result<(), String> {
    let parent = unpack_dir
        .parent()
        .expect("unpack dir is <cache>/<tag>/<triple>");
    std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    let url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        pin.repo, pin.release_tag, asset_file
    );
    let archive = parent.join(format!(".{asset_file}.part-{}", std::process::id()));
    let _ = std::fs::remove_file(&archive);
    // A token only avoids rate limiting on CI (public release assets need
    // none); if GitHub rejects the token form, the anonymous form still
    // has to be tried before giving up on the download.
    let token = env_non_empty("GITHUB_TOKEN").or_else(|| env_non_empty("GH_TOKEN"));
    let result = match token.as_deref() {
        Some(token) => curl_download(&url, &archive, Some(token))
            .or_else(|_| curl_download(&url, &archive, None)),
        None => curl_download(&url, &archive, None),
    };
    if let Err(error) = result {
        let _ = std::fs::remove_file(&archive);
        return Err(format!("download of {url} failed: {error}"));
    }

    let actual = sha256_file(&archive).map_err(|e| e.to_string())?;
    if actual != sha256.to_ascii_lowercase() {
        let _ = std::fs::remove_file(&archive);
        panic!(
            "mlx-sys: sha256 mismatch for {url}: pinned {sha256}, downloaded {actual}. The asset was \
             replaced or the pin in {PREBUILT_PIN_FILE} is wrong; refusing to use it. Set \
             MLX_PREBUILT=0 to build MLX from source meanwhile."
        );
    }

    let tmp = parent.join(format!(".unpack-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(&archive)
        .arg("-C")
        .arg(&tmp)
        .status()
        .map_err(|e| format!("cannot run tar: {e}"))?;
    let _ = std::fs::remove_file(&archive);
    if !status.success() {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(format!("tar -xzf of {asset_file} failed: {status}"));
    }
    if !prebuilt_layout_complete(&tmp) {
        let _ = std::fs::remove_dir_all(&tmp);
        panic!(
            "mlx-sys: {url} does not have the prebuilt layout (lib/libmlx.a, lib/mlx.metallib, \
             include/mlx); the release asset is malformed."
        );
    }
    // Another cargo (other target dir, same cache) may have finished first.
    if prebuilt_unpack_is_current(unpack_dir, &pin.release_tag, sha256) {
        let _ = std::fs::remove_dir_all(&tmp);
        return Ok(());
    }
    // Move a stale unpack aside with a rename instead of deleting it in
    // place, so a concurrent reader never sees a half-removed dir and the
    // final rename cannot fail on "directory not empty".
    let stale = parent.join(format!(".stale-{}", std::process::id()));
    if unpack_dir.exists() {
        std::fs::rename(unpack_dir, &stale).map_err(|e| {
            format!(
                "rename {} -> {}: {e}",
                unpack_dir.display(),
                stale.display()
            )
        })?;
    }
    std::fs::rename(&tmp, unpack_dir)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), unpack_dir.display()))?;
    let _ = std::fs::remove_dir_all(&stale);
    std::fs::write(
        unpack_dir.join(PREBUILT_MARKER),
        marker_content(&pin.release_tag, sha256),
    )
    .map_err(|e| format!("write marker: {e}"))?;
    Ok(())
}

/// A field of an unpacked artifact's `manifest.json`.
fn prebuilt_manifest_field(dir: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
    json_string_after(&text, key, 0)
}

fn prebuilt_manifest_deployment_target(dir: &Path) -> Option<String> {
    prebuilt_manifest_field(dir, "deployment_target")
}

/// Decide between the prebuilt artifact and the source build. Every knob
/// is a `rerun-if-env-changed` so flipping one re-runs this script.
fn resolve_mlx_source(
    manifest_dir: &Path,
    mlx_dir: &Path,
    triple: &str,
    build_metal: bool,
    deployment_target: Option<&str>,
) -> MlxSource {
    for key in [
        "MLX_PREBUILT",
        "MLX_PREBUILT_DIR",
        "MLX_PREBUILT_EXPORT",
        "MLX_PREBUILT_CACHE_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-changed={PREBUILT_PIN_FILE}");
    let source = |reason: &str| MlxSource::Source {
        reason: reason.to_string(),
        export: None,
    };

    if let Some(dir) = env_non_empty("MLX_PREBUILT_EXPORT") {
        return MlxSource::Source {
            reason: "MLX_PREBUILT_EXPORT is set".to_string(),
            export: Some(PathBuf::from(dir)),
        };
    }
    // The kill switch beats a stale MLX_PREBUILT_DIR left in the environment.
    if env::var("MLX_PREBUILT").is_ok_and(|v| v == "0") {
        return source("MLX_PREBUILT=0");
    }
    if let Some(dir) = env_non_empty("MLX_PREBUILT_DIR") {
        let dir = PathBuf::from(dir);
        if !dir.join("lib").join("libmlx.a").is_file() || !dir.join("include").join("mlx").is_dir()
        {
            panic!(
                "mlx-sys: MLX_PREBUILT_DIR={} has no lib/libmlx.a or include/mlx; expected an \
                 unpacked prebuilt MLX artifact (lib/, include/, manifest.json).",
                dir.display()
            );
        }
        if let (Some(recorded), Some(wanted)) =
            (prebuilt_manifest_deployment_target(&dir), deployment_target)
            && recorded != wanted
        {
            println!(
                "cargo:warning=mlx-sys: MLX_PREBUILT_DIR was built for macOS deployment target \
                 {recorded}, this build targets {wanted}"
            );
        }
        let label = prebuilt_manifest_field(&dir, "release_tag")
            .unwrap_or_else(|| "<no manifest.json>".to_string());
        return MlxSource::Prebuilt {
            label: format!("{label} (MLX_PREBUILT_DIR)"),
            dir,
        };
    }
    if triple != "aarch64-apple-darwin" || !build_metal {
        return source(&format!(
            "no prebuilt MLX for {triple} (Metal {})",
            if build_metal { "on" } else { "off" }
        ));
    }
    let Some(deployment_target) = deployment_target else {
        return source("no macOS deployment target");
    };

    let pin_path = manifest_dir.join(PREBUILT_PIN_FILE);
    let pin_text = match std::fs::read_to_string(&pin_path) {
        Ok(text) => text,
        Err(error) => return source(&format!("cannot read {PREBUILT_PIN_FILE}: {error}")),
    };
    let Some(pin) = parse_prebuilt_pin(&pin_text, triple) else {
        return source(&format!(
            "{PREBUILT_PIN_FILE} is missing repo/mlx_commit/release_tag"
        ));
    };
    let (Some(asset_file), Some(sha256)) = (
        pin.asset_file.clone(),
        pin.asset_sha256.clone().filter(|v| !v.is_empty()),
    ) else {
        // Never download without a digest to check it against.
        return source(&format!(
            "no trusted sha256 pinned for {triple} in {PREBUILT_PIN_FILE}"
        ));
    };
    let head = match submodule_state(mlx_dir) {
        Err(error) => return source(&format!("mlx submodule: {error}")),
        Ok((_, true)) => return source("mlx submodule has local modifications"),
        Ok((head, false)) if head != pin.mlx_commit => {
            return source(&format!(
                "mlx submodule is at {} but {PREBUILT_PIN_FILE} pins {}; push the gitlink bump so \
                 the `MLX Prebuilt` workflow publishes its release, then update the pin \
                 (docs/mlx-fork.md, \"Prebuilt MLX\")",
                &head[..12],
                &pin.mlx_commit[..12.min(pin.mlx_commit.len())]
            ));
        }
        Ok((head, false)) => head,
    };
    // The tag encodes the cmake build inputs too: a pin left pointing at a
    // release built with other defines (or another `cmake/` include) is
    // refused rather than linked.
    let defines = portable_cmake_defines(build_metal, "macos", "aarch64", Some(deployment_target));
    let build_key = match prebuilt_build_key(manifest_dir, &defines) {
        Ok(key) => key,
        Err(error) => return source(&format!("cannot compute the prebuilt build key: {error}")),
    };
    let expected_tag = release_tag_for(&head, &build_key);
    if pin.release_tag != expected_tag {
        return source(&format!(
            "{PREBUILT_PIN_FILE} pins release {} but this checkout's MLX build inputs name {}; the \
             cmake defines or crates/mlx-sys/cmake/ changed since that release was built, so a \
             push of build.rs publishes the new one and the pin needs updating",
            pin.release_tag, expected_tag
        ));
    }

    let unpack_dir = prebuilt_cache_root(manifest_dir)
        .join(&pin.release_tag)
        .join(triple);
    if !prebuilt_unpack_is_current(&unpack_dir, &pin.release_tag, &sha256)
        && let Err(error) = fetch_prebuilt(&pin, &asset_file, &sha256, &unpack_dir)
    {
        return source(&error);
    }
    // The archive's own manifest must agree with the pin; the digest only
    // proves the bytes are the release's, not that the release is the
    // right one.
    for (field, want) in [
        ("mlx_commit", head.as_str()),
        ("triple", triple),
        ("release_tag", expected_tag.as_str()),
    ] {
        match prebuilt_manifest_field(&unpack_dir, field) {
            Some(ref have) if have == want => {}
            have => {
                return source(&format!(
                    "prebuilt {} records {field} {} but this build needs {want}; the asset does not \
                     match the pin",
                    pin.release_tag,
                    have.as_deref().unwrap_or("<missing>")
                ));
            }
        }
    }
    match prebuilt_manifest_deployment_target(&unpack_dir) {
        Some(recorded) if recorded == deployment_target => {}
        recorded => {
            return source(&format!(
                "prebuilt {} was built for macOS deployment target {}, this build targets \
                 {deployment_target}",
                pin.release_tag,
                recorded.as_deref().unwrap_or("<unknown>")
            ));
        }
    }
    MlxSource::Prebuilt {
        label: pin.release_tag,
        dir: unpack_dir,
    }
}

fn copy_dir_recursive(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Write the prebuilt artifact set from a finished cmake install tree
/// (`MLX_PREBUILT_EXPORT`). No `build/` tree, bridge library or
/// `paged_attn.metallib`: those stay source-built on the consumer.
#[allow(clippy::too_many_arguments)]
fn export_prebuilt(
    dst: &Path,
    export_dir: &Path,
    mlx_commit: &str,
    release_tag: &str,
    build_key: &str,
    triple: &str,
    deployment_target: &str,
    metal_path_sentinel: &str,
    cmake_defines: &[(String, String)],
) -> io::Result<()> {
    let _ = std::fs::remove_dir_all(export_dir);
    let lib_dir = export_dir.join("lib");
    std::fs::create_dir_all(&lib_dir)?;
    for (name, required) in [
        ("libmlx.a", true),
        ("libjaccl.a", false),
        ("mlx.metallib", true),
    ] {
        let from = dst.join("lib").join(name);
        if from.is_file() {
            std::fs::copy(&from, lib_dir.join(name))?;
        } else if required {
            return Err(io::Error::other(format!(
                "export: {} is missing from the cmake install tree",
                from.display()
            )));
        }
    }
    copy_dir_recursive(&dst.join("include"), &export_dir.join("include"))?;
    let built_at = command_stdout(Command::new("date").args(["-u", "+%Y-%m-%dT%H:%M:%SZ"]))
        .unwrap_or_default();
    let defines = cmake_defines
        .iter()
        .map(|(k, v)| format!("    \"-D{}={}\"", json_escape(k), json_escape(v)))
        .collect::<Vec<_>>()
        .join(",\n");
    let manifest = format!(
        "{{\n  \"mlx_commit\": \"{mlx_commit}\",\n  \"release_tag\": \"{release_tag}\",\n  \
         \"build_key\": \"{build_key}\",\n  \"triple\": \"{triple}\",\n  \
         \"deployment_target\": \"{deployment_target}\",\n  \"metal_path_sentinel\": \"{}\",\n  \
         \"built_at\": \"{built_at}\",\n  \"cmake_defines\": [\n{defines}\n  ]\n}}\n",
        json_escape(metal_path_sentinel)
    );
    std::fs::write(export_dir.join("manifest.json"), manifest)?;
    Ok(())
}

/// Directories next to test binaries where a metallib must sit for the
/// colocated runtime lookups (MLX's `load_colocated_library` and
/// `paged_attn_metallib_path`): `target/<profile>/`, `target/<profile>/deps/`
/// and, for the `target/<triple>/<profile>` layout, the sibling `deps/`.
/// Derived from OUT_DIR (`.../<profile>/build/mlx-sys-<hash>/out`).
fn test_binary_sinks(out_dir: &Path) -> Vec<PathBuf> {
    let Some(profile_dir) =
        find_ancestor_with_name(out_dir, "build").and_then(|p| p.parent().map(Path::to_path_buf))
    else {
        return Vec::new();
    };
    let mut sinks = vec![profile_dir.clone(), profile_dir.join("deps")];
    if let Some(parent) = profile_dir.parent()
        && parent.file_name().is_some_and(|n| n != "target")
    {
        sinks.push(parent.join("deps"));
    }
    sinks.into_iter().filter(|sink| sink.exists()).collect()
}

/// Best effort, like the original paged_attn copy: a sink that cannot be
/// written only costs the colocated lookup in that one layout.
fn copy_to_sinks(file: &Path, sinks: &[PathBuf]) {
    let name = file.file_name().expect("metallib has a file name");
    for sink in sinks {
        let _ = std::fs::copy(file, sink.join(name));
    }
}

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=mlx");
    // `cmake/` feeds the prebuilt build key (`prebuilt_build_key`) in every
    // mode, not only the source build's configure, so an edit there must
    // re-run this script or a restored target keeps a prebuilt whose key no
    // longer matches.
    println!("cargo:rerun-if-changed=cmake");
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
    // Resolved once: it probes the host (`sw_vers`).
    let deployment_target = is_macos.then(macos_deployment_target);
    let paged_metallib_path = if build_metal {
        let deployment_target = deployment_target
            .as_deref()
            .expect("a Metal build targets macOS");
        assert_nax_buildable(deployment_target);
        Some(compile_paged_attn_metallib(
            &manifest_dir,
            &mlx_dir,
            &out_dir_path,
            deployment_target,
        ))
    } else {
        None
    };

    let triple = format!(
        "{target_arch}-{}",
        match target_os.as_str() {
            "macos" => "apple-darwin".to_string(),
            "linux" => format!(
                "unknown-linux-{}",
                env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "gnu".into())
            ),
            other => format!("unknown-{other}"),
        }
    );
    let mlx_source = resolve_mlx_source(
        &manifest_dir,
        &mlx_dir,
        &triple,
        build_metal,
        deployment_target.as_deref(),
    );

    // `mlx_root` holds `lib/` (libmlx.a, mlx.metallib) and `include/`:
    // the cmake install tree (== OUT_DIR) or the unpacked prebuilt dir.
    // `lib_dirs` are the link-search dirs; in source mode also the dirs
    // that receive a `paged_attn.metallib` copy.
    let (mlx_root, lib_dirs, prebuilt) = match mlx_source {
        MlxSource::Source { reason, export } => {
            println!("cargo:warning=mlx-sys: building MLX from source ({reason})");
            let export = export.map(|dir| {
                // The artifact's tag follows the commit actually built, not
                // the (possibly not yet updated) pin.
                let commit = match submodule_state(&mlx_dir) {
                    Ok((head, dirty)) => {
                        if dirty {
                            println!(
                                "cargo:warning=mlx-sys: exporting a prebuilt from a submodule with \
                                 local modifications"
                            );
                        }
                        head
                    }
                    Err(error) => panic!(
                        "mlx-sys: MLX_PREBUILT_EXPORT needs the submodule commit, but {error}"
                    ),
                };
                let portable = portable_cmake_defines(
                    build_metal,
                    &target_os,
                    &target_arch,
                    deployment_target.as_deref(),
                );
                let build_key = prebuilt_build_key(&manifest_dir, &portable)
                    .unwrap_or_else(|error| panic!("mlx-sys: prebuilt build key: {error}"));
                let tag = release_tag_for(&commit, &build_key);
                (dir, commit, build_key, tag)
            });
            let (dst, defines, sentinel) = build_mlx_from_source(
                &manifest_dir,
                &mlx_dir,
                &out_dir_path,
                build_metal,
                &target_os,
                &target_arch,
                deployment_target.as_deref(),
                // The sentinel is a Metal load path; a CPU/CUDA export has
                // no metallib to name with it.
                export
                    .as_ref()
                    .filter(|_| build_metal)
                    .map(|(_, _, _, tag)| tag.clone()),
            );
            if let Some((export_dir, commit, build_key, tag)) = export {
                export_prebuilt(
                    &dst,
                    &export_dir,
                    &commit,
                    &tag,
                    &build_key,
                    &triple,
                    deployment_target.as_deref().unwrap_or(""),
                    sentinel.as_deref().unwrap_or(""),
                    &defines,
                )?;
                println!(
                    "cargo:warning=mlx-sys: exported prebuilt MLX {tag} to {}",
                    export_dir.display()
                );
            }
            let lib_candidates = [
                dst.join("lib"),
                dst.join("build").join("lib"),
                dst.join("build").join("Release"),
                dst.join("build").join("mlx"),
                dst.join("build").join("mlx").join("lib"),
            ];
            let lib_dirs: Vec<PathBuf> = lib_candidates
                .into_iter()
                .filter(|candidate| candidate.exists())
                .collect();
            if lib_dirs.is_empty() {
                panic!(
                    "unable to locate MLX build artifacts under {}; expected lib directories to exist",
                    dst.display()
                );
            }
            (dst, lib_dirs, false)
        }
        MlxSource::Prebuilt { dir, label } => {
            println!(
                "cargo:warning=mlx-sys: using prebuilt MLX {label} from {}",
                dir.display()
            );
            // A wiped cache re-runs this script (and re-downloads) instead
            // of failing at link time.
            println!(
                "cargo:rerun-if-changed={}",
                dir.join("lib").join("libmlx.a").display()
            );
            (dir.clone(), vec![dir.join("lib")], true)
        }
    };
    for lib_dir in &lib_dirs {
        add_link_search(lib_dir);
    }

    // Co-locate `paged_attn.metallib` with `mlx.metallib`. Both must
    // ship next to the loaded binary at runtime (see
    // `packages/core/build.ts::copyMetallibs`, which ships the bound
    // build's `mlx.metallib` and `paged_attn.metallib` into the addon
    // directory) and next to cargo test binaries (`test_binary_sinks`),
    // so `cargo test` works without manual env var setup.
    //
    // Source mode: `paged_attn.metallib` goes into every lib dir of the
    // install tree (`out/lib` next to `mlx.metallib`). Prebuilt mode: the
    // prebuilt dir is shared across profiles, so nothing profile-specific
    // is written there; instead `OUT_DIR/lib` gets `paged_attn.metallib`
    // plus a hard link (copy if the cache is on another volume) of the
    // prebuilt `mlx.metallib`, so `OUT_DIR/lib` holds both metallibs in
    // both modes and build.ts never has to re-derive the cache location
    // from the environment (a relative MLX_PREBUILT_CACHE_DIR or
    // CARGO_TARGET_DIR resolves differently under cargo and under yarn).
    //
    // `mlx.metallib` is copied next to the test binaries in BOTH modes.
    // MLX loads a colocated `mlx.metallib` before the path baked into
    // libmlx.a, so once any build has put one there, every later build in
    // the same target dir runs against it: a source build after a prebuilt
    // one (or after a submodule bump) would otherwise pair fresh kernels
    // with a stale library and fail with "Unable to load kernel". Keeping
    // the copy in step with the linked libmlx closes that; prebuilt mode
    // needs it anyway, since its baked path names the producer's machine.
    if let Some(paged_metallib) = paged_metallib_path.as_ref() {
        let paged_dests: Vec<PathBuf> = if prebuilt {
            let lib = out_dir_path.join("lib");
            std::fs::create_dir_all(&lib)
                .map_err(|error| build_file_error("create", &lib, error))?;
            let src = mlx_root.join("lib").join("mlx.metallib");
            let dst = lib.join("mlx.metallib");
            let _ = std::fs::remove_file(&dst);
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst).map_err(|error| build_file_error("copy", &dst, error))?;
            }
            vec![lib]
        } else {
            lib_dirs.clone()
        };
        for dir in &paged_dests {
            let dst_path = dir.join("paged_attn.metallib");
            if let Err(e) = std::fs::copy(paged_metallib, &dst_path) {
                panic!(
                    "Failed to copy paged_attn.metallib to {}: {e}",
                    dst_path.display()
                );
            }
        }
        let sinks = test_binary_sinks(&out_dir_path);
        copy_to_sinks(paged_metallib, &sinks);
        copy_to_sinks(&mlx_root.join("lib").join("mlx.metallib"), &sinks);
    }

    println!("cargo:rustc-link-lib=static=mlx");

    if is_macos {
        link_clang_runtime(&macos_cxx_compiler());
        if build_metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=QuartzCore");
            // GPU core count for the K-quant split rule (mlx_kquant_metal.cpp).
            println!("cargo:rustc-link-lib=framework=IOKit");
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
    let include_generated = mlx_root.join("include");

    let mut bridge = cc::Build::new();
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
        // The build-tree origin of paged_attn.metallib, as MLX bakes its
        // METAL_PATH: a last-resort runtime lookup, and the string
        // `packages/core/metallib-select.ts` reads to bind the addon to this
        // mlx-sys build (with prebuilt MLX the METAL_PATH no longer names it).
        let paged_origin = paged_metallib_path
            .as_ref()
            .expect("a Metal build compiles paged_attn.metallib");
        bridge.define(
            "MLX_NODE_PAGED_ATTN_METALLIB",
            format!("\"{}\"", paged_origin.display()).as_str(),
        );
        build_bridge_preambles(&mut bridge, &mlx_dir, &src_dir, &out_dir_path)?;
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

/// Configure and run the MLX cmake build. Returns the install tree (==
/// OUT_DIR), the `-D` defines passed (recorded in an exported manifest) and,
/// with `export_release_tag`, the METAL_PATH string baked into libmlx.a.
#[allow(clippy::too_many_arguments)]
fn build_mlx_from_source(
    manifest_dir: &Path,
    mlx_dir: &Path,
    out_dir: &Path,
    build_metal: bool,
    target_os: &str,
    target_arch: &str,
    deployment_target: Option<&str>,
    export_release_tag: Option<String>,
) -> (PathBuf, Vec<(String, String)>, Option<String>) {
    let mut cfg = cmake::Config::new(mlx_dir);
    let mut defines =
        portable_cmake_defines(build_metal, target_os, target_arch, deployment_target);
    let mut define = |key: &str, value: &str| defines.push((key.to_string(), value.to_string()));

    // MLX bakes `${MLX_METAL_PATH}/mlx.metallib` into libmlx.a as the
    // last-resort load path (`device.cpp` `default_mtllib_path`) AND
    // writes/installs the built metallib from that directory
    // (`kernels/CMakeLists.txt`), so it must be a real, writable dir on the
    // producer. For an exported artifact it is a dir under OUT_DIR whose
    // tail `mlx-prebuilt/<release_tag>/kernels/mlx.metallib` the consumer
    // (`metallib-select.ts`) recognizes and resolves to its own unpacked
    // copy. Ordinary builds keep MLX's default (`<build>/.../kernels/`).
    let sentinel = export_release_tag.map(|tag| {
        let kernels = out_dir.join("mlx-prebuilt").join(tag).join("kernels");
        std::fs::create_dir_all(&kernels)
            .unwrap_or_else(|e| panic!("create {}: {e}", kernels.display()));
        define(
            "MLX_METAL_PATH",
            kernels.to_str().expect("OUT_DIR is UTF-8"),
        );
        format!("{}/mlx.metallib", kernels.display())
    });

    // Machine-specific defines (tool paths, SDK) follow; they are deliberately
    // not part of `portable_cmake_defines` and so not of the build key.
    if target_os == "macos" {
        let default_c_compiler = xcrun_find("clang").unwrap_or_else(|| "clang".to_string());
        let default_ar = xcrun_find("ar").unwrap_or_else(|| "/usr/bin/ar".to_string());
        let default_ranlib = xcrun_find("ranlib").unwrap_or_else(|| "/usr/bin/ranlib".to_string());
        let c_compiler = resolve_build_tool(
            "CC",
            &[default_c_compiler.as_str(), "/usr/bin/clang", "clang"],
        );
        let cxx_compiler = macos_cxx_compiler();
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
        define("CMAKE_C_COMPILER", &c_compiler);
        define("CMAKE_CXX_COMPILER", &cxx_compiler);
        define("CMAKE_AR", &ar);
        define("CMAKE_RANLIB", &ranlib);
        define("CMAKE_C_COMPILER_AR", &ar);
        define("CMAKE_CXX_COMPILER_AR", &ar);
        define("CMAKE_C_COMPILER_RANLIB", &ranlib);
        define("CMAKE_CXX_COMPILER_RANLIB", &ranlib);
        cfg.cflag(format!("-isysroot {sdk_path}"))
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
        define(
            "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
            switch_diagnostic
                .to_str()
                .expect("switch-exhaustiveness.cmake path is not valid UTF-8"),
        );
    }

    // Defines are applied together: `define` above only records them, so
    // the same list can be written into an exported artifact's manifest.
    for (key, value) in &defines {
        cfg.define(key, value);
    }
    let dst = cfg.build();
    (dst, defines, sentinel)
}

/// MLX kernel headers as source text for the JIT-built bridge kernels.
/// The precompiled-metallib build does not export them, so private copies
/// are generated from the linked sources so they cannot drift.
fn build_bridge_preambles(
    bridge: &mut cc::Build,
    mlx_dir: &Path,
    src_dir: &Path,
    out_dir: &Path,
) -> io::Result<()> {
    let preambles = out_dir.join("quantized-preambles");
    let script = mlx_dir.join("mlx/backend/metal/make_compiled_preamble.sh");
    for (source_name, name) in [
        ("utils", "utils"),
        ("steel/gemm/gemm", "gemm"),
        ("quantized_utils", "quantized_utils"),
        ("steel/gemm/nax", "nax"),
        ("steel/attn/kernels/steel_attention", "steel_attention"),
    ] {
        let status = Command::new("bash")
            .arg(&script)
            .arg(&preambles)
            .arg("clang")
            .arg(mlx_dir)
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
        std::fs::write(&path, source)
            .map_err(|error| build_file_error("write private quantized preamble", &path, error))?;
        bridge.file(path);
        println!(
            "cargo:rerun-if-changed={}",
            mlx_dir
                .join(format!("mlx/backend/metal/kernels/{source_name}.h"))
                .display()
        );
    }
    // The K-quant headers as source text, for the custom kernels that
    // reuse their decoders (the K-quant ops themselves run from the
    // prebuilt paged_attn.metallib). Their one project include, the
    // per-mode traits in kquant_mode.h, is inlined in place so each
    // preamble stays self-contained, in the generator's format.
    // kquant_mode.h, the grid tables and kquant_grid.h (in that order:
    // the grid decode needs the mode traits and the tables before it).
    let inlined_body = |name: &str| -> io::Result<String> {
        let header = src_dir.join(format!("metal/kquant/{name}.h"));
        Ok(read_build_source(&header)?
            .lines()
            .filter(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("#pragma once")
                    && !(trimmed.starts_with("#include \"") && trimmed.ends_with(".h\""))
            })
            .map(|line| format!("{line}\n"))
            .collect())
    };
    let mode_body = format!(
        "{}{}{}",
        inlined_body("kquant_mode")?,
        inlined_body("kquant_grid_tables")?,
        inlined_body("kquant_grid")?
    );
    for name in ["kquant", "kquant_nax"] {
        let header = src_dir.join(format!("metal/kquant/{name}.h"));
        let body: String = read_build_source(&header)?
            .lines()
            .filter(|line| !line.trim_start().starts_with("#pragma once"))
            .map(|line| {
                let trimmed = line.trim_start();
                if trimmed == "#include \"kquant_mode.h\"" {
                    mode_body.clone()
                } else if trimmed.starts_with("#include \"") && trimmed.ends_with(".h\"") {
                    String::new()
                } else {
                    format!("{line}\n")
                }
            })
            .collect();
        if body.contains(")preamble\"") {
            return Err(io::Error::other(format!(
                "{} contains the preamble delimiter",
                header.display()
            )));
        }
        let path = preambles.join(format!("{name}.cpp"));
        let source = format!(
            "namespace mlx::core::quantized_preamble {{\n\nconst char* {name}() {{\n  \
             return R\"preamble(\n#line 1 \"metal/kquant/{name}.h\"\n{body}\n)preamble\";\n}}\n\n}} \
             // namespace mlx::core::quantized_preamble\n"
        );
        std::fs::write(&path, source)
            .map_err(|error| build_file_error("write K-quant preamble", &path, error))?;
        bridge.file(path);
    }
    Ok(())
}
