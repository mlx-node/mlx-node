use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=metal/");

    // Only compile Metal shaders on macOS
    #[cfg(not(target_os = "macos"))]
    {
        // Set a placeholder path for non-macOS builds
        println!("cargo:rustc-env=PAGED_ATTN_METALLIB=");
        return;
    }

    #[cfg(target_os = "macos")]
    compile_metal_shaders();
}

#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let metal_dir = PathBuf::from(&manifest_dir).join("metal");

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

        // Compile Metal to AIR
        let status = Command::new("xcrun")
            .args([
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
