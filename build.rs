use std::process::Command;
use std::path::Path;
use std::env;
use std::fs;

fn main() {
    // Only compile Metal shaders on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let metal_src = format!("{}/src/kernels.metal", manifest_dir);
        let output_dir = format!("{}/lmul", manifest_dir);
        let metallib_path = Path::new(&output_dir).join("lmul.metallib");

        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");

        // Ensure xcrun is available
        if Command::new("xcrun").arg("-h").output().is_err() {
            panic!("xcrun not found. Make sure Xcode command line tools are installed.");
        }

        // Create temporary directory for intermediate files
        let temp_dir = format!("{}/tmp", env::var("OUT_DIR").unwrap());
        fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");

        // Compile .metal to .air
        let status = Command::new("xcrun")
            .args(&[
                "-sdk", "macosx", "metal",
                "-c", &metal_src,
                "-o", &format!("{}/kernel.air", temp_dir)
            ])
            .status()
            .expect("Failed to compile Metal shader");

        if !status.success() {
            panic!("Metal shader compilation failed");
        }

        // Create metallib
        let status = Command::new("xcrun")
            .args(&[
                "-sdk", "macosx", "metallib",
                &format!("{}/kernel.air", temp_dir),
                "-o", metallib_path.to_str().unwrap()
            ])
            .status()
            .expect("Failed to create metallib");

        if !status.success() {
            panic!("Metallib creation failed");
        }

        // Tell Cargo to rerun this script if the shader changes
        println!("cargo:rerun-if-changed={}", metal_src);
    }
}