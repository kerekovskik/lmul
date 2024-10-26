use std::process::Command;
use std::path::PathBuf;
use std::env;
use std::fs;

fn main() {
    // Only compile Metal shaders on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let metal_src = manifest_dir.join("src/kernels.metal");
        
        // Create output directories
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let package_dir = manifest_dir.join("lmul");
        
        fs::create_dir_all(&package_dir).expect("Failed to create package directory");
        
        // Paths for intermediate and final files
        let air_path = out_dir.join("kernel.air");
        let metallib_path = package_dir.join("lmul.metallib");

        // Ensure xcrun is available
        if Command::new("xcrun").arg("-h").output().is_err() {
            panic!("xcrun not found. Make sure Xcode command line tools are installed.");
        }

        println!("cargo:rerun-if-changed={}", metal_src.display());
        
        // Compile .metal to .air
        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx", "metal",
                "-c",
                &metal_src.to_string_lossy(),
                "-o",
                &air_path.to_string_lossy()
            ])
            .status()
            .expect("Failed to compile Metal shader");

        if !status.success() {
            panic!("Metal shader compilation failed");
        }

        // Create metallib
        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx", "metallib",
                &air_path.to_string_lossy(),
                "-o",
                &metallib_path.to_string_lossy()
            ])
            .status()
            .expect("Failed to create metallib");

        if !status.success() {
            panic!("Metallib creation failed");
        }

        // Print the final metallib path for debugging
        println!("cargo:warning=Created metallib at: {}", metallib_path.display());
    }
}