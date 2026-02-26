use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    #[cfg(target_os = "macos")]
    build_swift_ui();
}

#[cfg(target_os = "macos")]
fn build_swift_ui() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let swift_build_dir = format!("{}/swift-build", out_dir);
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let package_path = format!("{}/ui-macos", manifest_dir);

    // Build Swift package as a static library
    let status = Command::new("swift")
        .args([
            "build",
            "-c", "release",
            "--package-path", &package_path,
            "--build-path", &swift_build_dir,
        ])
        .status()
        .expect("Failed to run swift build. Is Xcode installed?");

    assert!(status.success(), "Swift build failed");

    // Find the built static library.
    // SPM puts it at <build-path>/release/libDemucsUI.a
    let lib_search_path = format!("{}/release", swift_build_dir);

    // Also check the architecture-specific path
    let lib_path = PathBuf::from(&lib_search_path).join("libDemucsUI.a");
    if !lib_path.exists() {
        // Try architecture-specific build directory
        let arch_dir = find_arch_lib_dir(&swift_build_dir);
        println!("cargo:rustc-link-search=native={}", arch_dir);
    } else {
        println!("cargo:rustc-link-search=native={}", lib_search_path);
    }

    println!("cargo:rustc-link-lib=static=DemucsUI");

    // Link required frameworks
    println!("cargo:rustc-link-lib=framework=AppKit");
    println!("cargo:rustc-link-lib=framework=SwiftUI");
    println!("cargo:rustc-link-lib=framework=UniformTypeIdentifiers");

    // Link Swift runtime libraries
    if let Some(swift_lib_path) = find_swift_lib_path() {
        println!("cargo:rustc-link-search=native={}", swift_lib_path);
    }

    // Rerun if Swift sources change
    println!("cargo:rerun-if-changed=ui-macos/Sources/");
    println!("cargo:rerun-if-changed=ui-macos/Package.swift");
}

/// Find the Swift runtime library path from the Xcode toolchain.
#[cfg(target_os = "macos")]
fn find_swift_lib_path() -> Option<String> {
    // Get the path to swiftc
    let output = Command::new("xcrun")
        .args(["--find", "swiftc"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let swiftc_path = String::from_utf8(output.stdout).ok()?.trim().to_string();
    // swiftc is at .../usr/bin/swiftc
    // Swift libs are at .../usr/lib/swift/macosx/
    let toolchain_path = PathBuf::from(&swiftc_path);
    let lib_path = toolchain_path
        .parent()? // bin/
        .parent()? // usr/
        .join("lib/swift/macosx");

    if lib_path.exists() {
        Some(lib_path.to_string_lossy().to_string())
    } else {
        // Fallback: try swift_static
        let static_path = toolchain_path
            .parent()?
            .parent()?
            .join("lib/swift_static/macosx");
        if static_path.exists() {
            Some(static_path.to_string_lossy().to_string())
        } else {
            None
        }
    }
}

/// Search for the library in architecture-specific build directories.
#[cfg(target_os = "macos")]
fn find_arch_lib_dir(swift_build_dir: &str) -> String {
    // Try common architecture patterns
    for arch in &["arm64-apple-macosx", "x86_64-apple-macosx"] {
        let path = format!("{}/{}/release", swift_build_dir, arch);
        if PathBuf::from(&path).join("libDemucsUI.a").exists() {
            return path;
        }
    }

    // Fallback: search for the file
    let output = Command::new("find")
        .args([swift_build_dir, "-name", "libDemucsUI.a", "-type", "f"])
        .output()
        .expect("Failed to search for libDemucsUI.a");

    let found = String::from_utf8(output.stdout)
        .unwrap_or_default()
        .lines()
        .next()
        .unwrap_or("")
        .to_string();

    if !found.is_empty() {
        PathBuf::from(&found)
            .parent()
            .unwrap()
            .to_string_lossy()
            .to_string()
    } else {
        panic!("Could not find libDemucsUI.a in {}", swift_build_dir);
    }
}
