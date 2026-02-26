// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DemucsUI",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "DemucsUI", type: .static, targets: ["DemucsUI"])
    ],
    targets: [
        .target(
            name: "CDemucsTypes",
            path: "Sources/CDemucsTypes",
            publicHeadersPath: "include"
        ),
        .target(
            name: "DemucsUI",
            dependencies: ["CDemucsTypes"],
            path: "Sources/DemucsUI"
        ),
    ]
)
