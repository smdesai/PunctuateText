// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PunctuateText",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        // Library for use in other projects
        .library(
            name: "PunctuateText",
            targets: ["PunctuateText"]),
        // Executable CLI tool
        .executable(
            name: "punctuate",
            targets: ["PunctuateCLI"]),
    ],
    dependencies: [
        .package(path: "./SegmentTextKit-Package")
    ],
    targets: [
        // Main library target
        .target(
            name: "PunctuateText",
            dependencies: [
                .product(name: "SegmentTextKit", package: "SegmentTextKit-Package")
            ],
            path: "Sources/PunctuateText",
            resources: [
                .copy("Resources/Punctuate.mlmodelc"),
                .copy("Resources/FullStopPunctuation.mlmodelc"),
                .copy("Resources/vocab.txt"),
                .copy("Resources/sentencepiece.bpe.model"),
            ]),

        // CLI executable target
        .executableTarget(
            name: "PunctuateCLI",
            dependencies: [
                "PunctuateText",
                .product(name: "SegmentTextKit", package: "SegmentTextKit-Package"),
            ],
            path: "Sources/PunctuateCLI"),
    ]
)
