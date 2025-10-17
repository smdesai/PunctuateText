# PunctuateText

A Swift package for adding punctuation and capitalization to text using a CoreML model based on DistilBERT.

## Structure

```
swift/
├── Package.swift
├── Sources/
│   ├── PunctuateText/          # Main library
│   │   ├── Tokenizer.swift     # BERT tokenizer with caching
│   │   ├── SegmentProcessor.swift # Process text segments through model
│   │   ├── Utilities.swift     # Helper functions and types
│   │   ├── Punctuate.swift     # Main punctuation function
│   │   └── PunctuateModel.swift # CoreML model wrapper
│   └── PunctuateCLI/           # Command-line interface
│       └── main.swift
└── Tests/
    └── PunctuateTextTests/     # Unit tests (optional)
```

## Building

### As a Swift Package

```bash
cd swift
swift build -c release
```

### Build just the CLI

```bash
cd swift
swift build -c release --product punctuate
```

The executable will be at `.build/release/punctuate`

## Usage

### Command Line

```bash
# With default example text
.build/release/punctuate

# With custom text
.build/release/punctuate "your text here without punctuation"

# With verbose timing
.build/release/punctuate "your text" --verbose
```

### As a Library

```swift
import PunctuateText
import CoreML

// Initialize with bundled resources
let tokenizer = try Tokenizer()
let model = try Punctuate()
let processor = SegmentProcessor(model: model, tokenizer: tokenizer)

// Punctuate text
let result = try punctuate(text: "your text here", processor: processor)
print(result)
```

## Requirements

- macOS 15.0+ / iOS 18.0+ / tvOS 18.0+ / watchOS 11.0+ / visionOS 2.0+
- Swift 5.9+
- CoreML framework

## Features

- Optimized tokenizer with caching
- Support for texts of any length using segmentation
- Efficient memory usage
- 24 punctuation/capitalization classes
- Context-aware capitalization

## Note

The package includes the model and vocabulary files as bundled resources, so they are automatically available when using the library or CLI tool.