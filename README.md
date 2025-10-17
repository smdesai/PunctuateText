# PunctuateText

PunctuateText restores punctuation and capitalization for ASR or otherwise unformatted text using Core ML models. The Swift package offers a reusable library plus a CLI utility, and it ships with scripts that convert the upstream HuggingFace checkpoints into `.mlmodelc` bundles.

## How It Works

- `Punctuate` (based on `unikei/distilbert-base-re-punctuate`) targets English biomedical transcripts and predicts 24 punctuation/casing tags per token.
- `FullStopPunctuation` (based on `oliverguhr/fullstop-punctuation-multilang-large`) predicts sentence-ending punctuation across several languages.
- Text is tokenized with a custom WordPiece tokenizer, batched into segments, and fed into the Core ML models via `MLModel` APIs.
- The CLI can switch between models and optionally apply sentence segmentation before punctuation repair.

## Conversion Pipeline

The `coreml-conversion/` folder contains Python utilities for exporting the Hugging Face checkpoints:

1. Install dependencies (`torch`, `transformers`, `coremltools`).
2. Run `python distilbert-convert.py` and `python full-stop-convert.py` to produce `.mlpackage` artifacts under `coreml-conversion/`.
3. Run `./compile.sh` to produce compiled models which is located in `MLModels/`
4. Copy the generated `.mlmodelc` directories into `swift/Sources/PunctuateText/Resources/` (already populated in this repo).

## Building the Swift Package

```bash
cd swift
swift build -c release
```

CLI binary lives at `swift/.build/release/punctuate`.

## Command Line Interface

```
Usage: punctuate [OPTIONS] [TEXT]

Options:
  --segment     Pre-segment text with SegmentTextKit before punctuation.
  --model2      Use the FullStopPunctuation model (default is Punctuate).
  --metrics     Run twice and report warm/loaded timings and throughput.
  --verbose     Print extra diagnostics (model choice, segmentation info).
  -h, --help    Show usage.
```

Examples:

```bash
swift run punctuate                                # default demo text
swift run punctuate "hello world how are you"      # custom text
swift run punctuate --segment --metrics "..."      # segmentation + stats
swift run punctuate --model2 "other languages"     # switch model
```

## Library Usage

```swift
import PunctuateText

let punctuator = try DualModelPunctuator(initialModel: .punctuate)
let result = try punctuator.process(text: "the atm protein is...")
// Or switch to the FullStop model:
try punctuator.switchModel(to: .fullStop)
```

## Resources

- Converted Core ML models and tokenizer files live under `swift/Sources/PunctuateText/Resources/`.
- Conversion scripts: `coreml-conversion/`
- Segmenter dependency: [`SegmentTextKit`](https://github.com/sachin-desai/SegmentText)

Requirements: macOS 15 / iOS 18 / tvOS 18 / watchOS 11 / visionOS 2 and Swift 5.9+
