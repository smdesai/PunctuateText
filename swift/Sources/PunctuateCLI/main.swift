import CoreML
import Foundation
import PunctuateText
import SegmentTextKit

// MARK: - Main Entry Point
@available(macOS 15.0, *)
@main
struct PunctuateCLI {
    static func main() {
        let args = CommandLine.arguments

        // Parse command line arguments
        let textArgs = args.dropFirst().filter { !$0.hasPrefix("--") }
        let useSegmentation = args.contains("--segment")
        let verbose = args.contains("--verbose")
        let showMetrics = args.contains("--metrics")
        let showHelp = args.contains("--help") || args.contains("-h")
        let useModel2 = args.contains("--model2")

        if showHelp {
            printHelp()
            return
        }

        let text: String
        if let firstTextArg = textArgs.first {
            text = firstTextArg
        } else {
            text =
                "the atm protein is a single high molecular weight protein predominantly confined to the nucleus of human fibroblasts but is present in both nuclear and microsomal fractions from human lymphoblast cells and peripheral blood lymphocytes atm protein levels and localization remain constant throughout all stages of the cell cycle truncated atm protein was not detected in lymphoblasts from ataxia telangiectasia patients homozygous for mutations leading to premature protein termination exposure of normal human cells to gamma irradiation and the radiomimetic drug neocarzinostatin had no effect on atm protein levels in contrast to a noted rise in p53 levels over the same time interval these findings are consistent with a role for the atm protein in ensuring the fidelity of dna repair and cell cycle regulation following genome damage"
        }

        do {
            let startTime = CFAbsoluteTimeGetCurrent()

            // Initialize DualModelPunctuator
            let initialModel: PunctuationModel = useModel2 ? .fullStop : .punctuate
            if verbose {
                print(
                    "Using model: \(initialModel == .punctuate ? "Punctuate" : "FullStopPunctuation")"
                )
            }
            let punctuator = try DualModelPunctuator(initialModel: initialModel)

            let initTime = CFAbsoluteTimeGetCurrent() - startTime

            // Process text
            print("Original text (\(text.split(separator: " ").count) words):")
            print(text)
            print("\n" + String(repeating: "-", count: 80) + "\n")

            if showMetrics {
                // Warm up the model first
                if verbose {
                    print("Warming up model...")
                }
                _ = try punctuator.processWithMetrics(
                    text: text,
                    useSegmentation: useSegmentation
                )

                // Now process with metrics (warmed up)
                let processStart = CFAbsoluteTimeGetCurrent()
                let (result, metrics) = try punctuator.processWithMetrics(
                    text: text,
                    useSegmentation: useSegmentation
                )
                let processTime = CFAbsoluteTimeGetCurrent() - processStart

                // Print result
                print("Punctuated text:")
                print(result)

                // Print metrics
                print("\n" + String(repeating: "-", count: 80) + "\n")
                print("Performance Metrics:")
                print(String(format: "  Initialization:        %.3f seconds", initTime))
                if useSegmentation {
                    print(
                        String(
                            format: "  Segmentation time:     %.3f seconds",
                            metrics.segmentationTime))
                    print(String(format: "  Sentences detected:    %d", metrics.sentenceCount))
                }
                print(
                    String(format: "  Punctuation time:      %.3f seconds", metrics.punctuationTime)
                )
                print(String(format: "  Total processing time: %.3f seconds", processTime))
                print(String(format: "  Words processed:       %d", metrics.wordCount))
                print(String(format: "  Characters processed:  %d", metrics.characterCount))
                if metrics.wordCount > 0 {
                    let wordsPerSecond = Double(metrics.wordCount) / metrics.punctuationTime
                    print(String(format: "  Words per second:      %.1f", wordsPerSecond))
                }
            } else {
                // Process without metrics
                let result: String
                if verbose && useSegmentation {
                    print("Using sentence segmentation...\n")
                }
                result = try punctuator.process(
                    text: text,
                    useSegmentation: useSegmentation
                )

                // Print result
                print("Punctuated text:")
                print(result)

                if verbose {
                    let totalTime = CFAbsoluteTimeGetCurrent() - startTime
                    print("\n" + String(repeating: "-", count: 80))
                    print(String(format: "Total execution time: %.3f seconds", totalTime))
                }
            }

        } catch {
            print("Error: \(error.localizedDescription)")
            exit(1)
        }
    }

    static func printHelp() {
        print(
            """
            Usage: punctuate [OPTIONS] [TEXT]

            Punctuate and capitalize text using a machine learning model.

            OPTIONS:
                --segment       Use sentence segmentation before punctuation
                --metrics       Show detailed performance metrics
                --verbose       Show timing information
                --model2        Use FullStopPunctuation model instead of default Punctuate model
                -h, --help      Show this help message

            If no TEXT is provided, a default example text will be used.

            Examples:
                punctuate "hello world how are you"
                punctuate --segment "this is a test it works well"
                punctuate --metrics --segment "example text"
                punctuate --verbose --metrics "analyze performance"
                punctuate --model2 "test with second model"
            """)
    }
}
