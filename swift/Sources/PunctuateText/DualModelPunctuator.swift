import CoreML
import Foundation
import SegmentTextKit

/// Enum to specify which punctuation model to use
public enum PunctuationModel: CaseIterable {
    case punctuate  // Original Punctuate model
    case fullStop  // FullStopPunctuation model
}

/// A punctuator that can use either Punctuate or FullStopPunctuation
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
public class DualModelPunctuator {
    private var punctuateProcessor: SegmentProcessor?
    private var fullStopModel: FullStopPunctuation?
    private let segmentKit: SegmentTextKit
    private var currentModel: PunctuationModel

    public init(initialModel: PunctuationModel = .punctuate) throws {
        self.segmentKit = try SegmentTextKit()
        self.currentModel = initialModel

        // Initialize the selected model
        try switchModel(to: initialModel)
    }

    /// Switch between models
    public func switchModel(to model: PunctuationModel) throws {
        currentModel = model

        switch model {
        case .punctuate:
            if punctuateProcessor == nil {
                let tokenizer = try Tokenizer()
                let punctuateModel = try loadPunctuateModel()
                self.punctuateProcessor = try SegmentProcessor(
                    model: punctuateModel, tokenizer: tokenizer)
            }

        case .fullStop:
            if fullStopModel == nil {
                do {
                    self.fullStopModel = try createFullStopPunctuation()
                } catch {
                    throw error
                }
            }

            // Verify initialization
            guard self.fullStopModel != nil else {
                throw NSError(
                    domain: "PunctuateText", code: 5,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Failed to initialize FullStopPunctuation"
                    ])
            }
        }
    }

    /// Get the current model type
    public var activeModel: PunctuationModel {
        return currentModel
    }

    /// Process text with the current model
    public func process(text: String, useSegmentation: Bool = false) throws -> String {
        if useSegmentation {
            return try processWithSegmentation(text: text)
        } else {
            return try processDirect(text: text)
        }
    }

    /// Process text with performance metrics
    public func processWithMetrics(text: String, useSegmentation: Bool = false) throws -> (
        result: String, metrics: PerformanceMetrics
    ) {
        let totalStart = CFAbsoluteTimeGetCurrent()
        var segmentationTime: TimeInterval = 0
        var sentenceCount = 1

        let processedText: String

        if useSegmentation {
            // Measure segmentation time
            let segmentStart = CFAbsoluteTimeGetCurrent()
            let sentences = segmentKit.splitSentences(text, threshold: 0.25)
            segmentationTime = CFAbsoluteTimeGetCurrent() - segmentStart

            sentenceCount = sentences.count

            // Join sentences, replacing all punctuation with spaces
            let punctuationSet = CharacterSet(charactersIn: ",.?!:")
            processedText =
                sentences
                .map { sentence in
                    let trimmed = sentence.trimmingCharacters(in: .whitespaces)
                    // Replace all punctuation characters with spaces
                    var cleaned = ""
                    for char in trimmed {
                        if punctuationSet.contains(Unicode.Scalar(String(char))!) {
                            cleaned += " "
                        } else {
                            cleaned += String(char)
                        }
                    }
                    // Clean up multiple spaces
                    return cleaned.replacingOccurrences(
                        of: "  +", with: " ", options: .regularExpression)
                }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
        } else {
            processedText = text
        }

        // Measure punctuation time
        let punctuationStart = CFAbsoluteTimeGetCurrent()
        let result = try processDirect(text: processedText)
        let punctuationTime = CFAbsoluteTimeGetCurrent() - punctuationStart

        let totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        // Calculate statistics
        let wordCount = processedText.split(separator: " ").count
        let characterCount = processedText.count

        let metrics = PerformanceMetrics(
            segmentationTime: segmentationTime,
            punctuationTime: punctuationTime,
            totalTime: totalTime,
            sentenceCount: sentenceCount,
            wordCount: wordCount,
            characterCount: characterCount
        )

        return (result: result, metrics: metrics)
    }

    // MARK: - Private Methods

    private func processDirect(text: String) throws -> String {
        switch currentModel {
        case .punctuate:
            guard let processor = punctuateProcessor else {
                throw NSError(
                    domain: "PunctuateText", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Punctuate processor not initialized"])
            }
            return try punctuate(text: text, processor: processor)

        case .fullStop:
            guard let model = fullStopModel else {
                throw NSError(
                    domain: "PunctuateText", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "FullStopPunctuation not initialized"])
            }
            return try model.restorePunctuation(text)
        }
    }

    private func processWithSegmentation(text: String) throws -> String {
        // Use SegmentTextKit to split into sentences
        let sentences = segmentKit.splitSentences(text)

        // Join sentences, replacing all punctuation with spaces
        let punctuationSet = CharacterSet(charactersIn: ",.?!:")
        let joinedText =
            sentences
            .map { sentence in
                let trimmed = sentence.trimmingCharacters(in: .whitespaces)
                // Replace all punctuation characters with spaces
                var cleaned = ""
                for char in trimmed {
                    if punctuationSet.contains(Unicode.Scalar(String(char))!) {
                        cleaned += " "
                    } else {
                        cleaned += String(char)
                    }
                }
                // Clean up multiple spaces
                return cleaned.replacingOccurrences(
                    of: "  +", with: " ", options: .regularExpression)
            }
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        // Apply punctuation
        return try processDirect(text: joinedText)
    }
}
