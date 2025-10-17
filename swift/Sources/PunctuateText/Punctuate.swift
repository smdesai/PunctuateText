import CoreML
import Foundation

// MARK: - Main Punctuation Function
/// Punctuate text of any length by splitting into segments and processing
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
public func punctuate(text: String, processor: SegmentProcessor) throws -> String {
    // Preprocess text
    let processedText = text.lowercased().replacingOccurrences(of: "\n", with: " ")
    let words = processedText.split(separator: " ", omittingEmptySubsequences: true).map(
        String.init)

    if words.isEmpty { return "" }

    // Split into segments
    let segments = splitToSegments(words: words, length: 150, overlap: 50)

    // Process segments
    if segments.count == 1 {
        // Single segment - process directly
        return try processor.processSegment(segments[0], startWord: 0)
    } else {
        // Multiple segments - process with proper overlap handling
        var results = [String](repeating: "", count: segments.count)

        // Process segments (could be made concurrent for better performance)
        for (index, segment) in segments.enumerated() {
            let startWord = index == 0 ? 0 : 50
            results[index] = try processor.processSegment(segment, startWord: startWord)
        }

        // Join results
        return results.joined(separator: " ").trimmingCharacters(in: .whitespaces)
    }
}
