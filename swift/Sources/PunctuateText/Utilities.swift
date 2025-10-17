import Foundation

// MARK: - Text Segment
/// Represents a segment of text with its position in the original text
public struct TextSegment {
    public let text: [String]  // Array of words
    public let startIdx: Int
    public let endIdx: Int

    public init(text: [String], startIdx: Int, endIdx: Int) {
        self.text = text
        self.startIdx = startIdx
        self.endIdx = endIdx
    }
}

// MARK: - Label Map
/// Maps model output indices to punctuation labels
internal let labelMap: [String] = [
    "UPPER_", "Upper_", "lower_",
    "UPPER.", "Upper.", "lower.",
    "UPPER,", "Upper,", "lower,",
    "UPPER!", "Upper!", "lower!",
    "UPPER?", "Upper?", "lower?",
    "UPPER:", "Upper:", "lower:",
    "UPPER;", "Upper;", "lower;",
    "UPPER-", "Upper-", "lower-",
]

// MARK: - Punctuation Functions
/// Apply punctuation and capitalization to a wordpiece based on its label
@inline(__always)
internal func punctuateWordpiece(_ wordpiece: String, _ label: String) -> String {
    var result = wordpiece

    // Fast path for no punctuation
    if label.hasSuffix("_") {
        // Handle capitalization only
        switch label.first {
        case "U":
            if label.hasPrefix("UPPER") {
                result = result.uppercased()
            } else {
                result = result.prefix(1).uppercased() + result.dropFirst()
            }
        default:
            break
        }
        return result
    }

    // Handle capitalization
    switch label.first {
    case "U":
        if label.hasPrefix("UPPER") {
            result = result.uppercased()
        } else {
            result = result.prefix(1).uppercased() + result.dropFirst()
        }
    default:
        break
    }

    // Add punctuation
    result.append(label.last!)

    return result
}

// MARK: - Text Segmentation
/// Split text into overlapping segments for processing
internal func splitToSegments(words: [String], length: Int, overlap: Int) -> [TextSegment] {
    var segments: [TextSegment] = []
    segments.reserveCapacity((words.count / length) + 2)

    var i = 0
    while true {
        let startIdx = length * i
        if startIdx >= words.count { break }

        let endIdx = min((length * (i + 1)) + overlap, words.count)
        let segment = TextSegment(
            text: Array(words[startIdx ..< endIdx]),
            startIdx: startIdx,
            endIdx: endIdx
        )

        segments.append(segment)
        i += 1
    }

    return segments
}
