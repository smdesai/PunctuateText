//
//  PerformanceMetrics.swift
//  PunctuateText
//

import Foundation

/// Structure to hold performance metrics for text processing
public struct PerformanceMetrics {
    public let segmentationTime: TimeInterval
    public let punctuationTime: TimeInterval
    public let totalTime: TimeInterval
    public let sentenceCount: Int
    public let wordCount: Int
    public let characterCount: Int

    // Computed properties for milliseconds
    public var segmentationTimeMs: Double {
        return segmentationTime * 1000
    }

    public var punctuationTimeMs: Double {
        return punctuationTime * 1000
    }

    public var totalTimeMs: Double {
        return totalTime * 1000
    }

    public var wordsPerSecond: Double {
        return punctuationTime > 0 ? Double(wordCount) / punctuationTime : 0
    }

    public var charactersPerSecond: Double {
        return punctuationTime > 0 ? Double(characterCount) / punctuationTime : 0
    }

    public init(
        segmentationTime: TimeInterval,
        punctuationTime: TimeInterval,
        totalTime: TimeInterval,
        sentenceCount: Int,
        wordCount: Int,
        characterCount: Int
    ) {
        self.segmentationTime = segmentationTime
        self.punctuationTime = punctuationTime
        self.totalTime = totalTime
        self.sentenceCount = sentenceCount
        self.wordCount = wordCount
        self.characterCount = characterCount
    }
}
