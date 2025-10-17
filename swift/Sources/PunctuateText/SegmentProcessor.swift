import CoreML
import Foundation

/// Processes text segments through the punctuation model
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
public final class SegmentProcessor {
    private let model: MLModel
    private let tokenizer: Tokenizer
    private let predictionOptions = MLPredictionOptions()

    internal init(model: MLModel, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Process a text segment and apply punctuation
    public func processSegment(_ segment: TextSegment, startWord: Int) throws -> String {
        let words = segment.text
        let sequenceLength = tokenizer.maxSequenceLength
        var aggregatedResult = ""
        var processedWords = 0

        while processedWords < words.count {
            let encoding = tokenizer.encodeChunk(
                words: words,
                startIndex: processedWords,
                maximumTokens: sequenceLength
            )

            let consumed = encoding.consumedWords
            if consumed == 0 {
                throw NSError(
                    domain: "PunctuateText",
                    code: 6,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Failed to encode segment chunk within model sequence length."
                    ]
                )
            }

            // Create MLMultiArrays for the chunk
            let shape: [NSNumber] = [1, NSNumber(value: sequenceLength)]
            let inputArray = try MLMultiArray(shape: shape, dataType: .float32)
            let maskArray = try MLMultiArray(shape: shape, dataType: .float32)

            let inputPointer = inputArray.dataPointer.assumingMemoryBound(to: Float32.self)
            let maskPointer = maskArray.dataPointer.assumingMemoryBound(to: Float32.self)

            for idx in 0 ..< sequenceLength {
                inputPointer[idx] = Float32(encoding.inputIds[idx])
                maskPointer[idx] = Float32(encoding.attentionMask[idx])
            }

            // Predict on the current chunk
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputArray),
                "attention_mask": MLFeatureValue(multiArray: maskArray),
            ])
            let prediction = try model.prediction(from: features, options: predictionOptions)

            guard let outputArray = prediction.featureValue(for: "output")?.multiArrayValue else {
                throw NSError(
                    domain: "PunctuateText",
                    code: 8,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Punctuate model prediction missing output tensor."
                    ]
                )
            }

            // Adjust startWord relative to the chunk
            let chunkStartWord = max(startWord - processedWords, 0)
            let chunkResult = processOutput(
                output: outputArray,
                wordPieces: encoding.wordPieces,
                wordIds: encoding.wordIds,
                startWord: chunkStartWord
            )

            if !aggregatedResult.isEmpty && !chunkResult.isEmpty && !aggregatedResult.hasSuffix("-")
            {
                aggregatedResult.append(" ")
            }
            aggregatedResult.append(chunkResult)

            processedWords += consumed
        }

        return aggregatedResult
    }

    private func processOutput(
        output: MLMultiArray, wordPieces: [String], wordIds: [Int?], startWord: Int
    ) -> String {
        var result = ""
        result.reserveCapacity(wordPieces.count * 10)  // Pre-allocate

        var lastWordId: Int? = nil

        for idx in 0 ..< wordPieces.count {
            // Skip special tokens
            guard let wordId = wordIds[idx] else { continue }

            // Skip words before startWord
            if wordId < startWord { continue }

            // Find max class
            var maxScore: Float = -Float.infinity
            var maxClass = 0

            // Get scores for this token position
            for classIdx in 0 ..< 24 {
                let score = output[[0, idx, classIdx] as [NSNumber]].floatValue
                if score > maxScore {
                    maxScore = score
                    maxClass = classIdx
                }
            }

            // Remove ## prefix for subword tokens
            let wordpiece =
                wordPieces[idx].hasPrefix("##")
                ? String(wordPieces[idx].dropFirst(2)) : wordPieces[idx]

            // Apply punctuation and capitalization
            let punctuated = punctuateWordpiece(wordpiece, labelMap[maxClass])

            // Add space between words
            if let lastId = lastWordId, wordId != lastId && !result.hasSuffix("-") {
                result.append(" ")
            }

            result.append(punctuated)
            lastWordId = wordId
        }

        return result
    }
}
