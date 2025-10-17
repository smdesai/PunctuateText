import CoreML
import Foundation

/// Processes text segments through the punctuation model
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
public final class SegmentProcessor {
    private let model: MLModel
    private let tokenizer: Tokenizer
    private let predictionOptions = MLPredictionOptions()
    private let sequenceLength: Int
    private let inputIdsArray: MLMultiArray
    private let attentionMaskArray: MLMultiArray
    private let featureProvider: MLDictionaryFeatureProvider

    internal init(model: MLModel, tokenizer: Tokenizer) throws {
        self.model = model
        self.tokenizer = tokenizer
        self.sequenceLength = tokenizer.maxSequenceLength

        let shape: [NSNumber] = [1, NSNumber(value: sequenceLength)]
        self.inputIdsArray = try MLMultiArray(shape: shape, dataType: .float32)
        self.attentionMaskArray = try MLMultiArray(shape: shape, dataType: .float32)
        self.featureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionMaskArray),
        ])
    }

    /// Process a text segment and apply punctuation
    public func processSegment(_ segment: TextSegment, startWord: Int) throws -> String {
        let words = segment.text
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
            let inputPointer = inputIdsArray.dataPointer.assumingMemoryBound(to: Float32.self)
            let maskPointer = attentionMaskArray.dataPointer.assumingMemoryBound(to: Float32.self)
            let totalCount = sequenceLength

            UnsafeMutableBufferPointer(start: inputPointer, count: totalCount).update(repeating: 0)
            UnsafeMutableBufferPointer(start: maskPointer, count: totalCount).update(repeating: 0)

            let copyCount = min(encoding.inputIds.count, totalCount)
            for idx in 0 ..< copyCount {
                inputPointer[idx] = Float32(encoding.inputIds[idx])
                maskPointer[idx] = Float32(encoding.attentionMask[idx])
            }

            // Predict on the current chunk
            let prediction = try model.prediction(from: featureProvider, options: predictionOptions)

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
        let logitsPointer = output.dataPointer.bindMemory(to: Float16.self, capacity: output.count)
        let classCount = labelMap.count
        let strides = output.strides.map { $0.intValue }
        let strideToken = strides.count > 1 ? strides[1] : classCount
        let strideClass = strides.count > 2 ? strides[2] : 1

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
            let tokenOffset = idx * strideToken
            for classIdx in 0 ..< classCount {
                let offset = tokenOffset + classIdx * strideClass
                guard offset < output.count else { continue }
                let score = Float(logitsPointer[offset])
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
