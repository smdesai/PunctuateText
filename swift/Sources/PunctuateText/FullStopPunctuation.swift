import CoreML
import Foundation
import SegmentTextKit
import SentencePieceWrapper

@available(macOS 15.0, iOS 17.0, *)
public class FullStopPunctuation {
    private let model: MLModel
    private let tokenizer: SentencePieceTokenizer
    private let inputIdsArray: MLMultiArray
    private let attentionMaskArray: MLMultiArray
    private let featureProvider: MLDictionaryFeatureProvider
    private let predictionOptions = MLPredictionOptions()
    private let enableLogging: Bool

    private let maxSequenceLength = 512
    private let classCount = 6

    // Label mapping for XLM-RoBERTa punctuation model
    private let id2label = [
        0: "0",
        1: ".",
        2: ",",
        3: "?",
        4: "-",
        5: ":",
    ]

    public init(model: MLModel, enableLogging: Bool = false) throws {
        self.model = model
        self.enableLogging = enableLogging

        let bundle = Bundle.module

        // Try without subdirectory first
        var tokenizerURL = bundle.url(forResource: "sentencepiece.bpe", withExtension: "model")

        // Then try with Resources subdirectory
        if tokenizerURL == nil {
            tokenizerURL = bundle.url(
                forResource: "sentencepiece.bpe", withExtension: "model", subdirectory: "Resources")
        }

        guard let finalURL = tokenizerURL else {
            throw SegmentTextError.modelNotFound("sentencepiece.bpe.model")
        }

        self.tokenizer = try SentencePieceTokenizer(modelPath: finalURL.path)

        self.inputIdsArray = try MLMultiArray(
            shape: [1, NSNumber(value: maxSequenceLength)], dataType: .float32)
        self.attentionMaskArray = try MLMultiArray(
            shape: [1, NSNumber(value: maxSequenceLength)], dataType: .float32)
        self.featureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionMaskArray),
        ])

        let warmupStart = CFAbsoluteTimeGetCurrent()
        _ = try? self.predictChunk("test")
        if enableLogging {
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart
            print("  FullStopPunctuation warmup \(String(format: "%.3f", warmupTime))s")
        }
    }

    func preprocess(_ text: String) -> [String] {
        // Remove punctuation except in numbers
        let pattern = "(?<!\\d)[.,;:!?](?!\\d)"
        let regex = try! NSRegularExpression(pattern: pattern, options: [])
        let cleanText = regex.stringByReplacingMatches(
            in: text,
            options: [],
            range: NSRange(location: 0, length: text.utf16.count),
            withTemplate: ""
        )

        return cleanText.split(separator: " ").map(String.init)
    }

    func predictChunk(_ text: String) throws -> [(entity: String, score: Float, word: String)] {
        // Tokenize using SentencePiece and get tokens
        let tokens = tokenizer.tokenize(text: text)
        let (inputIds, attentionMask) = tokenizer.encodeForModel(text: text)

        // Reuse multiarray buffers
        let totalCount = maxSequenceLength
        let inputPointer = inputIdsArray.dataPointer.assumingMemoryBound(to: Float32.self)
        let maskPointer = attentionMaskArray.dataPointer.assumingMemoryBound(to: Float32.self)

        UnsafeMutableBufferPointer(start: inputPointer, count: totalCount).update(repeating: 0)
        UnsafeMutableBufferPointer(start: maskPointer, count: totalCount).update(repeating: 0)

        let copyCount = min(inputIds.count, totalCount)
        for idx in 0 ..< copyCount {
            inputPointer[idx] = Float32(inputIds[idx])
            maskPointer[idx] = Float32(attentionMask[idx])
        }

        // Make prediction
        let prediction = try model.prediction(from: featureProvider, options: predictionOptions)

        guard let logits = prediction.featureValue(for: "output")?.multiArrayValue else {
            throw SegmentTextError.modelNotFound("FullStopPunctuation output tensor")
        }

        let logitsPointer = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
        let logitsStrides = logits.strides.map { $0.intValue }
        let strideToken = logitsStrides.count > 1 ? logitsStrides[1] : classCount
        let strideClass = logitsStrides.count > 2 ? logitsStrides[2] : 1

        // Process output for each token (excluding special tokens)
        var results: [(entity: String, score: Float, word: String)] = []

        // Skip CLS token at position 0
        let tokenCount = min(tokens.count + 1, inputIds.count)
        for i in 1 ..< tokenCount {
            // Skip if we hit SEP token or padding
            if inputIds[i] == 2 || inputIds[i] == 1 {
                break
            }

            // Get predictions for this token position
            var maxScore: Float = -Float.infinity
            var maxClass = 0

            let tokenOffset = i * strideToken
            for classIdx in 0 ..< classCount {
                let offset = tokenOffset + classIdx * strideClass
                guard offset < logits.count else { continue }
                let score = Float(logitsPointer[offset])
                if score > maxScore {
                    maxScore = score
                    maxClass = classIdx
                }
            }

            // Apply softmax to get probability
            var expSum: Float = 0
            for classIdx in 0 ..< classCount {
                let offset = tokenOffset + classIdx * strideClass
                guard offset < logits.count else { continue }
                let score = Float(logitsPointer[offset])
                expSum += exp(score - maxScore)
            }
            let probability = expSum > 0 ? (1 / expSum) : 0

            let entity = id2label[maxClass] ?? "0"
            let token = tokens[i - 1]  // Adjust for CLS token offset
            results.append((entity: entity, score: probability, word: token))
        }

        return results
    }

    func predict(_ words: [String], chunkSize: Int = 230) throws -> [(String, String, Float)] {
        let overlap = words.count > chunkSize ? 5 : 0
        var batches: [[String]] = []

        // Create overlapping chunks
        var i = 0
        while i < words.count {
            let end = min(i + chunkSize, words.count)
            batches.append(Array(words[i ..< end]))
            i += chunkSize - overlap
        }

        // Remove last batch if too small
        if let lastBatch = batches.last, lastBatch.count <= overlap && batches.count > 1 {
            batches.removeLast()
        }

        var taggedWords: [(String, String, Float)] = []

        for (batchIdx, batch) in batches.enumerated() {
            let isLastBatch = batchIdx == batches.count - 1
            let startWord = (batchIdx == 0) ? 0 : overlap
            let text = batch.joined(separator: " ")

            if enableLogging {
                print("Processing chunk: \(String(text.prefix(50)))...")
            }

            let tokenPredictions = try predictChunk(text)

            if enableLogging {
                print("Number of predictions: \(tokenPredictions.count)")
            }

            // Simple approach: map tokens to words by tracking word boundaries
            var wordResults: [(String, String, Float)] = []
            var currentWordTokens: [(entity: String, score: Float, word: String)] = []

            for pred in tokenPredictions {
                if pred.word.hasPrefix("▁") && !currentWordTokens.isEmpty {
                    // New word boundary - process accumulated tokens for previous word
                    let lastToken = currentWordTokens.last!
                    let wordIndex = wordResults.count
                    if wordIndex < batch.count {
                        wordResults.append((batch[wordIndex], lastToken.entity, lastToken.score))
                    }
                    currentWordTokens = [pred]
                } else {
                    currentWordTokens.append(pred)
                }
            }

            // Process last word
            if !currentWordTokens.isEmpty && wordResults.count < batch.count {
                let lastToken = currentWordTokens.last!
                wordResults.append((batch[wordResults.count], lastToken.entity, lastToken.score))
            }

            // Add results for the appropriate word range
            for (idx, result) in wordResults.enumerated() {
                if idx >= startWord && idx < (isLastBatch ? batch.count : batch.count - overlap) {
                    taggedWords.append(result)
                }
            }
        }

        assert(
            taggedWords.count == words.count,
            "Tagged words: \(taggedWords.count), Input words: \(words.count)")
        return taggedWords
    }

    func restorePunctuation(_ text: String, chunkSize: Int = 230) throws -> String {
        let words = preprocess(text)
        let predictions = try predict(words, chunkSize: chunkSize)
        return predictionToText(predictions)
    }

    private func predictionToText(_ predictions: [(String, String, Float)]) -> String {
        let tokens = predictions.map { prediction -> String in
            var word = prediction.0

            if word.hasPrefix("▁") {
                word.removeFirst()
            }

            if prediction.1 != "0", let punctuation = prediction.1.first {
                word.append(punctuation)
            }

            return word
        }

        return tokens.joined(separator: " ")
    }
}
