import CoreML
import Foundation
import SegmentTextKit
import SentencePieceWrapper

@available(macOS 15.0, iOS 17.0, *)
public class FullStopPunctuation {
    private let model: MLModel
    private let tokenizer: SentencePieceTokenizer

    // Label mapping for XLM-RoBERTa punctuation model
    private let id2label = [
        0: "0",
        1: ".",
        2: ",",
        3: "?",
        4: "-",
        5: ":",
    ]

    public init(model: MLModel) throws {
        self.model = model

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

        // Warm up the model with a dummy prediction
        print("  Warming up model...")
        let warmupStart = CFAbsoluteTimeGetCurrent()
        _ = try? self.predictChunk("test")
        let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart
        print("  Model warmup time: \(String(format: "%.3f", warmupTime)) seconds")
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

        // Create MLMultiArrays
        let inputArray = try MLMultiArray(shape: [1, 512], dataType: .float32)
        let maskArray = try MLMultiArray(shape: [1, 512], dataType: .float32)

        // Fill arrays
        for i in 0 ..< 512 {
            inputArray[i] = NSNumber(value: inputIds[i])
            maskArray[i] = NSNumber(value: attentionMask[i])
        }

        // Construct CoreML feature provider
        let featureInputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ])

        // Make prediction
        let prediction = try model.prediction(from: featureInputs)

        guard let logits = prediction.featureValue(for: "output")?.multiArrayValue else {
            throw SegmentTextError.modelNotFound("FullStopPunctuation output tensor")
        }

        // Process output for each token (excluding special tokens)
        var results: [(entity: String, score: Float, word: String)] = []

        // Skip CLS token at position 0
        for i in 1 ..< tokens.count + 1 {
            // Skip if we hit SEP token or padding
            if i >= tokens.count + 1 || inputIds[i] == 2 || inputIds[i] == 1 {
                break
            }

            // Get predictions for this token position
            var maxScore: Float = -Float.infinity
            var maxClass = 0

            for classIdx in 0 ..< 6 {
                let idx: [NSNumber] = [0, NSNumber(value: i), NSNumber(value: classIdx)]
                let score = logits[idx].floatValue
                if score > maxScore {
                    maxScore = score
                    maxClass = classIdx
                }
            }

            // Apply softmax to get probability
            var expSum: Float = 0
            for classIdx in 0 ..< 6 {
                let idx: [NSNumber] = [0, NSNumber(value: i), NSNumber(value: classIdx)]
                expSum += exp(logits[idx].floatValue - maxScore)
            }
            let probability = exp(0) / expSum

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

            print("Processing chunk: \(String(text.prefix(50)))...")

            let tokenPredictions = try predictChunk(text)
            print("Number of predictions: \(tokenPredictions.count)")

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
