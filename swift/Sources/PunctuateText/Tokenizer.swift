import Foundation

/// Optimized tokenizer for BERT-based models with caching support
public final class Tokenizer {
    private let vocab: [String: Int]
    private let unkTokenId: Int
    private let clsTokenId: Int
    private let sepTokenId: Int
    private let padTokenId: Int
    public let maxSequenceLength: Int = 256

    // Cache for tokenized words to avoid re-tokenization
    private var tokenCache = [String: [String]]()
    private let cacheLimit = 5000

    /// Initialize using vocabulary from bundle resources
    public init() throws {
        guard let vocabURL = Bundle.module.url(forResource: "vocab", withExtension: "txt") else {
            throw NSError(
                domain: "PunctuateText", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "vocab.txt not found in bundle"])
        }
        let content = try String(contentsOf: vocabURL, encoding: .utf8)

        // Initialize vocabulary
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(35000)  // Pre-allocate for typical vocab size

        content.enumerateLines { line, _ in
            vocab[line] = vocab.count
        }

        self.vocab = vocab
        self.unkTokenId = vocab["[UNK]"] ?? 100
        self.clsTokenId = vocab["[CLS]"] ?? 101
        self.sepTokenId = vocab["[SEP]"] ?? 102
        self.padTokenId = vocab["[PAD]"] ?? 0
    }

    // Optimized tokenizeWord with caching
    private func tokenizeWord(_ word: String) -> [String] {
        // Check cache first
        if let cached = tokenCache[word] {
            return cached
        }

        let lowerWord = word.lowercased()

        // Fast path: check if whole word exists
        if vocab[lowerWord] != nil {
            let result = [lowerWord]
            storeInCache(result, for: word)
            return result
        }

        // WordPiece tokenization with optimizations
        var tokens: [String] = []
        var start = 0
        let chars = Array(lowerWord)
        let length = chars.count

        while start < length {
            var found = false
            var end = length

            while start < end {
                let substr: String
                if start == 0 {
                    substr = String(chars[start ..< end])
                } else {
                    substr = "##" + String(chars[start ..< end])
                }

                if vocab[substr] != nil {
                    tokens.append(substr)
                    start = end
                    found = true
                    break
                }
                end -= 1
            }

            if !found {
                let result = ["[UNK]"]
                storeInCache(result, for: word)
                return result
            }
        }

        storeInCache(tokens, for: word)
        return tokens
    }

    private func storeInCache(_ tokens: [String], for key: String) {
        if tokenCache.count >= cacheLimit {
            tokenCache.removeAll(keepingCapacity: true)
        }
        tokenCache[key] = tokens
    }

    /// Encode a batch of word sequences
    public func encodeBatch(_ wordsList: [[String]]) -> [(
        inputIds: [Int32], attentionMask: [Int32], wordPieces: [String], wordIds: [Int?]
    )] {
        return wordsList.map { words in
            encode(words)
        }
    }

    /// Encode a single sequence of words
    public func encode(_ words: [String]) -> (
        inputIds: [Int32], attentionMask: [Int32], wordPieces: [String], wordIds: [Int?]
    ) {
        let encoded = encodeChunk(words: words, startIndex: 0, maximumTokens: maxSequenceLength)
        if encoded.consumedWords != words.count {
            assertionFailure(
                "Tokenizer.encode truncated input; use encodeChunk for long sequences.")
        }
        return (encoded.inputIds, encoded.attentionMask, encoded.wordPieces, encoded.wordIds)
    }

    /// Encode a slice of words ensuring it fits within the model's maximum sequence length.
    @inline(__always)
    internal func encodeChunk(words: [String], startIndex: Int, maximumTokens: Int) -> (
        inputIds: [Int32], attentionMask: [Int32], wordPieces: [String], wordIds: [Int?],
        consumedWords: Int
    ) {
        let limit = maximumTokens
        guard limit >= 2 else {
            return ([], [], [], [], 0)
        }

        var tokens = ["[CLS]"]
        var wordIds: [Int?] = [nil]

        tokens.reserveCapacity(limit)
        wordIds.reserveCapacity(limit)

        var consumedWords = 0
        var currentIndex = startIndex

        while currentIndex < words.count {
            let localWordIndex = consumedWords
            var wordTokens = tokenizeWord(words[currentIndex])

            // Ensure at least space for [SEP]
            if tokens.count + wordTokens.count + 1 > limit {
                if consumedWords == 0 {
                    // Fallback: truncate overly long words so we can make progress
                    let available = max(limit - tokens.count - 1, 0)
                    if available == 0 {
                        wordTokens = ["[UNK]"]
                    } else if wordTokens.count > available {
                        wordTokens = Array(wordTokens.prefix(available))
                        if wordTokens.isEmpty {
                            wordTokens = ["[UNK]"]
                        }
                    }

                    tokens.append(contentsOf: wordTokens)
                    for _ in 0 ..< wordTokens.count {
                        wordIds.append(localWordIndex)
                    }
                    consumedWords += 1
                    currentIndex += 1
                }
                break
            }

            tokens.append(contentsOf: wordTokens)
            for _ in 0 ..< wordTokens.count {
                wordIds.append(localWordIndex)
            }
            consumedWords += 1
            currentIndex += 1
        }

        // Append [SEP] token
        tokens.append("[SEP]")
        wordIds.append(nil)

        let tokenCountBeforePadding = tokens.count

        // Ensure final token is [SEP] while respecting the limit
        if tokens.count > limit {
            var trimmedTokens = Array(tokens.prefix(limit - 1))
            trimmedTokens.append("[SEP]")
            tokens = trimmedTokens

            var trimmedWordIds = Array(wordIds.prefix(limit - 1))
            trimmedWordIds.append(nil)
            wordIds = trimmedWordIds
        }

        // Pad up to limit
        if tokens.count < limit {
            let padCount = limit - tokens.count
            tokens.append(contentsOf: Array(repeating: "[PAD]", count: padCount))
            wordIds.append(contentsOf: Array(repeating: nil, count: padCount))
        }

        // Convert to IDs efficiently
        var inputIds = [Int32](repeating: Int32(padTokenId), count: limit)
        var attentionMask = [Int32](repeating: 0, count: limit)

        let effectiveCount = min(tokenCountBeforePadding, limit)
        for i in 0 ..< effectiveCount {
            inputIds[i] = Int32(vocab[tokens[i]] ?? unkTokenId)
            attentionMask[i] = 1
        }

        return (inputIds, attentionMask, tokens, wordIds, consumedWords)
    }
}
