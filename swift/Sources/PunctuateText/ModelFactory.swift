import CoreML
import Foundation

private var cachedDefaultPunctuateModel: MLModel?

/// Factory functions to create model instances
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
internal func loadPunctuateModel() throws -> MLModel {
    if let cached = cachedDefaultPunctuateModel {
        return cached
    }

    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine

    let model = try makePunctuateModel(configuration: config)
    cachedDefaultPunctuateModel = model
    return model
}

@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
internal func loadPunctuateModel(configuration: MLModelConfiguration) throws -> MLModel {
    return try makePunctuateModel(configuration: configuration)
}

@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
private func makePunctuateModel(configuration: MLModelConfiguration) throws -> MLModel {
    if let modelURL = Bundle.module.url(forResource: "Punctuate", withExtension: "mlmodelc") {
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    if let modelURL = Bundle.module.url(
        forResource: "Punctuate", withExtension: "mlmodelc", subdirectory: "Resources")
    {
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    throw NSError(
        domain: "PunctuateText", code: 7,
        userInfo: [NSLocalizedDescriptionKey: "Punctuate model not found in bundle"])
}
