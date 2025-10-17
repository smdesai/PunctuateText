import CoreML
import Foundation

/// Public factory function to create FullStopPunctuation instances
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
public func createFullStopPunctuation(enableLogging: Bool = false) throws -> FullStopPunctuation {

    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine

    let coreMLModel = try loadFullStopPunctuationModel(configuration: config)
    return try FullStopPunctuation(model: coreMLModel, enableLogging: enableLogging)
}

/// Factory functions to create FullStopPunctuation model instances
@available(macOS 15.0, iOS 17.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
internal func loadFullStopPunctuationModel(
    configuration: MLModelConfiguration = MLModelConfiguration()
) throws -> MLModel {
    if let modelURL = Bundle.module.url(
        forResource: "FullStopPunctuation", withExtension: "mlmodelc")
    {
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    if let modelURL = Bundle.module.url(
        forResource: "FullStopPunctuation", withExtension: "mlmodelc", subdirectory: "Resources")
    {
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    throw NSError(
        domain: "PunctuateText", code: 6,
        userInfo: [NSLocalizedDescriptionKey: "FullStopPunctuation model not found in bundle"])
}
